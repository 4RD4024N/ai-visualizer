"""
Duygu başlığı: ses modelinin 128 boyutlu embedding'i → (valence, arousal).

Veri: DEAM (MediaEval 2013–2015), 1802 şarkı, 15. saniyeden itibaren her 0,5 sn
için insanların işaretlediği duygu değerleri (−1…1):
  valence : olumsuz (hüzünlü / öfkeli)  ↔  olumlu (neşeli / huzurlu)
  arousal : sakin                        ↔  enerjik

Ses modeli DONUK kalır (embedding değişmez → rüya modu / emb_stats bozulmaz);
sadece küçük bir MLP eğitilir. Doğrulama şarkıları eğitimde hiç görülmez.

Kullanım:
  python train/prepare.py --zip D:/ai-visualizer-work/data/deam/DEAM_audio.zip --out D:/ai-visualizer-work/features/deam
  python train/emotion.py
Çıktı: models/emotion_head.npz  (numpy ile çalışır, canlıda ONNX gerekmez)
"""

import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_features import SR, HOP, N_MELS, FRAMES  # noqa: E402
import train as T  # noqa: E402

WORK = os.environ.get("AIVIS_WORK", r"D:\ai-visualizer-work")
DYN = "annotations/annotations averaged per song/dynamic (per second annotations)/{}.csv"
SHIFTS = (0, 8, 16, 24)          # canlıdaki çoklu hizalama ile aynı aralık (0–384 ms geri)


def load_labels(zip_path):
    z = zipfile.ZipFile(zip_path)
    out = {}
    for dim in ("valence", "arousal"):
        lines = z.read(DYN.format(dim)).decode("utf-8").splitlines()
        cols = [int(c.replace("sample_", "").replace("ms", "")) for c in lines[0].split(",")[1:]]
        for line in lines[1:]:
            p = line.split(",")
            sid = int(float(p[0]))
            for t, v in zip(cols, p[1:]):
                if v.strip():
                    out.setdefault(sid, {}).setdefault(t, [None, None])[0 if dim == "valence" else 1] = float(v)
    return {sid: {t: va for t, va in d.items() if None not in va} for sid, d in out.items()}


class Head(nn.Module):
    def __init__(self, d=128, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(h, h), nn.GELU(), nn.Dropout(0.3), nn.Linear(h, 2))

    def forward(self, x):
        return self.net(x)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=os.path.join(WORK, "features", "deam"))
    ap.add_argument("--labels", default=os.path.join(WORK, "data", "deam", "DEAM_Annotations.zip"))
    ap.add_argument("--ckpt", default=os.path.join(WORK, "checkpoints", "audio_latent_best.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    labels = load_labels(args.labels)
    idx = json.load(open(os.path.join(args.feats, "index.json"), encoding="utf-8"))
    feats = np.fromfile(os.path.join(args.feats, "feats.bin"), np.float16).reshape(-1, N_MELS)

    # Pencere listesi: (şarkı, bitiş karesi, etiket)
    songs, ends, ys = [], [], []
    for tr in idx["tracks"]:
        sid = int(os.path.splitext(os.path.basename(tr["name"]))[0])
        for t_ms, va in labels.get(sid, {}).items():
            end = int(round(t_ms / 1000 * SR / HOP))
            if FRAMES + max(SHIFTS) <= end <= tr["length"]:
                songs.append(sid); ends.append(tr["start"] + end); ys.append(va)
    songs, ends, ys = np.array(songs), np.array(ends), np.array(ys, np.float32)
    uniq = np.unique(songs)
    rng = np.random.default_rng(args.seed)
    val_songs = set(rng.choice(uniq, len(uniq) * 15 // 100, replace=False).tolist())
    is_val = np.array([s in val_songs for s in songs])
    print(f"{len(uniq)} şarkı, {len(ys)} etiketli an · doğrulama {len(val_songs)} şarkı")

    # Donuk encoder ile embedding'ler (her an için SHIFTS kadar hizalama)
    ck = torch.load(args.ckpt, map_location=dev)
    enc = T.AudioEncoder(ck["norm_scale"]).to(dev)
    enc.load_state_dict(ck["enc"])
    enc.eval()
    F_gpu = torch.from_numpy(feats).to(dev)
    ar = torch.arange(FRAMES, device=dev)
    E = np.zeros((len(ys), len(SHIFTS), 128), np.float32)
    with torch.no_grad():
        for i in range(0, len(ys), 2048):
            e_end = torch.from_numpy(ends[i:i + 2048]).to(dev)
            for j, k in enumerate(SHIFTS):
                ix = (e_end - k - FRAMES)[:, None] + ar[None, :]
                x = F_gpu[ix].float().transpose(1, 2).unsqueeze(1)
                E[i:i + 2048, j] = enc(x).float().cpu().numpy()
    print(f"embedding'ler hazır: {E.shape}")

    mu = E[~is_val].reshape(-1, 128).mean(0)
    sd = E[~is_val].reshape(-1, 128).std(0) + 1e-6
    En = (E - mu) / sd
    Xtr = torch.from_numpy(En[~is_val]).to(dev)                  # (N, S, 128)
    Ytr = torch.from_numpy(ys[~is_val]).to(dev)
    Xva = torch.from_numpy(En[is_val].mean(1)).to(dev)            # canlıdaki gibi: hizalama ortalaması
    Yva = ys[is_val]

    head = Head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state = -1e9, None
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            s = torch.randint(0, len(SHIFTS), (len(b),), device=dev)
            x = Xtr[b, s] + 0.05 * torch.randn_like(Xtr[b, s])
            loss = nn.functional.mse_loss(head(x), Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            p = head(Xva).cpu().numpy()
        rv = r2(Yva, p)
        score = rv.mean()
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if (ep + 1) % 10 == 0:
            cc = [np.corrcoef(Yva[:, d], p[:, d])[0, 1] for d in range(2)]
            print(f"epoch {ep + 1:3d}  kayıp {loss.item():.4f}  val R² valence {rv[0]:.3f} arousal {rv[1]:.3f}  "
                  f"korelasyon {cc[0]:.3f} / {cc[1]:.3f}", flush=True)

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        p = head(Xva).cpu().numpy()
    rv = r2(Yva, p)
    cc = [float(np.corrcoef(Yva[:, d], p[:, d])[0, 1]) for d in range(2)]
    # Şarkı düzeyinde: her şarkının ortalama tahmini vs ortalama etiketi
    vs = songs[is_val]
    sp = np.array([p[vs == s].mean(0) for s in np.unique(vs)])
    sy = np.array([Yva[vs == s].mean(0) for s in np.unique(vs)])
    rs = r2(sy, sp)
    print(f"EN İYİ · an düzeyi R² valence {rv[0]:.3f} arousal {rv[1]:.3f} · korelasyon {cc[0]:.3f} / {cc[1]:.3f} · "
          f"şarkı düzeyi R² {rs[0]:.3f} / {rs[1]:.3f}")

    sd_ = {k: v.numpy() for k, v in best_state.items()}
    out = os.path.join(ROOT, "models", "emotion_head.npz")
    np.savez(out, mu=mu, sd=sd,
             W1=sd_["net.0.weight"], b1=sd_["net.0.bias"], W2=sd_["net.3.weight"], b2=sd_["net.3.bias"],
             W3=sd_["net.6.weight"], b3=sd_["net.6.bias"],
             y_mean=ys.mean(0), y_std=ys.std(0),
             metrics=np.array([rv[0], rv[1], cc[0], cc[1], rs[0], rs[1]], np.float32))
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
