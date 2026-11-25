"""
Ses → 8 boyutlu görsel latent modeli.

Yöntem: zaman-contrastive self-supervised öğrenme (etiket gerekmez).
  • Aynı parçanın birbirine yakın iki anı (0.5–4 sn arayla) → "pozitif çift"
  • Batch'teki diğer parçalar → "negatif"
  • Kayıp: simetrik InfoNCE (NT-Xent)
  • Kararlılık: 16–256 ms kaymış iki pencerenin embedding'i neredeyse aynı olmalı
    (fark / şarkılar arası varyans oranı cezalandırılır). Görüntü doğrudan
    embedding'den üretildiği için bu, yumuşatmasız akıcı hareketi sağlar.
    Mimari de buna göre: max-pool yerine blur-pool (aliasing yok), zamanda
    sert maksimum yerine logsumexp.
Sonuç: aynı şarkının anları birbirine yakın, farklı şarkılar uzak düşen bir embedding.
Embedding'in en çok varyans taşıyan 8 PCA yönü (beyazlatılmış) → CPPN latent'i.
Benzer müzik → benzer görsel dünya, farklı müzik → farklı dünya.

Kullanım:
  python train/train.py                       # WORK/features altındaki tüm veri setleri
  python train/train.py --steps 12000 --data D:/ai-visualizer-work/features/fma
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from audio_features import SR, N_FFT, HOP, N_MELS, FRAMES, FMIN, FMAX  # noqa: E402

WORK = os.environ.get("AIVIS_WORK", r"D:\ai-visualizer-work")
LATENT = 8
EMB = 128


# ── Model ───────────────────────────────────────────────────────────────────

class BlurPool(nn.Module):
    """Max-blur-pool (Zhang 2019): maks (adım 1) → [1,2,1]² bulanıklık → adım 2.
    Girdi birkaç kare kaydığında çıktı zıplamaz (shift-invariance)."""

    def __init__(self, ch):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        k = (k[:, None] * k[None, :]) / 16.0
        self.register_buffer("k", k[None, None].repeat(ch, 1, 1, 1))
        self.ch = ch
        self.mp = nn.MaxPool2d(2, stride=1)

    def forward(self, x):
        x = F.pad(self.mp(F.pad(x, (0, 1, 0, 1), mode="replicate")), (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, self.k, stride=2, groups=self.ch)


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        BlurPool(cout),
    )


class AudioEncoder(nn.Module):
    """(B, 1, 64 mel, 128 kare) log-mel → (B, 128) embedding."""

    def __init__(self, norm_scale=1.0):
        super().__init__()
        self.register_buffer("norm_scale", torch.tensor(float(norm_scale)))
        self.features = nn.Sequential(conv_block(1, 32), conv_block(32, 64),
                                      conv_block(64, 128), conv_block(128, 256))
        self.fc = nn.Linear(512, EMB)

    def forward(self, x):
        # Pencere başına ortalama çıkar → ses seviyesinden bağımsız (sistem sesi değişse de aynı)
        x = (x - x.mean(dim=(2, 3), keepdim=True)) / self.norm_scale
        h = self.features(x).mean(dim=2)                    # frekans ekseninde ortalama → (B, 256, T')
        lse = torch.logsumexp(h, dim=2) - math.log(h.shape[2])
        h = torch.cat([h.mean(dim=2), lse], 1)              # zaman: ortalama + yumuşak maksimum
        return self.fc(h)


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(EMB, 256), nn.ReLU(inplace=True), nn.Linear(256, 128))

    def forward(self, e):
        return F.normalize(self.net(e), dim=1)


class LatentExport(nn.Module):
    """ONNX için: log-mel → (latent[8], embedding[128]). PCA + beyazlatma gömülü."""

    def __init__(self, enc, mu, comps, scale):
        super().__init__()
        self.enc = enc
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("proj", torch.as_tensor(comps.T / scale[None, :], dtype=torch.float32))

    def forward(self, logmel):
        e = self.enc(logmel)
        return (e - self.mu) @ self.proj, e


# ── Veri ────────────────────────────────────────────────────────────────────

def load_datasets(dirs, device):
    chunks, tracks, offset = [], [], 0
    for d in dirs:
        idx_path = os.path.join(d, "index.json")
        if not os.path.exists(idx_path):
            continue
        with open(idx_path, encoding="utf-8") as f:
            idx = json.load(f)
        arr = np.fromfile(os.path.join(d, "feats.bin"), dtype=np.float16)
        arr = arr[: idx["frames"] * N_MELS].reshape(-1, N_MELS)
        chunks.append(arr)
        for t in idx["tracks"]:
            if t["length"] >= FRAMES + 32:
                tracks.append((offset + t["start"], t["length"], os.path.basename(d)))
        offset += len(arr)
        print(f"  {d}: {len(idx['tracks'])} parça, {len(arr) * HOP / SR / 3600:.1f} saat")
    if not chunks:
        raise SystemExit("Veri bulunamadı. Önce train/prepare.py çalıştır ya da canlı kayıt yap.")
    feats = torch.from_numpy(np.concatenate(chunks)).to(device)            # (N, 64) float16, GPU'da
    return feats, tracks


class PairSampler:
    def __init__(self, feats, tracks, device):
        self.feats = feats
        self.start = torch.tensor([t[0] for t in tracks], device=device)
        self.length = torch.tensor([t[1] for t in tracks], device=device)
        self.n = len(tracks)
        self.device = device
        self.ar = torch.arange(FRAMES, device=device)

    def crop(self, track_idx, t):
        idx = self.start[track_idx, None] + t[:, None] + self.ar[None, :]   # (B, FRAMES)
        x = self.feats[idx].float()                                          # (B, FRAMES, 64)
        return x.transpose(1, 2).unsqueeze(1)                                # (B, 1, 64, FRAMES)

    def sample(self, batch, shift=False):
        ti = torch.randperm(self.n, device=self.device)[:batch]
        L = self.length[ti]
        maxt = (L - FRAMES).clamp(min=1)
        t1 = (torch.rand(len(ti), device=self.device) * maxt).long()
        if shift:
            # t1'den 1–16 kare (16–256 ms) kaymış pencere → kararlılık çifti
            k = torch.randint(1, 17, (len(ti),), device=self.device)
            t3 = torch.minimum(t1 + k, maxt - 1)
            t1 = torch.minimum(t1, t3 - 1).clamp(min=0)
        delta = torch.randint(32, 256, (len(ti),), device=self.device)       # 0.5–4 sn
        sign = torch.where(torch.rand(len(ti), device=self.device) < 0.5, -1, 1)
        t2 = (t1 + sign * delta).clamp(min=0)
        t2 = torch.minimum(t2, maxt - 1)
        if shift:
            return self.crop(ti, t1), self.crop(ti, t2), self.crop(ti, t3)
        return self.crop(ti, t1), self.crop(ti, t2)


def augment(x, seed=None):
    """seed verilirse aynı bozulma tekrar üretilir (kararlılık çifti için)."""
    B = x.shape[0]
    dev = x.device
    g = torch.Generator(device=dev)
    g.manual_seed(int(torch.randint(0, 2**62, (1,)).item()) if seed is None else seed)
    rand = lambda *sh: torch.rand(*sh, device=dev, generator=g)
    randint = lambda lo, hi, sh: torch.randint(lo, hi, sh, device=dev, generator=g)
    x = x + (rand(B, 1, 1, 1) * 4 - 2)                                       # kazanç ±8.7 dB
    tilt = torch.linspace(-1, 1, N_MELS, device=dev)[None, None, :, None]
    x = x + tilt * (rand(B, 1, 1, 1) * 2 - 1)                                # EQ eğimi
    noisy = rand(B, 1, 1, 1) < 0.4
    floor = x.mean(dim=(2, 3), keepdim=True) - 3 - rand(B, 1, 1, 1) * 4
    nz = torch.randn(x.shape, device=dev, generator=g)
    x = torch.where(noisy, torch.logaddexp(x, floor + 0.5 * nz), x)          # gürültü tabanı
    fill = x.mean(dim=(2, 3), keepdim=True)
    for _ in range(2):                                                       # SpecAugment
        w = randint(0, 9, (B, 1, 1, 1))
        f0 = (rand(B, 1, 1, 1) * (N_MELS - w)).long()
        fi = torch.arange(N_MELS, device=dev)[None, None, :, None]
        x = torch.where((fi >= f0) & (fi < f0 + w), fill, x)
        w = randint(0, 21, (B, 1, 1, 1))
        t0 = (rand(B, 1, 1, 1) * (FRAMES - w)).long()
        tj = torch.arange(FRAMES, device=dev)[None, None, None, :]
        x = torch.where((tj >= t0) & (tj < t0 + w), fill, x)
    return x


def stability(e1, e3):
    """Kaymış pencere farkı / batch'teki toplam varyans (ölçekten bağımsız).
    Farkı küçültürken şarkılar arası ayrımı da korumaya zorlar."""
    diff = (e1 - e3).pow(2).sum(1).mean()
    var = e1.var(0).sum() + e3.var(0).sum()
    return diff / (var + 1e-6)


def nt_xent(z1, z2, tau):
    logits = z1 @ z2.T / tau
    y = torch.arange(len(z1), device=z1.device)
    loss = (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2
    acc = (logits.argmax(1) == y).float().mean()
    return loss, acc


# ── Eğitim ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", help="özellik klasörleri (varsayılan: WORK/features/*)")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stab", type=float, default=4.0, help="kararlılık kaybı ağırlığı")
    ap.add_argument("--out", default=os.path.join(ROOT, "models"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {dev}  ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'})")

    dirs = args.data or sorted(glob.glob(os.path.join(WORK, "features", "*")))
    print("Veri setleri:")
    feats, tracks = load_datasets(dirs, dev)

    # Parça bazında train/val ayrımı
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tracks))
    n_val = max(64, len(tracks) // 20)
    val_tracks = [tracks[i] for i in perm[:n_val]]
    tr_tracks = [tracks[i] for i in perm[n_val:]]
    batch = min(args.batch, len(tr_tracks))
    print(f"Parça: {len(tr_tracks)} eğitim, {len(val_tracks)} doğrulama, batch {batch}")

    train_s = PairSampler(feats, tr_tracks, dev)
    val_s = PairSampler(feats, val_tracks, dev)

    # Normalizasyon ölçeği (pencere ortalaması çıkarıldıktan sonraki std)
    with torch.no_grad():
        x, _ = train_s.sample(batch)
        norm_scale = float((x - x.mean(dim=(2, 3), keepdim=True)).std())
    print(f"norm_scale = {norm_scale:.3f}")

    enc = AudioEncoder(norm_scale).to(dev)
    proj = Projector().to(dev)
    params = list(enc.parameters()) + list(proj.parameters())
    print(f"Parametre: {sum(p.numel() for p in enc.parameters()) / 1e6:.2f} M (encoder)")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    warm = min(300, args.steps // 10)

    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm)))

    use_amp = dev.type == "cuda"
    t0 = time.time()
    best_val, best_score, ema_loss, ema_acc, ema_stab = 0.0, -1e9, None, None, 0.0
    ckpt_dir = os.path.join(WORK, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "audio_latent_best.pt")

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        enc.train(); proj.train()
        a, b, c = train_s.sample(batch, shift=True)
        sd = int(torch.randint(0, 2**62, (1,)).item())
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            e1 = enc(augment(a, sd))
            e3 = enc(augment(c, sd))                  # aynı bozulma, 16–256 ms kayık
            z1 = proj(e1)
            z2 = proj(enc(augment(b)))
        loss_c, acc = nt_xent(z1.float(), z2.float(), args.tau)
        loss_s = stability(e1.float(), e3.float())
        loss = loss_c + args.stab * loss_s
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        ema_loss = loss_c.item() if ema_loss is None else ema_loss * 0.98 + loss_c.item() * 0.02
        ema_stab = loss_s.item() if step == 0 else ema_stab * 0.98 + loss_s.item() * 0.02
        ema_acc = acc.item() if ema_acc is None else ema_acc * 0.98 + acc.item() * 0.02

        if (step + 1) % 250 == 0 or step == args.steps - 1:
            enc.eval(); proj.eval()
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                accs, stabs = [], []
                for _ in range(4):
                    va, vb, vc = val_s.sample(min(batch, val_s.n), shift=True)
                    ea = enc(va).float()
                    _, vacc = nt_xent(proj(ea).float(), proj(enc(vb)).float(), args.tau)
                    accs.append(vacc.item())
                    stabs.append(stability(ea, enc(vc).float()).item())
            vacc = float(np.mean(accs))
            vjit = math.sqrt(float(np.mean(stabs)))   # kaymış pencere sapması / şarkılar arası yayılım
            el = time.time() - t0
            print(f"adım {step + 1:5d}/{args.steps}  kayıp {ema_loss:.3f}  kararlılık {ema_stab:.4f}  "
                  f"eğitim-acc {ema_acc:.3f}  val-acc {vacc:.3f}  val-titreme {vjit:.3f}  "
                  f"lr {lr_at(step):.2e}  {el / 60:.1f} dk", flush=True)
            # Seçim: hem ayırt edicilik hem kararlılık (titreme cezası)
            score = vacc - vjit
            if score >= best_score:
                best_score, best_val = score, vacc
                torch.save({"enc": enc.state_dict(), "norm_scale": norm_scale, "step": step + 1,
                            "val_acc": vacc, "val_jitter": vjit}, ckpt_path)

    print(f"En iyi val-acc: {best_val:.3f}  (şans seviyesi ≈ {1 / min(batch, val_s.n):.4f})")

    # ── PCA → 8 boyutlu latent ──
    ck = torch.load(ckpt_path, map_location=dev)
    enc.load_state_dict(ck["enc"])
    enc.eval()
    all_s = PairSampler(feats, tracks, dev)
    embs = []
    with torch.no_grad():
        for k in range(6):                                   # her parçadan 6 eşit aralıklı pencere
            for i in range(0, all_s.n, 1024):
                ti = torch.arange(i, min(i + 1024, all_s.n), device=dev)
                t = ((all_s.length[ti] - FRAMES).clamp(min=0).float() * (k + 0.5) / 6).long()
                embs.append(enc(all_s.crop(ti, t)).float().cpu())
    E = torch.cat(embs).numpy()
    mu = E.mean(0)
    U, S, Vt = np.linalg.svd(E - mu, full_matrices=False)
    ev = (S ** 2) / (len(E) - 1)
    comps = Vt[:LATENT]
    scale = np.sqrt(ev[:LATENT]) / 0.9                       # beyazlat, std ≈ 0.9
    explained = ev[:LATENT].sum() / ev.sum()
    print(f"PCA: ilk {LATENT} bileşen varyansın %{explained * 100:.1f}'ini açıklıyor")

    # ── ONNX export ──
    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "audio_latent.onnx")
    export = LatentExport(enc.float().cpu(), mu, comps, scale).eval()
    dummy = torch.randn(1, 1, N_MELS, FRAMES)
    kw = dict(input_names=["logmel"], output_names=["latent", "embedding"],
              dynamic_axes={"logmel": {0: "batch"}, "latent": {0: "batch"}, "embedding": {0: "batch"}},
              opset_version=17)
    try:
        torch.onnx.export(export, dummy, onnx_path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(export, dummy, onnx_path, **kw)

    cfg = {"sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "frames": FRAMES,
           "fmin": FMIN, "fmax": FMAX, "latent": LATENT, "emb": EMB,
           "val_acc": best_val, "val_jitter": ck.get("val_jitter"), "pca_explained": float(explained), "step": ck["step"],
           "tracks": len(tracks), "datasets": [os.path.basename(d) for d in dirs]}
    with open(os.path.join(args.out, "audio_latent.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Doğrulama: ONNX çıktısı PyTorch ile aynı mı?
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        o = sess.run(["latent"], {"logmel": dummy.numpy()})[0]
        with torch.no_grad():
            r = export(dummy)[0].numpy()
        print(f"ONNX kontrol: maks fark {np.abs(o - r).max():.2e}")
    except ImportError:
        pass
    print(f"Kaydedildi: {onnx_path}")


if __name__ == "__main__":
    main()
