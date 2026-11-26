"""
Eğitilmiş ses modelinin 128 boyutlu embedding'i için beyazlatma istatistikleri.

Rüya modu (neural_bg.DreamMapper) embedding'i doğrudan CPPN ağırlıklarına
çevirir. Ham embedding merkezlenmemiş ve bileşenleri ilişkili olduğundan
önce PCA ile beyazlatılır: her bileşen ≈ ortalama 0, varyans 1, ilişkisiz.

Kullanım (train.py'den sonra):
  python train/emb_stats.py --data work/features/fma
Çıktı: models/audio_latent_emb.npz  (mu[128], white[128,128])
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "audio_latent.onnx")
OUT = os.path.join(ROOT, "models", "audio_latent_emb.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="prepare.py çıktı klasörü")
    ap.add_argument("--n", type=int, default=6000, help="örnek kesit sayısı")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(os.path.splitext(MODEL)[0] + ".json", encoding="utf-8"))
    n_mels, frames = cfg["n_mels"], cfg["frames"]
    idx = json.load(open(os.path.join(args.data, "index.json"), encoding="utf-8"))
    feats = np.memmap(os.path.join(args.data, "feats.bin"), np.float16, "r").reshape(-1, n_mels)
    tracks = [t for t in idx["tracks"] if t["length"] > frames]

    rng = np.random.default_rng(args.seed)
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    embs = []
    for b in range(0, args.n, 256):
        batch = []
        for _ in range(min(256, args.n - b)):
            t = tracks[rng.integers(len(tracks))]
            s = t["start"] + rng.integers(t["length"] - frames)
            batch.append(feats[s:s + frames].T.astype(np.float32))
        x = np.stack(batch)[:, None]
        embs.append(sess.run(["embedding"], {"logmel": x})[0])
    E = np.concatenate(embs).astype(np.float64)

    mu = E.mean(0)
    cov = np.cov(E - mu, rowvar=False)
    ev, V = np.linalg.eigh(cov)
    ev = np.maximum(ev, ev.max() * 1e-4)          # çok küçük yönleri patlatma
    white = (V / np.sqrt(ev)).astype(np.float32)  # z = (e - mu) @ white
    Z = (E - mu) @ white
    np.savez(OUT, mu=mu.astype(np.float32), white=white)
    print(f"{len(E)} kesit · beyazlatılmış std: {Z.std(0).mean():.3f} · "
          f"ilk 8 yön varyansın %{100 * ev[::-1][:8].sum() / ev.sum():.1f}'i")
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
