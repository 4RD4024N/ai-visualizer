"""
Eğitilmiş modellerin doğrulama metriklerini yazdırır (yalnızca numpy — torch gerekmez).

  python tools/model_report.py            # tablo
  python tools/model_report.py --md       # README için Markdown tablo

Metrikler eğitim sırasında, eğitimde hiç görülmeyen doğrulama parçaları/şarkıları
üzerinde hesaplanıp model dosyalarına yazılır:
  models/audio_latent.json  ← train/train.py   (contrastive doğruluk, titreme, PCA)
  models/emotion_head.npz   ← train/emotion.py (DEAM valence/arousal: R², korelasyon)
"""

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")


def collect(models_dir=MODELS):
    rows = []
    p = os.path.join(models_dir, "audio_latent.json")
    if os.path.exists(p):
        m = json.load(open(p, encoding="utf-8"))
        n = f"{m.get('tracks', '?')} parça, {', '.join(m.get('datasets', []))}"
        rows += [
            ("Ses modeli", "Contrastive doğruluk (val)", m.get("val_acc"),
             f"bir anın eşini batch'teki diğer parçalar arasından bulma (eğitim: {n})"),
            ("Ses modeli", "Titreme (val)", m.get("val_jitter"),
             "16–256 ms kaymış pencerede sapma / şarkılar arası yayılım — düşük = akıcı"),
            ("Ses modeli", "PCA açıklanan varyans", m.get("pca_explained"),
             "8 boyutlu latent'in embedding varyansındaki payı"),
        ]
    p = os.path.join(models_dir, "emotion_head.npz")
    if os.path.exists(p):
        r2v, r2a, cv, ca, sv, sa = (float(x) for x in np.load(p)["metrics"])
        rows += [
            ("Duygu başlığı", "R² valence (an)", r2v, "DEAM, görülmemiş şarkılar, 0,5 sn çözünürlük"),
            ("Duygu başlığı", "R² arousal (an)", r2a, ""),
            ("Duygu başlığı", "Korelasyon valence / arousal", (cv, ca), ""),
            ("Duygu başlığı", "R² valence (şarkı)", sv, "şarkı ortalaması tahmini vs etiketi"),
            ("Duygu başlığı", "R² arousal (şarkı)", sa, ""),
        ]
    return rows


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, tuple):
        return " / ".join(f"{x:.3f}" for x in v)
    return f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    rows = collect()
    if not rows:
        print("models/ altında metrik içeren model yok.")
        return
    if args.md:
        print("| Model | Metrik | Değer | Not |\n|---|---|---|---|")
        for r in rows:
            print(f"| {r[0]} | {r[1]} | {fmt(r[2])} | {r[3]} |")
    else:
        for r in rows:
            print(f"{r[0]:<14} {r[1]:<30} {fmt(r[2]):>14}  {r[3]}")


if __name__ == "__main__":
    main()
