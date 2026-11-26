"""
CLAP kavram istatistikleri: her kavramın "genel müzikte" ortalama benzerliği.

Bazı kavramlar (ör. "balloons", "cold blue light") her sese biraz benzer ve her
şarkıda üste çıkar (hubness). Canlıda ham benzerlik yerine z-skoru kullanılır:
  z = (benzerlik − kavram_ortalaması) / kavram_sapması
Böylece o şarkıya ÖZGÜ kavramlar seçilir.

Kullanım:  python train/concept_stats.py
Çıktı   :  models/clap_concepts.npz  (kavramlar: mean/std/text, ruh halleri: m_mean/m_std/m_text)
"""

import os
import subprocess
import sys
import zipfile

WORK = os.environ.get("AIVIS_WORK") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work")
os.environ["HF_HOME"] = os.path.join(WORK, "hf")   # sistem HF_HOME'u başka projeyi gösterebilir
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from concepts import CLAP_TEXT, MOOD_CLAP_TEXT  # noqa: E402

CLAP = "laion/larger_clap_music_and_speech"
SOURCES = [(os.path.join(WORK, "data", "deam", "DEAM_audio.zip"), 200),
           (os.path.join(WORK, "data", "fma_small.zip"), 200)]


def features(out):
    return torch.nn.functional.normalize(getattr(out, "pooler_output", out), dim=-1)


def main():
    from transformers import ClapModel, ClapProcessor
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClapModel.from_pretrained(CLAP).to(dev).eval()
    proc = ClapProcessor.from_pretrained(CLAP)
    with torch.no_grad():
        T = features(model.get_text_features(**proc(text=CLAP_TEXT, return_tensors="pt", padding=True).to(dev)))
        TM = features(model.get_text_features(**proc(text=MOOD_CLAP_TEXT, return_tensors="pt", padding=True).to(dev)))
    rng = np.random.default_rng(0)
    sims, msims = [], []
    for path, n in SOURCES:
        z = zipfile.ZipFile(path)
        names = [x for x in z.namelist() if x.endswith(".mp3")]
        for name in rng.choice(names, n, replace=False):
            pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ss", str(rng.integers(0, 18)), "-t", "10",
                                  "-f", "f32le", "-ac", "1", "-ar", "48000", "pipe:1"],
                                 input=z.read(name), capture_output=True).stdout
            x = np.frombuffer(pcm, np.float32)
            if len(x) < 48000 * 5:
                continue
            with torch.no_grad():
                A = features(model.get_audio_features(**proc(audio=[x], sampling_rate=48000,
                                                              return_tensors="pt").to(dev)))
            sims.append((A @ T.T)[0].cpu().numpy())
            msims.append((A @ TM.T)[0].cpu().numpy())
        print(f"{os.path.basename(path)}: {n} klip", flush=True)
    S, SM = np.stack(sims), np.stack(msims)
    out = os.path.join(ROOT, "models", "clap_concepts.npz")
    np.savez(out, mean=S.mean(0).astype(np.float32), std=(S.std(0) + 1e-4).astype(np.float32),
             text=T.cpu().numpy().astype(np.float32), model=np.array(CLAP),
             m_mean=SM.mean(0).astype(np.float32), m_std=(SM.std(0) + 1e-4).astype(np.float32),
             m_text=TM.cpu().numpy().astype(np.float32))
    order = np.argsort(-S.mean(0))[:5]
    print(f"{len(S)} klip · en 'genel' kavramlar (z-skoru bunları dengeler): {[CLAP_TEXT[i][18:] for i in order]}")
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
