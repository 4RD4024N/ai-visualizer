"""
Söz → ilk kare gecikme ölçümü (gerçek modellerle; kullanıcının makinesinde, venv ile).

  .venv/Scripts/python.exe tools/bench_latency.py --wav sarki.wav
  .venv/Scripts/python.exe tools/bench_latency.py --wav sarki.wav --legacy

SD-Turbo + TAESD + CLAP (diffusion_layer.DiffusionLayer) ve Whisper (lyrics.LyricsListener)
uygulamadaki gibi yüklenir; --wav dosyası gerçek zamanlı akıyormuş gibi (saatle) beslenir.
Ana döngü 60 Hz'de son kareyi "çizer"; söz satırının ilk karesi görülünce gecikme kaydedilir.

  --legacy : eski davranış (pencere 6 sn / adım 1,5 sn, Türkçede iki generate, söz önceliği
             yok, encoder yeniden kullanımı yok, diff_fast kapalı) → önce / sonra karşılaştırması.

"Söz anı" yaklaşığı uygulamadakiyle aynıdır (latency.py: pencereye yeni giren sesin ortası).
--wav verilmezse Whisper ÖLÇÜLMEZ: sentetik ses + 2 sn'de bir yapay söz satırı ile sadece
SD tarafı (sahne + kare) ölçülür ve bu açıkça yazılır.
"""

import argparse
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import latency  # noqa: E402


def load_wav(path):
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(1)
    except ImportError:
        from scipy.io import wavfile
        sr, x = wavfile.read(path)
        if x.dtype.kind == "i":
            x = x.astype(np.float32) / np.iinfo(x.dtype).max
        elif x.dtype.kind == "u":
            x = (x.astype(np.float32) - 128) / 128
        x = x.astype(np.float32)
        if x.ndim == 2:
            x = x.mean(1)
    return np.ascontiguousarray(x, np.float32), int(sr)


class WavTap:
    """AudioTap.latest_raw taklidi: saat ilerledikçe dosyanın son ~10 sn'si."""

    def __init__(self, x, sr, keep=10.0):
        self.x, self.sr, self.keep = x, sr, keep
        self.t0 = None

    def start(self):
        self.t0 = time.perf_counter()

    def pos(self):
        if self.t0 is None:
            return 0
        return min(len(self.x), int((time.perf_counter() - self.t0) * self.sr))

    def done(self):
        return self.t0 is not None and self.pos() >= len(self.x)

    def latest_raw(self):
        p = self.pos()
        n = int(self.keep * self.sr)
        out = np.zeros(n, np.float32)
        seg = self.x[max(0, p - n):p]
        if len(seg):
            out[-len(seg):] = seg
        return out, self.sr


def dream_input():
    """Uygulamadaki rüya modu girdisine benzer bir CPPN stili (duygu başlığı + görsel üretici)."""
    try:
        from neural_bg import EmotionMapper
        m = EmotionMapper()
        st = m.style(m.h["mu"].copy(), None)
        return st, np.zeros(8, np.float32), m.va.copy()
    except Exception as e:
        print(f"[Bench] EmotionMapper yok ({e}) — rastgele stil")
        rng = np.random.default_rng(0)
        st = {"W": rng.normal(0, 1, (48, 16)).astype(np.float32),
              "B": rng.normal(0, 0.5, (13, 4)).astype(np.float32),
              "acts": np.full((3, 3), 1 / 3, np.float32),
              "palette": [(0.2, 0.4, 1.0), (0.9, 0.1, 0.8), (1.0, 0.5, 0.1), (0.1, 0.9, 0.5)]}
        return st, np.zeros(8, np.float32), np.zeros(2, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", help="sözlü bir şarkı (wav / flac; soundfile yoksa sadece wav)")
    ap.add_argument("--seconds", type=float, default=0, help="en fazla bu kadar sn çal (0 = tamamı)")
    ap.add_argument("--model", default=None, help="Whisper modeli (varsayılan config.json lyrics_model)")
    ap.add_argument("--legacy", action="store_true", help="eski ayarlarla ölç (karşılaştırma)")
    ap.add_argument("--compile", action="store_true", help="diff_compile açık")
    args = ap.parse_args()

    import config
    import diffusion_layer
    import lyrics as lyrics_mod
    cfg = config.load()
    if args.legacy:
        cfg.update(lyrics_window=6.0, lyrics_step=1.5, lyrics_translate_first=False,
                   lyrics_reuse_encoder=False, lyric_priority=False, diff_fast=False)
    if args.compile:
        cfg["diff_compile"] = True
    model = args.model or cfg["lyrics_model"]

    synthetic = not args.wav
    if synthetic:
        print("=" * 72)
        print("UYARI: --wav verilmedi. Whisper ÖLÇÜLMÜYOR; sentetik ses + yapay söz satırları ile")
        print("       yalnızca SD tarafı (sahne kurma + ilk kare + çizim) ölçülecek.")
        print("       Gerçek söz→ekran için: --wav <sözlü şarkı.wav>")
        print("=" * 72)
        sr = 48000
        x = (0.05 * np.random.default_rng(0).standard_normal(sr * int(args.seconds or 40))).astype(np.float32)
    else:
        x, sr = load_wav(args.wav)
        print(f"[Bench] {args.wav}: {len(x) / sr:.1f} sn, {sr} Hz")
    if args.seconds:
        x = x[:int(args.seconds * sr)]

    tap = WavTap(x, sr)
    tracker = latency.LatencyTracker(enabled=True)
    diff = diffusion_layer.DiffusionLayer(tap=tap, tracker=tracker, lyric_priority=cfg["lyric_priority"],
                                          fast=cfg["diff_fast"], compile=cfg["diff_compile"])
    lyr = lyrics_mod.LyricsListener(tap, model=model, window=cfg["lyrics_window"], step=cfg["lyrics_step"],
                                    translate_first=cfg["lyrics_translate_first"],
                                    hud_original=cfg["lyrics_hud_original"],
                                    reuse_encoder=cfg["lyrics_reuse_encoder"], tracker=tracker)
    lyr.on_line.append(diff.wake)
    diff.lyrics = lyr
    print(f"[Bench] ayarlar: {'ESKİ (legacy)' if args.legacy else 'yeni'}  pencere {lyr.WINDOW:g} sn, "
          f"adım {lyr.EVERY:g} sn, translate_first {lyr.translate_first}, söz önceliği {diff.lyric_priority}, "
          f"diff_fast {diff.fast}, compile {diff.compile}")

    t0 = time.time()
    diff.start()
    while not (diff.ready or diff.failed):
        time.sleep(0.1)
    if diff.failed:
        sys.exit(f"[Bench] Difüzyon yüklenemedi: {diff.failed}")
    if not synthetic:
        lyr.start()                                       # transformers import'u: difüzyondan sonra
        while not (lyr.ready or lyr.failed):
            time.sleep(0.1)
        if lyr.failed:
            sys.exit(f"[Bench] Whisper yüklenemedi: {lyr.failed}")
    else:
        lyr.ready = True
    print(f"[Bench] modeller hazır ({time.time() - t0:.0f} sn) — akış başlıyor")

    st, z, va = dream_input()
    stop = threading.Event()

    def feeder():                                         # ana döngü: her ~16 ms rüya girdisi
        while not stop.is_set():
            diff.submit(st, z, va)
            time.sleep(1 / 60)

    def fake_lines():                                     # sentetik mod: 2 sn'de bir yapay satır
        words = ["burning streets tonight", "ocean in the night", "dance until the sun comes up",
                 "wolves running through the snow", "a city made of glass"]
        i = 0
        while not stop.is_set():
            time.sleep(2.0)
            t = latency.now()
            lyr._push(f"we walk the {words[i % len(words)]} {i}", None, "en",
                      {"heard": t, "window": t, "whisper": t})
            i += 1

    threading.Thread(target=feeder, daemon=True).start()
    tap.start()
    if synthetic:
        threading.Thread(target=fake_lines, daemon=True).start()
    fade = cfg["lyric_fade"]
    infer, seen = [], -1
    try:
        while not tap.done():
            fid, frame, lid = diff.get_frame_info()
            if fid != seen:
                seen = fid
                if lid is not None:
                    f = fade if not args.legacy else min(0.5, max(0.08, 1.0 / max(diff.fps, 1e-3)))
                    tracker.finish(lid, extra=f)
            if lyr.infer_ms and (not infer or infer[-1] != lyr.infer_ms):
                infer.append(lyr.infer_ms)
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        print("[Bench] durduruldu")
    stop.set()
    lyr.stop()
    diff.stop()

    n, avg, p95, parts = tracker.summary()
    print("\n" + "=" * 72)
    print(f"Ayarlar: {'ESKİ (legacy)' if args.legacy else 'yeni'}   Whisper: {model if not synthetic else 'ÖLÇÜLMEDİ'}")
    if n == 0:
        print("Hiç söz satırı ekrana ulaşmadı (vokal algılanmadı ya da filtre eledi). Son ham çıktı:",
              lyr.last_raw)
    else:
        xs = list(tracker.history)
        print(f"Söz → ekran  n={n}  ortalama {avg:.2f} sn   p95 {p95:.2f} sn   en kötü {max(xs):.2f} sn   "
              f"≤2 sn: {sum(v <= 2.0 for v in xs)}/{n}")
        print("Adım ortalamaları: " + ", ".join(f"{k} {v:.2f}" for k, v in parts.items()))
    if infer:
        print(f"Whisper pencere başına: ortalama {np.mean(infer):.0f} ms, p95 {np.percentile(infer, 95):.0f} ms "
              f"({len(infer)} pencere)")
    print(f"SD kare hızı (son): {diff.fps:.1f} fps")
    print("Not: 'söz anı' = pencereye yeni giren sesin ortası (yaklaşık; ±adım/2).")


if __name__ == "__main__":
    main()
