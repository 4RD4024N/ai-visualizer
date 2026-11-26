"""
Canlı ses → kendi modelimiz → CPPN latent.

AudioTap          : ses callback'inden ham blokları alır, 16 kHz'e çevirir, son 4 sn'yi tutar.
FeatureRecorder   : (isteğe bağlı) dinlenen müziğin SADECE log-mel özelliklerini kaydeder.
                    Ses dosyası oluşmaz; parçalar sessizlik aralarından ayrılır.
AudioLatentModel  : ONNX modeli 10 Hz'de çalıştırır → yumuşatılmış 8 boyutlu latent.
"""

import collections
import json
import os
import threading
import time

import numpy as np

from audio_features import SR, HOP, N_MELS, FRAMES, WINDOW_SAMPLES, logmel, StreamResampler

ROOT = os.path.dirname(os.path.abspath(__file__))
from config import WORK_DIR  # noqa: E402
LIVE_DIR = os.path.join(WORK_DIR, "features", "live")
MODEL_PATH = os.path.join(ROOT, "models", "audio_latent.onnx")


class FeatureRecorder:
    SILENCE_RMS = 1e-3
    MIN_SEC = 12.0
    MAX_SEC = 360.0

    def __init__(self, out_dir=LIVE_DIR):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.bin_path = os.path.join(out_dir, "feats.bin")
        self.idx_path = os.path.join(out_dir, "index.json")
        self.buf, self.n, self.silent = [], 0, 0.0
        self.saved_tracks = 0

    def feed(self, x):
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        if rms < self.SILENCE_RMS:
            self.silent += len(x) / SR
        else:
            self.silent = 0.0
            self.buf.append(x.copy())
            self.n += len(x)
        sec = self.n / SR
        if (self.silent > 1.5 and sec > 0) or sec > self.MAX_SEC:
            self.flush()

    def flush(self):
        sec = self.n / SR
        buf, self.buf, self.n = self.buf, [], 0
        if sec < self.MIN_SEC:
            return
        feats = logmel(np.concatenate(buf)).T.astype(np.float16)    # ses burada atılır
        if os.path.exists(self.idx_path):
            with open(self.idx_path, encoding="utf-8") as f:
                idx = json.load(f)
        else:
            idx = {"n_mels": N_MELS, "frames": 0, "tracks": []}
        with open(self.bin_path, "ab") as f:
            f.write(feats.tobytes())
        name = time.strftime("live_%Y%m%d_%H%M%S") + f"_{len(idx['tracks']):04d}"
        idx["tracks"].append({"name": name, "start": idx["frames"], "length": int(len(feats))})
        idx["frames"] += int(len(feats))
        tmp = self.idx_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(idx, f)
        os.replace(tmp, self.idx_path)
        self.saved_tracks += 1
        print(f"[Kayıt] {name}: {sec:.0f} sn özellik kaydedildi (toplam {len(idx['tracks'])} parça)")


class AudioTap:
    def __init__(self):
        self._q = collections.deque(maxlen=1024)
        self._sr = None
        self._rs = None
        self.ring = np.zeros(SR * 4, np.float32)
        self.lock = threading.Lock()
        self.recorder = None
        self.running = False
        self.total = 0                    # ring'e yazılan toplam örnek (hop hizası için)
        self.raw_sr = None                # ham ses (CLAP: ~10 sn, cihaz örnekleme hızı)
        self.raw = np.zeros(0, np.float32)

    def push(self, block, sr):
        """Ses callback'inden çağrılır — hafif olmalı."""
        self._q.append((block, sr))

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            blocks = []
            while self._q:
                blocks.append(self._q.popleft())
            if not blocks:
                time.sleep(0.01)
                continue
            sr = blocks[-1][1]
            if sr != self._sr:
                self._sr, self._rs = sr, StreamResampler(sr)
            raw = np.concatenate([b for b, _ in blocks]).astype(np.float32)
            x = self._rs(raw)
            with self.lock:
                n = min(len(x), len(self.ring))
                self.ring = np.roll(self.ring, -n)
                self.ring[-n:] = x[-n:]
                self.total += len(x)
                if self.raw_sr != sr:
                    self.raw_sr, self.raw = sr, np.zeros(int(sr * 10), np.float32)
                m = min(len(raw), len(self.raw))
                self.raw = np.roll(self.raw, -m)
                self.raw[-m:] = raw[-m:]
            rec = self.recorder
            if rec is not None:
                rec.feed(x)

    def latest(self, n):
        with self.lock:
            return self.ring[-n:].copy()

    def latest_raw(self):
        """Son ~10 sn ham ses ve örnekleme hızı (CLAP kavram seçimi için)."""
        with self.lock:
            return self.raw.copy(), self.raw_sr

    def latest_aligned(self, n, hop):
        """Son n örnek, ama sonu her zaman hop ızgarasına oturur: ardışık çağrılarda
        spektrogram kareleri aynı yerden kesilir (rastgele blok sınırı → ek titreme yok)."""
        with self.lock:
            end = len(self.ring) - (self.total % hop)
            return self.ring[end - n:end].copy()

    def toggle_recording(self):
        if self.recorder is None:
            self.recorder = FeatureRecorder()
            print(f"[Kayıt] Başladı → {LIVE_DIR}")
            return True
        rec, self.recorder = self.recorder, None
        rec.flush()
        print("[Kayıt] Durdu")
        return False

    def stop(self):
        self.running = False
        if self.recorder is not None:
            self.recorder.flush()
            self.recorder = None


class AudioLatentModel:
    """Ses modelini sürekli çalıştırır (sessizlik dahil — model sessizliği de 'duyar').

    raw()      : modelin o anki ham çıktısı (latent[8], embedding[128]) — rüya modu.
    get()      : yumuşatılmış latent — eski (bant + model karışık) mod.
    Pencere ~2 sn, 30 Hz'de kayar. Her çalıştırmada model aynı son sesin
    n_align farklı hizalamasına (0–span kare geri) tek batch'te bakar ve
    ortalamasını verir: pencere kaymasından gelen titreme ~3 kat azalır,
    zaman bazlı yumuşatma gerekmez.
    """

    def __init__(self, tap, path=MODEL_PATH, hz=30.0, smooth_sec=0.8, n_align=8, span=24):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.offs = np.linspace(0, span, n_align).astype(int) if n_align > 1 else np.zeros(1, int)
        self.span = int(self.offs.max())
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.tap = tap
        self.dt = 1.0 / hz
        self.smooth = smooth_sec
        self.latent = np.zeros(8, np.float32)
        self._raw = None                  # (latent, embedding) — son ham çıktı
        self.ready = False
        self.running = False
        self.infer_ms = 0.0
        cfg_path = os.path.splitext(path)[0] + ".json"
        self.cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        a = self.dt / (self.smooth + self.dt)
        while self.running:
            t0 = time.perf_counter()
            x = self.tap.latest_aligned(WINDOW_SAMPLES + self.span * HOP, HOP)
            M = logmel(x)                                  # (N_MELS, FRAMES + span)
            T = M.shape[1]
            m = np.stack([M[:, T - FRAMES - k:T - k] for k in self.offs])[:, None]
            z, e = self.sess.run(["latent", "embedding"], {"logmel": m})
            z, e = z.mean(0).astype(np.float32), e.mean(0).astype(np.float32)
            self._raw = (z, e)                            # tek atamayla değiş — thread güvenli
            self.latent += (np.clip(z, -3.0, 3.0) - self.latent) * a
            self.ready = True
            self.infer_ms = (time.perf_counter() - t0) * 1000
            time.sleep(max(0.0, self.dt - (time.perf_counter() - t0)))

    def get(self):
        return self.latent.copy()

    def raw(self):
        return self._raw

    def stop(self):
        self.running = False
