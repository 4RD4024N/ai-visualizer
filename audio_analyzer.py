"""
Ses yakalama + analiz: WASAPI loopback → FFT bantları, RMS, spektral akış,
beat takibi (beat.py) ve bölüm algılama (structure.py).

sounddevice yalnızca `start()` içinde import edilir; analiz kısmı ses donanımı
olmadan (testlerde, CI'da) çalışır.
"""

import threading

import numpy as np

from beat import BeatTracker
from structure import SectionDetector

CHUNK = 1024
FFT_SIZE = 4096
BANDS = 8
BAND_NAMES = ["sub_bass", "bass", "low_mid", "mid", "high_mid", "presence", "brilliance", "air"]
BAND_RANGES = [(20, 60), (60, 250), (250, 500), (500, 2000), (2000, 4000), (4000, 6000),
               (6000, 12000), (12000, 20000)]


class AudioAnalyzer:
    def __init__(self, device_hint=""):
        self.device_hint = device_hint.strip().lower()
        self.stream = None
        self.lock = threading.Lock()
        self.running = False
        self.channels = 1
        self.tap = None                 # AudioTap: ses modeli + canlı kayıt için ham ses
        self.recorder = None            # VideoRecorder: video kaydına ses
        self._set_rate(48000)

    def _set_rate(self, sr):
        self._sample_rate = sr
        self.buffer = np.zeros(FFT_SIZE)
        self.bands = np.zeros(BANDS)
        self.bands_smooth = np.zeros(BANDS)
        self.rms = 0.0
        self.peak = 0.0
        self._prev_mag = None
        self._window = np.hanning(FFT_SIZE)
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sr)
        self._masks = [(freqs >= lo) & (freqs < hi) for lo, hi in BAND_RANGES]
        dt = CHUNK / float(sr)
        self.beats = BeatTracker(dt)
        self.sections = SectionDetector(dt)
        # Ruh hali istatistikleri — uzun pencereli
        self._stat_bands = np.zeros(BANDS)
        self._stat_rms = 0.0
        self._rms_hist = []             # son ~3 sn RMS (dinamik aralık için)
        self._flux_hist = []

    @property
    def sample_rate(self):
        return self._sample_rate

    # ── Cihaz ──
    def _find_loopback(self, sd):
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        candidates = []
        for i, dev in enumerate(devices):
            name_lower = dev["name"].lower()
            ch_in = int(dev["max_input_channels"])
            if ch_in == 0:
                continue
            api_name = hostapis[dev["hostapi"]]["name"]
            score = 0
            if self.device_hint and self.device_hint in name_lower:
                score = 200                         # config.json "device" önceliklidir
            elif "cable output" in name_lower:
                score = 100
            elif "stereo mix" in name_lower or "what u hear" in name_lower:
                score = 60
            if "WASAPI" in api_name:
                score += 10
            if score > 0:
                candidates.append((score, i, min(ch_in, 2), dev["name"]))
        if self.device_hint and not any(c[0] >= 200 for c in candidates):
            print(f"[Audio] '{self.device_hint}' adlı cihaz bulunamadı — otomatik seçim")
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            _, idx, ch, name = candidates[0]
            print(f"[Audio] Loopback: [{idx}] {name} ({ch}ch)")
            return idx, ch
        return None, 1

    def start(self):
        import sounddevice as sd
        dev_idx, ch = self._find_loopback(sd)
        if dev_idx is not None:
            for sr in [48000, 44100, 96000, 32000]:
                try:
                    self._set_rate(sr)
                    self.stream = sd.InputStream(device=dev_idx, channels=ch, samplerate=sr,
                                                 blocksize=CHUNK, dtype="float32", callback=self._cb)
                    self.stream.start()
                    self.running, self.channels = True, ch
                    print(f"[Audio] Loopback aktif ({sr}Hz)")
                    return
                except Exception as e:
                    print(f"[Audio] {sr}Hz: {e}")
        try:
            self._set_rate(48000)
            self.stream = sd.InputStream(channels=1, samplerate=48000, blocksize=CHUNK,
                                         dtype="float32", callback=self._cb)
            self.stream.start()
            self.running, self.channels = True, 1
            print("[Audio] Mikrofon fallback")
        except Exception as e:
            print(f"[Audio] Hata: {e}")

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def _cb(self, indata, frames, time_info, status):
        s = indata[:, 0].copy()
        if self.tap is not None:
            self.tap.push(s, self._sample_rate)
        rec = self.recorder
        if rec is not None:
            rec.push_audio(indata)
        self.process(s)

    # ── Analiz (ses thread'i; testlerde doğrudan çağrılabilir) ──
    def process(self, s):
        with self.lock:
            n = len(s)
            self.buffer = np.roll(self.buffer, -n)
            self.buffer[-n:] = s
            self._analyze(s)

    def _analyze(self, s):
        self.rms = float(np.sqrt(np.mean(s ** 2)))
        self.peak = float(np.max(np.abs(s)))
        mag = np.abs(np.fft.rfft(self.buffer * self._window))
        for i, mask in enumerate(self._masks):
            energy = float(np.mean(mag[mask])) / 50.0 if mask.any() else 0.0
            self.bands[i] = self.bands[i] * 0.5 + energy * 0.5
        self.bands_smooth = self.bands_smooth * 0.85 + self.bands * 0.15

        # Spektral akış (onset gücü): sadece pozitif değişimlerin toplamı
        if self._prev_mag is not None:
            diff = mag - self._prev_mag
            flux = float(np.sum(diff[diff > 0]))
            bass_mask = self._masks[0] | self._masks[1]
            bd = diff[bass_mask]
            bass_flux = float(np.sum(bd[bd > 0]))
        else:
            flux = bass_flux = 0.0
        self._prev_mag = mag

        self.beats.push(flux, bass_flux)
        self.sections.push(self.bands, self.rms, flux)

        self._stat_bands = self._stat_bands * 0.98 + self.bands * 0.02
        self._stat_rms = self._stat_rms * 0.98 + self.rms * 0.02
        self._rms_hist.append(self.rms)
        if len(self._rms_hist) > 140:
            self._rms_hist.pop(0)
        self._flux_hist.append(flux)
        if len(self._flux_hist) > 140:
            self._flux_hist.pop(0)

    # ── Okuma (ana thread) ──
    def snapshot(self):
        with self.lock:
            if len(self._rms_hist) > 10:
                rms_var = float(np.std(self._rms_hist))
                dyn_range = float(np.max(self._rms_hist) - np.min(self._rms_hist))
            else:
                rms_var = dyn_range = 0.0
            flux_avg = float(np.mean(self._flux_hist)) if self._flux_hist else 0.0
            sb = self._stat_bands
            low, mid, high = float(np.sum(sb[0:3])), float(np.sum(sb[3:5])), float(np.sum(sb[5:8]))
            total = low + mid + high + 1e-6
            return {
                "bands": {k: round(float(v), 3) for k, v in zip(BAND_NAMES, self.bands_smooth)},
                "rms": round(self.rms, 4),
                "rms_avg": round(self._stat_rms, 4),
                "bpm": round(self.beats.bpm, 1),
                "beat_confidence": round(self.beats.confidence, 2),
                "section": self.sections.label,
                "dynamics": round(rms_var, 4),
                "dynamic_range": round(dyn_range, 4),
                "onset_activity": round(flux_avg, 2),
                "spectral_balance": {"low": round(low / total, 2), "mid": round(mid / total, 2),
                                     "high": round(high / total, 2)},
            }

    def get(self):
        with self.lock:
            return self.bands.copy().astype(float), float(self.rms), float(self.peak)

    def rhythm(self):
        """(yeni beat'ler, yeni bölümler, faz, güven, bpm, bölüm)"""
        with self.lock:
            b, s = self.beats, self.sections
            return b.pop_events(), s.pop_events(), b.phase(), b.confidence, b.bpm, s.label
