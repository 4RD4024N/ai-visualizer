"""
AI Music Visualizer v2
WASAPI Loopback + pyglet 2.x + LM Studio
DSP-driven görsel katmanlar + AI renk paleti
"""

import pyglet
import pyglet.shapes as shapes
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import math
import json
import random
import urllib.request

# ── Sabitler ────────────────────────────────────────────────────────────────
CHUNK       = 1024
FFT_SIZE    = 4096
BANDS       = 8
AI_INTERVAL = 1.5
TARGET_FPS  = 60
WINDOW_W    = 1280
WINDOW_H    = 720

BAND_NAMES  = ["sub_bass","bass","low_mid","mid","high_mid","presence","brilliance","air"]
BAND_RANGES = [(20,60),(60,250),(250,500),(500,2000),(2000,4000),(4000,6000),(6000,12000),(12000,20000)]
LM_BASE_URL = "http://localhost:1234/v1"

AI_PROMPT = """You are the creative director of a music visualizer. Based on the audio analysis, choose the animation scene and look. Respond with ONLY a JSON object, no explanation, no markdown.

Required keys:
- scene: pick ONE that fits the music character: "flow" (smooth/melodic), "galaxy" (spacious/epic), "web" (intricate/glitchy), "cosmic" (dreamy/ambient), "storm" (aggressive/intense), "minimal" (calm/sparse), "full" (high energy/everything)
- complexity: 0.0-1.0 (sparse to dense/intricate). High for busy music, low for calm.
- speed: 0.5-2.0 animation speed. High for fast/energetic music.
- palette: array of 4 colors, each {"r":0-1,"g":0-1,"b":0-1}. Vivid neon for energy, cool pastel for calm.
- bg_r,bg_g,bg_b: 0.0-0.06 dark background
- mood: energetic/calm/dark/ethereal/aggressive/dreamy
- shape_rain: true ONLY on very high energy moments, else false

Map from audio: high rms_avg+dynamics+bpm=energetic scenes(full/storm)+vivid+fast. low=calm scenes(minimal/cosmic)+pastel+slow. high spectral_balance.low(bass)=storm/galaxy. high spectral_balance.high(treble)=web/flow. Always derive from the actual numbers, vary your choices."""

# ── Ses Analizi ──────────────────────────────────────────────────────────────
class AudioAnalyzer:
    def __init__(self):
        self.stream       = None
        self.buffer       = np.zeros(FFT_SIZE)
        self.lock         = threading.Lock()
        self.running      = False
        self.bands        = np.zeros(BANDS)
        self.bands_smooth = np.zeros(BANDS)
        self.rms          = 0.0
        self.peak         = 0.0
        self.bpm_est      = 120.0
        self._sample_rate = 48000
        # Onset envelope — BPM autocorrelation için (~6sn @ ~47fps)
        self._onset_env   = np.zeros(280)
        self._prev_mag    = None
        self._frame_dt    = CHUNK / 48000.0   # bir frame'in süresi
        self._bpm_cooldown = 0.0
        # Mood istatistikleri — uzun pencereli ortalama
        self._stat_bands  = np.zeros(BANDS)   # uzun ortalama band enerjisi
        self._stat_rms    = 0.0
        self._rms_hist    = []                # son ~3sn RMS (dinamik aralık için)
        self._flux_hist   = []                # spektral akış geçmişi

    def _find_loopback(self):
        devices  = sd.query_devices()
        hostapis = sd.query_hostapis()
        candidates = []
        for i, dev in enumerate(devices):
            name_lower = dev["name"].lower()
            ch_in = int(dev["max_input_channels"])
            if ch_in == 0:
                continue
            api_name = hostapis[dev["hostapi"]]["name"]
            score = 0
            if "cable output" in name_lower:   score = 100
            elif "stereo mix" in name_lower:   score = 60
            elif "what u hear" in name_lower:  score = 60
            if "WASAPI" in api_name:           score += 10
            if score > 0:
                candidates.append((score, i, min(ch_in, 2), dev["name"]))
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            _, idx, ch, name = candidates[0]
            print(f"[Audio] Loopback: [{idx}] {name} ({ch}ch)")
            return idx, ch
        return None, 1

    def start(self):
        dev_idx, ch = self._find_loopback()
        if dev_idx is not None:
            for sr in [48000, 44100, 96000, 32000]:
                try:
                    self.stream = sd.InputStream(
                        device=dev_idx, channels=ch,
                        samplerate=sr, blocksize=CHUNK,
                        dtype="float32", callback=self._cb,
                    )
                    self.stream.start()
                    self.running = True
                    self._sample_rate = sr
                    print(f"[Audio] Loopback aktif ({sr}Hz)")
                    return
                except Exception as e:
                    print(f"[Audio] {sr}Hz: {e}")
        try:
            self.stream = sd.InputStream(
                channels=1, samplerate=48000,
                blocksize=CHUNK, dtype="float32", callback=self._cb,
            )
            self.stream.start()
            self.running = True
            print("[Audio] Mikrofon fallback")
        except Exception as e:
            print(f"[Audio] Hata: {e}")

    def _cb(self, indata, frames, time_info, status):
        s = indata[:, 0].copy()
        with self.lock:
            self.buffer = np.roll(self.buffer, -len(s))
            self.buffer[-len(s):] = s
            self._analyze(s)

    def _analyze(self, s):
        self.rms  = float(np.sqrt(np.mean(s**2)))
        self.peak = float(np.max(np.abs(s)))
        win   = np.hanning(len(self.buffer))
        mag   = np.abs(np.fft.rfft(self.buffer * win))
        freqs = np.fft.rfftfreq(len(self.buffer), 1.0 / self._sample_rate)
        for i, (lo, hi) in enumerate(BAND_RANGES):
            mask   = (freqs >= lo) & (freqs < hi)
            energy = float(np.mean(mag[mask])) / 50.0 if mask.any() else 0.0
            self.bands[i] = self.bands[i] * 0.5 + energy * 0.5
        self.bands_smooth = self.bands_smooth * 0.85 + self.bands * 0.15

        # ── Spektral flux (onset gücü) ──────────────────────────────
        # Sadece pozitif değişimlerin toplamı = onset envelope değeri
        if self._prev_mag is not None and len(self._prev_mag) == len(mag):
            diff = mag - self._prev_mag
            flux = float(np.sum(diff[diff > 0]))
        else:
            flux = 0.0
        self._prev_mag = mag

        # Onset envelope kaydır
        self._onset_env = np.roll(self._onset_env, -1)
        self._onset_env[-1] = flux

        # ── İstatistik toplama (mood için) ──────────────────────────
        self._stat_bands = self._stat_bands * 0.98 + self.bands * 0.02
        self._stat_rms   = self._stat_rms * 0.98 + self.rms * 0.02
        self._rms_hist.append(self.rms)
        if len(self._rms_hist) > 140: self._rms_hist.pop(0)
        self._flux_hist.append(flux)
        if len(self._flux_hist) > 140: self._flux_hist.pop(0)

        # ── BPM: autocorrelation (her ~0.5sn) ───────────────────────
        self._bpm_cooldown -= self._frame_dt
        if self._bpm_cooldown <= 0:
            self._bpm_cooldown = 0.5
            self._estimate_bpm()

    def _estimate_bpm(self):
        env = self._onset_env.copy()
        if np.max(env) < 1e-6:
            return
        # Hafif yumuşatma — gürültü azalt
        kernel = np.array([0.25, 0.5, 0.25])
        env = np.convolve(env, kernel, mode="same")
        env = env - np.mean(env)
        env[env < 0] = 0.0   # yarı dalga doğrultma (onset vurgusu)

        # Autocorrelation
        ac = np.correlate(env, env, mode="full")[len(env)-1:]
        if ac[0] <= 1e-9:
            return
        ac = ac / ac[0]

        min_bpm, max_bpm = 60, 200
        min_lag = max(1, int((60.0 / max_bpm) / self._frame_dt))
        max_lag = int((60.0 / min_bpm) / self._frame_dt)
        max_lag = min(max_lag, len(ac) - 1)
        if max_lag <= min_lag:
            return

        # Harmonik güçlendirme: her lag için katlarını da topla
        # (gerçek tempo, alt-katlarda da tepe yapar → birleştir)
        scores = np.zeros(max_lag + 1)
        for lag in range(min_lag, max_lag + 1):
            s = ac[lag]
            for mult, w in ((2, 0.5), (3, 0.33), (4, 0.25)):
                if lag * mult <= len(ac) - 1:
                    s += ac[lag * mult] * w
            scores[lag] = s

        # Tempo önceli: 120 BPM civarını tercih et (log-Gauss ağırlık)
        # → oktav hatalarını (60↔120↔240) azaltır
        lags = np.arange(min_lag, max_lag + 1)
        bpms = 60.0 / (lags * self._frame_dt)
        pref = 120.0
        sigma = 0.55
        weight = np.exp(-0.5 * (np.log2(bpms / pref) / sigma) ** 2)
        weighted = scores[min_lag:max_lag + 1] * weight

        best_i = int(np.argmax(weighted))
        best_lag = min_lag + best_i
        peak_val = scores[best_lag]
        # Güven: tepe ne kadar baskın
        confidence = peak_val / (np.mean(scores[min_lag:max_lag+1]) + 1e-6)

        if peak_val < 0.08 or confidence < 1.3:
            return  # zayıf/belirsiz → mevcut tahmini koru

        bpm = 60.0 / (best_lag * self._frame_dt)
        while bpm < 70:  bpm *= 2
        while bpm > 190: bpm /= 2

        # Güvene göre adaptif yumuşatma: yüksek güven → hızlı kilitlen
        alpha = min(0.5, 0.15 + confidence * 0.08)
        self.bpm_est = self.bpm_est * (1 - alpha) + bpm * alpha

    def snapshot(self):
        with self.lock:
            # Dinamik aralık: yüksek = ritmik/vurgulu, düşük = düz/ambient
            if len(self._rms_hist) > 10:
                rms_var = float(np.std(self._rms_hist))
                rms_max = float(np.max(self._rms_hist))
                rms_min = float(np.min(self._rms_hist))
                dyn_range = rms_max - rms_min
            else:
                rms_var = dyn_range = 0.0
            flux_avg = float(np.mean(self._flux_hist)) if self._flux_hist else 0.0

            # Frekans dengesi: bass-ağırlık mı, treble-ağırlık mı
            sb = self._stat_bands
            low  = float(np.sum(sb[0:3]))   # sub_bass..low_mid
            mid  = float(np.sum(sb[3:5]))   # mid..high_mid
            high = float(np.sum(sb[5:8]))   # presence..air
            total = low + mid + high + 1e-6

            return {
                "bands": {k: round(float(v), 3) for k, v in zip(BAND_NAMES, self.bands_smooth)},
                "rms": round(self.rms, 4),
                "rms_avg": round(self._stat_rms, 4),
                "bpm": round(self.bpm_est, 1),
                "dynamics": round(rms_var, 4),
                "dynamic_range": round(dyn_range, 4),
                "onset_activity": round(flux_avg, 2),
                "spectral_balance": {
                    "low": round(low/total, 2),
                    "mid": round(mid/total, 2),
                    "high": round(high/total, 2),
                },
            }

    def get(self):
        with self.lock:
            return self.bands.copy().astype(float), float(self.rms), float(self.peak)

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()


# ── AI Palette Director ───────────────────────────────────────────────────────
DEFAULT_PALETTE = [
    (0.2, 0.4, 1.0),
    (0.9, 0.1, 0.8),
    (1.0, 0.5, 0.1),
    (0.1, 0.9, 0.5),
]

def _parse_json(raw):
    import re
    raw = raw.strip()
    if "</think>" in raw:
        raw = raw[raw.rfind("</think>") + len("</think>"):]
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not raw: return None
    if "```" in raw:
        for part in raw.split("```")[1::2]:
            part = part.lstrip("json").strip()
            if part.startswith("{"): raw = part; break
    if not raw.startswith("{"):
        idx = raw.find("{")
        if idx >= 0: raw = raw[idx:]
        else: return None
    depth, end = 0, -1
    for i, ch in enumerate(raw):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: end = i; break
    if end >= 0: raw = raw[:end+1]
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    try: return json.loads(raw)
    except: return None

def _lm_call(base_url, model, system_prompt, user_prompt, max_tokens=200, temperature=0.5):
    """Genel LM Studio HTTP çağrısı."""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=body,
        headers={"Content-Type":"application/json","Authorization":"Bearer lm-studio"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"].strip()

def _get_model(base_url):
    req = urllib.request.Request(f"{base_url}/models",
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())
    models = data.get("data", [])
    if not models: raise RuntimeError("Model yok")
    mid = models[0]["id"]
    print(f"[AI] Model: {mid}")
    return mid




class AIDirector:
    """Tek model — renk paleti + compose parametreleri (her 3sn)"""
    def __init__(self, audio, q):
        self.audio   = audio
        self.queue   = q
        self.running = False
        self.model   = None

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running and not self.model:
            try: self.model = _get_model(LM_BASE_URL)
            except Exception as e:
                print(f"[AI] {e} — 5sn")
                time.sleep(5)
        while self.running:
            try:
                snap = self.audio.snapshot()
                body = json.dumps({
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": AI_PROMPT},
                        {"role": "user",   "content": f"Audio:{json.dumps(snap)}"},
                    ],
                    "max_tokens": 800,
                    "temperature": 0.85,
                    "stream": False,
                    "thinking": {"type": "disabled"},
                }).encode()
                req = urllib.request.Request(
                    f"{LM_BASE_URL}/chat/completions", data=body,
                    headers={"Content-Type":"application/json","Authorization":"Bearer lm-studio"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=20) as r:
                    raw = json.loads(r.read())["choices"][0]["message"]["content"].strip()
                d = _parse_json(raw)
                if d and ("palette" in d or "mood" in d):
                    d["_type"] = "palette"
                    if "mandala_speed" in d: d["_type"] = "full"
                    if self.queue.full():
                        try: self.queue.get_nowait()
                        except: pass
                    self.queue.put(d)
                    print(f"[AI] mood:{d.get('mood','?')} mandala:{d.get('mandala_speed','—')}")
                else:
                    print(f"[AI] fail ({len(raw)} chars): {repr(raw[:200])}")
            except Exception as e:
                print(f"[AI] {e}")
            time.sleep(1.5)

    def stop(self): self.running = False



# ── Renk Paleti (lerp destekli) ───────────────────────────────────────────────
class Palette:
    def __init__(self):
        self._cur = [list(c) for c in DEFAULT_PALETTE]
        self._tgt = [list(c) for c in DEFAULT_PALETTE]
        self.bg   = [0.02, 0.0, 0.08]
        self._tgt_bg = [0.02, 0.0, 0.08]
        self.mood = "hazir"
        self.hue_shift   = 0.0
        self.drift_speed = 0.025      # yavaş renk kayması (0 = kapalı)
        self._draw = [list(c) for c in DEFAULT_PALETTE]  # çizimde kullanılan (kaymış)

    def randomize(self):
        """Rastgele canlı palet."""
        import colorsys
        base_hue = random.random()
        for i in range(4):
            hue = (base_hue + i * 0.22 + random.uniform(-0.05, 0.05)) % 1.0
            sat = random.uniform(0.7, 1.0)
            val = random.uniform(0.8, 1.0)
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            self._tgt[i] = [r, g, b]

    def set(self, d):
        pal = d.get("palette", [])
        for i in range(min(4, len(pal))):
            self._tgt[i] = [
                float(pal[i].get("r", self._tgt[i][0])),
                float(pal[i].get("g", self._tgt[i][1])),
                float(pal[i].get("b", self._tgt[i][2])),
            ]
        self._tgt_bg = [
            float(d.get("bg_r", 0.02)),
            float(d.get("bg_g", 0.0)),
            float(d.get("bg_b", 0.08)),
        ]
        self.mood = d.get("mood", "?")

    def update(self, dt):
        import colorsys
        spd = dt * 1.5
        for i in range(4):
            for j in range(3):
                self._cur[i][j] += (self._tgt[i][j] - self._cur[i][j]) * min(1.0, spd)
        for j in range(3):
            self.bg[j] += (self._tgt_bg[j] - self.bg[j]) * min(1.0, spd)
        # Yavaş hue kayması — renkler canlı kalsın (frame başına 4 dönüşüm)
        self.hue_shift = (self.hue_shift + dt * self.drift_speed) % 1.0
        if self.drift_speed > 0:
            for i in range(4):
                r, g, b = self._cur[i]
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                h = (h + self.hue_shift) % 1.0
                self._draw[i] = list(colorsys.hsv_to_rgb(h, s, v))
        else:
            self._draw = [list(c) for c in self._cur]

    def get(self, idx, alpha=1.0):
        c = self._draw[idx % 4]
        return (
            max(0, min(255, int(c[0]*255))),
            max(0, min(255, int(c[1]*255))),
            max(0, min(255, int(c[2]*255))),
            max(0, min(255, int(alpha*255))),
        )

    def bg_gl(self):
        return self.bg[0], self.bg[1], self.bg[2]


# ── DSP Görsel Katmanlar ──────────────────────────────────────────────────────

def clamp8(v): return max(0, min(255, int(v)))

# Katman 1: Merkezi mandala — sub_bass + bass
class MandalaLayer:
    def __init__(self, batch):
        self.batch   = batch
        self.angle   = 0.0
        self.rings   = []
        self._build()

    def _build(self):
        for r in self.rings:
            r.delete()
        self.rings = []
        cx, cy = WINDOW_W//2, WINDOW_H//2
        for i in range(10):
            r = shapes.Arc(cx, cy, 20 + i*32, color=(255,255,255,30), batch=self.batch)
            self.rings.append(r)

    def update(self, dt, bands, palette, t, speed_mult=1.0):
        sub  = float(bands[0])
        bass = float(bands[1])
        mid  = float(bands[2])
        energy = (sub * 0.3 + bass * 0.5 + mid * 0.2)
        self.angle += dt * (80 + energy * 400) * speed_mult
        cx, cy = WINDOW_W//2, WINDOW_H//2
        n = len(self.rings)
        for i, ring in enumerate(self.rings):
            phase    = i / n
            # Her ring farkli frekansta pulse eder
            freq     = 2.0 + i * 0.8
            pulse    = 1.0 + energy * 3.5 * abs(math.sin(t * freq + phase * math.pi * 2))
            base_r   = 20 + i * 32
            ring.radius   = float(base_r * max(0.3, pulse))
            ring.x        = cx
            ring.y        = cy
            # Her ring farkli hizda, alternating yon
            spd_mult = (0.3 + phase * 1.4) * (1 if i%2==0 else -1.3)
            ring.rotation = float(self.angle * spd_mult)
            alpha = 0.08 + energy * 0.7 + 0.15 * math.sin(t * 1.5 + i)
            c = palette.get(i % 4, alpha=min(1.0, alpha))
            try: ring.color = c
            except: pass

    def delete(self):
        for r in self.rings: r.delete()


# Katman 2: Dönen yıldız geometri — low_mid + mid
class StarGeometryLayer:
    def __init__(self, batch):
        self.batch  = batch
        self.stars  = []
        self.angles = []
        self._build()

    def _build(self):
        for s in self.stars: s.delete()
        self.stars   = []
        self.angles  = []
        self.phases  = []  # her yildiz icin bagimsiz faz
        cx, cy = WINDOW_W//2, WINDOW_H//2
        configs = [
            (130, 5),
            (80,  7),
            (45,  3),
            (60,  6),
            (60,  4),
            (50,  8),
            (40,  5),
            (40,  3),
        ]
        for sz, spk in configs:
            star = shapes.Star(cx, cy, sz, sz*0.38, num_spikes=spk,
                               rotation=0.0, color=(255,255,255,180), batch=self.batch)
            self.stars.append(star)
            self.angles.append(random.uniform(0, 360))
            self.phases.append(random.uniform(0, math.tau))

    def update(self, dt, bands, palette, t):
        mid_energy = (bands[2] + bands[3]) * 0.5
        speeds     = [80, -130, 200]
        sizes      = [120, 75, 45]
        for i, (star, spd, base_sz) in enumerate(zip(self.stars, speeds, sizes)):
            self.angles[i] = (self.angles[i] + dt * spd * (1 + mid_energy * 3)) % 360
            pulse = 1.0 + mid_energy * 1.8 * math.sin(t * math.pi * 6 + i * 1.2)
            sz    = base_sz * max(0.3, pulse)
            try: star.rotation = self.angles[i]
            except: pass
            star.outer_radius = sz
            star.inner_radius = sz * 0.38
            c = palette.get(i, alpha=0.4 + mid_energy * 0.5)
            try: star.color = c
            except: pass


# Katman 3: Frekans çubuğu (radyal) — tüm bandlar
class RadialBarsLayer:
    def __init__(self, batch):
        self.batch = batch
        self.lines = []
        self.N     = 128
        cx, cy = WINDOW_W//2, WINDOW_H//2
        for i in range(self.N):
            line = shapes.Line(cx, cy, cx+1, cy+1,
                               thickness=2, color=(255,255,255,100), batch=batch)
            self.lines.append(line)

    def update(self, dt, bands, palette, t, full_fft=None):
        cx, cy = WINDOW_W//2, WINDOW_H//2
        N = self.N
        for i, line in enumerate(self.lines):
            angle    = (i / N) * math.tau
            band_idx = int((i / N) * BANDS)
            energy   = float(bands[band_idx]) if band_idx < BANDS else 0.0
            # Komsu band ile blend — daha akici
            next_idx = min(band_idx + 1, BANDS - 1)
            energy   = energy * 0.7 + float(bands[next_idx]) * 0.3
            energy  *= (1 + 0.4 * math.sin(t * 4 + i * 0.15))
            inner_r  = 140
            outer_r  = inner_r + max(8, energy * 280)
            x1 = float(cx + math.cos(angle) * inner_r)
            y1 = float(cy + math.sin(angle) * inner_r)
            x2 = float(cx + math.cos(angle) * outer_r)
            y2 = float(cy + math.sin(angle) * outer_r)
            line.x  = x1; line.y  = y1
            line.x2 = x2; line.y2 = y2
            alpha = 0.2 + energy * 0.8
            c = palette.get(band_idx % 4, alpha=min(1.0, alpha))
            try: line.color = c
            except: pass

    def delete(self):
        for l in self.lines: l.delete()


# Katman: Serbest dolasan sekiller — tum bandlar, ekran genelinde hareket
class FloatingShapesLayer:
    """Ekranda serbestce yuzen, sekip duran, frekansa pulse eden sekiller."""
    SHAPE_TYPES = ["circle", "ring", "star", "triangle", "polygon", "hexstar"]

    def __init__(self, batch, count=34):
        self.batch  = batch
        self.items  = []
        self.glows  = []
        self._build(count)

    def _make_shape(self, kind, x, y, sz, spk):
        if kind == "circle":
            return shapes.Circle(x, y, sz, color=(255,255,255,180), batch=self.batch)
        elif kind == "ring":
            return shapes.Arc(x, y, sz, color=(255,255,255,180), batch=self.batch)
        elif kind == "star":
            return shapes.Star(x, y, sz, sz*0.42, num_spikes=5,
                               rotation=0.0, color=(255,255,255,180), batch=self.batch)
        elif kind == "triangle":
            return shapes.Star(x, y, sz, sz*0.5, num_spikes=3,
                               rotation=0.0, color=(255,255,255,180), batch=self.batch)
        elif kind == "hexstar":
            return shapes.Star(x, y, sz, sz*0.4, num_spikes=6,
                               rotation=0.0, color=(255,255,255,180), batch=self.batch)
        else:  # polygon
            return shapes.Star(x, y, sz, sz*0.88, num_spikes=spk,
                               rotation=0.0, color=(255,255,255,180), batch=self.batch)

    def _build(self, count):
        for it in self.items: it["obj"].delete()
        for g in self.glows: g.delete()
        self.items = []
        self.glows = []
        for i in range(count):
            kind  = random.choice(self.SHAPE_TYPES)
            sz    = random.uniform(14, 46)
            spk   = random.randint(4, 8)
            x     = random.uniform(60, WINDOW_W - 60)
            y     = random.uniform(60, WINDOW_H - 60)
            obj   = self._make_shape(kind, x, y, sz, spk)
            glow  = shapes.Circle(x, y, sz*1.7, color=(255,255,255,0), batch=self.batch)
            self.items.append({
                "obj": obj, "glow": glow, "kind": kind,
                "x": x, "y": y,
                "vx": random.uniform(-70, 70),
                "vy": random.uniform(-70, 70),
                "base": sz, "spk": spk,
                "angle": random.uniform(0, 360),
                "rot": random.uniform(-120, 120),
                "phase": random.uniform(0, math.tau),
                "band": i % BANDS,
            })

    def update(self, dt, bands, palette, t, energy_mult=1.0,
               mouse=None, mouse_mode=0):
        mx, my = (mouse if mouse else (None, None))
        for i, it in enumerate(self.items):
            band   = it["band"]
            energy = float(bands[band]) * energy_mult
            # Hareket
            it["x"] += it["vx"] * dt
            it["y"] += it["vy"] * dt
            # Kenar seğmesi
            if it["x"] < 40:            it["vx"] = abs(it["vx"]);  it["x"] = 40
            if it["x"] > WINDOW_W - 40: it["vx"] = -abs(it["vx"]); it["x"] = WINDOW_W - 40
            if it["y"] < 40:            it["vy"] = abs(it["vy"]);  it["y"] = 40
            if it["y"] > WINDOW_H - 40: it["vy"] = -abs(it["vy"]); it["y"] = WINDOW_H - 40

            # Bass vurusunda hız artar (enerji ivmesi)
            bass = float(bands[1])
            if bass > 0.5:
                it["vx"] *= (1 + bass * dt * 2)
                it["vy"] *= (1 + bass * dt * 2)
            # Hız sınırı
            sp = math.hypot(it["vx"], it["vy"])
            maxsp = 260
            if sp > maxsp:
                it["vx"] *= maxsp / sp; it["vy"] *= maxsp / sp

            # Fare etkisi
            if mx is not None and mouse_mode != 0:
                dx = mx - it["x"]; dy = my - it["y"]
                d2 = dx*dx + dy*dy + 200
                d  = math.sqrt(d2)
                if d < 320:
                    f = 14000 / d2
                    fx = (dx/d)*f; fy = (dy/d)*f
                    if mouse_mode == 1:
                        it["vx"] += fx*dt*60; it["vy"] += fy*dt*60
                    else:
                        it["vx"] -= fx*dt*60; it["vy"] -= fy*dt*60

            # Dönüş — enerjiye göre hızlanır
            it["angle"] = (it["angle"] + it["rot"] * dt * (1 + energy * 4)) % 360

            # Pulse
            pulse = 1.0 + energy * 2.2 * abs(math.sin(t * 3 + it["phase"]))
            sz    = it["base"] * max(0.3, pulse)

            obj = it["obj"]
            obj.x = float(it["x"]); obj.y = float(it["y"])
            try: obj.rotation = float(it["angle"])
            except: pass
            if isinstance(obj, shapes.Circle):
                obj.radius = sz
            elif isinstance(obj, shapes.Arc):
                obj.radius = sz
            elif isinstance(obj, shapes.Star):
                obj.outer_radius = sz
                inner_ratio = 0.88 if it["kind"]=="polygon" else (0.5 if it["kind"]=="triangle" else 0.42)
                obj.inner_radius = sz * inner_ratio

            alpha = 0.35 + energy * 0.6
            c = palette.get(band % 4, alpha=min(1.0, alpha))
            try: obj.color = c
            except: pass

            # Glow halesi
            g = it["glow"]
            g.x = float(it["x"]); g.y = float(it["y"])
            g.radius = sz * 1.8
            gc = palette.get(band % 4, alpha=min(0.5, 0.08 + energy * 0.5))
            try: g.color = gc
            except: pass

    def delete(self):
        for it in self.items: it["obj"].delete()
        for it in self.items: it["glow"].delete()


# ═══════════════════════════════════════════════════════════════════
#  YARATICI KATMANLAR
# ═══════════════════════════════════════════════════════════════════

# Şok dalgaları — beat anında merkezden genişleyen halkalar
class ShockwaveLayer:
    POOL = 16
    def __init__(self, batch):
        self.batch = batch
        self.rings = []
        cx, cy = WINDOW_W//2, WINDOW_H//2
        for _ in range(self.POOL):
            arc = shapes.Arc(cx, cy, 10, thickness=3,
                             color=(255,255,255,0), batch=batch)
            self.rings.append({"obj": arc, "active": False,
                               "r": 0.0, "max": 400, "life": 0.0,
                               "col": 0, "x": cx, "y": cy})
        self._prev_bass = 0.0

    def _spawn(self, palette, intensity, x, y):
        for ring in self.rings:
            if ring["active"]: continue
            ring["active"] = True
            ring["r"]      = 8.0
            ring["max"]    = 200 + intensity * 500
            ring["life"]   = 0.0
            ring["col"]    = random.randint(0, 3)
            ring["x"]      = x
            ring["y"]      = y
            return

    def update(self, dt, bands, palette, t, opacity=1.0, cx=None, cy=None):
        if cx is None: cx, cy = WINDOW_W//2, WINDOW_H//2
        bass = float(bands[0] + bands[1]) * 0.5
        # Beat: bass yükseliş anı
        if bass - self._prev_bass > 0.18 and bass > 0.35:
            self._spawn(palette, bass, cx, cy)
        self._prev_bass = bass

        for ring in self.rings:
            obj = ring["obj"]
            if not ring["active"]:
                obj.color = (0,0,0,0); continue
            ring["life"] += dt
            ring["r"]    += dt * (300 + ring["max"]) * 0.7
            prog = ring["r"] / ring["max"]
            if prog >= 1.0:
                ring["active"] = False
                obj.color = (0,0,0,0); continue
            obj.x = ring["x"]; obj.y = ring["y"]
            obj.radius = float(ring["r"])
            alpha = (1.0 - prog) ** 1.5
            c = palette.get(ring["col"], alpha=min(1.0, alpha * 0.9) * opacity)
            try: obj.color = c
            except: pass

    def delete(self):
        for r in self.rings: r["obj"].delete()


# Yıldız takımı ağı — yüzen düğümler, yakın olanlar çizgiyle bağlanır
class ConstellationLayer:
    NODES = 22
    MAX_LINES = 60
    THRESH = 220
    def __init__(self, batch):
        self.batch = batch
        self.nodes = []
        self.dots  = []
        for i in range(self.NODES):
            x = random.uniform(60, WINDOW_W-60)
            y = random.uniform(60, WINDOW_H-60)
            dot = shapes.Circle(x, y, 3, color=(255,255,255,200), batch=batch)
            self.dots.append(dot)
            self.nodes.append({"x":x,"y":y,
                               "vx":random.uniform(-40,40),
                               "vy":random.uniform(-40,40),
                               "band": i % BANDS})
        self.lines = [shapes.Line(0,0,1,1, thickness=1.2,
                      color=(255,255,255,0), batch=batch)
                      for _ in range(self.MAX_LINES)]

    def update(self, dt, bands, palette, t, complexity=0.6, opacity=1.0, mouse=None, mouse_mode=0):
        mx, my = (mouse if mouse else (None,None))
        # complexity: bağlantı mesafesi → daha yoğun ağ
        self.THRESH = 140 + complexity * 180
        # Düğümleri hareket ettir
        for nd in self.nodes:
            energy = float(bands[nd["band"]])
            nd["x"] += nd["vx"] * dt * (1 + energy)
            nd["y"] += nd["vy"] * dt * (1 + energy)
            if nd["x"]<30 or nd["x"]>WINDOW_W-30: nd["vx"]*=-1; nd["x"]=max(30,min(WINDOW_W-30,nd["x"]))
            if nd["y"]<30 or nd["y"]>WINDOW_H-30: nd["vy"]*=-1; nd["y"]=max(30,min(WINDOW_H-30,nd["y"]))
            if mx is not None and mouse_mode!=0:
                dx=mx-nd["x"]; dy=my-nd["y"]; d2=dx*dx+dy*dy+200; d=math.sqrt(d2)
                if d<300:
                    f=8000/d2
                    if mouse_mode==1: nd["vx"]+=(dx/d)*f*dt*60; nd["vy"]+=(dy/d)*f*dt*60
                    else: nd["vx"]-=(dx/d)*f*dt*60; nd["vy"]-=(dy/d)*f*dt*60
        # Düğümleri çiz
        for i, nd in enumerate(self.nodes):
            energy = float(bands[nd["band"]])
            dot = self.dots[i]
            dot.x = float(nd["x"]); dot.y = float(nd["y"])
            dot.radius = 2.5 + energy * 8
            c = palette.get(nd["band"] % 4, alpha=min(1.0, 0.5 + energy) * opacity)
            try: dot.color = c
            except: pass
        # Yakın çiftleri bağla
        li = 0
        avg_energy = float(np.mean(bands))
        for a in range(self.NODES):
            if li >= self.MAX_LINES: break
            for b in range(a+1, self.NODES):
                if li >= self.MAX_LINES: break
                dx = self.nodes[a]["x"]-self.nodes[b]["x"]
                dy = self.nodes[a]["y"]-self.nodes[b]["y"]
                dist = math.hypot(dx, dy)
                if dist < self.THRESH:
                    ln = self.lines[li]; li += 1
                    ln.x  = float(self.nodes[a]["x"]); ln.y  = float(self.nodes[a]["y"])
                    ln.x2 = float(self.nodes[b]["x"]); ln.y2 = float(self.nodes[b]["y"])
                    fade = (1 - dist/self.THRESH)
                    c = palette.get(1, alpha=min(0.8, fade * (0.3 + avg_energy)) * opacity)
                    try: ln.color = c
                    except: pass
        for j in range(li, self.MAX_LINES):
            self.lines[j].color = (0,0,0,0)

    def delete(self):
        for d in self.dots: d.delete()
        for l in self.lines: l.delete()


# Lissajous / harmonograf — morphing matematiksel eğri
class LissajousLayer:
    SEG = 160
    def __init__(self, batch):
        self.batch = batch
        self.lines = [shapes.Line(0,0,1,1, thickness=2,
                      color=(255,255,255,0), batch=batch)
                      for _ in range(self.SEG)]
        self._phase = 0.0
        self._a = 3.0; self._b = 2.0

    def update(self, dt, bands, palette, t, complexity=0.6, opacity=1.0, cx=None, cy=None):
        if cx is None: cx, cy = WINDOW_W//2, WINDOW_H//2
        mid  = float(bands[3] + bands[4])
        high = float(bands[5] + bands[6])
        rms_like = float(np.mean(bands))
        # complexity: temel a/b oranını yükseltir → daha çok lob/karmaşa
        cbase = 1.0 + complexity * 5.0
        self._a += (cbase + mid * 4.0 - self._a) * dt * 0.5
        self._b += (cbase * 0.7 + high * 5.0 - self._b) * dt * 0.5
        self._phase += dt * (0.6 + rms_like * 2.0)
        A = min(WINDOW_W, WINDOW_H) * (0.18 + rms_like * 0.22)
        B = A
        pts = []
        for i in range(self.SEG):
            th = (i / self.SEG) * math.tau
            x = cx + A * math.sin(self._a * th + self._phase)
            y = cy + B * math.sin(self._b * th)
            pts.append((x, y))
        col_a = palette.get(2)
        col_b = palette.get(3)
        for i in range(self.SEG):
            ln = self.lines[i]
            x1, y1 = pts[i]
            x2, y2 = pts[(i+1) % self.SEG]
            ln.x  = float(x1); ln.y  = float(y1)
            ln.x2 = float(x2); ln.y2 = float(y2)
            # Eğri boyunca renk geçişi
            f = i / self.SEG
            r = int(col_a[0]*(1-f) + col_b[0]*f)
            g = int(col_a[1]*(1-f) + col_b[1]*f)
            b = int(col_a[2]*(1-f) + col_b[2]*f)
            a = int((120 + rms_like * 135) * opacity)
            try: ln.color = (r, g, b, min(255, a))
            except: pass

    def delete(self):
        for l in self.lines: l.delete()


# Akan kurdeleler — yatay sinüs bantları, farklı frekans/renk/hız
class RibbonLayer:
    RIBBONS = 4
    SEG = 90
    def __init__(self, batch):
        self.batch   = batch
        self.ribbons = []
        for r in range(self.RIBBONS):
            lines = [shapes.Line(0,0,1,1, thickness=3,
                     color=(255,255,255,0), batch=batch)
                     for _ in range(self.SEG)]
            self.ribbons.append({
                "lines": lines,
                "y": WINDOW_H * (0.25 + r * 0.18),
                "phase": random.uniform(0, math.tau),
                "speed": 0.8 + r * 0.4,
                "band": [1, 3, 5, 6][r % 4],
                "col": r % 4,
            })

    def update(self, dt, bands, palette, t, complexity=0.6, opacity=1.0):
        # complexity: kaç kurdele görünür + harmonik zenginliği
        visible = 1 + int(complexity * (self.RIBBONS - 1) + 0.5)
        harm    = 1.0 + complexity * 3.0   # ek harmonik gücü
        for ri, rb in enumerate(self.ribbons):
            if ri >= visible:
                for ln in rb["lines"]: ln.color = (0,0,0,0)
                continue
            rb["phase"] += dt * rb["speed"] * 2
            energy = float(bands[rb["band"]])
            amp = WINDOW_H * (0.04 + energy * 0.16)
            base_y = rb["y"]
            col = palette.get(rb["col"], alpha=min(1.0, 0.3 + energy * 0.6) * opacity)
            lines = rb["lines"]
            xs = [WINDOW_W * i / (self.SEG-1) for i in range(self.SEG)]
            for i in range(self.SEG-1):
                p1 = rb["phase"] + (i/self.SEG)*math.pi*5
                p2 = rb["phase"] + ((i+1)/self.SEG)*math.pi*5
                y1 = base_y + math.sin(p1)*amp + math.sin(p1*2.3+1)*amp*0.4*harm*0.3 + math.sin(p1*4.1+2)*amp*0.2*complexity
                y2 = base_y + math.sin(p2)*amp + math.sin(p2*2.3+1)*amp*0.4*harm*0.3 + math.sin(p2*4.1+2)*amp*0.2*complexity
                ln = lines[i]
                ln.x  = float(xs[i]);   ln.y  = float(y1)
                ln.x2 = float(xs[i+1]); ln.y2 = float(y2)
                try: ln.color = col
                except: pass
            lines[self.SEG-1].color = (0,0,0,0)

    def delete(self):
        for rb in self.ribbons:
            for l in rb["lines"]: l.delete()


# Spiral galaksi — dönen spiral kollar, bass ile nefes alır
class SpiralLayer:
    ARMS = 3
    PER_ARM = 26
    def __init__(self, batch):
        self.batch = batch
        self.dots  = []
        cx, cy = WINDOW_W//2, WINDOW_H//2
        for arm in range(self.ARMS):
            for j in range(self.PER_ARM):
                d = shapes.Circle(cx, cy, 3, color=(255,255,255,0), batch=batch)
                self.dots.append({"obj": d, "arm": arm, "idx": j,
                                  "band": j % BANDS})
        self.angle = 0.0

    def update(self, dt, bands, palette, t, complexity=0.6, opacity=1.0, cx=None, cy=None):
        if cx is None: cx, cy = WINDOW_W//2, WINDOW_H//2
        bass = float(bands[0] + bands[1]) * 0.5
        self.angle += dt * (0.4 + bass * 1.5)
        scale = (min(WINDOW_W, WINDOW_H) * 0.45) * (0.7 + bass * 0.5)
        # complexity: kıvrım sıkılığı (winding) + görünür nokta oranı
        winding = 3.0 + complexity * 5.0
        visible_frac = 0.3 + complexity * 0.7
        for d in self.dots:
            arm   = d["arm"]
            j     = d["idx"]
            frac  = j / self.PER_ARM
            if frac > visible_frac:
                d["obj"].color = (0,0,0,0); continue
            theta = self.angle + arm * (math.tau / self.ARMS) + frac * winding
            radius = frac * scale
            x = cx + math.cos(theta) * radius
            y = cy + math.sin(theta) * radius
            energy = float(bands[d["band"]])
            obj = d["obj"]
            obj.x = float(x); obj.y = float(y)
            obj.radius = 2 + frac * 4 + energy * 6
            alpha = 0.3 + energy * 0.6 + (1-frac) * 0.2
            c = palette.get(arm % 4, alpha=min(1.0, alpha) * opacity)
            try: obj.color = c
            except: pass

    def delete(self):
        for d in self.dots: d["obj"].delete()


# ── Çift sarmal (DNA helix) — dönen ikili sarmal + basamaklar ────────
class HelixLayer:
    N = 44
    def __init__(self, batch):
        self.batch = batch
        self.a = [shapes.Circle(0,0,4, color=(255,255,255,0), batch=batch) for _ in range(self.N)]
        self.b = [shapes.Circle(0,0,4, color=(255,255,255,0), batch=batch) for _ in range(self.N)]
        self.rungs = [shapes.Line(0,0,1,1, thickness=2, color=(255,255,255,0), batch=batch)
                      for _ in range(self.N)]
        self.phase = 0.0

    def update(self, dt, bands, palette, t, complexity=0.6, opacity=1.0, cx=None, cy=None):
        if cx is None: cx, cy = WINDOW_W//2, WINDOW_H//2
        mid  = float(bands[2] + bands[3])
        bass = float(bands[1])
        self.phase += dt * (1.0 + bass * 2.5)
        radius = 60 + complexity * 120 + mid * 80
        span   = WINDOW_H * 0.8
        turns  = 2.0 + complexity * 3.0
        for i in range(self.N):
            frac = i / (self.N - 1)
            y = cy - span/2 + frac * span
            ang = self.phase + frac * math.tau * turns
            energy = float(bands[i % BANDS])
            xa = cx + math.cos(ang) * radius
            xb = cx + math.cos(ang + math.pi) * radius
            # ön/arka derinlik → boyut
            za = (math.sin(ang) + 1) * 0.5
            zb = (math.sin(ang + math.pi) + 1) * 0.5
            da = self.a[i]; db = self.b[i]
            da.x = float(xa); da.y = float(y); da.radius = 2 + za*5 + energy*4
            db.x = float(xb); db.y = float(y); db.radius = 2 + zb*5 + energy*4
            ca = palette.get(0, alpha=min(1.0, (0.3 + za*0.6)) * opacity)
            cb = palette.get(2, alpha=min(1.0, (0.3 + zb*0.6)) * opacity)
            try: da.color = ca; db.color = cb
            except: pass
            # basamak (her 2 noktada bir görünür)
            rg = self.rungs[i]
            if i % 2 == 0:
                rg.x = float(xa); rg.y = float(y)
                rg.x2 = float(xb); rg.y2 = float(y)
                rc = palette.get(1, alpha=min(0.6, 0.2 + mid*0.4) * opacity)
                try: rg.color = rc
                except: pass
            else:
                rg.color = (0,0,0,0)

    def delete(self):
        for o in self.a + self.b + self.rungs: o.delete()


# ── Tünel / solucan deliği — içe doğru kayan çokgenler ───────────────
class TunnelLayer:
    RINGS = 18
    def __init__(self, batch):
        self.batch = batch
        self.rings = []
        cx, cy = WINDOW_W//2, WINDOW_H//2
        for i in range(self.RINGS):
            sides = random.choice([3, 4, 5, 6, 6, 8])
            arc = shapes.Star(cx, cy, 40, 38, num_spikes=sides, rotation=0.0,
                              color=(255,255,255,0), batch=batch)
            self.rings.append({"obj": arc, "z": i / self.RINGS, "spin": random.uniform(-1,1)})

    def update(self, dt, bands, palette, t, complexity=0.6, opacity=1.0, cx=None, cy=None):
        if cx is None: cx, cy = WINDOW_W//2, WINDOW_H//2
        bass = float(bands[0] + bands[1]) * 0.5
        speed = 0.15 + bass * 0.5
        maxr  = min(WINDOW_W, WINDOW_H) * 0.7
        for r in self.rings:
            r["z"] -= dt * speed
            if r["z"] <= 0.02:
                r["z"] += 1.0
            # perspektif: küçük z = uzak (küçük), büyük z = yakın (büyük)
            radius = maxr * (r["z"] ** 1.8)
            obj = r["obj"]
            obj.x = cx; obj.y = cy
            obj.outer_radius = max(4, radius)
            obj.inner_radius = max(3, radius * 0.92)
            try: obj.rotation = float(t * 30 * r["spin"])
            except: pass
            depth_alpha = (1.0 - r["z"]) * (0.4 + bass * 0.6)
            col_idx = int(r["z"] * 4) % 4
            c = palette.get(col_idx, alpha=min(1.0, depth_alpha) * opacity)
            try: obj.color = c
            except: pass

    def delete(self):
        for r in self.rings: r["obj"].delete()


# ── Yıldız patlaması ışınları — merkezden uzanan, pulse eden ışınlar ─
class BurstRaysLayer:
    MAX = 72
    def __init__(self, batch):
        self.batch = batch
        self.rays = [shapes.Line(0,0,1,1, thickness=2, color=(255,255,255,0), batch=batch)
                     for _ in range(self.MAX)]
        self.angle = 0.0

    def update(self, dt, bands, palette, t, complexity=0.6, opacity=1.0, cx=None, cy=None):
        if cx is None: cx, cy = WINDOW_W//2, WINDOW_H//2
        bass = float(bands[0] + bands[1]) * 0.5
        n = 16 + int(complexity * (self.MAX - 16))
        self.angle += dt * (20 + bass * 80)
        inner = 30 + bass * 60
        for i, ray in enumerate(self.rays):
            if i >= n:
                ray.color = (0,0,0,0); continue
            ang = math.radians(self.angle) + (i / n) * math.tau
            band = i % BANDS
            energy = float(bands[band])
            length = inner + 60 + energy * 320
            x1 = cx + math.cos(ang) * inner
            y1 = cy + math.sin(ang) * inner
            x2 = cx + math.cos(ang) * length
            y2 = cy + math.sin(ang) * length
            ray.x = float(x1); ray.y = float(y1)
            ray.x2 = float(x2); ray.y2 = float(y2)
            c = palette.get(band % 4, alpha=min(1.0, 0.25 + energy * 0.7) * opacity)
            try: ray.color = c
            except: pass

    def delete(self):
        for r in self.rays: r.delete()


# ── Synthwave ızgara — perspektif zemin, bass ile tepe yapar ─────────
class GridWaveLayer:
    COLS = 16
    ROWS = 12
    def __init__(self, batch):
        self.batch = batch
        # yatay çizgiler
        self.hlines = [shapes.Line(0,0,1,1, thickness=2, color=(255,255,255,0), batch=batch)
                       for _ in range(self.ROWS)]
        # dikey çizgiler
        self.vlines = [shapes.Line(0,0,1,1, thickness=2, color=(255,255,255,0), batch=batch)
                       for _ in range(self.COLS+1)]
        self.scroll = 0.0

    def update(self, dt, bands, palette, t, complexity=0.6, opacity=1.0):
        bass = float(bands[0] + bands[1]) * 0.5
        self.scroll = (self.scroll + dt * (0.3 + bass)) % 1.0
        horizon = WINDOW_H * 0.55
        cx = WINDOW_W / 2
        col = palette.get(2, alpha=min(1.0, 0.4 + bass * 0.5) * opacity)
        # Yatay çizgiler — perspektifle aşağı doğru sıklaşır
        for i, ln in enumerate(self.hlines):
            f = ((i + self.scroll) / self.ROWS)
            persp = f ** 2.2
            y = horizon - persp * horizon * 0.95
            bump = math.sin(t*3 + i) * bass * 30
            ln.x = 0; ln.y = float(y + bump)
            ln.x2 = WINDOW_W; ln.y2 = float(y + bump)
            a = (1 - f) * (0.4 + bass*0.5)
            try: ln.color = palette.get(2, alpha=min(1.0, a) * opacity)
            except: pass
        # Dikey çizgiler — ufuktan tabana yelpaze
        for j, ln in enumerate(self.vlines):
            fx = j / self.COLS
            x_far  = cx + (fx - 0.5) * WINDOW_W * 0.15
            x_near = cx + (fx - 0.5) * WINDOW_W * 2.2
            ln.x = float(x_far);  ln.y = float(horizon)
            ln.x2 = float(x_near); ln.y2 = 0.0
            try: ln.color = col
            except: pass

    def delete(self):
        for l in self.hlines + self.vlines: l.delete()


# ── Güneş sistemi — merkez yıldız + yörüngedeki gezegenler ───────────
class OrbitsLayer:
    PLANETS = 7
    def __init__(self, batch):
        self.batch = batch
        cx, cy = WINDOW_W//2, WINDOW_H//2
        self.sun = shapes.Circle(cx, cy, 20, color=(255,255,255,0), batch=batch)
        self.orbits = [shapes.Arc(cx, cy, 60+i*55, color=(255,255,255,0), batch=batch)
                       for i in range(self.PLANETS)]
        self.planets = [shapes.Circle(cx, cy, 8, color=(255,255,255,0), batch=batch)
                        for i in range(self.PLANETS)]
        self.ang = [random.uniform(0, math.tau) for _ in range(self.PLANETS)]

    def update(self, dt, bands, palette, t, complexity=0.6, opacity=1.0, cx=None, cy=None):
        if cx is None: cx, cy = WINDOW_W//2, WINDOW_H//2
        bass = float(bands[0] + bands[1]) * 0.5
        visible = 2 + int(complexity * (self.PLANETS - 2))
        # Güneş
        self.sun.x = cx; self.sun.y = cy
        self.sun.radius = 16 + bass * 30
        self.sun.color = palette.get(0, alpha=min(1.0, 0.6 + bass*0.4) * opacity)
        for i in range(self.PLANETS):
            orbit = self.orbits[i]; planet = self.planets[i]
            if i >= visible:
                orbit.color = (0,0,0,0); planet.color = (0,0,0,0); continue
            r = 60 + i * 55
            spd = (0.8 - i*0.08) * (1 + bass)
            self.ang[i] += dt * spd
            px = cx + math.cos(self.ang[i]) * r
            py = cy + math.sin(self.ang[i]) * r
            energy = float(bands[i % BANDS])
            orbit.x = cx; orbit.y = cy; orbit.radius = r
            orbit.color = palette.get(1, alpha=0.12 * opacity)
            planet.x = float(px); planet.y = float(py)
            planet.radius = 5 + energy * 14
            planet.color = palette.get((i % 3)+1, alpha=min(1.0, 0.5 + energy*0.5) * opacity)

    def delete(self):
        self.sun.delete()
        for o in self.orbits: o.delete()
        for p in self.planets: p.delete()


# Sürpriz katman: yukarıdan düşen şekiller (event tetiklemeli)
class ShapeRainLayer:
    POOL = 50
    def __init__(self, batch):
        self.batch = batch
        self.items = []
        for _ in range(self.POOL):
            kind = random.choice(["circle","star","tri"])
            if kind == "circle":
                obj = shapes.Circle(0, -50, 6, color=(255,255,255,0), batch=batch)
            elif kind == "star":
                obj = shapes.Star(0, -50, 12, 5, num_spikes=5, color=(255,255,255,0), batch=batch)
            else:
                obj = shapes.Star(0, -50, 12, 6, num_spikes=3, color=(255,255,255,0), batch=batch)
            self.items.append({"obj": obj, "active": False, "kind": kind,
                               "x": 0, "y": -50, "vy": 0, "vr": 0,
                               "angle": 0, "size": 10, "col": 0})
        self.active = False
        self._spawn_acc = 0.0

    def burst(self, n=24):
        """Bir grup şekli düşmeye başlat."""
        self.active = True
        spawned = 0
        for it in self.items:
            if it["active"]: continue
            if spawned >= n: break
            it["active"] = True
            it["x"]     = random.uniform(40, WINDOW_W-40)
            it["y"]     = WINDOW_H + random.uniform(20, 200)
            it["vy"]    = random.uniform(120, 300)
            it["vr"]    = random.uniform(-180, 180)
            it["angle"] = random.uniform(0, 360)
            it["size"]  = random.uniform(8, 26)
            it["col"]   = random.randint(0, 3)
            spawned += 1

    def update(self, dt, bands, palette, t):
        any_active = False
        for it in self.items:
            obj = it["obj"]
            if not it["active"]:
                obj.color = (0,0,0,0); continue
            any_active = True
            it["y"]     -= it["vy"] * dt
            it["angle"] += it["vr"] * dt
            it["vy"]    += 200 * dt    # yerçekimi
            if it["y"] < -40:
                it["active"] = False
                obj.color = (0,0,0,0); continue
            energy = float(bands[1])
            sz = it["size"] * (1 + energy * 0.8)
            obj.x = float(it["x"]); obj.y = float(it["y"])
            try: obj.rotation = float(it["angle"])
            except: pass
            if isinstance(obj, shapes.Circle):
                obj.radius = sz
            elif isinstance(obj, shapes.Star):
                obj.outer_radius = sz
                obj.inner_radius = sz * (0.5 if it["kind"]=="tri" else 0.42)
            c = palette.get(it["col"], alpha=0.85)
            try: obj.color = c
            except: pass
        self.active = any_active

    def delete(self):
        for it in self.items: it["obj"].delete()


# Katman 4: Parçacık sistemi — high_mid + presence + brilliance
class ParticleLayer:
    MAX = 300

    def __init__(self, batch):
        self.batch     = batch
        self.particles = []
        self._pool     = []
        cx, cy = WINDOW_W//2, WINDOW_H//2
        for _ in range(self.MAX):
            p = shapes.Circle(cx, cy, 2, color=(255,255,255,0), batch=batch)
            self._pool.append({
                "obj": p, "x": cx, "y": cy,
                "vx": 0.0, "vy": 0.0,
                "life": 0.0, "max_life": 1.0,
                "r": 2.0, "alive": False,
                "col_idx": 0,
            })

    def _spawn(self, n, bands, palette):
        cx, cy = WINDOW_W//2, WINDOW_H//2
        energy = float(bands[4] + bands[5] + bands[6])
        for p in self._pool:
            if p["alive"]: continue
            if n <= 0: break
            angle    = random.uniform(0, math.tau)
            speed    = random.uniform(100, 320) * (1 + energy * 3)
            p["x"]   = cx + random.uniform(-20, 20)
            p["y"]   = cy + random.uniform(-20, 20)
            p["vx"]  = math.cos(angle) * speed
            p["vy"]  = math.sin(angle) * speed
            p["life"]     = 0.0
            p["max_life"] = random.uniform(0.6, 2.0)
            p["r"]        = random.uniform(2.0, 8.0) * (1 + energy * 1.5)
            p["alive"]    = True
            p["col_idx"]  = random.randint(0, 3)
            n -= 1

    def update(self, dt, bands, palette, t, rate_mult=1.0, opacity=1.0,
               mouse=None, mouse_mode=0):
        high_energy = float(bands[4] + bands[5] + bands[6]) / 3.0
        spawn_n = int(high_energy * 25 * rate_mult + float(bands[1]) * 10)
        if spawn_n > 0:
            self._spawn(spawn_n, bands, palette)

        mx, my = (mouse if mouse else (None, None))

        for p in self._pool:
            if not p["alive"]: continue
            p["life"] += dt
            if p["life"] >= p["max_life"]:
                p["alive"] = False
                p["obj"].color = (0, 0, 0, 0)
                continue
            prog  = p["life"] / p["max_life"]
            alpha = int((1.0 - prog) * 220)

            # Fare etkilesimi: mode 1=cek, 2=it
            if mx is not None and mouse_mode != 0:
                dx = mx - p["x"]; dy = my - p["y"]
                dist2 = dx*dx + dy*dy + 100
                dist  = math.sqrt(dist2)
                if dist < 350:
                    force = 9000 / dist2
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    if mouse_mode == 1:    # cek
                        p["vx"] += fx * dt * 60
                        p["vy"] += fy * dt * 60
                    else:                   # it
                        p["vx"] -= fx * dt * 60
                        p["vy"] -= fy * dt * 60

            p["vx"] *= (1 - dt * 0.9)
            p["vy"] *= (1 - dt * 0.9)
            p["x"]  += p["vx"] * dt
            p["y"]  += p["vy"] * dt
            r     = p["r"] * (1.0 - prog * 0.6)
            obj   = p["obj"]
            obj.x = p["x"]
            obj.y = p["y"]
            obj.radius = max(0.5, r)
            c = palette.get(p["col_idx"], alpha=(alpha/255) * opacity)
            try: obj.color = c
            except: pass

    def delete(self):
        for p in self._pool: p["obj"].delete()


# Katman 5: Dalgalı çizgi (ses dalgası) — rms
class WaveformLayer:
    def __init__(self, batch):
        self.batch  = batch
        self.N      = 160
        self.lines  = []
        for i in range(self.N - 1):
            l = shapes.Line(0,0,1,1, thickness=1, color=(200,200,255,60), batch=batch)
            self.lines.append(l)
        self._phase = 0.0

    def update(self, dt, bands, palette, t, rms, amp_mult=1.0):
        self._phase += dt * (3 + rms * 12)
        mid   = float(bands[2] + bands[3])
        high  = float(bands[4] + bands[5])
        bass  = float(bands[1])
        amp   = float((rms * 0.4 + mid * 0.35 + high * 0.25) * WINDOW_H * 0.35) * amp_mult
        amp   = max(15, amp)
        # Ust dalga
        y_top = WINDOW_H * 0.82
        # Alt dalga (yansima)
        y_bot = WINDOW_H * 0.18
        xs = [int(WINDOW_W * i / (self.N - 1)) for i in range(self.N)]
        c_top = palette.get(2, alpha=min(1.0, 0.3 + rms * 0.6))
        c_bot = palette.get(3, alpha=min(1.0, 0.2 + bass * 0.5))
        half = len(self.lines) // 2
        for i in range(min(half, self.N - 1)):
            phase2 = self._phase + (i / self.N) * math.pi * 5
            dy = math.sin(phase2) * amp * 0.55
            dy += math.sin(phase2 * 2.1 + 0.8) * amp * 0.3
            dy += math.sin(phase2 * 0.5 + 2.1) * amp * 0.15
            # Ust
            self.lines[i].x  = xs[i];   self.lines[i].y  = y_top + dy
            self.lines[i].x2 = xs[i+1]; self.lines[i].y2 = y_top + dy * 0.9
            try: self.lines[i].color = c_top
            except: pass
        for i in range(min(half, self.N - 1)):
            phase2 = self._phase * 0.7 + (i / self.N) * math.pi * 4
            dy = math.sin(phase2) * amp * 0.45
            dy += math.sin(phase2 * 1.7 + 1.2) * amp * 0.35
            j = half + i
            if j < len(self.lines):
                self.lines[j].x  = xs[i];   self.lines[j].y  = y_bot - dy
                self.lines[j].x2 = xs[i+1]; self.lines[j].y2 = y_bot - dy * 0.9
                try: self.lines[j].color = c_bot
                except: pass

    # Ikinci dalga (yansıma)
    def update2(self, dt, bands, palette, t, rms):
        pass  # Tek dalga yeterli


# ── Ana Pencere ───────────────────────────────────────────────────────────────
class VisualizerWindow(pyglet.window.Window):
    def __init__(self):
        super().__init__(
            width=WINDOW_W, height=WINDOW_H,
            caption="AI Music Visualizer",
            resizable=True, vsync=True,
        )
        self.set_minimum_size(640, 360)

        # Blending aktif — additive glow için
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)

        self.audio       = AudioAnalyzer()
        self.dir_queue   = queue.Queue(maxsize=8)
        self.ai = AIDirector(self.audio, self.dir_queue)
        self.palette     = Palette()
        # Compose parametreleri (defaults)
        self.mandala_speed  = 1.0
        self.particle_rate  = 1.0
        self.wave_amp       = 1.0
        self.radial_bars    = 64
        self._tgt_mandala_speed = 1.0
        self._tgt_particle_rate = 1.0
        self._tgt_wave_amp      = 1.0

        self.batch  = pyglet.graphics.Batch()
        self.batch2 = pyglet.graphics.Batch()  # HUD

        # Fade overlay
        self.fade = shapes.Rectangle(0, 0, WINDOW_W, WINDOW_H,
                                     color=(0, 0, 0, 35))

        # Katmanlar (çizim sırasına göre — arka → ön)
        self.layer_grid    = GridWaveLayer(self.batch)
        self.layer_tunnel  = TunnelLayer(self.batch)
        self.layer_ribbon  = RibbonLayer(self.batch)
        self.layer_spiral  = SpiralLayer(self.batch)
        self.layer_orbits  = OrbitsLayer(self.batch)
        self.layer_rays    = BurstRaysLayer(self.batch)
        self.layer_liss    = LissajousLayer(self.batch)
        self.layer_helix   = HelixLayer(self.batch)
        self.layer_const   = ConstellationLayer(self.batch)
        self.layer_shock   = ShockwaveLayer(self.batch)
        self.layer_parts   = ParticleLayer(self.batch)

        # ShapeRain (sürpriz katman)
        self.layer_rain = ShapeRainLayer(self.batch)

        # Beat flash overlay (additive)
        self.flash = shapes.Rectangle(0, 0, WINDOW_W, WINDOW_H,
                                      color=(255, 255, 255, 0), batch=None)
        self._flash_intensity = 0.0

        # AI kreatif direktör state
        self.ai_driving      = False
        self.complexity      = 0.6
        self.anim_speed      = 1.0
        self._tgt_complexity = 0.6
        self._tgt_speed      = 1.0
        self._cur_scene_name = "full"
        # AI sahne adı → aktif katmanlar
        self.ai_scene_map = {
            "flow":    ["ribbon","liss","parts","helix"],
            "galaxy":  ["spiral","rays","shock","parts"],
            "web":     ["const","liss","shock","grid"],
            "cosmic":  ["orbits","const","helix","parts"],
            "storm":   ["grid","rays","spiral","shock","parts"],
            "minimal": ["liss","orbits"],
            "full":    ["tunnel","ribbon","spiral","rays","liss","const","shock","parts"],
        }

        # ── Sahne sistemi ──────────────────────────────────────────
        # Aktif katmanlar (sahne sistemi bunu yönetir)
        self.layers_on    = {"grid":False, "tunnel":False, "ribbon":True,
                             "spiral":True, "orbits":False, "rays":False,
                             "liss":True, "helix":False, "const":True,
                             "shock":True, "parts":True}
        # Yumuşak geçiş: her katmanın anlık opaklığı + nefes fazı
        self.layer_op     = {k: 1.0 for k in self.layers_on}
        self.layer_phase  = {k: random.uniform(0, math.tau) for k in self.layers_on}
        # Her sahne = aktif katman kombinasyonu. Otomatik döner.
        self.scenes = [
            {"name":"Akis",      "on":["ribbon","liss","parts"]},
            {"name":"Galaksi",   "on":["spiral","rays","shock","parts"]},
            {"name":"Ag",        "on":["const","liss","shock"]},
            {"name":"Helix",     "on":["helix","const","parts"]},
            {"name":"Tunel",     "on":["tunnel","rays","parts"]},
            {"name":"Synthwave", "on":["grid","rays","shock"]},
            {"name":"Yorunge",   "on":["orbits","const","parts"]},
            {"name":"Patlama",   "on":["rays","spiral","shock","parts"]},
            {"name":"Tam",       "on":["ribbon","spiral","liss","const","shock","parts","rays"]},
            {"name":"Minimal",   "on":["liss","parts"]},
            {"name":"Kozmik",    "on":["spiral","const","helix"]},
        ]
        self.auto_scenes   = True
        self.scene_idx     = 3          # "Tam" ile başla
        self.scene_timer   = 0.0
        self.scene_dur     = 22.0       # sahne süresi (sn)
        self._apply_scene(self.scene_idx)

        # ── Drop algılama ──────────────────────────────────────────
        self._energy_avg   = 0.0        # uzun ortalama enerji
        self._drop_cooldown = 0.0

        # Etkileşim state
        self.mouse_pos    = None
        self.mouse_mode   = 0       # 0=off, 1=cek, 2=it
        self.show_hud     = True
        self.show_help    = False
        self.sensitivity  = 1.0
        self._prev_bass   = 0.0

        # HUD
        self.fps_lbl  = pyglet.text.Label("", x=14, y=self.height-24,
                                          font_name="Consolas", font_size=11,
                                          color=(200,200,210,150), batch=self.batch2)
        self.mood_lbl = pyglet.text.Label("Baslatiliyor...", x=14, y=14,
                                          font_name="Consolas", font_size=14,
                                          color=(150,200,255,210), batch=self.batch2)
        self.hint_lbl = pyglet.text.Label("[H] yardim  [TAB] HUD",
                                          x=14, y=40, font_name="Consolas",
                                          font_size=10, color=(120,130,150,140),
                                          batch=self.batch2)
        self.scene_lbl = pyglet.text.Label("", x=14, y=66,
                                           font_name="Consolas", font_size=12,
                                           color=(255,200,120,200), batch=self.batch2)
        # Yardim overlay (ayri batch)
        self.help_batch = pyglet.graphics.Batch()
        help_text = ("KONTROLLER\n\n"
                     "1-5    Katmanlari ac/kapa\n"
                     "       (1:dalga 2:radyal 3:mandala\n"
                     "        4:yildiz 5:parcacik)\n"
                     "M      Fare modu (kapali/cek/it)\n"
                     "+/-    Hassasiyet\n"
                     "F      Tam ekran\n"
                     "SPACE  Renk paletini karistir\n"
                     "TAB    HUD goster/gizle\n"
                     "H      Bu menu\n"
                     "ESC/Q  Cikis")
        self.help_bg = shapes.Rectangle(0, 0, 420, 300,
                                        color=(10, 12, 22, 230), batch=self.help_batch)
        self.help_lbl = pyglet.text.Label(help_text, x=0, y=0,
                                          font_name="Consolas", font_size=13,
                                          color=(210,220,240,255), multiline=True,
                                          width=400, batch=self.help_batch)

        self.t       = 0.0
        self._fps_dt = 0.0; self._fps_fr = 0; self._fps_v = 0.0
        # Görsel yumuşatma + zoom pulse efekt state
        self.vis_bands   = np.zeros(BANDS)
        self._zoom       = 1.0
        self._zoom_pulse = 0.0

        self.audio.start()
        self.ai.start()
        pyglet.clock.schedule_interval(self.update, 1.0 / TARGET_FPS)
        print("[App] Hazir!")

    def update(self, dt):
        self.t += dt

        # AI direktifi — kreatif direktör modu
        try:
            while True:
                d = self.dir_queue.get_nowait()
                self.palette.set(d)
                self.mood_lbl.text = f"AI: {d.get('mood','?')}"
                # AI sahne seçtiyse uygula (otomatik rotasyonu devralır)
                scene = d.get("scene")
                if scene and scene in self.ai_scene_map:
                    self.ai_driving = True
                    self.auto_scenes = False
                    layers = self.ai_scene_map[scene]
                    for key in self.layers_on:
                        self.layers_on[key] = (key in layers)
                    self._cur_scene_name = scene
                    mode = "AI"
                    self.scene_lbl.text = f"Sahne: {scene} [{mode}]"
                # Karmaşıklık & hız hedefleri
                self._tgt_complexity = float(d.get("complexity", 0.6))
                self._tgt_speed      = float(d.get("speed", 1.0))
                self._tgt_particle_rate = 0.5 + self._tgt_complexity * 2.0
                if d.get("shape_rain"):
                    self.layer_rain.burst(n=22)
        except queue.Empty:
            pass
        # Smooth interpolate
        spd = dt * 1.2
        self.complexity    += (self._tgt_complexity - self.complexity) * min(1.0, spd)
        self.anim_speed    += (self._tgt_speed      - self.anim_speed) * min(1.0, spd)
        self.particle_rate += (self._tgt_particle_rate - self.particle_rate) * min(1.0, spd)

        self.palette.update(dt)

        bands_raw, rms, peak = self.audio.get()
        bands_raw = bands_raw * self.sensitivity
        rms_s = rms * self.sensitivity

        # ── Görsel yumuşatma: envelope follower ────────────────────
        # Hızlı atak (vuruşu yakala) + yavaş bırakma (akıcı iz) → pürüzsüz hareket
        for i in range(BANDS):
            target = float(bands_raw[i])
            if target > self.vis_bands[i]:
                self.vis_bands[i] += (target - self.vis_bands[i]) * min(1.0, dt * 22)  # atak
            else:
                self.vis_bands[i] += (target - self.vis_bands[i]) * min(1.0, dt * 7)   # bırakma
        bands = self.vis_bands

        # Beat algıla — bass yükseliş anı → flash + zoom pulse
        bass = float(bands_raw[1])
        if bass - self._prev_bass > 0.25 and bass > 0.4:
            self._flash_intensity = min(1.0, self._flash_intensity + bass * 0.6)
            self._zoom_pulse = min(0.10, self._zoom_pulse + bass * 0.06)  # zoom darbesi
        self._prev_bass = bass
        self._flash_intensity *= (1 - dt * 5.0)
        # Zoom pulse yumuşak sönüm (yaylı geri dönüş)
        self._zoom_pulse *= (1 - dt * 4.0)
        self._zoom = 1.0 + self._zoom_pulse

        # ── Sahne otomatik rotasyonu ───────────────────────────────
        if self.auto_scenes:
            self.scene_timer += dt
            if self.scene_timer >= self.scene_dur:
                self.scene_timer = 0.0
                self.scene_idx = (self.scene_idx + 1) % len(self.scenes)
                self._apply_scene(self.scene_idx)

        # ── Drop algılama → özel an ────────────────────────────────
        cur_energy = float(np.mean(bands))
        self._energy_avg = self._energy_avg * 0.99 + cur_energy * 0.01
        self._drop_cooldown -= dt
        # Ani büyük enerji sıçraması = drop
        if (cur_energy > self._energy_avg * 2.2 and cur_energy > 0.5
                and self._drop_cooldown <= 0):
            self._trigger_drop(cur_energy)
            self._drop_cooldown = 4.0   # 4sn boşluk

        # ShapeRain her zaman güncellenir (aktif şekiller düşsün)
        self.layer_rain.update(dt, bands, self.palette, self.t)

        # Fade alpha: sessizken daha koyu
        fade_a = max(12, min(50, 32 - int(rms_s * 120)))
        bg = self.palette.bg_gl()
        pyglet.gl.glClearColor(bg[0], bg[1], bg[2], 1.0)
        self.fade.color = (
            clamp8(bg[0]*255), clamp8(bg[1]*255), clamp8(bg[2]*255), fade_a
        )

        # Flash rengi — paletin en parlak rengiyle
        fc = self.palette.get(0, alpha=1.0)
        self.flash.color = (fc[0], fc[1], fc[2],
                            clamp8(self._flash_intensity * 60))

        # ── Katman opaklıkları: yumuşak fade + organik nefes ───────
        for k in self.layers_on:
            tgt = 1.0 if self.layers_on[k] else 0.0
            # ~0.6sn'de fade (in: hızlı, out: biraz daha yavaş hisset)
            rate = 3.2 if tgt > self.layer_op[k] else 2.0
            self.layer_op[k] += (tgt - self.layer_op[k]) * min(1.0, dt * rate)
            # Yavaş nefes — transparanlık ara ara dalgalanır
            self.layer_phase[k] += dt * 0.35
        def op_of(k):
            breath = 0.82 + 0.18 * math.sin(self.layer_phase[k])
            return max(0.0, self.layer_op[k]) * breath

        # Katman güncellemeleri — complexity + speed + opacity ile
        cx_v = self.complexity
        sp_v = self.anim_speed
        dt_s = dt * sp_v
        VIS = 0.01
        P = self.palette
        m, mm = self.mouse_pos, self.mouse_mode

        def up(k, layer, **kw):
            if self.layer_op[k] > VIS:
                layer.update(dt_s, bands, P, self.t, opacity=op_of(k), **kw)
                setattr(self, f"_cleared_{k}", False)
            elif not getattr(self, f"_cleared_{k}", False):
                self.layer_op[k] = 0.0
                setattr(self, f"_cleared_{k}", True)
                self._clear_layer(layer)

        up("grid",   self.layer_grid,   complexity=cx_v)
        up("tunnel", self.layer_tunnel, complexity=cx_v)
        up("ribbon", self.layer_ribbon, complexity=cx_v)
        up("spiral", self.layer_spiral, complexity=cx_v)
        up("orbits", self.layer_orbits, complexity=cx_v)
        up("rays",   self.layer_rays,   complexity=cx_v)
        up("liss",   self.layer_liss,   complexity=cx_v)
        up("helix",  self.layer_helix,  complexity=cx_v)
        up("const",  self.layer_const,  complexity=cx_v, mouse=m, mouse_mode=mm)
        up("shock",  self.layer_shock)
        up("parts",  self.layer_parts,  rate_mult=self.particle_rate, mouse=m, mouse_mode=mm)

        # FPS
        self._fps_dt += dt; self._fps_fr += 1
        if self._fps_dt >= 0.5:
            self._fps_v  = self._fps_fr / self._fps_dt
            self._fps_dt = 0.0; self._fps_fr = 0
        snap = self.audio.snapshot()
        mode_txt = ["fare:kapali", "fare:cek", "fare:it"][self.mouse_mode]
        self.fps_lbl.text = (f"FPS:{self._fps_v:.0f}  RMS:{rms:.3f}  "
                             f"BPM:{snap['bpm']:.0f}  x{self.sensitivity:.1f}  {mode_txt}")
        self.fps_lbl.y = self.height - 24

    def on_draw(self):
        gl = pyglet.gl
        self.clear()

        # 1) Fade overlay — normal blend (trail efekti)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.fade.width  = self.width
        self.fade.height = self.height
        self.fade.draw()

        # 2) Tüm görsel katmanlar — ADDITIVE blend + zoom pulse
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
        z = self._zoom
        if abs(z - 1.0) > 0.001:
            from pyglet.math import Mat4, Vec3
            cx, cy = self.width / 2, self.height / 2
            # Merkez etrafında ölçekle: T(c) · S(z) · T(-c)
            self.view = (Mat4.from_translation(Vec3(cx, cy, 0)) @
                         Mat4.from_scale(Vec3(z, z, 1)) @
                         Mat4.from_translation(Vec3(-cx, -cy, 0)))
            self.batch.draw()
            self.view = Mat4()   # sıfırla (HUD/flash etkilenmesin)
        else:
            self.batch.draw()

        # 3) Beat flash — additive
        if self._flash_intensity > 0.01:
            self.flash.width  = self.width
            self.flash.height = self.height
            self.flash.draw()

        # 4) HUD — normal blend
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        if self.show_hud:
            self.batch2.draw()

        # 5) Yardım menüsü
        if self.show_help:
            self.help_bg.x  = self.width // 2 - 210
            self.help_bg.y  = self.height // 2 - 160
            self.help_lbl.x = self.width // 2 - 190
            self.help_lbl.y = self.height // 2 + 130
            self.help_batch.draw()

    def on_resize(self, w, h):
        super().on_resize(w, h)
        self.fps_lbl.y = h - 24

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_pos = (x, y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, mod):
        self.mouse_pos = (x, y)

    def on_mouse_leave(self, x, y):
        self.mouse_pos = None

    def on_key_press(self, sym, mod):
        key = pyglet.window.key
        if sym in (key.ESCAPE, key.Q):
            self.close()
        elif sym == key._1: self.layers_on["ribbon"] = not self.layers_on["ribbon"]
        elif sym == key._2: self.layers_on["spiral"] = not self.layers_on["spiral"]
        elif sym == key._3: self.layers_on["liss"]   = not self.layers_on["liss"]
        elif sym == key._4: self.layers_on["const"]  = not self.layers_on["const"]
        elif sym == key._5: self.layers_on["shock"]  = not self.layers_on["shock"]
        elif sym == key._6: self.layers_on["parts"]  = not self.layers_on["parts"]
        elif sym == key._7: self.layers_on["helix"]  = not self.layers_on["helix"]
        elif sym == key._8: self.layers_on["tunnel"] = not self.layers_on["tunnel"]
        elif sym == key._9: self.layers_on["rays"]   = not self.layers_on["rays"]
        elif sym == key._0: self.layers_on["grid"]   = not self.layers_on["grid"]
        elif sym == key.O:  self.layers_on["orbits"] = not self.layers_on["orbits"]
        elif sym == key.M:
            self.mouse_mode = (self.mouse_mode + 1) % 3
        elif sym in (key.PLUS, key.EQUAL, key.NUM_ADD):
            self.sensitivity = min(3.0, self.sensitivity + 0.1)
        elif sym in (key.MINUS, key.NUM_SUBTRACT):
            self.sensitivity = max(0.2, self.sensitivity - 0.1)
        elif sym == key.F:
            self.set_fullscreen(not self.fullscreen)
        elif sym == key.SPACE:
            self.palette.randomize()
        elif sym == key.TAB:
            self.show_hud = not self.show_hud
        elif sym == key.H:
            self.show_help = not self.show_help
        elif sym == key.A:
            self.auto_scenes = not self.auto_scenes
            self._apply_scene(self.scene_idx)
        elif sym == key.N:
            self.scene_timer = 0.0
            self.scene_idx = (self.scene_idx + 1) % len(self.scenes)
            self._apply_scene(self.scene_idx)
        elif sym == key.R:
            self.layer_rain.burst(28)  # manuel şekil yağmuru

    def _clear_layer(self, layer):
        """Sönen katmanın tüm şekillerini görünmez yap (kalıntı kalmasın)."""
        def hide(o):
            try: o.color = (0,0,0,0)
            except: pass
        try:
            for attr in ("lines", "rays", "rungs", "a", "b", "hlines", "vlines", "orbits", "planets"):
                if hasattr(layer, attr):
                    for o in getattr(layer, attr): hide(o)
            if hasattr(layer, "sun"): hide(layer.sun)
            if hasattr(layer, "dots"):
                for d in layer.dots:
                    hide(d["obj"] if isinstance(d, dict) else d)
            if hasattr(layer, "rings"):
                for r in layer.rings:
                    hide(r["obj"] if isinstance(r, dict) else r)
            if hasattr(layer, "ribbons"):
                for rb in layer.ribbons:
                    for l in rb["lines"]: hide(l)
            if hasattr(layer, "_pool"):
                for p in layer._pool:
                    hide(p["obj"]); p["alive"] = False
        except Exception:
            pass

    def _apply_scene(self, idx):
        """Sahnenin katmanlarını aç, diğerlerini kapat."""
        scene = self.scenes[idx]
        on = set(scene["on"])
        for key in self.layers_on:
            self.layers_on[key] = (key in on)
        if hasattr(self, "scene_lbl"):
            mode = "AUTO" if self.auto_scenes else "MANUEL"
            self.scene_lbl.text = f"Sahne: {scene['name']} [{mode}]"

    def _trigger_drop(self, energy):
        """Müzik patladığında özel an: şekil yağmuru + güçlü flash + palet."""
        self.layer_rain.burst(n=int(18 + energy * 20))
        self._flash_intensity = min(1.2, self._flash_intensity + 0.9)
        # Birden fazla şok dalgası
        if "shock" in self.layers_on:
            cx, cy = self.width//2, self.height//2
            for _ in range(3):
                self.layer_shock._spawn(self.palette, energy,
                                        cx + random.uniform(-120,120),
                                        cy + random.uniform(-120,120))
        print(f"[Event] DROP! enerji={energy:.2f}")

    def on_close(self):
        print("[App] Kapatiliyor...")
        self.ai.stop()
        self.audio.stop()
        pyglet.app.exit()


if __name__ == "__main__":
    print("=" * 52)
    print("  AI Music Visualizer v2")
    print("  DSP Layers + AI Palette + pyglet 2.x")
    print("=" * 52)
    VisualizerWindow()
    pyglet.app.run()