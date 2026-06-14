"""
AI Music Visualizer
WASAPI Loopback + pyglet 2.x + LM Studio (local)
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
SAMPLE_RATE = 44100
CHUNK       = 1024
FFT_SIZE    = 4096
BANDS       = 8
AI_INTERVAL = 2.5
TARGET_FPS  = 60
WINDOW_W    = 1280
WINDOW_H    = 720

BAND_NAMES  = ["sub_bass","bass","low_mid","mid","high_mid","presence","brilliance","air"]
BAND_RANGES = [(20,60),(60,250),(250,500),(500,2000),(2000,4000),(4000,6000),(6000,12000),(12000,20000)]
LM_BASE_URL = "http://localhost:1234/v1"

SYSTEM_PROMPT = """Output ONLY valid JSON, no text before or after. No markdown. Start with {

{"shapes":[{"type":"circle","count":3,"size":0.2,"color_r":1.0,"color_g":0.0,"color_b":0.5,"alpha":200,"sides":6,"spikes":5,"rotation_speed":3.0,"pulse":true,"glow":true}],"bg_r":5,"bg_g":0,"bg_b":20,"bg_fade_alpha":20,"global_scale":1.0,"chaos":0.3,"mood":"energetic"}

Rules for the JSON values based on audio data:
- type: circle/ring/arc/sector/star/polygon/triangle/line
- count: 1-12 (total across all shapes max 20)
- size: 0.1-0.6
- color_r/g/b: 0.0-1.0
- alpha: 80-240
- rotation_speed: 2.0-8.0 (always positive or negative, never 0)
- chaos: 0.0-1.0
- mood: energetic/calm/dark/ethereal/aggressive/dreamy
- Max 3 shape objects
- High bass: large circles/polygons, dark bg, high chaos
- High treble: stars/arcs, bright colors
- Low RMS: fewer shapes, lower alpha
- Use vivid varied colors, not monotone
"""

# ── Ses Analizi (sounddevice — WASAPI loopback) ──────────────────────────────
class AudioAnalyzer:
    def __init__(self):
        self.stream     = None
        self.buffer     = np.zeros(FFT_SIZE)
        self.lock       = threading.Lock()
        self.running    = False
        self.bands      = np.zeros(BANDS)
        self.rms        = 0.0
        self.peak       = 0.0
        self.bpm_est    = 120.0
        self._beat_buf  = []
        self._last_beat = 0.0
        self._sample_rate = SAMPLE_RATE

    def _find_loopback_device(self):
        """
        Oncelik: CABLE Output (VB-Audio) > Stereo Mix > fallback
        """
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
            if "cable output" in name_lower:
                score = 100
            elif "stereo mix" in name_lower or "stereo karisim" in name_lower:
                score = 60
            elif "what u hear" in name_lower:
                score = 60
            if "WASAPI" in api_name:
                score += 10
            if score > 0:
                candidates.append((score, i, min(ch_in, 2), dev["name"]))

        if candidates:
            candidates.sort(key=lambda x: -x[0])
            _, idx, ch, name = candidates[0]
            print(f"[Audio] Loopback secildi: [{idx}] {name} ({ch}ch)")
            return idx, ch, False

        print("[Audio] Loopback cihaz bulunamadi")
        return None, 1, False

    def start(self):
        dev_idx, channels, _ = self._find_loopback_device()
        channels = max(1, min(channels, 2))

        if dev_idx is not None:
            for sr in [48000, 44100, 96000, 32000]:
                try:
                    self.stream = sd.InputStream(
                        device=dev_idx,
                        channels=channels,
                        samplerate=sr,
                        blocksize=CHUNK,
                        dtype="float32",
                        callback=self._cb,
                    )
                    self.stream.start()
                    self.running = True
                    self._sample_rate = sr
                    print(f"[Audio] Loopback aktif ({channels}ch, {sr}Hz)")
                    return
                except Exception as e:
                    print(f"[Audio] {sr}Hz denendi, hata: {e}")

        # Fallback: default mikrofon
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK,
                dtype="float32",
                callback=self._cb,
            )
            self.stream.start()
            self.running = True
            self._sample_rate = SAMPLE_RATE
            print("[Audio] Mikrofon fallback aktif")
        except Exception as e:
            print(f"[Audio] Mikrofon da acilamadi: {e}")

    def _cb(self, indata, frames, time_info, status):
        # indata shape: (frames, channels)
        s = indata[:, 0].copy()   # mono — sol kanal
        with self.lock:
            self.buffer = np.roll(self.buffer, -len(s))
            self.buffer[-len(s):] = s
            self._analyze(s)

    def _analyze(self, s):
        self.rms  = float(np.sqrt(np.mean(s**2)))
        self.peak = float(np.max(np.abs(s)))
        win       = np.hanning(len(self.buffer))
        mag       = np.abs(np.fft.rfft(self.buffer * win))
        freqs     = np.fft.rfftfreq(len(self.buffer), 1.0 / getattr(self, "_sample_rate", SAMPLE_RATE))
        for i, (lo, hi) in enumerate(BAND_RANGES):
            mask   = (freqs >= lo) & (freqs < hi)
            energy = float(np.mean(mag[mask])) / 50.0 if mask.any() else 0.0
            self.bands[i] = self.bands[i] * 0.6 + energy * 0.4
        bass = self.bands[1]
        now  = time.time()
        if bass > 0.4 and (now - self._last_beat) > 0.25:
            iv = now - self._last_beat
            self._beat_buf.append(iv)
            if len(self._beat_buf) > 8: self._beat_buf.pop(0)
            if len(self._beat_buf) >= 2:
                self.bpm_est = 60.0 / float(np.mean(self._beat_buf))
            self._last_beat = now

    def snapshot(self):
        with self.lock:
            return {
                "bands": dict(zip(BAND_NAMES, self.bands.tolist())),
                "rms":  round(self.rms, 4),
                "peak": round(self.peak, 4),
                "bpm":  round(self.bpm_est, 1),
            }

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()


def _safe_parse_json(raw: str):
    """Bozuk JSON'u temizleyip parse etmeye calis."""
    import re
    raw = raw.strip()
    # Markdown fence temizle
    if "```" in raw:
        for part in raw.split("```")[1::2]:
            part = part.lstrip("json").strip()
            if part.startswith("{"):
                raw = part; break
    # { ile baslayan kismi bul
    if not raw.startswith("{"):
        idx = raw.find("{")
        if idx >= 0: raw = raw[idx:]
        else: return None
    # Acik JSON'u kapat — son } bul, sonrasini kes
    depth = 0
    end   = -1
    for i, ch in enumerate(raw):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end >= 0:
        raw = raw[:end+1]
    # Trailing comma temizle: ,} ve ,]
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    try:
        return json.loads(raw)
    except Exception:
        return None

# ── LM Studio AI ────────────────────────────────────────────────────────────
class AIDirector:
    def __init__(self, audio, q):
        self.audio   = audio
        self.queue   = q
        self.running = False
        self.thread  = None
        self.model   = None

    def _get_model(self):
        req = urllib.request.Request(
            f"{LM_BASE_URL}/models",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        models = data.get("data", [])
        if not models:
            raise RuntimeError("Yuklu model bulunamadi")
        mid = models[0]["id"]
        print(f"[AI] Model: {mid}")
        return mid

    def _call(self, prompt):
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": 800,
            "temperature": 0.3,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{LM_BASE_URL}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer lm-studio"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running and not self.model:
            try:
                self.model = self._get_model()
            except Exception as e:
                print(f"[AI] Baglanamadi: {e} — 5sn bekleniyor")
                time.sleep(5)
        while self.running:
            try:
                snap = self.audio.snapshot()
                raw  = self._call(f"Ses verisi:\n{json.dumps(snap)}")
                raw  = raw.strip()
                if "```" in raw:
                    for part in raw.split("```")[1::2]:
                        part = part.lstrip("json").strip()
                        if part.startswith("{"):
                            raw = part; break
                if not raw.startswith("{"):
                    idx = raw.find("{")
                    if idx >= 0: raw = raw[idx:]
                d = _safe_parse_json(raw)
                if d is None:
                    print(f"[AI] JSON parse basarisiz. Ham cikti: {repr(raw[:300])}")
                else:
                    if self.queue.full():
                        try: self.queue.get_nowait()
                        except queue.Empty: pass
                    self.queue.put(d)
                    print(f"[AI] Mood:{d.get('mood','?')} Shapes:{len(d.get('shapes',[]))}")
            except json.JSONDecodeError as e:
                print(f"[AI] JSON hatasi: {e}")
            except Exception as e:
                print(f"[AI] Hata: {e}")
            time.sleep(AI_INTERVAL)

    def stop(self):
        self.running = False


# ── Shape Instance ────────────────────────────────────────────────────────────
def clamp_color(v):
    return max(0, min(255, int(v)))

class ShapeInstance:
    """Tek animasyonlu shape — pyglet 2.x uyumlu (Star, Circle, Arc, Sector, Polygon, Triangle, Line)"""

    VALID_TYPES = {"circle","ring","arc","sector","star","polygon","triangle","line"}

    def __init__(self, spec, batch):
        self.spec  = spec
        self.batch = batch
        self.angle = random.uniform(0.0, 360.0)
        self.phase = random.uniform(0.0, math.tau)
        self.x     = random.uniform(0.15, 0.85) * WINDOW_W
        self.y     = random.uniform(0.15, 0.85) * WINDOW_H
        self.vx    = random.uniform(-80, 80)
        self.vy    = random.uniform(-80, 80)
        self._obj  = None
        self._glow = None
        # Interpolasyon state — AI direktifinden hedefe yumusak gecis
        self._cur_r   = float(spec.get("color_r", 1.0))
        self._cur_g   = float(spec.get("color_g", 1.0))
        self._cur_b   = float(spec.get("color_b", 1.0))
        self._cur_a   = float(spec.get("alpha", 180))
        self._cur_sz  = float(spec.get("size", 0.15))
        self._cur_rs  = float(spec.get("rotation_speed", 0.5))
        self._tgt_r   = self._cur_r
        self._tgt_g   = self._cur_g
        self._tgt_b   = self._cur_b
        self._tgt_a   = self._cur_a
        self._tgt_sz  = self._cur_sz
        self._tgt_rs  = self._cur_rs
        self._build()

    # ── Renk yardımcısı ──────────────────────────────────────────────────────
    def _rgba(self, alpha=None):
        s = self.spec
        a = int(s.get("alpha", 180)) if alpha is None else int(alpha)
        return (
            clamp_color(float(s.get("color_r", 1.0)) * 255),
            clamp_color(float(s.get("color_g", 1.0)) * 255),
            clamp_color(float(s.get("color_b", 1.0)) * 255),
            clamp_color(a),
        )

    def _glow_rgba(self):
        c = self._rgba()
        return (c[0], c[1], c[2], max(8, c[3] // 5))

    def _base_size(self, scale=1.0):
        return self._cur_sz * min(WINDOW_W, WINDOW_H) * 0.5 * scale

    # ── İnşa ─────────────────────────────────────────────────────────────────
    def _build(self):
        s    = self.spec
        t    = s.get("type", "circle")
        if t not in self.VALID_TYPES:
            t = "circle"
        sz   = self._base_size()
        c    = self._rgba()
        cg   = self._glow_rgba()
        sid  = max(3, int(s.get("sides", 6)))
        spk  = max(3, int(s.get("spikes", 5)))
        x, y = self.x, self.y

        if t == "circle":
            if s.get("glow"):
                self._glow = shapes.Circle(x, y, sz * 1.7, color=cg, batch=self.batch)
            self._obj = shapes.Circle(x, y, sz, color=c, batch=self.batch)

        elif t == "ring":
            # Arc ile tam daire = ring efekti
            if s.get("glow"):
                self._glow = shapes.Arc(x, y, sz * 1.6, color=cg, batch=self.batch)
            self._obj = shapes.Arc(x, y, sz, color=c, batch=self.batch)

        elif t == "arc":
            span = random.uniform(math.pi * 0.5, math.pi * 1.5)
            self._obj = shapes.Arc(x, y, sz, angle=span, color=c, batch=self.batch)

        elif t == "sector":
            span = random.uniform(math.pi * 0.3, math.pi * 1.2)
            self._obj = shapes.Sector(x, y, sz, angle=span, color=c, batch=self.batch)
            if s.get("glow"):
                self._glow = shapes.Circle(x, y, sz * 1.4, color=cg, batch=self.batch)

        elif t == "star":
            outer = sz
            inner = sz * 0.42
            self._obj = shapes.Star(x, y, outer, inner, num_spikes=spk,
                                    color=c, batch=self.batch)
            if s.get("glow"):
                self._glow = shapes.Circle(x, y, sz * 1.4, color=cg, batch=self.batch)

        elif t == "polygon":
            # Polygon: düzenli n-gen — Star ile inner≈outer yaparak simüle
            outer = sz
            inner = sz * 0.88
            self._obj = shapes.Star(x, y, outer, inner, num_spikes=sid,
                                    color=c, batch=self.batch)
            if s.get("glow"):
                self._glow = shapes.Circle(x, y, sz * 1.4, color=cg, batch=self.batch)

        elif t == "triangle":
            # Eşkenar üçgen: 3-spike Star
            self._obj = shapes.Star(x, y, sz, sz * 0.5, num_spikes=3,
                                    color=c, batch=self.batch)

        elif t == "line":
            x2 = x + math.cos(math.radians(self.angle)) * sz
            y2 = y + math.sin(math.radians(self.angle)) * sz
            self._obj = shapes.Line(x, y, x2, y2, width=max(1, int(sz * 0.04)),
                                    color=c, batch=self.batch)

        else:
            self._obj = shapes.Circle(x, y, sz, color=c, batch=self.batch)

    # ── Update ───────────────────────────────────────────────────────────────
    def update(self, dt, t, rms, bands, global_scale, chaos):
        # Interpolasyon — renk, boyut, hiz hedefe dogru kayar
        spd = 6.0
        self._cur_r  = self._lerp(self._cur_r,  self._tgt_r,  dt, spd)
        self._cur_g  = self._lerp(self._cur_g,  self._tgt_g,  dt, spd)
        self._cur_b  = self._lerp(self._cur_b,  self._tgt_b,  dt, spd)
        self._cur_a  = self._lerp(self._cur_a,  self._tgt_a,  dt, spd)
        self._cur_sz = self._lerp(self._cur_sz, self._tgt_sz, dt, spd)
        self._cur_rs = self._lerp(self._cur_rs, self._tgt_rs, dt, spd)

        rot_spd = self._cur_rs
        self.angle = (self.angle + rot_spd * dt * 80) % 360

        jx = random.uniform(-1, 1) * chaos * 10
        jy = random.uniform(-1, 1) * chaos * 10

        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x < 60 or self.x > WINDOW_W - 60:  self.vx *= -1
        if self.y < 60 or self.y > WINDOW_H - 60:  self.vy *= -1

        bass  = bands.get("bass", 0.0)
        treble = bands.get("brilliance", 0.0)
        base_sz = self._base_size(global_scale)

        if self.spec.get("pulse"):
            treble = bands.get("brilliance", 0.0)
            beat   = (bass * 5.0 + treble * 2.0) * abs(math.sin(t * math.pi * 3 + self.phase))
            sz     = base_sz * max(0.15, 1.0 + beat)
        else:
            sz = base_sz

        px = self.x + jx
        py = self.y + jy

        typ = self.spec.get("type", "circle")

        # Interpolated rengi her frame uygula
        cr = clamp_color(self._cur_r * 255)
        cg = clamp_color(self._cur_g * 255)
        cb = clamp_color(self._cur_b * 255)
        ca = clamp_color(self._cur_a)
        ca_glow = max(8, ca // 5)

        for obj, factor in [(self._glow, 1.6), (self._obj, 1.0)]:
            if obj is None:
                continue
            obj.x        = px
            obj.y        = py
            obj.rotation = self.angle
            # Rengi canli guncelle
            try:
                if obj is self._glow:
                    obj.color = (cr, cg, cb, ca_glow)
                else:
                    obj.color = (cr, cg, cb, ca)
            except Exception:
                pass

            fsz = sz * factor

            if isinstance(obj, shapes.Circle):
                obj.radius = fsz
            elif isinstance(obj, shapes.Arc):
                obj.radius = fsz
            elif isinstance(obj, shapes.Sector):
                obj.radius = fsz
            elif isinstance(obj, shapes.Star):
                obj.outer_radius = fsz
                obj.inner_radius = fsz * (0.42 if typ == "star" else 0.88)
            elif isinstance(obj, shapes.Line):
                x2 = px + math.cos(math.radians(self.angle)) * fsz
                y2 = py + math.sin(math.radians(self.angle)) * fsz
                obj.x  = px; obj.y  = py
                obj.x2 = x2; obj.y2 = y2

    def set_target(self, spec):
        """AI'dan yeni direktif gelince hedef degerleri guncelle."""
        self._tgt_r  = float(spec.get("color_r", self._cur_r))
        self._tgt_g  = float(spec.get("color_g", self._cur_g))
        self._tgt_b  = float(spec.get("color_b", self._cur_b))
        self._tgt_a  = float(spec.get("alpha", self._cur_a))
        self._tgt_sz = float(spec.get("size", self._cur_sz))
        self._tgt_rs = float(spec.get("rotation_speed", self._cur_rs))

    def _lerp(self, cur, tgt, dt, speed=3.0):
        """Exponential lerp — her frame hedefe yaklasir."""
        return cur + (tgt - cur) * min(1.0, dt * speed)

    def delete(self):
        for o in (self._obj, self._glow):
            if o: o.delete()


# ── Ana Pencere ───────────────────────────────────────────────────────────────
class VisualizerWindow(pyglet.window.Window):

    DEFAULT_DIRECTIVE = {
        "shapes": [
            {"type":"circle","count":3,"size":0.13,"color_r":0.2,"color_g":0.4,
             "color_b":1.0,"alpha":150,"sides":6,"spikes":5,"rotation_speed":1.5,
             "pulse":True,"glow":True},
            {"type":"star","count":2,"size":0.09,"color_r":0.9,"color_g":0.2,
             "color_b":0.8,"alpha":130,"sides":5,"spikes":5,"rotation_speed":-2.0,
             "pulse":True,"glow":False},
        ],
        "bg_r":4,"bg_g":3,"bg_b":18,"bg_fade_alpha":18,
        "global_scale":1.0,"chaos":0.15,"mood":"hazir",
    }

    def __init__(self):
        super().__init__(
            width=WINDOW_W, height=WINDOW_H,
            caption="AI Music Visualizer",
            resizable=True, vsync=True,
        )
        self.set_minimum_size(640, 360)

        self.audio     = AudioAnalyzer()
        self.dir_queue = queue.Queue(maxsize=4)
        self.ai        = AIDirector(self.audio, self.dir_queue)

        self.batch     = pyglet.graphics.Batch()
        self.instances : list[ShapeInstance] = []

        self.bg        = [4, 3, 18]
        self.fade_a    = 18
        self.global_scale = 1.0
        self.chaos     = 0.15
        self.t         = 0.0
        self._fps_dt   = 0.0
        self._fps_fr   = 0
        self._fps_val  = 0.0

        # Fade overlay — kendi draw() çağrısıyla çizilecek
        self.fade_rect = shapes.Rectangle(0, 0, WINDOW_W, WINDOW_H,
                                          color=(4, 3, 18, 18))

        self.hud = pyglet.graphics.Batch()
        self.fps_lbl  = pyglet.text.Label(
            "", x=12, y=self.height - 22,
            font_name="Consolas", font_size=11,
            color=(200, 200, 200, 180), batch=self.hud,
        )
        self.mood_lbl = pyglet.text.Label(
            "LM Studio bekleniyor...", x=12, y=12,
            font_name="Consolas", font_size=13,
            color=(140, 190, 255, 210), batch=self.hud,
        )

        self._apply(self.DEFAULT_DIRECTIVE)
        self.audio.start()
        self.ai.start()
        pyglet.clock.schedule_interval(self.update, 1.0 / TARGET_FPS)
        print("[App] Hazir — muzik cal!")

    # ── Direktif ─────────────────────────────────────────────────────────────
    def _clear(self):
        for inst in self.instances:
            inst.delete()
        self.instances.clear()

    # Interpolasyon hedefleri
    _tgt_bg      = [4, 3, 18]
    _tgt_fade_a  = 18
    _tgt_scale   = 1.0
    _tgt_chaos   = 0.15
    # Smooth bg interpolasyon
    _cur_bg      = [4.0, 3.0, 18.0]
    _cur_fade_a  = 18.0
    _cur_scale   = 1.0
    _cur_chaos   = 0.15

    def _apply(self, d):
        # Arka plan ve global parametreler icin hedef guncelle
        self._tgt_bg     = [int(d.get("bg_r",4)), int(d.get("bg_g",3)), int(d.get("bg_b",18))]
        self._tgt_fade_a = float(d.get("bg_fade_alpha", 18))
        self._tgt_scale  = float(d.get("global_scale", 1.0))
        self._tgt_chaos  = float(d.get("chaos", 0.15))
        self.mood_lbl.text = f"AI Mood: {d.get('mood','?')}"

        # Sekiller: tipte degisim varsa yeniden olustur, yoksa sadece hedef guncelle
        new_specs = []
        for spec in d.get("shapes", []):
            cnt = max(1, min(20, int(spec.get("count", 1))))
            for _ in range(cnt):
                new_specs.append(spec)

        # Mevcut instance sayisi ile yeni spec sayisini esit yap
        cur_count = len(self.instances)
        new_count = len(new_specs)

        if new_count > cur_count:
            # Yeni sekiller ekle
            for i in range(cur_count, new_count):
                self.instances.append(ShapeInstance(new_specs[i], self.batch))
        elif new_count < cur_count:
            # Fazla sekilleri sil
            for inst in self.instances[new_count:]:
                inst.delete()
            self.instances = self.instances[:new_count]

        # Hepsinin hedefini guncelle (tip ayni kalir, parametreler lerp ile gider)
        for i, inst in enumerate(self.instances):
            if i < len(new_specs):
                inst.set_target(new_specs[i])

    # ── Update ───────────────────────────────────────────────────────────────
    def update(self, dt):
        self.t += dt
        try:
            d = self.dir_queue.get_nowait()
            self._apply(d)
        except queue.Empty:
            pass

        # Arka plan ve global parametreleri smooth interpolate et
        spd = 4.0
        for i in range(3):
            self._cur_bg[i] += (self._tgt_bg[i] - self._cur_bg[i]) * min(1.0, dt * spd)
        self._cur_fade_a += (self._tgt_fade_a - self._cur_fade_a) * min(1.0, dt * spd)
        self._cur_scale  += (self._tgt_scale  - self._cur_scale)  * min(1.0, dt * spd)
        self._cur_chaos  += (self._tgt_chaos  - self._cur_chaos)  * min(1.0, dt * spd)

        bg_r = clamp_color(self._cur_bg[0])
        bg_g = clamp_color(self._cur_bg[1])
        bg_b = clamp_color(self._cur_bg[2])
        fa   = clamp_color(self._cur_fade_a)
        pyglet.gl.glClearColor(bg_r/255, bg_g/255, bg_b/255, 1.0)
        self.fade_rect.color = (bg_r, bg_g, bg_b, fa)

        snap = self.audio.snapshot()
        for inst in self.instances:
            inst.update(dt, self.t, snap["rms"], snap["bands"],
                        self._cur_scale, self._cur_chaos)

        self._fps_dt += dt
        self._fps_fr += 1
        if self._fps_dt >= 0.5:
            self._fps_val = self._fps_fr / self._fps_dt
            self._fps_dt  = 0.0
            self._fps_fr  = 0
        self.fps_lbl.text = (
            f"FPS:{self._fps_val:.0f}  "
            f"RMS:{snap['rms']:.3f}  "
            f"BPM:{snap['bpm']:.0f}  "
            f"Obj:{len(self.instances)}"
        )
        self.fps_lbl.y = self.height - 22

    # ── Draw ─────────────────────────────────────────────────────────────────
    def on_draw(self):
        self.clear()
        self.fade_rect.width  = self.width
        self.fade_rect.height = self.height
        self.fade_rect.draw()
        self.batch.draw()
        self.hud.draw()

    def on_resize(self, w, h):
        super().on_resize(w, h)
        self.fps_lbl.y  = h - 22
        self.mood_lbl.y = 12

    def on_key_press(self, sym, mod):
        if sym in (pyglet.window.key.ESCAPE, pyglet.window.key.Q):
            self.close()

    def on_close(self):
        print("[App] Kapatiliyor...")
        self.ai.stop()
        self.audio.stop()
        pyglet.app.exit()


# ── Giriş ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 52)
    print("  AI Music Visualizer")
    print("  WASAPI Loopback + pyglet 2.x + LM Studio")
    print("=" * 52)
    print("Cikis: ESC veya Q\n")
    VisualizerWindow()
    pyglet.app.run()