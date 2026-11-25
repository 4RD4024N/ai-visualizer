"""Eski (v1) katmanlar — sahne sisteminde kullanılmıyor, referans için duruyor."""

import math
import random

import pyglet.shapes as shapes

from .common import WINDOW_W, WINDOW_H, BANDS


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
