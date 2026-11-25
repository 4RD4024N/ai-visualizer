"""Geometrik katmanlar: Lissajous, şerit, spiral, helix, tünel, ışınlar, ızgara, yörüngeler."""

import math
import random

import numpy as np
import pyglet.shapes as shapes

from .common import WINDOW_W, WINDOW_H, BANDS


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
