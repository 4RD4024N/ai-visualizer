"""Efekt katmanları: şok dalgası, takımyıldız, şekil yağmuru, parçacıklar."""

import math
import random

import numpy as np
import pyglet.shapes as shapes

from .common import WINDOW_W, WINDOW_H, BANDS


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
