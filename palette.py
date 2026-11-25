"""Renk paleti — hedefe yumuşak geçiş (lerp) + yavaş renk tonu kayması."""

import colorsys
import random

DEFAULT_PALETTE = [
    (0.2, 0.4, 1.0),
    (0.9, 0.1, 0.8),
    (1.0, 0.5, 0.1),
    (0.1, 0.9, 0.5),
]


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
