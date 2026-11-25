"""Katmanların paylaştığı sabitler. Katmanlar sabit 1280×720 tuvalde çizer."""

WINDOW_W = 1280
WINDOW_H = 720
BANDS = 8


def clamp8(v):
    return max(0, min(255, int(v)))
