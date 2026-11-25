"""
Yerel sahne direktörü (LLM yok): ses istatistiklerinden sahne, palet, hız ve ruh hali seçer.

Ayrı thread'de INTERVAL saniyede bir `audio.snapshot()` okur ve kararını kuyruğa koyar.
"""

import colorsys
import queue
import random
import threading
import time


class LocalDirector:
    """Ses istatistiklerinden sahne/palet/hız seçer — dış servis yok.
    Çıktı formatı eski AIDirector ile aynı (dir_queue'ya dict)."""
    INTERVAL = 3.0

    def __init__(self, audio, q):
        self.audio   = audio
        self.queue   = q
        self.running = False
        self._ref    = {"rms": 1e-3, "flux": 1e-3, "dyn": 1e-4}   # uyarlanır tavan
        self._scene  = None
        self._cand   = None
        self._hue    = random.random()

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self): self.running = False

    def _norm(self, key, v):
        # Yavaş sönen tepe değere göre 0-1 — cihaz/ses seviyesinden bağımsız
        self._ref[key] = max(v, self._ref[key] * 0.97)
        return min(1.0, v / (self._ref[key] + 1e-9))

    def _decide(self, s):
        if s["rms_avg"] < 1e-4:
            return None                                          # sessizlik
        loud = self._norm("rms", s["rms_avg"])
        flux = self._norm("flux", s["onset_activity"])
        dyn  = self._norm("dyn", s["dynamics"])
        tempo = min(1.0, max(0.0, (s["bpm"] - 70) / 100))
        energy = 0.4 * loud + 0.25 * flux + 0.2 * dyn + 0.15 * tempo
        low, high = s["spectral_balance"]["low"], s["spectral_balance"]["high"]

        if energy > 0.75:   cand = "storm" if low > 0.5 else "full"
        elif energy > 0.5:  cand = "galaxy" if low > 0.5 else ("web" if high > 0.25 else "flow")
        elif energy > 0.28: cand = "flow" if high > 0.2 else "cosmic"
        else:               cand = "minimal"
        # Histerezis: aynı aday üst üste iki kez gelirse sahneyi değiştir
        if cand == self._cand or self._scene is None:
            self._scene = cand
        self._cand = cand

        # Palet: bas ağırlığı sıcak (kırmızı/mor), tiz ağırlığı soğuk (camgöbeği/mavi)
        target_hue = (0.92 + 0.6 * (high - low * 0.5)) % 1.0
        dh = ((target_hue - self._hue + 0.5) % 1.0) - 0.5
        self._hue = (self._hue + dh * 0.3 + random.uniform(-0.03, 0.03)) % 1.0
        sat = 0.45 + 0.55 * energy
        val = 0.75 + 0.25 * energy
        spread = 0.08 + 0.25 * energy                            # enerjik → daha zıt renkler
        pal = []
        for i in range(4):
            r, g, b = colorsys.hsv_to_rgb((self._hue + i * spread) % 1.0,
                                          min(1.0, sat + random.uniform(-0.05, 0.05)), val)
            pal.append({"r": r, "g": g, "b": b})
        br, bgc, bb = colorsys.hsv_to_rgb(self._hue, 0.8, 0.02 + 0.04 * (1 - energy))

        if energy > 0.75:   mood = "aggressive" if low > 0.5 else "energetic"
        elif energy > 0.5:  mood = "energetic"
        elif energy > 0.28: mood = "dreamy" if high > 0.2 else "ethereal"
        else:               mood = "dark" if low > 0.5 else "calm"

        return {
            "scene": self._scene,
            "complexity": round(0.2 + 0.8 * (0.6 * energy + 0.4 * flux), 2),
            "speed": round(0.6 + 1.3 * (0.5 * energy + 0.5 * tempo), 2),
            "palette": pal,
            "bg_r": br, "bg_g": bgc, "bg_b": bb,
            "mood": mood,
            "shape_rain": bool(energy > 0.88 and dyn > 0.7),
            "_type": "full",
        }

    def _loop(self):
        while self.running:
            try:
                d = self._decide(self.audio.snapshot())
                if d:
                    if self.queue.full():
                        try: self.queue.get_nowait()
                        except queue.Empty: pass
                    self.queue.put(d)
            except Exception as e:
                print(f"[Director] {e}")
            time.sleep(self.INTERVAL)
