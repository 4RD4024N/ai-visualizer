"""
Şarkı yapısı algılama — bölüm sınırları (giriş / verse / nakarat / drop / breakdown).

Ses thread'i her blokta `push(bands, rms, flux)` çağırır. Özellikler 0,5 sn'lik
segmentlerde ortalanır: log bant enerjileri + log RMS + log onset gücü.

Yenilik (novelty): son 2 sn'nin ortalaması ile ondan önceki 6 sn'nin ortalaması
arasındaki uzaklık (özellikler son 2 dakika üzerinden z-skorlanır, böylece her şarkıya
uyum sağlar). Yenilik uyarlanır eşiği aşıp tepe yaptığında bölüm değişmiştir.

Etiket, yeni bölümün enerjisinin şarkının o ana kadarki enerji dağılımındaki yerine
ve enerji sıçramasına göre verilir:
  drop      : büyük enerji artışı + yüksek enerji
  breakdown : büyük enerji düşüşü
  chorus    : üst enerji dilimi
  calm      : alt enerji dilimi
  verse     : arası
2 sn'den uzun sessizlik geçmişi sıfırlar (yeni şarkı = "intro").
"""

import collections

import numpy as np


class SectionDetector:
    SEG = 0.5            # segment süresi (sn)
    RECENT = 4           # 2 sn
    PAST = 12            # 6 sn
    HISTORY = 240        # 2 dk
    MIN_GAP = 8.0        # iki bölüm arası en kısa süre (sn)
    MIN_NOVELTY = 0.8
    SILENCE_RMS = 1e-4
    MAX_EVENTS = 16

    def __init__(self, frame_dt):
        self.dt = float(frame_dt)
        self.t = 0.0
        self.label = "intro"
        self.since = 0.0
        self.novelty = 0.0
        self.events = []                 # [(zaman, etiket, yenilik)]
        self._reset()

    def _reset(self):
        self._acc = []
        self._acc_t = 0.0
        self._silent = 0.0
        self._segs = collections.deque(maxlen=self.HISTORY)
        self._nov = collections.deque(maxlen=self.HISTORY)

    def push(self, bands, rms, flux):
        self.t += self.dt
        if rms < self.SILENCE_RMS:
            self._silent += self.dt
            if self._silent > 2.0 and self._segs:
                self._reset()
                self._set("intro", 0.0, force=True)
            return
        self._silent = 0.0
        b = np.asarray(bands, np.float64)
        self._acc.append(np.concatenate([np.log1p(np.maximum(b, 0) * 10),
                                         [np.log(rms + 1e-5), np.log1p(max(flux, 0.0))]]))
        self._acc_t += self.dt
        if self._acc_t >= self.SEG:
            self._segment(np.mean(self._acc, axis=0))
            self._acc, self._acc_t = [], 0.0

    def pop_events(self):
        ev, self.events = self.events, []
        return ev

    def _segment(self, f):
        self._segs.append(f)
        need = self.RECENT + self.PAST
        if len(self._segs) < need:
            self._nov.append(0.0)
            return
        X = np.asarray(self._segs)
        Z = (X - X.mean(0)) / (X.std(0) + 1e-3)
        recent, past = Z[-self.RECENT:].mean(0), Z[-need:-self.RECENT].mean(0)
        nov = float(np.linalg.norm(recent - past) / np.sqrt(Z.shape[1]))
        self._nov.append(nov)
        self.novelty = nov
        if len(self._nov) < 3:
            return
        prev2, prev, cur = self._nov[-3], self._nov[-2], self._nov[-1]
        hist = np.asarray(self._nov)[:-1]
        hist = hist[hist > 0]
        thr = self.MIN_NOVELTY
        if len(hist) >= 10:                              # medyan + MAD: geçmiş tepeler eşiği şişirmez
            med = float(np.median(hist))
            thr = max(thr, med + 4.0 * 1.4826 * float(np.median(np.abs(hist - med))))
        # Bir önceki segment tepe mi? (yükseldi, şimdi düşüyor)
        if not (prev > thr and prev >= prev2 and prev > cur):
            return
        if self.t - self.since < self.MIN_GAP:
            return
        e = X[:, -2]                                     # log RMS
        e_recent, e_past = e[-self.RECENT:].mean(), e[-need:-self.RECENT].mean()
        rank = float((e <= e_recent).mean())
        delta = e_recent - e_past
        if delta > 0.6 and rank > 0.6:
            lab = "drop"
        elif delta < -0.6:
            lab = "breakdown"
        elif rank > 0.7:
            lab = "chorus"
        elif rank < 0.3:
            lab = "calm"
        else:
            lab = "verse"
        self._set(lab, prev)

    def _set(self, label, nov, force=False):
        if not force and self.t - self.since < self.MIN_GAP:
            return
        self.label, self.since = label, self.t
        self.events.append((self.t, label, nov))
        if len(self.events) > self.MAX_EVENTS:
            del self.events[0]
