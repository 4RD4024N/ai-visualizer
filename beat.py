"""
Beat takibi — tempo + faz (PLL) + ölçü başı.

Ses thread'i her blokta `push(flux, bass)` çağırır (flux = spektral akış, onset gücü).

1) Tempo   : onset zarfının otokorelasyonu, harmonik toplama ve 120 BPM civarı
             öncel (oktav hatalarını azaltır). Her 0,5 sn'de bir.
2) Faz     : bir beat ızgarası tutulur (`next_beat`). Yerel onset tepeleri en yakın
             ızgara noktasına göre hata ölçer, ızgara hatanın bir kısmı kadar kaydırılır
             (faz kilitli döngü). Beat'ler tepeyi beklemeden ızgaraya göre yayılır,
             böylece görseller vuruşla aynı anda tetiklenir.
3) Güven   : son beat'lerin ne kadarına gerçek bir onset denk geldi (0–1). Güven
             düşükken güçlü bir tepe ızgarayı yeniden kurar.
4) Ölçü    : 4 vuruşluk döngüde bas enerjisi en yüksek slot = ölçü başı (downbeat).

Zaman "ses zamanı"dır (işlenen örnek sayısı / örnekleme hızı) — duvar saatinden bağımsız,
bu yüzden testlerde sentetik onset dizisiyle birebir çalışır.
"""

import numpy as np


class BeatTracker:
    MIN_BPM, MAX_BPM = 60, 200
    PEAK_RATIO = 1.5          # onset / ortalama — bu oranın altındaki tepeler yok sayılır
    WINDOW = 0.2              # ızgara noktasına uzaklık toleransı (periyodun oranı)
    PLL_GAIN = 0.3            # hatanın ne kadarı düzeltilir
    MAX_EVENTS = 32

    def __init__(self, frame_dt, history_sec=6.0):
        self.dt = float(frame_dt)
        self._env = np.zeros(max(16, int(history_sec / self.dt)))
        self.t = 0.0                    # ses zamanı (sn)
        self.bpm = 120.0
        self.period = 0.5
        self.next_beat = None
        self.last_beat = -1e9
        self.beat_index = -1
        self.confidence = 0.0
        self._hit = False               # son yayılan beat'e onset denk geldi mi
        self._hit_next = False          # sıradaki beat'e (erken gelen) onset denk geldi mi
        self._tempo_cd = 0.0
        self._flux_avg = 0.0
        self._recent = [0.0, 0.0, 0.0]  # son 3 normalize onset (tepe seçimi)
        self._bar = np.zeros(4)         # slot başına bas enerjisi (ölçü başı tahmini)
        self._bass_hist = np.zeros(max(4, int(0.15 / self.dt) + 1))   # son ~150 ms bas
        self._pend = None               # [slot, bas, bitiş zamanı] — beat çevresindeki bas toplanıyor
        self.events = []                # [(zaman, indeks, ölçü_başı_mı)]

    # ── Giriş ──
    def push(self, flux, bass=0.0):
        self.t += self.dt
        flux = float(flux)
        self._env = np.roll(self._env, -1)
        self._env[-1] = flux
        self._flux_avg = flux if self._flux_avg == 0.0 else self._flux_avg * 0.995 + flux * 0.005
        o = flux / (self._flux_avg + 1e-9)
        r = self._recent
        r.pop(0)
        r.append(o)
        bass = float(bass)
        self._bass_hist = np.roll(self._bass_hist, -1)
        self._bass_hist[-1] = bass
        if self._pend is not None:                  # beat'in ±pencere içindeki bas tepesi o beat'e ait
            self._pend[1] = max(self._pend[1], bass)
            if self.t >= self._pend[2]:
                self._bar *= 0.97
                self._bar[self._pend[0]] += self._pend[1]
                self._pend = None

        self._tempo_cd -= self.dt
        if self._tempo_cd <= 0:
            self._tempo_cd = 0.5
            self._estimate_tempo()

        if r[1] > self.PEAK_RATIO and r[1] >= r[0] and r[1] > r[2]:
            self._on_peak(self.t - self.dt, r[1])

        if self.next_beat is not None and self.t >= self.next_beat:
            self._emit(self.next_beat)
            self.next_beat += self.period
            if self.next_beat <= self.t:            # çok geride kaldıysa (ör. tempo değişti)
                self.next_beat = self.t + self.period

    # ── Çıkış ──
    def phase(self):
        """Beat içindeki konum: 0 = beat anı, 1'e doğru sıradaki beat'e yaklaşır."""
        if self.next_beat is None:
            return 0.0
        return float(np.clip(1.0 - (self.next_beat - self.t) / self.period, 0.0, 1.0))

    def downbeat_slot(self):
        return int(np.argmax(self._bar)) if self._bar.any() else 0

    def pop_events(self):
        ev, self.events = self.events, []
        return ev

    # ── İç ──
    def _emit(self, tb):
        self.confidence = 0.8 * self.confidence + 0.2 * (1.0 if self._hit else 0.0)
        self._hit, self._hit_next = self._hit_next, False
        self.beat_index += 1
        if self._pend is not None:                  # önceki beat'in penceresi kapanmadıysa şimdi işle
            self._bar *= 0.97
            self._bar[self._pend[0]] += self._pend[1]
        w = self.WINDOW * self.period
        n = min(len(self._bass_hist), max(1, int(w / self.dt) + 1))
        self._pend = [self.beat_index % 4, float(self._bass_hist[-n:].max()), tb + w]
        self.last_beat = tb
        down = (self.beat_index % 4) == self.downbeat_slot()
        self.events.append((tb, self.beat_index, down))
        if len(self.events) > self.MAX_EVENTS:
            del self.events[0]

    def _on_peak(self, tp, strength):
        p, w = self.period, self.WINDOW * self.period
        if self.next_beat is None:
            self._emit(tp)
            self._hit = True
            self.next_beat = tp + p
            return
        e_last, e_next = tp - self.last_beat, tp - self.next_beat
        late = abs(e_last) <= abs(e_next)           # tepe son beat'e mi yakın, sıradakine mi
        err = e_last if late else e_next
        if abs(err) <= w:
            self.next_beat += self.PLL_GAIN * err
            if late:
                self._hit = True
            else:
                self._hit_next = True
        elif self.confidence < 0.3 and strength > 2.0:
            self._emit(tp)                          # kilit yok: ızgarayı bu tepeye kur
            self._hit = True
            self.next_beat = tp + p

    def _estimate_tempo(self):
        env = self._env
        if np.max(env) < 1e-9:
            return
        env = np.convolve(env, [0.25, 0.5, 0.25], mode="same")
        env = env - env.mean()
        env[env < 0] = 0.0
        ac = np.correlate(env, env, mode="full")[len(env) - 1:]
        if ac[0] <= 1e-12:
            return
        ac = ac / ac[0]
        min_lag = max(1, int((60.0 / self.MAX_BPM) / self.dt))
        max_lag = min(int((60.0 / self.MIN_BPM) / self.dt), len(ac) - 1)
        if max_lag <= min_lag:
            return
        lags = np.arange(min_lag, max_lag + 1)
        scores = ac[lags].copy()
        for mult, wgt in ((2, 0.5), (3, 0.33), (4, 0.25)):     # harmonik toplama
            m = lags * mult
            ok = m < len(ac)
            scores[ok] += ac[m[ok]] * wgt
        bpms = 60.0 / (lags * self.dt)
        prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.55) ** 2)
        i = int(np.argmax(scores * prior))
        peak = scores[i]
        conf = peak / (scores.mean() + 1e-6)
        if peak < 0.08 or conf < 1.3:
            return
        # Parabolik ara değerleme: kare çözünürlüğünden daha hassas periyot
        lag = float(lags[i])
        if 0 < i < len(scores) - 1:
            a, b, c = scores[i - 1], scores[i], scores[i + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-9:
                lag += float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        bpm = 60.0 / (lag * self.dt)
        while bpm < 70:
            bpm *= 2
        while bpm > 190:
            bpm /= 2
        alpha = min(0.5, 0.15 + conf * 0.08)
        self.bpm = self.bpm * (1 - alpha) + bpm * alpha
        self.period = 60.0 / self.bpm
