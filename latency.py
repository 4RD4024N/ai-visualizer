"""
Söz → ekran gecikme ölçümü — torch gerektirmez (birim testli).

Her kabul edilen söz satırı için zaman damgaları (time.perf_counter, süreç içi ortak saat):

  heard   : sözün duyulduğu anın YAKLAŞIĞI. Sözün tam söylendiği an bilinemez (Whisper
            zaman damgası vermez); pencere sonu (sesin alındığı an) ile bir önceki
            pencerenin sonu arasındaki "yeni gelen ses" diliminin ortası alınır. Yeni söz
            bu dilimde bir yerdedir → ortalama durumu verir. (En kötü durum: dilimin başı,
            yani + adım/2; adım 0,5 sn iken ±0,25 sn belirsizlik.)
  window  : ses penceresinin alındığı an (pencere sonu)
  whisper : Whisper çıktısı hazır (çeviri / transcribe bitti)
  accept  : satır filtrelerden geçti, line_id arttı
  scene   : SD thread'i bu satırla yeni sahneyi kurdu (prompt + CLAP metin encode)
  frame   : bu satırın prompt'uyla üretilen ilk kare hazır (CPU'ya kopyalandı)
  draw    : o kare ekrana ilk kez çizildi (+ çapraz geçiş süresi tamamen görünür olana dek)

Log satırı:
  [Gecikme] söz→ekran 1.62 sn (bekleme 0.25, whisper 0.71, sahne 0.02, kare 0.06, çizim 0.05)
    bekleme = window − heard          (yeni sesin ortası → pencere alınması)
    whisper = accept − window          (yeniden örnekleme + Whisper + filtre)
    sahne   = scene − accept           (SD thread'inin satırı görmesi + prompt kurma)
    kare    = frame − scene            (ilk SD karesi; o an süren kareyi bitirmek dahil)
    çizim   = draw − frame (+ geçiş)   (ana döngünün kareyi alıp çizmesi)
"""

import collections
import threading
import time

STAGES = ("heard", "window", "whisper", "accept", "scene", "frame", "draw")
PARTS = (("bekleme", "heard", "window"), ("whisper", "window", "accept"),
         ("sahne", "accept", "scene"), ("kare", "scene", "frame"), ("çizim", "frame", "draw"))


def now():
    return time.perf_counter()


def heard_estimate(window_t, prev_window_t, max_gap):
    """Yeni gelen ses diliminin ortası. İlk pencerede (ya da uzun aradan sonra) dilim
    max_gap ile sınırlanır — pencere boyunu aşan "yeni ses" olmaz."""
    if prev_window_t is None:
        gap = max_gap
    else:
        gap = min(max(0.0, window_t - prev_window_t), max_gap)
    return window_t - gap / 2.0


def breakdown(st):
    """Damga sözlüğü → (toplam, [(ad, süre)]) ya da eksikse None."""
    if any(k not in st for k in ("heard", "draw")):
        return None
    parts = []
    for name, a, b in PARTS:
        if a in st and b in st:
            parts.append((name, max(0.0, st[b] - st[a])))
    return st["draw"] - st["heard"], parts


def format_line(total, parts):
    inner = ", ".join(f"{n} {v:.2f}" for n, v in parts)
    return f"[Gecikme] söz→ekran {total:.2f} sn ({inner})"


def percentile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class LatencyTracker:
    """Thread-safe: Whisper, SD ve ana thread aynı satırın farklı damgalarını yazar.
    Bir satır 'draw' damgasını alınca tamamlanır, log basılır (enabled ise), geçmişe eklenir."""

    MAX_OPEN = 16

    def __init__(self, enabled=True, printer=print):
        self.enabled = enabled
        self.printer = printer
        self._open = collections.OrderedDict()      # line_id → {aşama: t}
        self._lock = threading.Lock()
        self.last = None                            # (toplam, parçalar) — HUD için
        self.history = collections.deque(maxlen=200)
        self.history_parts = collections.deque(maxlen=200)

    def stamp(self, line_id, stage, t=None, only_first=True):
        if line_id is None:
            return
        t = now() if t is None else t
        with self._lock:
            st = self._open.get(line_id)
            if st is None:
                st = self._open[line_id] = {}
                while len(self._open) > self.MAX_OPEN:
                    self._open.popitem(last=False)
            if only_first and stage in st:
                return
            st[stage] = t

    def stamps(self, line_id):
        with self._lock:
            return dict(self._open.get(line_id, {}))

    def finish(self, line_id, t=None, extra=0.0):
        """Ekrana çizildi. extra: çapraz geçiş süresi (tamamen görünene kadar)."""
        t = (now() if t is None else t) + extra
        with self._lock:
            st = self._open.pop(line_id, None)
        if st is None:
            return None
        st["draw"] = t
        res = breakdown(st)
        if res is None:
            return None
        total, parts = res
        self.last = res
        self.history.append(total)
        self.history_parts.append(dict(parts))
        if self.enabled and self.printer is not None:
            self.printer(format_line(total, parts))
        return res

    def drop(self, line_id):
        with self._lock:
            self._open.pop(line_id, None)

    def summary(self):
        """(n, ortalama, p95, {parça: ortalama}) — bench için."""
        xs = list(self.history)
        if not xs:
            return 0, float("nan"), float("nan"), {}
        avg_parts = {}
        for d in self.history_parts:
            for k, v in d.items():
                avg_parts.setdefault(k, []).append(v)
        return (len(xs), sum(xs) / len(xs), percentile(xs, 0.95),
                {k: sum(v) / len(v) for k, v in avg_parts.items()})
