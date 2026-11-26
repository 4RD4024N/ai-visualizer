"""Söz → ekran gecikme yolu: GPU'suz, sahte Whisper / SD nesneleriyle."""

import threading
import time

import numpy as np
import pytest

import latency
from diffusion_layer import DiffusionLayer, scene_due
from lyrics import LyricsListener


# ── latency.py ──
def test_heard_estimate_is_middle_of_new_audio():
    assert latency.heard_estimate(10.0, 9.5, 4.0) == pytest.approx(9.75)
    # ilk pencere / uzun ara: yeni ses pencere boyuyla sınırlı
    assert latency.heard_estimate(10.0, None, 4.0) == pytest.approx(8.0)
    assert latency.heard_estimate(10.0, 1.0, 4.0) == pytest.approx(8.0)


def test_tracker_breakdown_and_log():
    out = []
    tr = latency.LatencyTracker(printer=out.append)
    base = 100.0
    for stage, dt in (("heard", 0.0), ("window", 0.25), ("whisper", 0.9), ("accept", 0.91),
                      ("scene", 0.93), ("frame", 1.01)):
        tr.stamp(7, stage, base + dt)
    tr.stamp(7, "scene", base + 5.0)            # ikinci damga ilkini ezmez
    total, parts = tr.finish(7, t=base + 1.03, extra=0.06)
    assert total == pytest.approx(1.09)
    d = dict(parts)
    assert d["bekleme"] == pytest.approx(0.25)
    assert d["whisper"] == pytest.approx(0.66)
    assert d["sahne"] == pytest.approx(0.02)
    assert d["kare"] == pytest.approx(0.08)
    assert d["çizim"] == pytest.approx(0.08)
    assert out == ["[Gecikme] söz→ekran 1.09 sn (bekleme 0.25, whisper 0.66, sahne 0.02, kare 0.08, çizim 0.08)"]
    assert tr.last[0] == pytest.approx(1.09)
    assert tr.finish(7) is None                 # aynı satır iki kez bitmez
    n, avg, p95, _ = tr.summary()
    assert n == 1 and avg == pytest.approx(1.09) and p95 == pytest.approx(1.09)


def test_tracker_disabled_is_silent_and_bounded():
    out = []
    tr = latency.LatencyTracker(enabled=False, printer=out.append)
    for i in range(50):
        tr.stamp(i, "heard", 0.0)
    assert len(tr._open) == tr.MAX_OPEN
    tr.finish(49, t=1.0)
    assert out == [] and tr.last[0] == pytest.approx(1.0)


def test_percentile():
    assert latency.percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert latency.percentile(list(range(101)), 0.95) == pytest.approx(95)


# ── Söz önceliği / sahne tetikleme ──
def test_scene_due_lyric_priority():
    kw = dict(min_scene=1.2, scene_sec=3.0, lyric_min_scene=0.3)
    # yeni söz satırı MIN_SCENE'i beklemez
    assert scene_due(0.4, False, False, True, lyric_priority=True, **kw)
    assert not scene_due(0.4, False, False, True, lyric_priority=False, **kw)
    # art arda satırlarda küçük taban
    assert not scene_due(0.1, False, False, True, lyric_priority=True, **kw)
    # ölçü isteği MIN_SCENE'e uyar (eski davranış)
    assert not scene_due(0.5, False, True, False, **kw)
    assert scene_due(1.3, False, True, False, **kw)
    # müzik değişimi ve uzun sahne yaştan bağımsız
    assert scene_due(0.0, True, False, False, **kw)
    assert scene_due(3.1, False, False, False, **kw)
    assert not scene_due(1.0, False, False, False, **kw)


# ── Söz dinleyici: pencere zamanlaması, translate-first, filtre ──
class FakeTap:
    def __init__(self, sr=48000, sec=10.0, seed=0):
        self.sr = sr
        self.rng = np.random.default_rng(seed)
        self.x = (0.1 * self.rng.standard_normal(int(sr * sec))).astype(np.float32)

    def advance(self, sec=0.5):
        n = int(self.sr * sec)
        self.x = np.concatenate([self.x[n:], (0.1 * self.rng.standard_normal(n)).astype(np.float32)])

    def latest_raw(self):
        return self.x.copy(), self.sr


def resample(x, sr):
    return x[::3]


def make_listener(results, originals=None, **kw):
    tr = latency.LatencyTracker(printer=None)
    ls = LyricsListener(FakeTap(), tracker=tr, **kw)
    calls = []
    ls._transcribe = lambda x16: (calls.append(("transcribe", len(x16))), results.pop(0))[1]
    ls._original = lambda ctx: (calls.append(("original", ctx)), originals.pop(0))[1]
    return ls, tr, calls


def test_window_is_trimmed_and_resampled():
    ls, _, calls = make_listener([None], window=4.0)
    ls._step(resample)
    assert calls == [("transcribe", 4 * 16000)]


def test_translate_first_publishes_before_original(capsys):
    order = []
    ls, tr, calls = make_listener([("we will dance until the sun comes up", None, "tr", -0.3, "ctx")],
                                  originals=[("güneş doğana dek dans edeceğiz", -0.4)])
    ls.on_line.append(lambda lid: order.append(("wake", lid, list(calls))))
    lid = ls._step(resample)
    assert lid == 1 and ls.line_id == 1
    # SD thread'i uyandırıldığında Türkçe orijinal henüz üretilmemişti
    assert order == [("wake", 1, [("transcribe", 4 * 16000)])]
    assert calls[-1] == ("original", "ctx")
    assert ls.current() == "we will dance until the sun comes up"
    assert ls.current_original() == ("güneş doğana dek dans edeceğiz", "tr")
    assert ls.current_with_id() == ("we will dance until the sun comes up", 1)
    st = tr.stamps(1)
    assert st["heard"] <= st["window"] <= st["whisper"] <= st["accept"]
    assert "[Söz] [tr] » güneş doğana dek dans edeceğiz" in capsys.readouterr().out


def test_blacklisted_original_retracts_line():
    ls, _, _ = make_listener([("thanks a lot my dear friends forever", None, "tr", -0.3, "ctx")],
                             originals=[("izlediğiniz için teşekkürler", -0.2)])
    assert ls._step(resample) is None
    assert ls.current() is None


def test_hud_original_off_skips_second_generate():
    ls, _, calls = make_listener([("we will dance until the sun comes up", None, "tr", -0.3, "ctx")],
                                 originals=[], hud_original=False)
    assert ls._step(resample) == 1
    assert [c[0] for c in calls] == ["transcribe"]
    assert ls.current_original()[0] == "we will dance until the sun comes up"


def test_overlapping_window_is_not_a_new_line():
    ls, _, _ = make_listener([("your eyes are like the ocean", "your eyes are like the ocean", "en", -0.2, None),
                              ("your eyes are like the ocean in the night",
                               "your eyes are like the ocean in the night", "en", -0.2, None)])
    woke = []
    ls.on_line.append(woke.append)
    assert ls._step(resample) == 1
    ls.tap.advance()
    assert ls._step(resample) is None
    assert woke == [1] and ls.line_id == 1
    assert ls.current() == "your eyes are like the ocean in the night"   # daha uzun olan tutulur


def test_low_logprob_and_hallucination_rejected():
    ls, _, _ = make_listener([("we will dance until the sun comes up", "x", "en", -2.0, None),
                              ("Thank you for watching!", "Thank you for watching!", "en", -0.1, None)])
    assert ls._step(resample) is None
    ls.tap.advance()
    assert ls._step(resample) is None
    assert ls.line_id == 0


def test_stalled_audio_and_silence_are_skipped():
    ls, _, calls = make_listener([None, None])
    ls._step(resample)
    ls._step(resample)                           # ses akışı durdu: aynı pencere tekrar işlenmez
    assert len(calls) == 1
    ls.tap.x[:] = 0.0
    ls.tap.x[-1] = 1e-6
    ls._step(resample)                           # sessizlik: Whisper koşmaz
    assert len(calls) == 1


def test_short_buffer_waits():
    ls, _, calls = make_listener([None])
    ls.tap.x = ls.tap.x[:48000]
    ls._step(resample)
    assert calls == []


def test_bekleme_tracks_step():
    """Adım 0,5 sn iken bekleme yaklaşığı ≈ 0,25 sn (eski 1,5 sn aralıkta ≈ 0,75)."""
    ls, tr, _ = make_listener([None, ("we will dance until the sun comes up",
                                      "we will dance until the sun comes up", "en", -0.2, None)])
    ls._step(resample)
    time.sleep(0.05)
    ls.tap.advance()
    ls._step(resample)
    st = tr.stamps(1)
    assert 0.0 < st["window"] - st["heard"] < 0.2


# ── Difüzyon döngüsü: uyanma, ilk kare etiketi, damgalar (sahte torch) ──
class _FakeStream:
    def __init__(self, priority=0):
        pass

    @staticmethod
    def priority_range():
        return (0, -1)


class _FakeCuda:
    Stream = _FakeStream

    @staticmethod
    def set_stream(s):
        pass


class _FakeTorch:
    cuda = _FakeCuda


class FakeLyrics:
    enabled = True

    def __init__(self):
        self.line = None
        self.line_id = 0

    def current_with_id(self):
        return self.line, self.line_id


class FakeDiffusion(DiffusionLayer):
    """GPU'suz: _load / CLAP / üretim sahte; sahne kararı ve döngü gerçek."""
    FRAME = 0.02

    def _load(self):
        self.torch = _FakeTorch()
        self.clap = None

    def _clap_loop(self):
        pass

    def _update_concepts(self):
        lyric, lid = self.lyrics.current_with_id()
        new_line = lyric is not None and lid != self._lyric_seen
        age = time.time() - self._scene_t
        if scene_due(age, False, False, new_line, self.MIN_SCENE, 99.0, self.lyric_priority,
                     self.LYRIC_MIN_SCENE) and new_line:
            self._lyric_seen = lid
            self._scene_t = time.time()
            self._lyric_pending = lid
            self.tracker.stamp(lid, "scene")

    def _generate(self, st, z, va):
        time.sleep(self.FRAME)
        return np.zeros((2, 2, 3), np.uint8)


def test_diffusion_loop_tags_first_lyric_frame():
    tr = latency.LatencyTracker(printer=None)
    d = FakeDiffusion(tracker=tr)
    d.lyrics = FakeLyrics()
    d._scene_t = time.time()                    # sahne yeni başladı: MIN_SCENE dolmadı
    d.start()
    stop = threading.Event()

    def feeder():                               # ana thread: her ~16 ms girdi
        while not stop.is_set():
            d.submit({}, np.zeros(8), np.zeros(2))
            time.sleep(0.016)

    threading.Thread(target=feeder, daemon=True).start()
    try:
        t0 = time.time()
        while not d.ready and time.time() - t0 < 2:
            time.sleep(0.01)
        time.sleep(0.35)                        # LYRIC_MIN_SCENE (0,3) dolsun, MIN_SCENE (1,2) dolmasın
        tr.stamp(1, "heard")
        tr.stamp(1, "accept")
        d.lyrics.line, d.lyrics.line_id = "we will dance until the sun comes up", 1
        d.wake()
        seen = None
        t0 = time.time()
        while time.time() - t0 < 1.0:
            fid, frame, lid = d.get_frame_info()
            if lid == 1:
                seen = fid
                break
            time.sleep(0.002)
        assert seen is not None, "söz satırının ilk karesi etiketlenmedi"
        res = tr.finish(1)
        assert res is not None
        total, parts = res
        # MIN_SCENE (1,2 sn) beklenmedi: satır → kare ≈ bir-iki kare süresi
        assert total < 0.5
        assert dict(parts)["kare"] < 0.2
        time.sleep(0.1)
        assert d.get_frame_info()[2] is None    # sonraki kareler etiketsiz
    finally:
        stop.set()
        d.stop()
