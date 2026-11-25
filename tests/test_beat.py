import numpy as np
import pytest

from beat import BeatTracker

DT = 1024 / 48000


def run_click_track(bpm, seconds=20.0, accent_every=4, seed=0):
    rng = np.random.default_rng(seed)
    per = 60.0 / bpm
    onsets = {int(round(k * per / DT)): k for k in range(int(seconds / per) + 1)}
    bt, beats = BeatTracker(DT), []
    for i in range(int(seconds / DT)):
        k = onsets.get(i)
        flux = 0.2 + 0.05 * rng.random() + (5.0 if k is not None else 0.0)
        bass = 1.0 if (k is not None and k % accent_every == 0) else 0.1
        bt.push(flux, bass)
        beats += bt.pop_events()
    return bt, beats, per


@pytest.mark.parametrize("bpm", [90, 120, 128, 150, 174])
def test_tempo_and_phase_lock(bpm):
    bt, beats, per = run_click_track(bpm)
    assert abs(bt.bpm - bpm) < 3
    assert bt.confidence > 0.8
    t = np.array([b[0] for b in beats if b[0] > 10])
    assert abs(np.diff(t).mean() - per) < 0.01
    phase_err = np.abs((t / per) - np.round(t / per)) * per
    assert phase_err.mean() < 0.04          # 40 ms


def test_downbeat_follows_accent():
    bt, beats, per = run_click_track(120)
    downs = {int(round(b[0] / per)) % 4 for b in beats if b[2] and b[0] > 10}
    assert downs == {0}


def test_silence_emits_nothing():
    bt = BeatTracker(DT)
    for _ in range(2000):
        bt.push(0.0)
    assert bt.pop_events() == []
    assert bt.phase() == 0.0


def test_phase_in_range():
    bt, _, _ = run_click_track(120, seconds=5)
    assert 0.0 <= bt.phase() <= 1.0
