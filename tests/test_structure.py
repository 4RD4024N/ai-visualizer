import numpy as np

from structure import SectionDetector

DT = 1024 / 48000


class Feeder:
    def __init__(self):
        self.sd = SectionDetector(DT)
        self.rng = np.random.default_rng(1)

    def feed(self, seconds, level, bands):
        r = self.rng
        for _ in range(int(seconds / DT)):
            b = np.array(bands) * level * (1 + 0.1 * r.standard_normal(8))
            self.sd.push(b, 0.02 * level * (1 + 0.1 * r.standard_normal()), 3 * level * (1 + 0.2 * r.random()))
        return self.sd.pop_events()


QUIET = [1, 1, .5, .5, .3, .2, .1, .05]
LOUD = [2, 2, 1, 1, .8, .6, .4, .2]


def test_stationary_music_has_no_boundaries():
    assert Feeder().feed(40, 0.3, QUIET) == []


def test_energy_jump_is_a_drop_detected_quickly():
    f = Feeder()
    f.feed(30, 0.3, QUIET)
    ev = f.feed(20, 1.5, LOUD)
    assert len(ev) == 1
    t, label, _ = ev[0]
    assert 30 < t < 35
    assert label == "drop"


def test_energy_fall_is_a_breakdown():
    f = Feeder()
    f.feed(30, 0.3, QUIET)
    f.feed(20, 1.5, LOUD)
    ev = f.feed(20, 0.25, [.3, .3, .5, .8, .8, .6, .4, .2])
    assert [e[1] for e in ev] == ["breakdown"]
    assert 50 < ev[0][0] < 55


def test_silence_resets_to_intro():
    f = Feeder()
    f.feed(20, 1.0, LOUD)
    ev = f.feed(3, 0.0, QUIET)
    assert [e[1] for e in ev] == ["intro"]
