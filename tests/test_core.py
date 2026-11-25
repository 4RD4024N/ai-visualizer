"""Ses donanımı / OpenGL gerektirmeyen modüllerin testleri."""

import json
import queue

import numpy as np
import pytest

import config
from audio_analyzer import AudioAnalyzer, CHUNK, BAND_NAMES
from audio_features import mel_filterbank, logmel, N_MELS, SR
from director import LocalDirector


# ── config ──
def test_config_missing_file_gives_defaults(tmp_path):
    assert config.load(str(tmp_path / "yok.json")) == config.DEFAULTS


def test_config_roundtrip_and_coercion(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"sensitivity": "1.5", "beat_sync": "false", "record_fps": 24.6,
                             "bilinmeyen": 1, "width": "abc"}), encoding="utf-8")
    cfg = config.load(str(p))
    assert cfg["sensitivity"] == 1.5
    assert cfg["beat_sync"] is False
    assert cfg["record_fps"] == 25
    assert cfg["width"] == config.DEFAULTS["width"]        # geçersiz → varsayılan
    assert "bilinmeyen" not in cfg
    cfg["flash"] = 0.3
    config.save(cfg, str(p))
    assert config.load(str(p))["flash"] == 0.3


def test_config_corrupt_file(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{bozuk", encoding="utf-8")
    assert config.load(str(p)) == config.DEFAULTS


# ── audio_features ──
def test_mel_filterbank_shape_and_coverage():
    fb = mel_filterbank()
    assert fb.shape[0] == N_MELS
    assert (fb >= 0).all()
    assert (fb.sum(1) > 0).all()                            # her mel bandı en az bir FFT kutusu alır


def test_logmel_frames():
    x = np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.1
    m = logmel(x)
    assert m.ndim == 2 and N_MELS in m.shape
    assert np.isfinite(m).all()


# ── audio_analyzer ──
def tone(freq, seconds, sr=48000, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("freq,band", [(100, "bass"), (1000, "mid"), (8000, "brilliance")])
def test_analyzer_puts_tone_in_right_band(freq, band):
    a = AudioAnalyzer()
    x = tone(freq, 1.0)
    for i in range(0, len(x) - CHUNK, CHUNK):
        a.process(x[i:i + CHUNK])
    bands, rms, peak = a.get()
    assert BAND_NAMES[int(np.argmax(bands))] == band
    assert abs(rms - 0.5 / np.sqrt(2)) < 0.02
    snap = a.snapshot()
    assert set(snap) >= {"bands", "rms", "bpm", "section", "spectral_balance"}


def test_analyzer_tracks_kick_drum_tempo():
    sr, bpm = 48000, 125
    x = np.zeros(int(16 * sr), np.float32)
    kick = tone(60, 0.08) * np.exp(-np.arange(int(0.08 * sr)) / (0.02 * sr)).astype(np.float32)
    for k in range(int(16 * bpm / 60)):
        s = int(k * 60 / bpm * sr)
        x[s:s + len(kick)] += kick[: len(x) - s]
    x += 0.01 * np.random.default_rng(0).standard_normal(len(x)).astype(np.float32)
    a = AudioAnalyzer()
    for i in range(0, len(x) - CHUNK, CHUNK):
        a.process(x[i:i + CHUNK])
    beats, _, phase, conf, est, _ = a.rhythm()
    assert abs(est - bpm) < 4
    assert conf > 0.6
    assert 0 <= phase <= 1


# ── director ──
def snap(rms_avg=0.02, flux=50.0, dyn=0.01, bpm=128, low=0.6, high=0.15):
    return {"rms_avg": rms_avg, "onset_activity": flux, "dynamics": dyn, "bpm": bpm,
            "spectral_balance": {"low": low, "mid": 1 - low - high, "high": high}}


def test_director_silence_returns_none():
    d = LocalDirector(audio=None, q=queue.Queue())
    assert d._decide(snap(rms_avg=0.0)) is None


def test_director_output_is_valid():
    d = LocalDirector(audio=None, q=queue.Queue())
    for s in (snap(), snap(rms_avg=0.001, flux=1, bpm=80, low=0.2, high=0.5)):
        out = d._decide(s)
        assert out["scene"] in {"minimal", "cosmic", "flow", "web", "galaxy", "full", "storm"}
        assert len(out["palette"]) == 4
        for c in out["palette"]:
            assert all(0.0 <= c[k] <= 1.0 for k in "rgb")
        assert 0.2 <= out["complexity"] <= 1.0
        assert out["speed"] > 0


def test_director_loud_bassy_music_is_intense():
    d = LocalDirector(audio=None, q=queue.Queue())
    out = None
    for _ in range(3):                                      # histerezis: aynı aday iki kez
        out = d._decide(snap(rms_avg=0.05, flux=200, dyn=0.05, bpm=170, low=0.7))
    assert out["scene"] == "storm"
    assert out["mood"] == "aggressive"
