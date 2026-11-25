"""Model metriklerinin gerilememesi + kayıt hattının (ffmpeg) uçtan uca testi."""

import ctypes
import os
import shutil
import subprocess
import time

import numpy as np
import pytest

from recorder import VideoRecorder
from tools.model_report import collect

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


# ── Model metrikleri: yeniden eğitimde kötüleşme olursa test düşer ──
def test_model_metrics_present_and_above_floor():
    rows = {r[1]: r[2] for r in collect(MODELS)}
    assert rows["Contrastive doğruluk (val)"] >= 0.95
    assert rows["Titreme (val)"] <= 0.06
    assert rows["R² arousal (şarkı)"] >= 0.60
    assert rows["R² valence (şarkı)"] >= 0.35


def test_onnx_model_runs():
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(os.path.join(MODELS, "audio_latent.onnx"),
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    outs = sess.run(None, {inp.name: np.zeros(shape, np.float32)})
    assert all(np.isfinite(o).all() for o in outs)


# ── Kayıt ──
class FakeGL:
    """glReadPixels yerine renk geçişli bir kare üretir."""
    GL_PACK_ALIGNMENT = GL_RGB = GL_UNSIGNED_BYTE = 0

    def __init__(self):
        self.n = 0

    def glPixelStorei(self, *a):
        pass

    def glReadPixels(self, x, y, w, h, fmt, typ, buf):
        self.n += 1
        ctypes.memset(buf, (self.n * 7) % 256, w * h * 3)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg yok")
def test_recorder_produces_synced_mp4(tmp_path):
    rec = VideoRecorder(str(tmp_path), fps=30, crf=30)
    assert rec.start(321, 241, 48000, 2)                    # tek boyutlar çifte yuvarlanmalı
    assert rec.size == (320, 240)
    gl = FakeGL()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        rec.capture(gl)
        rec.push_audio(np.zeros((1024, 2), np.float32) + 0.1)
        time.sleep(1024 / 48000)
    rec.stop()
    rec.wait()
    assert rec.last_output and os.path.exists(rec.last_output)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
                          "-of", "csv=p=0", rec.last_output], capture_output=True, text=True).stdout
    durs = {line.split(",")[0]: float(line.split(",")[1]) for line in out.strip().splitlines()}
    assert set(durs) == {"video", "audio"}
    assert abs(durs["video"] - 2.0) < 0.3                   # duvar saatine göre kare sayısı
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp_")]
