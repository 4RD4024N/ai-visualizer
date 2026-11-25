"""
Ortak log-mel spektrogram — hem eğitim hem canlı çıkarım AYNI kodu kullanır.
Sadece numpy + scipy (visualizer'a ağır bağımlılık eklemez).
"""

import numpy as np
from scipy.signal import resample_poly, lfilter, firwin

SR      = 16000
N_FFT   = 1024
HOP     = 256
N_MELS  = 64
FMIN    = 30.0
FMAX    = 8000.0
FRAMES  = 128                               # model girdisi: 128 kare ≈ 2.05 sn
WINDOW_SAMPLES = N_FFT + (FRAMES - 1) * HOP  # 128 kare için gereken örnek sayısı


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    hz = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), np.float32)
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    fb *= (2.0 / (hz[2:] - hz[:-2]))[:, None]   # alan normalizasyonu
    return fb


_FB = mel_filterbank()
_WIN = np.hanning(N_FFT).astype(np.float32)


def logmel(x):
    """16 kHz mono float sinyal → (N_MELS, T) float32 log-mel."""
    x = np.asarray(x, np.float32)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n = 1 + (len(x) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, N_FFT), strides=(x.strides[0] * HOP, x.strides[0]), writeable=False)
    spec = np.abs(np.fft.rfft(frames * _WIN, axis=1)) ** 2
    mel = spec @ _FB.T
    return np.log(mel + 1e-6).T.astype(np.float32)


class StreamResampler:
    """Canlı akış için durum korumalı yeniden örnekleme (→ 16 kHz)."""

    def __init__(self, sr_in):
        self.sr_in = int(sr_in)
        self.factor = self.sr_in // SR if self.sr_in % SR == 0 else 0
        if self.factor > 1:
            # Tam sayı oran (48k → 16k): anti-alias FIR + faz takipli seyreltme, kesintisiz
            self.taps = firwin(63, 0.9 / self.factor).astype(np.float32)
            self.zi = np.zeros(len(self.taps) - 1, np.float32)
            self.phase = 0
        elif self.factor == 0:
            g = np.gcd(self.sr_in, SR)
            self.up, self.down = SR // g, self.sr_in // g

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        if self.factor == 1:
            return x
        if self.factor > 1:
            y, self.zi = lfilter(self.taps, 1.0, x, zi=self.zi)
            out = y[self.phase::self.factor]
            self.phase = (self.phase - len(x)) % self.factor
            return out.astype(np.float32)
        return resample_poly(x, self.up, self.down).astype(np.float32)   # 44.1k vb.
