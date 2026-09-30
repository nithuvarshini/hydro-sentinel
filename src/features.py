"""
Acoustic feature extraction.

Two feature views are produced per 1-second clip, echoing the source
paper's "LLDs + statistical functions" pipeline (openSMILE, 6373-dim) but
implemented from scratch with numpy/scipy so the project has no heavy
external audio dependency:

1. `handcrafted_features(clip)`  -> a compact (27-dim) vector of classic
   low-level descriptors and their statistics: zero-crossing rate, RMS
   energy, spectral centroid/bandwidth/rolloff/flatness, band-energy
   ratios, and time-domain moments (mean, std, skew, kurtosis, min/max).
   This is the input to the Logistic Regression baseline.

2. `spectrogram_features(clip)`  -> a fixed-size log-magnitude
   spectrogram (N_FREQ_BINS x N_TIME_BINS), flattened, giving the deep
   models a richer, less hand-engineered view of the signal (closer in
   spirit to feeding a CNN a mel-spectrogram image).

The final feature vector fed to the neural models is the concatenation of
both views, which is a common, cheap way to give an MLP CNN-like spectral
detail without needing convolutional layers.
"""
import numpy as np
from scipy import signal as sps
from scipy.stats import skew, kurtosis

from . import config


def _band_energy_ratios(freqs, power, n_bands=6):
    """Energy fraction in `n_bands` log-spaced frequency bands -- a cheap
    stand-in for MFCC-style spectral-shape descriptors."""
    edges = np.geomspace(max(freqs[1], 1.0), freqs[-1], n_bands + 1)
    total = power.sum() + 1e-12
    ratios = []
    for i in range(n_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        ratios.append(power[mask].sum() / total)
    return np.array(ratios)


def handcrafted_features(clip):
    sr = config.SAMPLE_RATE
    x = clip.astype(np.float64)

    # --- time domain ---
    zcr = np.mean(np.abs(np.diff(np.sign(x)))) / 2.0
    rms = np.sqrt(np.mean(x ** 2))
    mean_abs = np.mean(np.abs(x))
    sd = np.std(x)
    sk = skew(x)
    ku = kurtosis(x)
    peak = np.max(np.abs(x))
    crest = peak / (rms + 1e-9)

    # --- frequency domain ---
    freqs, power = sps.periodogram(x, fs=sr)
    power = power + 1e-12
    power_norm = power / power.sum()

    centroid = np.sum(freqs * power_norm)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power_norm))
    cumulative = np.cumsum(power_norm)
    rolloff_idx = np.searchsorted(cumulative, 0.85)
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
    flatness = np.exp(np.mean(np.log(power))) / (np.mean(power) + 1e-12)

    band_ratios = _band_energy_ratios(freqs, power, n_bands=6)

    # first-order delta of the waveform (paper's "Delta" LLDs)
    dx = np.diff(x)
    d_rms = np.sqrt(np.mean(dx ** 2))
    d_zcr = np.mean(np.abs(np.diff(np.sign(dx)))) / 2.0

    feats = np.concatenate([
        [zcr, rms, mean_abs, sd, sk, ku, peak, crest,
         centroid, bandwidth, rolloff, flatness, d_rms, d_zcr],
        band_ratios,
    ])
    return feats.astype(np.float32)


HANDCRAFTED_NAMES = [
    "zcr", "rms", "mean_abs", "std", "skew", "kurtosis", "peak", "crest_factor",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness",
    "delta_rms", "delta_zcr",
] + [f"band_energy_ratio_{i+1}" for i in range(6)]


def spectrogram_features(clip, n_freq=None, n_time=None):
    n_freq = n_freq or config.N_FREQ_BINS
    n_time = n_time or config.N_TIME_BINS
    sr = config.SAMPLE_RATE

    f, t, Sxx = sps.spectrogram(clip, fs=sr, nperseg=128, noverlap=96)
    Sxx = np.log1p(Sxx)

    # pool to a fixed (n_freq x n_time) grid regardless of input size
    freq_edges = np.linspace(0, Sxx.shape[0], n_freq + 1).astype(int)
    time_edges = np.linspace(0, Sxx.shape[1], n_time + 1).astype(int)
    pooled = np.zeros((n_freq, n_time), dtype=np.float32)
    for i in range(n_freq):
        for j in range(n_time):
            block = Sxx[freq_edges[i]:max(freq_edges[i] + 1, freq_edges[i + 1]),
                        time_edges[j]:max(time_edges[j] + 1, time_edges[j + 1])]
            pooled[i, j] = block.mean() if block.size else 0.0

    # per-clip normalization
    pooled = (pooled - pooled.mean()) / (pooled.std() + 1e-6)
    return pooled.flatten().astype(np.float32)


def extract_all_features(waveforms):
    """Vectorized feature extraction over an (N, N_SAMPLES) array of clips.
    Returns (handcrafted, spectrogram, combined)."""
    hc = np.stack([handcrafted_features(w) for w in waveforms])
    sp = np.stack([spectrogram_features(w) for w in waveforms])
    combined = np.concatenate([hc, sp], axis=1)
    return hc, sp, combined
