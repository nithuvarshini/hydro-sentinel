"""
Synthetic underwater acoustic signal generator.

WHY SYNTHETIC DATA
-------------------
The case-study paper (Bao et al., 2025, Applied Acoustics 235:110701) uses
proprietary hydrophone recordings the authors explicitly cannot share. The
closest public alternatives (ShipsEar, DeepShip) require manual registration
or are hosted on servers unreachable from an automated environment. To keep
this project fully reproducible end-to-end, we generate synthetic hydrophone
signals whose statistical structure matches the acoustic phenomena described
in the peer-reviewed literature:

  * Ship-radiated noise = tonal "line spectra" from rotating machinery
    (engine/propeller shaft harmonics, typically < 500 Hz) superimposed on
    broadband hull/flow noise, per Bao et al. (2025) and the ShipsEar /
    DeepShip characterisation papers.
  * Ambient (no-ship) noise = colored ocean background noise, occasionally
    punctuated by short broadband transients (snapping shrimp / biologics),
    which is a deliberate source of confusable negatives.
  * Domain shift between "Source" and "Target" sites is modeled the same
    way the paper models Lake vs. Ocean1 vs. Ocean2: different fundamental
    engine frequency bands, different noise coloration, and different SNR.

Swap this module for a real loader (e.g. reading ShipsEar/DeepShip .wav
files with soundfile/librosa) and every downstream script keeps working
unchanged, since everything past this file only depends on
(waveform, sample_rate, label).
"""
import numpy as np
from scipy import signal as sps

from . import config


def _pink_noise(n, rng):
    """Generate ~1/f 'pink' noise of length n via spectral shaping of white noise."""
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]  # avoid divide-by-zero at DC
    X = X / np.sqrt(freqs)
    pink = np.fft.irfft(X, n)
    return pink / (np.std(pink) + 1e-9)


def _harmonic_tonal(t, f0, n_harmonics, decay, rng, jitter=0.02):
    """Sum of decaying harmonics of f0 with slight random phase/amplitude jitter,
    mimicking a ship engine/propeller shaft line-spectrum signature."""
    sig = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        amp = decay ** (k - 1) * (1.0 + jitter * rng.standard_normal())
        phase = rng.uniform(0, 2 * np.pi)
        # small random FM wobble to avoid perfectly periodic (too-easy) tones
        wobble = 1.0 + 0.003 * np.sin(2 * np.pi * 0.7 * t + phase)
        sig += amp * np.sin(2 * np.pi * f0 * k * wobble * t + phase)
    return sig


def _broadband_hull_noise(n, rng, tilt=1.2):
    """Colored broadband component representing hull/flow noise, slightly
    brighter (less steep roll-off) than ambient pink ocean noise."""
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    X = X / (freqs ** (tilt / 2))
    out = np.fft.irfft(X, n)
    return out / (np.std(out) + 1e-9)


def _amplitude_envelope(t, rng, depth=0.15):
    """Slow amplitude modulation mimicking sea-state / range fluctuations."""
    f_mod = rng.uniform(0.3, 0.9)
    return 1.0 + depth * np.sin(2 * np.pi * f_mod * t + rng.uniform(0, 2 * np.pi))


def _add_transient_clicks(sig, rng, n_clicks_range=(0, 2)):
    """Occasionally inject short broadband clicks (snapping shrimp / biologics)
    into a signal -- a realistic source of confusable negatives."""
    n = len(sig)
    n_clicks = rng.integers(*n_clicks_range) if n_clicks_range[1] > 0 else 0
    for _ in range(n_clicks):
        pos = rng.integers(0, max(1, n - 40))
        width = rng.integers(5, 25)
        click = rng.standard_normal(width) * np.hanning(width) * rng.uniform(1.5, 3.0)
        end = min(pos + width, n)
        sig[pos:end] += click[: end - pos]
    return sig


DOMAIN_PARAMS = {
    "source": dict(f0_range=(90, 170), n_harm_range=(3, 6), decay_range=(0.45, 0.62),
                   snr_db_range=(-5, 5), ambient_tilt=1.5, hump_prob=0.30),
    "target": dict(f0_range=(110, 190), n_harm_range=(3, 5), decay_range=(0.40, 0.58),
                   snr_db_range=(-11, -1), ambient_tilt=1.15, hump_prob=0.40),
}


def _spurious_resonance_hump(t, rng, freq_range=(40, 400)):
    """A weak, noise-like spectral 'hump' (reef / biologic chorus resonance)
    that can appear in AMBIENT clips too, deliberately confusable with a
    faint ship tonal so the task is not trivially separable by energy alone."""
    f_c = rng.uniform(*freq_range)
    bw = rng.uniform(8, 25)
    n_tones = rng.integers(2, 5)
    sig = np.zeros_like(t)
    for _ in range(n_tones):
        f = f_c + rng.normal(0, bw)
        phase = rng.uniform(0, 2 * np.pi)
        sig += np.sin(2 * np.pi * f * t + phase)
    sig = sig / (np.std(sig) + 1e-9)
    amp = rng.uniform(0.15, 0.45)
    return amp * sig


def generate_clip(label, domain, rng):
    """Generate one 1-second synthetic hydrophone clip.

    Parameters
    ----------
    label : int -- 0 = no ship (ambient only), 1 = ship present
    domain: str -- 'source' or 'target' (controls frequency band / SNR / noise color)
    rng   : np.random.Generator

    Returns
    -------
    waveform : np.ndarray, shape (N_SAMPLES,), unit-ish scale
    """
    p = DOMAIN_PARAMS[domain]
    t = np.arange(config.N_SAMPLES) / config.SAMPLE_RATE

    ambient = _pink_noise(config.N_SAMPLES, rng)
    ambient = ambient / (np.std(ambient) + 1e-9)

    if label == 0:
        clip = ambient.copy()
        clip = _add_transient_clicks(clip, rng, n_clicks_range=(0, 2))
        # hard negatives: sometimes a spurious resonance hump appears with
        # NO ship present, deliberately confusable with a faint ship tonal
        if rng.uniform() < p["hump_prob"]:
            clip = clip + _spurious_resonance_hump(t, rng)
    else:
        f0 = rng.uniform(*p["f0_range"])
        n_harm = rng.integers(*p["n_harm_range"])
        decay = rng.uniform(*p["decay_range"])
        tonal = _harmonic_tonal(t, f0, n_harm, decay, rng)
        tonal = tonal / (np.std(tonal) + 1e-9)
        hull = _broadband_hull_noise(config.N_SAMPLES, rng)
        ship_signal = 0.7 * tonal + 0.55 * hull
        ship_signal *= _amplitude_envelope(t, rng)
        ship_signal = ship_signal / (np.std(ship_signal) + 1e-9)

        snr_db = rng.uniform(*p["snr_db_range"])
        snr_lin = 10 ** (snr_db / 20)
        clip = snr_lin * ship_signal + ambient
        clip = _add_transient_clicks(clip, rng, n_clicks_range=(0, 1))

    # gentle high-pass to remove any DC drift, then normalize
    sos = sps.butter(2, 15, btype="highpass", fs=config.SAMPLE_RATE, output="sos")
    clip = sps.sosfilt(sos, clip)
    clip = clip / (np.max(np.abs(clip)) + 1e-9)
    return clip.astype(np.float32)


def generate_dataset(n_per_class, domain, seed):
    """Generate a balanced dataset of (waveforms, labels) for one domain."""
    rng = np.random.default_rng(seed)
    waveforms, labels = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            waveforms.append(generate_clip(label, domain, rng))
            labels.append(label)
    waveforms = np.stack(waveforms).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    # shuffle
    idx = rng.permutation(len(labels))
    return waveforms[idx], labels[idx]
