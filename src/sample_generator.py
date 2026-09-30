"""
Generates realistic benchmark sample underwater hydrophone recordings (.wav)
for instant demonstration and operational testing in the web dashboard.
"""
import os
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sps

from . import config
from .signal_generator import (
    _pink_noise, _harmonic_tonal, _broadband_hull_noise,
    _amplitude_envelope, _add_transient_clicks, _spurious_resonance_hump
)

SAMPLE_DIR = os.path.join(config.DATA_DIR, "sample_recordings")
os.makedirs(SAMPLE_DIR, exist_ok=True)


def generate_all_samples():
    sr = config.SAMPLE_RATE
    rng = np.random.default_rng(2026)
    samples_info = []

    # 1. Merchant Cargo Vessel (6 seconds) - Sustained engine tonals + cavitation
    dur1 = 6.0
    n1 = int(dur1 * sr)
    t1 = np.arange(n1) / sr
    amb1 = _pink_noise(n1, rng)
    tonal1 = _harmonic_tonal(t1, f0=118.0, n_harmonics=4, decay=0.55, rng=rng)
    hull1 = _broadband_hull_noise(n1, rng, tilt=1.1)
    ship1 = 0.75 * tonal1 + 0.5 * hull1
    ship1 *= (1.0 + 0.12 * np.sin(2 * np.pi * 0.5 * t1))  # blade pass modulation
    snr_lin1 = 10 ** (3.0 / 20.0)  # +3 dB SNR
    sig1 = snr_lin1 * (ship1 / np.std(ship1)) + (amb1 / np.std(amb1))
    sos = sps.butter(2, 15, btype="highpass", fs=sr, output="sos")
    sig1 = sps.sosfilt(sos, sig1)
    sig1 = sig1 / (np.max(np.abs(sig1)) + 1e-9)
    wavfile.write(os.path.join(SAMPLE_DIR, "merchant_vessel_transit.wav"), sr, (sig1 * 32767).astype(np.int16))
    samples_info.append({
        "id": "merchant_vessel_transit.wav",
        "title": "Merchant Cargo Vessel Transit (6.0s)",
        "category": "Target Present",
        "expected": "Vessel Detected",
        "description": "Commercial cargo carrier with strong low-frequency diesel engine tonals (118 Hz) and cavitation noise."
    })

    # 2. Deep Ocean Ambient (5 seconds) - Pure ambient noise
    dur2 = 5.0
    n2 = int(dur2 * sr)
    t2 = np.arange(n2) / sr
    amb2 = _pink_noise(n2, rng)
    amb2 = amb2 / (np.std(amb2) + 1e-9)
    amb2 = sps.sosfilt(sos, amb2)
    amb2 = amb2 / (np.max(np.abs(amb2)) + 1e-9)
    wavfile.write(os.path.join(SAMPLE_DIR, "ocean_ambient_deep_sea.wav"), sr, (amb2 * 32767).astype(np.int16))
    samples_info.append({
        "id": "ocean_ambient_deep_sea.wav",
        "title": "Deep Ocean Ambient Baseline (5.0s)",
        "category": "Ambient Baseline",
        "expected": "No Target",
        "description": "Undisturbed deep-sea background noise with typical ~1/f acoustic coloration and wave motion."
    })

    # 3. Harbor Tug (6 seconds, Challenging Low SNR -8 dB)
    dur3 = 6.0
    n3 = int(dur3 * sr)
    t3 = np.arange(n3) / sr
    amb3 = _pink_noise(n3, rng)
    tonal3 = _harmonic_tonal(t3, f0=145.0, n_harmonics=3, decay=0.48, rng=rng)
    hull3 = _broadband_hull_noise(n3, rng, tilt=1.3)
    ship3 = 0.65 * tonal3 + 0.6 * hull3
    snr_lin3 = 10 ** (-8.0 / 20.0)  # -8 dB SNR
    sig3 = snr_lin3 * (ship3 / np.std(ship3)) + (amb3 / np.std(amb3))
    sig3 = _add_transient_clicks(sig3, rng, n_clicks_range=(1, 3))
    sig3 = sps.sosfilt(sos, sig3)
    sig3 = sig3 / (np.max(np.abs(sig3)) + 1e-9)
    wavfile.write(os.path.join(SAMPLE_DIR, "harbor_tug_challenging_snr.wav"), sr, (sig3 * 32767).astype(np.int16))
    samples_info.append({
        "id": "harbor_tug_challenging_snr.wav",
        "title": "Harbor Tug Maneuvering (Low SNR -8dB, 6.0s)",
        "category": "Challenging Acoustic Channel",
        "expected": "Low-SNR Detection",
        "description": "Maneuvering harbor tugboat submerged in heavy background noise; tests model transfer robustness."
    })

    # 4. Biologic Coral Reef (5 seconds) - Snapping shrimp + spurious acoustic resonance (Hard Negative)
    dur4 = 5.0
    n4 = int(dur4 * sr)
    t4 = np.arange(n4) / sr
    amb4 = _pink_noise(n4, rng)
    amb4 = _add_transient_clicks(amb4, rng, n_clicks_range=(4, 8))
    # Add spurious biological reef resonance hump
    hump4 = _spurious_resonance_hump(t4, rng, freq_range=(150, 320))
    sig4 = (amb4 / np.std(amb4)) + 0.45 * (hump4 / np.std(hump4))
    sig4 = sps.sosfilt(sos, sig4)
    sig4 = sig4 / (np.max(np.abs(sig4)) + 1e-9)
    wavfile.write(os.path.join(SAMPLE_DIR, "biologic_coral_reef_ambient.wav"), sr, (sig4 * 32767).astype(np.int16))
    samples_info.append({
        "id": "biologic_coral_reef_ambient.wav",
        "title": "Coral Reef Biologics & Shrimp (5.0s)",
        "category": "Hard Negative Ambient",
        "expected": "No Target / Ambient",
        "description": "High-frequency biological transients (snapping shrimp) and reef resonance without ship presence."
    })

    # 5. Coastal Patrol Craft Transit (8 seconds) - Dynamic entry and exit!
    dur5 = 8.0
    n5 = int(dur5 * sr)
    t5 = np.arange(n5) / sr
    amb5 = _pink_noise(n5, rng)
    tonal5 = _harmonic_tonal(t5, f0=162.0, n_harmonics=4, decay=0.50, rng=rng)
    hull5 = _broadband_hull_noise(n5, rng, tilt=1.15)
    ship5 = 0.7 * tonal5 + 0.5 * hull5
    # Smooth bell-shaped envelope peaking at t=4.5s (entry from 2s, peak at 4.5s, exit by 6.5s)
    envelope = np.exp(-0.5 * ((t5 - 4.5) / 1.3) ** 2)
    # Background ambient throughout, vessel enters and recedes
    sig5 = (amb5 / np.std(amb5)) + 1.8 * envelope * (ship5 / np.std(ship5))
    sig5 = sps.sosfilt(sos, sig5)
    sig5 = sig5 / (np.max(np.abs(sig5)) + 1e-9)
    wavfile.write(os.path.join(SAMPLE_DIR, "coastal_patrol_intermittent.wav"), sr, (sig5 * 32767).astype(np.int16))
    samples_info.append({
        "id": "coastal_patrol_intermittent.wav",
        "title": "Coastal Patrol Craft Passage (8.0s)",
        "category": "Dynamic Multi-Interval Transit",
        "expected": "Target Active in Intervals 2s-6s",
        "description": "Fast patrol vessel entering the hydrophone acoustic detection cone at 2s, peaking, and fading out by 6s."
    })

    print(f"Generated {len(samples_info)} benchmark recordings in {SAMPLE_DIR}")
    return samples_info


if __name__ == "__main__":
    generate_all_samples()
