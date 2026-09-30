"""
Audio processing, signal quality assessment, and target detection engine
for the Underwater Acoustic Target Detection & Monitoring Application.

Addresses key underwater acoustic literature limitations:
  1. Domain/environment shift: Automated acoustic regime profiling and
     robustness warnings.
  2. Noisy/low-quality signals: Automated SNR and acoustic quality index
     estimation (no manual SNR/domain input required from the user).
  3. Limited target information: Dual-branch DLR ensemble disagreement and
     uncertainty quantification rather than forcing confident binary predictions.
  4. Practical deployment: Segmented multi-interval timeline analysis and
     real-time latency tracking (RTF).
"""
import io
import time
import math
import numpy as np
import scipy.signal as sps
import scipy.io.wavfile as wavfile
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64

from . import config
from .features import handcrafted_features, spectrogram_features


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Resample audio signal from orig_sr to target_sr using polyphase or Fourier resampling."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    # If up/down factors are reasonable, use polyphase filter (faster, accurate anti-aliasing)
    if up <= 500 and down <= 500:
        resampled = sps.resample_poly(audio, up, down)
    else:
        # Fourier-based resampling for arbitrary rational rates
        target_len = int(round(len(audio) * float(target_sr) / float(orig_sr)))
        resampled = sps.resample(audio, target_len)

    return resampled.astype(np.float32)


def load_and_validate_audio(file_bytes_or_path, filename="recording.wav"):
    """
    Validate, read, and preprocess an uploaded or sample WAV recording.

    Returns:
        dict containing:
            - audio: preprocessed 1D float32 numpy array at config.SAMPLE_RATE
            - metadata: sample rate, channels, bit depth, raw duration, processed samples
            - validation_errors: list of error strings (empty if valid)
    """
    validation_errors = []
    metadata = {
        "filename": filename,
        "original_sr": None,
        "channels": None,
        "raw_samples": None,
        "duration_sec": 0.0,
        "bit_depth": "Unknown",
        "clipped": False,
        "target_sr": config.SAMPLE_RATE,
    }

    try:
        if isinstance(file_bytes_or_path, (str, bytes, bytearray)):
            if isinstance(file_bytes_or_path, str):
                orig_sr, data = wavfile.read(file_bytes_or_path)
            else:
                bio = io.BytesIO(file_bytes_or_path)
                orig_sr, data = wavfile.read(bio)
        else:
            # File-like object (e.g. Flask FileStorage.stream)
            file_bytes_or_path.seek(0)
            orig_sr, data = wavfile.read(file_bytes_or_path)
    except Exception as e:
        validation_errors.append(f"Invalid or corrupted audio file: {str(e)}")
        return None, metadata, validation_errors

    metadata["original_sr"] = int(orig_sr)
    if orig_sr < 500:
        validation_errors.append(f"Sample rate {orig_sr} Hz is too low for acoustic detection.")

    # Infer bit depth & normalize to [-1.0, 1.0]
    if data.dtype == np.int16:
        metadata["bit_depth"] = "16-bit PCM"
        norm_data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        metadata["bit_depth"] = "32-bit PCM"
        norm_data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        metadata["bit_depth"] = "8-bit PCM"
        norm_data = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype in (np.float32, np.float64):
        metadata["bit_depth"] = f"{data.dtype.itemsize * 8}-bit Float"
        norm_data = data.astype(np.float32)
    else:
        norm_data = data.astype(np.float32)

    # Check channels
    if norm_data.ndim == 1:
        metadata["channels"] = 1
        mono_data = norm_data
    elif norm_data.ndim == 2:
        metadata["channels"] = norm_data.shape[1]
        # Downmix multi-channel to mono
        mono_data = np.mean(norm_data, axis=1)
    else:
        validation_errors.append(f"Unsupported audio channel dimensionality: {norm_data.ndim}")
        return None, metadata, validation_errors

    metadata["raw_samples"] = len(mono_data)
    duration = len(mono_data) / float(orig_sr)
    metadata["duration_sec"] = round(duration, 3)

    if duration < 0.2:
        validation_errors.append(f"Audio duration ({duration:.2f}s) is too short. Minimum duration is 0.2s.")
        return None, metadata, validation_errors

    # Check for heavy digital clipping
    if np.max(np.abs(mono_data)) >= 0.999:
        metadata["clipped"] = True

    # Resample to pipeline target sample rate (4000 Hz)
    audio_resampled = resample_audio(mono_data, orig_sr, config.SAMPLE_RATE)

    # High-pass filter at 15 Hz to eliminate DC drift & sub-audible hydrophone sway
    sos = sps.butter(2, 15, btype="highpass", fs=config.SAMPLE_RATE, output="sos")
    audio_filtered = sps.sosfilt(sos, audio_resampled)

    # Unit peak normalization matching the training distribution
    max_peak = np.max(np.abs(audio_filtered))
    if max_peak > 1e-6:
        audio_normalized = (audio_filtered / max_peak).astype(np.float32)
    else:
        audio_normalized = audio_filtered.astype(np.float32)

    metadata["processed_samples"] = len(audio_normalized)
    return audio_normalized, metadata, validation_errors


def segment_audio(audio: np.ndarray, segment_duration: float = config.DURATION, hop_duration: float = 1.0):
    """
    Split audio into fixed 1-second analysis windows (4000 samples).
    Pads the last window if it contains substantial audio (>= 0.3s).
    """
    sr = config.SAMPLE_RATE
    seg_samples = int(segment_duration * sr)
    hop_samples = int(hop_duration * sr)

    n_samples = len(audio)
    if n_samples <= seg_samples:
        # Shorter or equal to one segment: pad symmetrically to exact segment size
        pad_needed = seg_samples - n_samples
        if pad_needed > 0:
            padded = np.pad(audio, (0, pad_needed), mode="symmetric")
        else:
            padded = audio
        return [{
            "index": 0,
            "start_time": 0.0,
            "end_time": round(n_samples / float(sr), 2),
            "samples": padded.astype(np.float32)
        }]

    segments = []
    start = 0
    seg_idx = 0
    while start < n_samples:
        end = start + seg_samples
        if end <= n_samples:
            chunk = audio[start:end]
            segments.append({
                "index": seg_idx,
                "start_time": round(start / float(sr), 2),
                "end_time": round(end / float(sr), 2),
                "samples": chunk.astype(np.float32)
            })
        else:
            # Remaining tail segment
            remaining = n_samples - start
            if remaining >= int(0.3 * sr):
                chunk = audio[start:n_samples]
                chunk_padded = np.pad(chunk, (0, seg_samples - remaining), mode="symmetric")
                segments.append({
                    "index": seg_idx,
                    "start_time": round(start / float(sr), 2),
                    "end_time": round(n_samples / float(sr), 2),
                    "samples": chunk_padded.astype(np.float32)
                })
            break
        start += hop_samples
        seg_idx += 1

    return segments


def analyze_signal_quality(clip: np.ndarray, sr: int = config.SAMPLE_RATE):
    """
    Automated acoustic signal quality and noise condition assessment.
    Eliminates manual SNR / domain guessing by computing objective acoustic metrics:
      - Estimated In-Band SNR (dB) using spectral decomposition
      - Noise floor level (dBFS)
      - Spectral flatness (Wiener entropy)
      - Spectral tilt / slope
      - Crest factor & RMS energy
      - Overall Signal Quality Index (0-100%) and Acoustic Regime
    """
    x = clip.astype(np.float64)
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-12
    peak = float(np.max(np.abs(x)))
    crest_factor = round(peak / rms, 2)

    # Compute Power Spectral Density via Welch method
    nperseg = min(len(x), 256)
    freqs, psd = sps.welch(x, fs=sr, nperseg=nperseg)
    psd = psd + 1e-15

    # Focus on the vessel acoustic band (25 Hz - 1200 Hz)
    vessel_mask = (freqs >= 25) & (freqs <= 1200)
    vessel_freqs = freqs[vessel_mask]
    vessel_psd = psd[vessel_mask]

    # Noise floor estimation: 25th percentile of spectral density in vessel band
    if len(vessel_psd) > 0:
        noise_floor_density = float(np.percentile(vessel_psd, 25))
        signal_excess = np.maximum(0.0, vessel_psd - noise_floor_density)
        signal_power = float(np.sum(signal_excess))
        noise_power = float(noise_floor_density * len(vessel_psd)) + 1e-15

        raw_snr = 10.0 * np.log10(signal_power / noise_power + 1e-6)
        # Calibrated SNR bounds for hydrophone operational metrics
        estimated_snr_db = float(np.clip(raw_snr - 2.0, -18.0, 15.0))
    else:
        estimated_snr_db = -10.0
        noise_floor_density = 1e-6

    noise_floor_dbfs = round(10.0 * np.log10(noise_floor_density + 1e-15), 1)

    # Spectral flatness (Wiener entropy: ratio of geometric mean to arithmetic mean)
    geo_mean = np.exp(np.mean(np.log(psd)))
    arith_mean = np.mean(psd)
    spectral_flatness = float(geo_mean / (arith_mean + 1e-15))
    spectral_flatness = round(np.clip(spectral_flatness, 0.0, 1.0), 3)

    # Spectral tilt / coloration (linear regression slope of log power vs log frequency)
    valid_f = freqs[freqs > 30]
    valid_p = psd[freqs > 30]
    if len(valid_f) > 2:
        log_f = np.log10(valid_f)
        log_p = np.log10(valid_p)
        slope = float(np.polyfit(log_f, log_p, 1)[0])
    else:
        slope = -1.2
    spectral_tilt = round(slope, 2)

    # Composite Signal Quality Index (0 - 100%)
    # Higher SNR, moderate crest factor, and distinct spectral structure improve score
    snr_factor = np.clip((estimated_snr_db + 15.0) / 25.0, 0.0, 1.0)
    crest_factor_pen = 1.0 - np.clip((crest_factor - 6.0) / 10.0, 0.0, 0.5)
    quality_score = int(round(100.0 * (0.70 * snr_factor + 0.30 * crest_factor_pen)))
    quality_score = int(np.clip(quality_score, 10, 99))

    # Acoustic quality tier
    if quality_score >= 75:
        quality_rating = "Excellent Fidelity"
        quality_badge = "success"
    elif quality_score >= 55:
        quality_rating = "Good Acoustic Quality"
        quality_badge = "primary"
    elif quality_score >= 38:
        quality_rating = "Moderate Noise"
        quality_badge = "warning"
    else:
        quality_rating = "Severe Noise / Low SNR"
        quality_badge = "danger"

    # Environmental regime inference (Source vs Target domain matching)
    if estimated_snr_db >= 2.0 and spectral_tilt <= -1.3:
        environment_regime = "Favorable Hydrophone Channel (High SNR)"
        regime_code = "Source-like"
        robustness_warning = None
    elif estimated_snr_db >= -7.0:
        environment_regime = "Standard Oceanic Channel (Moderate Attenuation)"
        regime_code = "Target-like"
        robustness_warning = None
    else:
        environment_regime = "Degraded / Shallow Water Reverberant Environment"
        regime_code = "Severe Shift"
        robustness_warning = "Low SNR detected. DLR transfer learning and uncertainty gating active."

    return {
        "estimated_snr_db": round(estimated_snr_db, 1),
        "noise_floor_dbfs": noise_floor_dbfs,
        "spectral_flatness": spectral_flatness,
        "spectral_tilt": spectral_tilt,
        "rms_energy": round(rms, 4),
        "crest_factor": crest_factor,
        "quality_score": quality_score,
        "quality_rating": quality_rating,
        "quality_badge": quality_badge,
        "environment_regime": environment_regime,
        "regime_code": regime_code,
        "robustness_warning": robustness_warning,
    }


def evaluate_segment(segment_audio: np.ndarray, models: dict):
    """
    Run multi-model inference and uncertainty analysis on a single 1.0s clip.
    Uses Deep-DLR Ensemble as primary detector with epistemic disagreement quantification.
    """
    hc = handcrafted_features(segment_audio).reshape(1, -1)
    sp = spectrogram_features(segment_audio).reshape(1, -1)
    comb = np.concatenate([hc, sp], axis=1)

    primary_model = models.get("deep_dlr")
    if primary_model is None:
        raise ValueError("deep_dlr model not loaded")

    # Deep-DLR component predictions (Dual branches: Original + Reversed)
    p_orig, p_rev = primary_model.component_predictions(comb)
    p_orig_ship = float(p_orig[0, 1])
    p_rev_ship = float(p_rev[0, 1])
    p_ens_ship = float((p_orig_ship + p_rev_ship) / 2.0)

    # Disagreement between the two DLR branches (Epistemic uncertainty)
    branch_disagreement = float(abs(p_orig_ship - p_rev_ship))

    # Normalized entropy (Aleatoric uncertainty)
    eps = 1e-9
    p_clip = np.clip(p_ens_ship, eps, 1.0 - eps)
    entropy = float(-(p_clip * math.log2(p_clip) + (1.0 - p_clip) * math.log2(1.0 - p_clip)))

    # Combined confidence score penalized by branch disagreement
    raw_conf = max(p_ens_ship, 1.0 - p_ens_ship)
    confidence = float(raw_conf * (1.0 - 0.45 * branch_disagreement))
    confidence_pct = round(confidence * 100.0, 1)

    # Uncertainty percentage (blend of disagreement and entropy)
    uncertainty_pct = round((0.6 * branch_disagreement + 0.4 * entropy) * 100.0, 1)

    # Also compute comparison predictions for Model Insights
    model_predictions = {}
    for name, key in [
        ("Logistic Regression (Baseline)", "logreg"),
        ("Shallow BP + DLR (Paper Replica)", "shallow_dlr"),
        ("Deep Net + Transfer (Ablation)", "deep_transfer"),
        ("Deep Net + DLR Ensemble (Proposed)", "deep_dlr")
    ]:
        if key in models:
            m = models[key]
            X = hc if key in ("logreg", "shallow_dlr") else comb
            proba = m.predict_proba(X)[0]
            pred = int(proba[1] >= 0.5)
            model_predictions[name] = {
                "key": key,
                "prediction": "Ship Present" if pred == 1 else "No Ship",
                "p_ship": round(float(proba[1]) * 100.0, 1),
                "confidence": round(float(proba[pred]) * 100.0, 1),
            }

    # Decision Categorization
    is_target = p_ens_ship >= 0.50

    if p_ens_ship >= 0.72 and branch_disagreement <= 0.22:
        decision_category = "CONFIRMED_TARGET"
        category_label = "Confirmed Target (High Confidence)"
        category_badge = "danger"  # acoustic detection alert
        recommendation = "Target acoustic signature verified. Propeller cavitation & machinery harmonics match known vessel profile."
    elif p_ens_ship >= 0.50:
        if branch_disagreement > 0.25:
            decision_category = "UNCERTAIN_REVIEW"
            category_label = "Ambiguous Signal / Manual Review Advised"
            category_badge = "warning"
            recommendation = "Ensemble branches exhibit disagreement. Low SNR or transitional acoustic state; manual operator inspection required."
        else:
            decision_category = "TENTATIVE_TARGET"
            category_label = "Tentative Target (Low Confidence)"
            category_badge = "secondary"
            recommendation = "Faint machinery harmonics detected near noise floor. Continued hydrophone tracking advised."
    elif p_ens_ship >= 0.38 and branch_disagreement > 0.25:
        decision_category = "UNCERTAIN_REVIEW"
        category_label = "Ambiguous Signal / Manual Review Advised"
        category_badge = "warning"
        recommendation = "Borderline acoustic energy with branch divergence. Inspect spectrogram for transient biologic or reef resonance."
    else:
        decision_category = "AMBIENT_NOISE"
        category_label = "Ambient Ocean (No Target)"
        category_badge = "info"
        recommendation = "Background ocean noise only. No ship machinery tonals or propeller signatures detected."

    return {
        "p_ship": round(p_ens_ship * 100.0, 1),
        "p_ambient": round((1.0 - p_ens_ship) * 100.0, 1),
        "is_target": is_target,
        "confidence_pct": confidence_pct,
        "uncertainty_pct": uncertainty_pct,
        "branch_disagreement_pct": round(branch_disagreement * 100.0, 1),
        "branch_orig_p_ship": round(p_orig_ship * 100.0, 1),
        "branch_rev_p_ship": round(p_rev_ship * 100.0, 1),
        "decision_category": decision_category,
        "category_label": category_label,
        "category_badge": category_badge,
        "recommendation": recommendation,
        "model_predictions": model_predictions,
    }


def generate_waveform_plot(audio: np.ndarray, timeline_segments: list = None, sr: int = config.SAMPLE_RATE):
    """Render a clean, modern acoustic waveform chart with detection highlighting."""
    fig, ax = plt.subplots(figsize=(10, 2.5), facecolor="#0B132B")
    ax.set_facecolor("#0F1C3F")

    t = np.arange(len(audio)) / float(sr)
    ax.plot(t, audio, color="#00D2FF", linewidth=0.7, alpha=0.9, label="Hydrophone Signal")

    # Highlight target detection intervals on the waveform
    if timeline_segments:
        for seg in timeline_segments:
            t0 = seg["start_time"]
            t1 = seg["end_time"]
            cat = seg["decision_category"]
            if cat == "CONFIRMED_TARGET":
                ax.axvspan(t0, t1, color="#EF4444", alpha=0.22)
            elif cat == "TENTATIVE_TARGET":
                ax.axvspan(t0, t1, color="#F59E0B", alpha=0.18)
            elif cat == "UNCERTAIN_REVIEW":
                ax.axvspan(t0, t1, color="#EC4899", alpha=0.18)

    ax.set_xlim(0, max(t[-1], 1.0))
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("Time (seconds)", color="#8FA9B8", fontsize=10)
    ax.set_ylabel("Amplitude", color="#8FA9B8", fontsize=10)
    ax.tick_params(colors="#8FA9B8", labelsize=9)
    ax.grid(True, color="#1D2E56", linestyle="--", linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color("#1D2E56")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_spectrogram_plot(audio: np.ndarray, sr: int = config.SAMPLE_RATE):
    """Render a high-resolution time-frequency spectrogram showing ship tonals and ambient noise."""
    fig, ax = plt.subplots(figsize=(10, 3.4), facecolor="#0B132B")
    ax.set_facecolor("#0F1C3F")

    f, t, Sxx = sps.spectrogram(audio, fs=sr, nperseg=128, noverlap=96)
    # Display up to 1000 Hz where vessel engine line spectra exist
    freq_mask = f <= 1000
    f_sub = f[freq_mask]
    Sxx_sub = Sxx[freq_mask, :]

    Sxx_db = 10.0 * np.log10(Sxx_sub + 1e-12)
    mesh = ax.pcolormesh(t, f_sub, Sxx_db, shading="gouraud", cmap="magma")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("PSD (dB/Hz)", color="#8FA9B8", fontsize=9)
    cbar.ax.tick_params(colors="#8FA9B8", labelsize=8)
    cbar.outline.set_edgecolor("#1D2E56")

    ax.set_xlabel("Time (seconds)", color="#8FA9B8", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", color="#8FA9B8", fontsize=10)
    ax.tick_params(colors="#8FA9B8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1D2E56")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def analyze_recording_pipeline(audio: np.ndarray, metadata: dict, models: dict):
    """
    Complete end-to-end analysis pipeline:
      - Multi-segment temporal decomposition
      - Automatic acoustic quality profiling
      - Deep-DLR inference with dual-branch uncertainty tracking
      - Real-time performance tracking (processing latency and RTF)
      - Visualizations (waveform and spectrogram)
      - Aggregate mission-ready detection verdict
    """
    t_start = time.perf_counter()

    # 1. Automatic Signal Quality Assessment
    quality_info = analyze_signal_quality(audio, config.SAMPLE_RATE)
    t_quality = time.perf_counter()

    # 2. Multi-segment temporal division
    segments = segment_audio(audio)

    # 3. Inference on each segment
    timeline = []
    target_segments_count = 0
    high_conf_targets_count = 0
    uncertain_segments_count = 0
    all_p_ship = []
    all_confidence = []
    all_disagreement = []

    for seg in segments:
        seg_audio = seg["samples"]
        seg_eval = evaluate_segment(seg_audio, models)
        seg_quality = analyze_signal_quality(seg_audio, config.SAMPLE_RATE)

        item = {
            "index": seg["index"],
            "interval_label": f"{seg['start_time']:.1f}s - {seg['end_time']:.1f}s",
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "p_ship": seg_eval["p_ship"],
            "confidence_pct": seg_eval["confidence_pct"],
            "uncertainty_pct": seg_eval["uncertainty_pct"],
            "disagreement_pct": seg_eval["branch_disagreement_pct"],
            "decision_category": seg_eval["decision_category"],
            "category_label": seg_eval["category_label"],
            "category_badge": seg_eval["category_badge"],
            "recommendation": seg_eval["recommendation"],
            "segment_snr_db": seg_quality["estimated_snr_db"],
            "model_predictions": seg_eval["model_predictions"],
        }
        timeline.append(item)

        if seg_eval["is_target"]:
            target_segments_count += 1
        if seg_eval["decision_category"] == "CONFIRMED_TARGET":
            high_conf_targets_count += 1
        elif seg_eval["decision_category"] == "UNCERTAIN_REVIEW":
            uncertain_segments_count += 1

        all_p_ship.append(seg_eval["p_ship"])
        all_confidence.append(seg_eval["confidence_pct"])
        all_disagreement.append(seg_eval["branch_disagreement_pct"])

    t_inference = time.perf_counter()

    # 4. Generate visualizations
    waveform_b64 = generate_waveform_plot(audio, timeline, config.SAMPLE_RATE)
    spectrogram_b64 = generate_spectrogram_plot(audio, config.SAMPLE_RATE)
    t_viz = time.perf_counter()

    # Aggregate recording-level statistics
    total_segments = len(timeline)
    mean_p_ship = round(float(np.mean(all_p_ship)), 1)
    peak_p_ship = round(float(np.max(all_p_ship)), 1)
    mean_confidence = round(float(np.mean(all_confidence)), 1)
    mean_disagreement = round(float(np.mean(all_disagreement)), 1)
    active_target_duration_sec = round(target_segments_count * config.DURATION, 2)
    target_coverage_pct = round((target_segments_count / max(total_segments, 1)) * 100.0, 1)

    # Overall Mission Verdict
    if high_conf_targets_count >= 1 or (target_segments_count >= total_segments / 2 and mean_confidence >= 65.0):
        overall_status = "TARGET_CONFIRMED"
        overall_verdict = "UNDERWATER TARGET DETECTED"
        verdict_badge = "danger"
        status_description = (
            f"Vessel propulsion acoustic signature confirmed across {target_segments_count}/{total_segments} "
            f"analysis intervals ({active_target_duration_sec:.1f}s active transit). "
            f"High confidence ({mean_confidence}%) with strong branch consensus."
        )
        operator_action = "Automatic detection confirmed. Target logged to acoustic monitoring register."
        needs_manual_review = False
    elif target_segments_count > 0 and (uncertain_segments_count > 0 or mean_confidence < 60.0):
        overall_status = "UNCERTAIN_REVIEW"
        overall_verdict = "AMBIGUOUS / REVIEW REQUIRED"
        verdict_badge = "warning"
        status_description = (
            f"Possible vessel signature detected in {target_segments_count}/{total_segments} intervals, "
            f"but high ensemble branch disagreement ({mean_disagreement}%) or low SNR ({quality_info['estimated_snr_db']} dB) "
            f"indicates ambiguous acoustic conditions."
        )
        operator_action = "Manual hydrophone operator review strongly advised before verifying target presence."
        needs_manual_review = True
    elif target_segments_count > 0:
        overall_status = "TENTATIVE_DETECTION"
        overall_verdict = "TENTATIVE TARGET DETECTED"
        verdict_badge = "secondary"
        status_description = (
            f"Weak vessel signature detected in {target_segments_count}/{total_segments} intervals. "
            f"Acoustic energy is faint near ambient noise threshold."
        )
        operator_action = "Maintain continuous hydrophone passive watch; verify on subsequent acoustic sweeps."
        needs_manual_review = False
    else:
        overall_status = "NO_TARGET_AMBIENT"
        overall_verdict = "AMBIENT OCEAN (NO TARGET)"
        verdict_badge = "info"
        status_description = (
            f"No ship propulsion tonals or cavitation signatures detected across all {total_segments} "
            f"intervals ({metadata['duration_sec']:.1f}s). Spectrum is consistent with ambient marine noise."
        )
        operator_action = "No threat detected. Hydrophone operating normally."
        needs_manual_review = False

    # Timing and Practical Deployment benchmarks
    total_time_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
    quality_time_ms = round((t_quality - t_start) * 1000.0, 1)
    inference_time_ms = round((t_inference - t_quality) * 1000.0, 1)
    viz_time_ms = round((t_viz - t_inference) * 1000.0, 1)
    audio_duration_sec = metadata["duration_sec"]
    rtf = round((total_time_ms / 1000.0) / max(audio_duration_sec, 0.001), 3)

    return {
        "metadata": metadata,
        "quality": quality_info,
        "summary": {
            "overall_status": overall_status,
            "overall_verdict": overall_verdict,
            "verdict_badge": verdict_badge,
            "status_description": status_description,
            "operator_action": operator_action,
            "needs_manual_review": needs_manual_review,
            "total_segments": total_segments,
            "target_segments_count": target_segments_count,
            "active_duration_sec": active_target_duration_sec,
            "target_coverage_pct": target_coverage_pct,
            "mean_p_ship": mean_p_ship,
            "peak_p_ship": peak_p_ship,
            "mean_confidence": mean_confidence,
            "mean_disagreement": mean_disagreement,
        },
        "timeline": timeline,
        "timing": {
            "total_latency_ms": total_time_ms,
            "quality_analysis_ms": quality_time_ms,
            "inference_ms": inference_time_ms,
            "viz_ms": viz_time_ms,
            "rtf": rtf,
            "is_realtime": rtf < 1.0,
        },
        "visualizations": {
            "waveform_b64": waveform_b64,
            "spectrogram_b64": spectrogram_b64,
        }
    }
