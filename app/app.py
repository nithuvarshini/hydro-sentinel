"""
Flask Web Application: Underwater Acoustic Target Detection & Monitoring System.

Practical, production-grade passive hydrophone monitoring platform powered by
the Deep-DLR Ensemble acoustic classification pipeline (extending Bao et al. 2025).

Key Capabilities:
  - Real WAV audio upload, validation, and automated multi-rate preprocessing.
  - Automated acoustic signal quality and in-band SNR estimation (no manual SNR/domain input).
  - High-confidence vs tentative vs uncertain detection categorization with
    operator review recommendations based on DLR ensemble disagreement.
  - Multi-segment temporal timeline tracking for continuous audio recordings.
  - Persistent scan history and printable acoustic mission detection reports.
  - Dedicated 'Model Insights' section for academic verification and AML report metrics.
"""
import os
import sys
import json
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.audio_processor import (
    load_and_validate_audio, analyze_recording_pipeline, analyze_signal_quality
)
from src.report_generator import (
    save_analysis_to_history, get_history_list, get_scan_by_id, generate_html_report
)
from src.sample_generator import SAMPLE_DIR, generate_all_samples
from src.data_pipeline import load_domain

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

MODELS = {}


def load_all_models():
    """Load all trained model checkpoints into memory."""
    model_files = {
        "logreg": "logreg_baseline.joblib",
        "shallow_dlr": "shallow_dlr.joblib",
        "deep_transfer": "deep_transfer.joblib",
        "deep_dlr": "deep_dlr.joblib",
    }
    for key, fn in model_files.items():
        p = os.path.join(config.MODELS_DIR, fn)
        if os.path.exists(p):
            MODELS[key] = joblib.load(p)
            print(f"Loaded model [{key}] from {fn}")
        else:
            print(f"Warning: Model file not found: {p}")


# Ensure benchmark audio samples exist
if not os.path.exists(SAMPLE_DIR) or len(os.listdir(SAMPLE_DIR)) < 5:
    try:
        generate_all_samples()
    except Exception as e:
        print(f"Sample generation notice: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def list_samples():
    """Return catalog of available benchmark hydrophone recordings."""
    catalog = [
        {
            "id": "merchant_vessel_transit.wav",
            "title": "Merchant Cargo Vessel (6.0s)",
            "category": "Target Present",
            "badge": "danger",
            "description": "Commercial vessel with diesel engine shaft tonals (118 Hz) and cavitation noise."
        },
        {
            "id": "ocean_ambient_deep_sea.wav",
            "title": "Deep Ocean Ambient Baseline (5.0s)",
            "category": "Ambient Baseline",
            "badge": "info",
            "description": "Undisturbed deep-sea background noise with natural ~1/f acoustic coloration."
        },
        {
            "id": "harbor_tug_challenging_snr.wav",
            "title": "Harbor Tugboat (Low SNR -8dB, 6.0s)",
            "category": "Challenging Channel",
            "badge": "warning",
            "description": "Maneuvering tugboat in high-noise harbor basin; tests transfer learning robustness."
        },
        {
            "id": "biologic_coral_reef_ambient.wav",
            "title": "Coral Reef Biologics & Shrimp (5.0s)",
            "category": "Hard Negative Ambient",
            "badge": "secondary",
            "description": "Snapping shrimp acoustic clicks and reef resonance hump without ship presence."
        },
        {
            "id": "coastal_patrol_intermittent.wav",
            "title": "Coastal Patrol Craft Passage (8.0s)",
            "category": "Multi-Interval Transit",
            "badge": "primary",
            "description": "Fast vessel entering the hydrophone acoustic cone, peaking, and receding."
        }
    ]
    return jsonify(catalog)


@app.route("/api/samples/audio/<filename>")
def serve_sample_audio(filename):
    """Stream sample audio file to browser for audio player playback."""
    clean_name = os.path.basename(filename)
    path = os.path.join(SAMPLE_DIR, clean_name)
    if os.path.exists(path):
        return send_file(path, mimetype="audio/wav")
    return jsonify({"error": "Sample file not found"}), 404


@app.route("/api/analyze_sample/<filename>", methods=["POST"])
def analyze_sample(filename):
    """Analyze one of the pre-packaged hydrophone recordings."""
    clean_name = os.path.basename(filename)
    path = os.path.join(SAMPLE_DIR, clean_name)
    if not os.path.exists(path):
        return jsonify({"error": f"Sample recording '{clean_name}' not found"}), 404

    audio, meta, errs = load_and_validate_audio(path, filename=clean_name)
    if errs:
        return jsonify({"error": "; ".join(errs)}), 400

    result = analyze_recording_pipeline(audio, meta, MODELS)
    scan_id = save_analysis_to_history(result)
    result["scan_id"] = scan_id
    return jsonify(result)


@app.route("/api/analyze_upload", methods=["POST"])
def analyze_upload():
    """Validate, preprocess, and analyze an uploaded real hydrophone WAV file."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided in request"}), 400

    file = request.files["audio"]
    if not file or file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename
    if not filename.lower().endswith(".wav"):
        return jsonify({"error": "Unsupported file format. Please upload an uncompressed WAV audio file."}), 400

    audio, meta, errs = load_and_validate_audio(file.stream, filename=filename)
    if errs:
        return jsonify({"error": "; ".join(errs)}), 400

    result = analyze_recording_pipeline(audio, meta, MODELS)
    scan_id = save_analysis_to_history(result)
    result["scan_id"] = scan_id
    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def get_history():
    """Retrieve history log of past acoustic analyses."""
    return jsonify(get_history_list())


@app.route("/api/history/<scan_id>", methods=["GET"])
def get_scan(scan_id):
    """Retrieve full analysis telemetry for a previous scan ID."""
    record = get_scan_by_id(scan_id)
    if not record:
        return jsonify({"error": f"Scan ID '{scan_id}' not found"}), 404
    return jsonify(record)


@app.route("/api/report/<scan_id>", methods=["GET"])
def download_report(scan_id):
    """Generate and return self-contained printable HTML report."""
    record = get_scan_by_id(scan_id)
    if not record:
        return f"Scan ID '{scan_id}' not found", 404

    full_res = record.get("full_result", {})
    html = generate_html_report(full_res, scan_id=scan_id)
    return Response(html, mimetype="text/html")


@app.route("/api/model_insights", methods=["GET"])
def get_model_insights():
    """Return academic research metrics, benchmark comparisons, and robustness summary."""
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics_table.csv")
    robustness_path = os.path.join(config.RESULTS_DIR, "robustness_summary.json")
    timings_path = os.path.join(config.RESULTS_DIR, "timings.json")

    metrics_data = []
    if os.path.exists(metrics_path):
        import pandas as pd
        df = pd.read_csv(metrics_path)
        for _, row in df.iterrows():
            metrics_data.append({
                "model": row["model"],
                "accuracy": round(row["accuracy"] * 100, 2),
                "precision": round(row["precision"] * 100, 2),
                "recall": round(row["recall"] * 100, 2),
                "f1": round(row["f1"] * 100, 2),
                "roc_auc": round(row["roc_auc"], 4),
            })

    robustness_data = {}
    if os.path.exists(robustness_path):
        with open(robustness_path) as f:
            rob_raw = json.load(f)
            summary = rob_raw.get("summary", {})
            for m_name, s in summary.items():
                robustness_data[m_name] = {
                    "acc_mean": round(s["acc_mean"] * 100, 2),
                    "acc_std": round(s["acc_std"] * 100, 2),
                    "f1_mean": round(s["f1_mean"] * 100, 2),
                    "f1_std": round(s["f1_std"] * 100, 2),
                }

    timings_data = {}
    if os.path.exists(timings_path):
        with open(timings_path) as f:
            timings_data = json.load(f)

    figures = [
        {"id": "fig_accuracy_comparison.png", "title": "Held-Out Test Accuracy Comparison across 6 Architectures"},
        {"id": "fig_confusion_matrices.png", "title": "Confusion Matrices on Held-Out Test Target Domain"},
        {"id": "fig_roc_curves.png", "title": "ROC Curves & Area Under Curve (AUC)"},
        {"id": "fig_robustness.png", "title": "Robustness Across 5 Independent Random Seeds (Mean ± Std)"},
        {"id": "fig_error_by_snr.png", "title": "Error Analysis: Performance vs Signal Energy (SNR Proxy)"},
    ]

    return jsonify({
        "metrics": metrics_data,
        "robustness": robustness_data,
        "timings": timings_data,
        "figures": figures,
    })


@app.route("/results_viz/<filename>")
def serve_results_viz(filename):
    """Serve evaluation figures for the Model Insights tab."""
    clean_name = os.path.basename(filename)
    return send_from_directory(config.RESULTS_DIR, clean_name)


if __name__ == "__main__":
    load_all_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
else:
    load_all_models()
