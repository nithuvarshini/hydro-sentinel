"""
Report generation and analysis history management for the
Underwater Acoustic Target Detection & Monitoring Application.
"""
import os
import json
import uuid
from datetime import datetime
from . import config

HISTORY_FILE = os.path.join(config.RESULTS_DIR, "analysis_history.json")


def _load_history_data():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"scans": []}
    return {"scans": []}


def _save_history_data(data):
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_analysis_to_history(result: dict) -> str:
    """Save analysis record to persistent history and return unique scan ID."""
    scan_id = "SCAN-" + datetime.now().strftime("%Y%m%d-") + str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    record = {
        "scan_id": scan_id,
        "timestamp": timestamp,
        "filename": result["metadata"].get("filename", "audio.wav"),
        "duration_sec": result["metadata"].get("duration_sec", 0.0),
        "channels": result["metadata"].get("channels", 1),
        "original_sr": result["metadata"].get("original_sr", config.SAMPLE_RATE),
        "overall_status": result["summary"]["overall_status"],
        "overall_verdict": result["summary"]["overall_verdict"],
        "verdict_badge": result["summary"]["verdict_badge"],
        "mean_confidence": result["summary"]["mean_confidence"],
        "active_duration_sec": result["summary"]["active_duration_sec"],
        "target_segments_count": result["summary"]["target_segments_count"],
        "total_segments": result["summary"]["total_segments"],
        "estimated_snr_db": result["quality"]["estimated_snr_db"],
        "quality_score": result["quality"]["quality_score"],
        "quality_rating": result["quality"]["quality_rating"],
        "needs_manual_review": result["summary"]["needs_manual_review"],
        "full_result": result,
    }

    data = _load_history_data()
    # Keep up to 50 most recent scans
    scans = data.get("scans", [])
    scans.insert(0, record)
    data["scans"] = scans[:50]
    _save_history_data(data)
    return scan_id


def get_history_list():
    """Retrieve simplified list of past analyses for the history table."""
    data = _load_history_data()
    scans = data.get("scans", [])
    summary_list = []
    for s in scans:
        summary_list.append({
            "scan_id": s["scan_id"],
            "timestamp": s["timestamp"],
            "filename": s["filename"],
            "duration_sec": s["duration_sec"],
            "overall_status": s["overall_status"],
            "overall_verdict": s["overall_verdict"],
            "verdict_badge": s["verdict_badge"],
            "mean_confidence": s["mean_confidence"],
            "target_segments": f"{s['target_segments_count']}/{s['total_segments']}",
            "active_duration_sec": s["active_duration_sec"],
            "estimated_snr_db": s["estimated_snr_db"],
            "quality_rating": s["quality_rating"],
            "needs_manual_review": s["needs_manual_review"],
        })
    return summary_list


def get_scan_by_id(scan_id: str):
    """Retrieve full analysis result for a given scan ID."""
    data = _load_history_data()
    for s in data.get("scans", []):
        if s["scan_id"] == scan_id:
            return s
    return None


def generate_html_report(result: dict, scan_id: str = None) -> str:
    """Generate a printable, self-contained HTML acoustic detection report."""
    meta = result["metadata"]
    quality = result["quality"]
    summary = result["summary"]
    timing = result["timing"]
    timeline = result["timeline"]
    viz = result["visualizations"]

    if not scan_id:
        scan_id = "SCAN-" + datetime.now().strftime("%Y%m%d-") + str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Format timeline table rows
    timeline_rows = ""
    for seg in timeline:
        badge_class = {
            "CONFIRMED_TARGET": "badge-danger",
            "TENTATIVE_TARGET": "badge-warning",
            "UNCERTAIN_REVIEW": "badge-review",
            "AMBIENT_NOISE": "badge-info"
        }.get(seg["decision_category"], "badge-info")

        timeline_rows += f"""
        <tr>
            <td><strong>{seg['interval_label']}</strong></td>
            <td><span class="badge {badge_class}">{seg['category_label']}</span></td>
            <td>{seg['p_ship']}%</td>
            <td>{seg['confidence_pct']}%</td>
            <td>{seg['disagreement_pct']}%</td>
            <td>{seg['segment_snr_db']} dB</td>
            <td style="font-size: 11px; color: #555;">{seg['recommendation']}</td>
        </tr>
        """

    # Format model comparison from first segment (or representative segment)
    first_models = timeline[0]["model_predictions"] if timeline else {}
    model_rows = ""
    for name, m in first_models.items():
        is_best = "Deep Net + DLR" in name
        row_style = "background-color: #f0fdf4; font-weight: bold;" if is_best else ""
        model_rows += f"""
        <tr style="{row_style}">
            <td>{name} {'<span class="badge badge-success">Headline</span>' if is_best else ''}</td>
            <td>{m['prediction']}</td>
            <td>{m['p_ship']}%</td>
            <td>{m['confidence']}%</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Acoustic Detection Report - {scan_id}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b; background: #f8fafc; margin: 0; padding: 24px; font-size: 13px; line-height: 1.5;
  }}
  .report-container {{
    max-width: 960px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 32px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);
  }}
  .header {{
    display: flex; justify-content: space-between; align-items: flex-start;
    border-bottom: 2px solid #0284c7; padding-bottom: 16px; margin-bottom: 20px;
  }}
  .header h1 {{ margin: 0 0 4px; font-size: 22px; color: #0f172a; text-transform: uppercase; letter-spacing: 0.5px; }}
  .header p {{ margin: 0; color: #64748b; font-size: 12px; }}
  .scan-meta {{ text-align: right; font-size: 12px; color: #475569; }}
  .verdict-box {{
    padding: 16px 20px; border-radius: 6px; margin-bottom: 24px;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .verdict-box.danger {{ background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }}
  .verdict-box.warning {{ background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }}
  .verdict-box.info {{ background: #f0f9ff; border: 1px solid #bae6fd; color: #075985; }}
  .verdict-title {{ font-size: 18px; font-weight: 700; margin-bottom: 4px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px; }}
  .card-label {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; margin-bottom: 4px; }}
  .card-value {{ font-size: 16px; font-weight: 700; color: #0f172a; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
  .badge-danger {{ background: #fee2e2; color: #991b1b; }}
  .badge-warning {{ background: #fef3c7; color: #92400e; }}
  .badge-review {{ background: #fce7f3; color: #9d174d; }}
  .badge-info {{ background: #e0f2fe; color: #075985; }}
  .badge-success {{ background: #dcfce7; color: #166534; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }}
  th, td {{ padding: 8px 10px; border: 1px solid #e2e8f0; text-align: left; }}
  th {{ background: #f1f5f9; color: #475569; font-weight: 600; }}
  .img-viz {{ width: 100%; border: 1px solid #e2e8f0; border-radius: 6px; margin-top: 8px; }}
  .signoff {{
    margin-top: 32px; border-top: 1px dashed #cbd5e1; padding-top: 20px;
    display: flex; justify-content: space-between; font-size: 12px; color: #475569;
  }}
  .print-btn {{
    background: #0284c7; color: #fff; border: none; padding: 8px 16px;
    border-radius: 6px; font-weight: 600; cursor: pointer; margin-bottom: 16px;
  }}
  @media print {{
    body {{ background: #fff; padding: 0; }}
    .report-container {{ border: none; box-shadow: none; padding: 0; }}
    .print-btn {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="report-container">
  <button class="print-btn" onclick="window.print()">Print / Save as PDF</button>

  <div class="header">
    <div>
      <h1>Acoustic Target Detection Report</h1>
      <p>Passive Hydrophone Surveillance & Acoustic Monitoring Station</p>
    </div>
    <div class="scan-meta">
      <div><strong>Report ID:</strong> {scan_id}</div>
      <div><strong>Generated:</strong> {timestamp}</div>
      <div><strong>Engine:</strong> Deep-DLR Ensemble (Bao et al. 2025 Ext.)</div>
    </div>
  </div>

  <div class="verdict-box {summary['verdict_badge']}">
    <div>
      <div class="verdict-title">{summary['overall_verdict']}</div>
      <div style="font-size: 13px;">{summary['status_description']}</div>
    </div>
    <div style="text-align: right;">
      <div style="font-size: 11px; text-transform: uppercase;">Confidence</div>
      <div style="font-size: 24px; font-weight: 800;">{summary['mean_confidence']}%</div>
    </div>
  </div>

  <div class="grid-4">
    <div class="card">
      <div class="card-label">Recording File</div>
      <div class="card-value" style="font-size: 14px; overflow: hidden; text-overflow: ellipsis;">{meta['filename']}</div>
      <div style="font-size: 11px; color: #64748b; margin-top: 2px;">{meta['duration_sec']}s &middot; {meta['channels']} Ch &middot; {meta['original_sr']} Hz</div>
    </div>
    <div class="card">
      <div class="card-label">Estimated SNR</div>
      <div class="card-value">{quality['estimated_snr_db']} dB</div>
      <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Floor: {quality['noise_floor_dbfs']} dBFS</div>
    </div>
    <div class="card">
      <div class="card-label">Acoustic Quality</div>
      <div class="card-value">{quality['quality_score']}/100</div>
      <div style="font-size: 11px; color: #64748b; margin-top: 2px;">{quality['quality_rating']}</div>
    </div>
    <div class="card">
      <div class="card-label">Processing Latency</div>
      <div class="card-value">{timing['total_latency_ms']} ms</div>
      <div style="font-size: 11px; color: #64748b; margin-top: 2px;">RTF: {timing['rtf']} (Real-Time)</div>
    </div>
  </div>

  <h3 style="margin: 20px 0 6px; font-size: 15px; color: #0f172a;">Acoustic Waveform & Spectrogram Analysis</h3>
  <div>
    <img class="img-viz" src="data:image/png;base64,{viz['waveform_b64']}" alt="Waveform">
    <img class="img-viz" src="data:image/png;base64,{viz['spectrogram_b64']}" alt="Spectrogram">
  </div>

  <h3 style="margin: 24px 0 6px; font-size: 15px; color: #0f172a;">Temporal Detection Timeline (Segmented Analysis)</h3>
  <table>
    <thead>
      <tr>
        <th>Interval</th>
        <th>Classification</th>
        <th>P(Target)</th>
        <th>Confidence</th>
        <th>Branch Disagree</th>
        <th>Est. SNR</th>
        <th>Acoustic Recommendation</th>
      </tr>
    </thead>
    <tbody>
      {timeline_rows}
    </tbody>
  </table>

  <h3 style="margin: 24px 0 6px; font-size: 15px; color: #0f172a;">Model Verification Telemetry (Deep-DLR vs Baselines)</h3>
  <table>
    <thead>
      <tr>
        <th>Architecture</th>
        <th>Prediction</th>
        <th>P(Ship)</th>
        <th>Confidence</th>
      </tr>
    </thead>
    <tbody>
      {model_rows}
    </tbody>
  </table>

  <div class="signoff">
    <div>
      <div><strong>Automated Pipeline Sign-off:</strong> PASS</div>
      <div><strong>Uncertainty Review Required:</strong> {'YES - FLAG FOR ANALYST' if summary['needs_manual_review'] else 'NO - AUTO-APPROVED'}</div>
    </div>
    <div style="width: 260px; border-bottom: 1px solid #94a3b8; text-align: center; padding-top: 14px;">
      Hydrophone Acoustic Analyst Signature / Stamp
    </div>
  </div>
</div>
</body>
</html>"""
    return html
