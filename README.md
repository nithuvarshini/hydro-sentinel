# Underwater Acoustic Target Detection & Monitoring Application

**HYDRO-SENTINEL PRO** &middot; Advanced Machine Learning (AML) Project extending Bao et al. (2025), *A dual-label-reversed ensemble transfer learning strategy for underwater target detection* (Applied Acoustics 235, 110701).

---

## Overview

This project transforms research into a complete, production-grade **Underwater Acoustic Target Detection & Monitoring Application**. Instead of a laboratory playground requiring manual parameter inputs (such as manually entering ground truth labels or SNR sliders), the system operates as a genuine passive hydrophone surveillance station that accepts real underwater audio recordings (`.wav`), automatically evaluates noise conditions, detects target vessels across segmented time intervals, quantifies epistemic and aleatoric uncertainty, and produces standardized mission reports.

At the same time, the project preserves the rigorous academic transfer learning and ensemble research required for the AML report, including model comparisons, 5-seed robustness validation, and ablation studies.

---

## Headline Research Results

Replication of Bao et al. (2025) and extension to deep architectures evaluated across 5 independent random seeds on held-out target domain data:

| Model Architecture | Mean Accuracy | Std Dev | Mean F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression (Baseline) | 81.64% | &plusmn;3.02% | 81.48% | 0.9251 |
| Shallow BP (No Transfer, Paper's BP_200) | 82.84% | &plusmn;3.34% | 83.11% | 0.9212 |
| Shallow BP + Transfer (Paper's Transfer_Original) | 83.56% | &plusmn;1.02% | 83.58% | 0.9222 |
| Shallow BP + DLR Ensemble (Paper's Proposed Method) | 83.64% | &plusmn;2.47% | 83.70% | 0.9197 |
| Deep Net + Transfer (Depth-Only Ablation) | 88.40% | &plusmn;1.54% | 87.80% | 0.9467 |
| **Deep Net + DLR Ensemble (Our Proposed Fix)** | **88.96%** | **&plusmn;1.75%** | **88.34%** | **0.9539** |

---

## Literature Limitations Addressed

| Literature Limitation | Operational Solution Implemented |
|---|---|
| **Domain / Environment Shift** | Dual-Label-Reversed (DLR) transfer learning with domain-specific scalers and automated acoustic regime profiling. |
| **Noisy / Low-Quality Signals** | Automated in-band SNR estimation, spectral flatness (Wiener entropy), and dynamic noise floor tracking (no user guessing). |
| **Limited Target Information** | Dual-branch DLR ensemble disagreement ($\Delta_{\text{ensemble}}$) and entropy gating to flag ambiguous signals instead of forcing brittle predictions. |
| **Practical Deployment** | Multi-segment temporal decomposition, sub-second inference latency, and Real-Time Factor tracking ($\text{RTF} < 0.25$). |
| **Poor Real-World Usability** | Modern tactical monitoring dashboard with audio playback, detection timeline, scan history, and printable mission reports. |

---

## Application Workflow & UI Sections

1. **Dashboard:** Station status, active monitoring mode, key system telemetry, and quick-start testing shortcuts.
2. **Analyze Audio:**
   - Upload any real hydrophone recording (`.wav`) or select from 5 curated operational benchmark scenarios (Merchant Vessel, Deep Ocean Ambient, Low-SNR Harbor Tug, Coral Reef Biologics, Coastal Patrol Passage).
   - Audio playback directly inside the browser.
   - Automated signal quality profiling (in-band SNR, noise floor, spectral flatness, composite quality index).
   - Detection verdict with confidence score, uncertainty percentage, and operational action.
   - High-resolution acoustic waveform (with highlighted target intervals) and time-frequency spectrogram.
3. **Detection Timeline:**
   - Multi-segment temporal decomposition dividing audio into 1.0-second analysis windows.
   - Visual timeline ribbon and detailed interval table showing exact timestamps when targets enter, transit, and exit.
4. **History & Reports:**
   - Searchable scan register storing previous hydrophone analyses.
   - One-click generation of self-contained, printable HTML/PDF mission detection reports with operator sign-off sections.
5. **Model Insights (Academic Evaluation):**
   - Separate evaluation tab presenting theoretical methodology, 6-model benchmark comparison table, 5-seed robustness statistics, confusion matrices, and ROC curves.

---

## Project Structure

```
AML_project_code/
├── app/
│   ├── app.py                      # Production Flask web server & REST API
│   └── templates/
│       └── index.html              # Tactical underwater monitoring UI
├── src/
│   ├── config.py                   # Central parameters (sample rate 4000 Hz, paths)
│   ├── audio_processor.py          # WAV validation, resampling, SNR assessment, pipeline
│   ├── features.py                 # Handcrafted (27-dim) + Spectrogram (1024-dim) features
│   ├── models.py                   # LogReg, Shallow BP, DeepNet, TransferModel, DLREnsemble
│   ├── report_generator.py         # Printable HTML reports & scan history persistence
│   ├── sample_generator.py         # Curated benchmark hydrophone scenarios (.wav)
│   ├── signal_generator.py         # Physics-informed synthetic acoustic generator
│   ├── data_pipeline.py            # Source, target, and test dataset splits
│   ├── train.py                    # Trains all 6 models and saves .joblib checkpoints
│   ├── evaluate.py                 # Generates metrics tables, ROC curves, confusion matrices
│   ├── eda.py                      # Exploratory data analysis scripts & figures
│   └── robustness_check.py         # 5-seed statistical validation
├── data/
│   ├── sample_recordings/          # Pre-packaged benchmark hydrophone WAV files
│   ├── source_domain.npz
│   ├── target_domain.npz
│   └── test_domain.npz
├── models/                         # Saved .joblib model weights
├── results/                        # Figures, tables, metrics, and scan history
└── requirements.txt
```

---

## Running the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
python -m app.app
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

### 3. Reproducing Academic Training & Evaluation (Optional)
```bash
python -m src.data_pipeline        # Build domain splits (~10s)
python -m src.train                 # Train all 6 models (~15s)
python -m src.evaluate                # Produce metrics tables & figures
python -m src.eda                      # Produce EDA figures
python -m src.robustness_check           # 5-seed validation (~70s)
```

---

## Dependencies

- Python 3.10+
- numpy, scipy, pandas, scikit-learn, matplotlib, seaborn, joblib, Flask.
