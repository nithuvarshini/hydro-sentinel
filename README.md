# Underwater Acoustic Target Detection & Monitoring Application

**HYDRO-SENTINEL PRO** — An Advanced Machine Learning (AML) project for underwater acoustic target detection using Deep Multi-Layer Perceptron (MLP) models and Dual-Label-Reversed (DLR) ensemble transfer learning.

This project is based on the Dual-Label-Reversed transfer learning strategy described by Bao et al. (2025) and extends the approach with a deeper MLP architecture, additional acoustic features, robustness evaluation, and a web-based monitoring application.

---

## Overview

HYDRO-SENTINEL PRO is a web-based prototype for detecting underwater acoustic targets from WAV recordings.

The system accepts an audio recording, preprocesses the signal, divides it into 1-second analysis segments, extracts acoustic features, evaluates signal quality, and predicts whether each segment contains a target signal.

The project combines two parts:

* **AML research component** — comparison of six machine learning models, transfer learning, Dual-Label-Reversed (DLR) ensemble learning, and 5-seed robustness evaluation.
* **Application component** — a Flask-based web interface for audio analysis, detection timelines, signal visualizations, scan history, and report generation.

The experimental datasets used for model development are physics-informed synthetic underwater acoustic datasets generated specifically for this project.

---

## Research Approach

The project investigates whether a deeper MLP combined with transfer learning and DLR can improve underwater acoustic target detection under domain shift.

### Main models evaluated

1. Logistic Regression baseline
2. Shallow MLP without transfer learning
3. Shallow MLP with transfer learning
4. Shallow MLP with DLR ensemble
5. Deep MLP with transfer learning
6. Deep MLP with DLR ensemble

The deep architecture uses a **256 → 128 → 64** hidden-layer structure with ReLU activations.

The DLR approach trains two branches using the original and reversed target labels and combines their predictions during inference.

---

## Dataset

The project uses three physics-informed synthetic acoustic domains:

| Dataset              | Samples | Classes                | Purpose                       |
| -------------------- | ------: | ---------------------- | ----------------------------- |
| Source Domain        |   1,600 | 800 Ambient / 800 Ship | Source-domain training        |
| Target Domain        |     200 | 100 Ambient / 100 Ship | Transfer-learning fine-tuning |
| Held-out Test Domain |     500 | 250 Ambient / 250 Ship | Final evaluation              |

Each recording is **1 second long at 4,000 Hz**.

The source domain contains higher-SNR acoustic conditions, while the target domain represents a lower-SNR and shifted acoustic environment.

The held-out test domain is used for final model comparison and error analysis.

### Important note

The dataset is **synthetically generated rather than collected from real hydrophone deployments**. The signal generator is physics-informed and is used to create controlled source, target, and test domains. This makes controlled experimentation possible but is also a limitation when considering performance on real-world underwater recordings.

---

## Feature Extraction

The model uses **1,044 input features**:

### Handcrafted acoustic features — 20

The handcrafted feature set contains:

* 8 time-domain features
* 4 spectral features
* 2 delta features
* 6 band-energy ratio features

### Spectrogram features — 1,024

The short-time Fourier transform (STFT) representation is pooled into a **32 × 32** representation, producing 1,024 spectrogram features.

### Total

**20 handcrafted + 1,024 spectrogram = 1,044 features**

The baseline Logistic Regression and shallow models use the handcrafted features, while the deep model uses the combined feature representation.

---

## Preprocessing

The audio preprocessing pipeline includes:

* WAV validation
* Mono conversion for multi-channel audio
* Resampling to 4,000 Hz
* Second-order Butterworth high-pass filtering at 15 Hz
* Peak normalization
* Feature extraction
* Domain-specific feature scaling

Separate scalers are used for the source and target domains to reduce the effect of distribution differences between domains.

The application also calculates signal-quality indicators such as estimated in-band SNR, noise floor, spectral flatness, spectral tilt, and a composite quality score.

---

## Model Evaluation

The models were evaluated on the held-out target-domain test set.

### Single Test-Split Results

| Model               |  Accuracy |  Precision |    Recall |         F1 |    ROC-AUC |
| ------------------- | --------: | ---------: | --------: | ---------: | ---------: |
| Logistic Regression |     84.0% |     85.42% |     82.0% |     83.67% |     0.9251 |
| Shallow No Transfer |     83.8% |     83.40% |     84.4% |     83.90% |     0.9212 |
| Shallow Transfer    |     83.4% |     83.27% |     83.6% |     83.43% |     0.9222 |
| Shallow DLR         |     83.8% |     83.67% |     84.0% |     83.83% |     0.9197 |
| Deep Transfer       |     87.0% |     93.02% |     80.0% |     86.02% |     0.9467 |
| **Deep DLR**        | **87.8%** | **92.76%** | **82.0%** | **87.05%** | **0.9539** |

---

## 5-Seed Robustness Evaluation

To examine sensitivity to random initialization and training variation, the models were evaluated across five random seeds.

| Model               | Mean Accuracy | Accuracy Std. Dev. |    Mean F1 | F1 Std. Dev. |
| ------------------- | ------------: | -----------------: | ---------: | -----------: |
| Logistic Regression |        81.64% |             ±3.02% |     81.48% |       ±3.01% |
| Shallow No Transfer |        82.84% |             ±3.34% |     83.11% |       ±3.33% |
| Shallow Transfer    |        83.56% |             ±1.02% |     83.58% |       ±1.13% |
| Shallow DLR         |        83.64% |             ±2.47% |     83.70% |       ±2.66% |
| Deep Transfer       |        88.20% |             ±1.80% |     87.80% |       ±1.87% |
| **Deep DLR**        |    **88.64%** |         **±1.97%** | **88.34%** |   **±2.18%** |

The five-seed results show that the deep models achieve higher average performance than the shallow models. Deep DLR has the highest mean accuracy and F1 among the evaluated models, although its performance varies across random seeds.

---

## Error Analysis

On the held-out test set:

* Shallow DLR produced **81 errors**
* Deep DLR produced **61 errors**
* Shallow DLR produced **41 false positives**
* Deep DLR produced **16 false positives**

This corresponds to a **61% reduction in false positives** from Shallow DLR to Deep DLR on the evaluated test set.

The results also show that the deep model reduces false alarms, while false negatives remain an important source of error.

---

## Application Features

### 1. Dashboard

Provides a central interface for:

* System status
* Monitoring mode
* Quick analysis access
* Application telemetry

### 2. Audio Analysis

The application can process WAV recordings and perform:

* Audio validation
* Resampling and preprocessing
* Signal-quality analysis
* Acoustic feature extraction
* Target prediction
* Confidence estimation
* Waveform visualization
* Spectrogram visualization

### 3. Detection Timeline

Audio is divided into **1-second analysis windows**.

The application displays:

* Segment timestamps
* Target/non-target predictions
* Confidence values
* Detection intervals
* Timeline visualization

### 4. Signal Quality Analysis

The application calculates:

* Estimated in-band SNR
* Noise floor
* Spectral flatness
* Spectral tilt
* Composite signal-quality score

### 5. Model Insights

The application includes an academic evaluation section containing:

* Model comparison
* Accuracy and F1 results
* Confusion matrices
* ROC curves
* 5-seed robustness results
* Error-analysis information

### 6. History and Reports

Previous analyses can be stored in the application history.

The system can generate printable analysis reports containing the detection results and signal-analysis information.

---

## Project Structure

```text
AML_project_code/
│
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── src/
│   ├── config.py
│   ├── audio_processor.py
│   ├── features.py
│   ├── models.py
│   ├── report_generator.py
│   ├── sample_generator.py
│   ├── signal_generator.py
│   ├── data_pipeline.py
│   ├── train.py
│   ├── evaluate.py
│   ├── eda.py
│   └── robustness_check.py
│
├── data/
│   ├── source_domain.npz
│   ├── target_domain.npz
│   └── test_domain.npz
│
├── models/
│   └── *.joblib
│
├── results/
│   ├── EDA figures
│   ├── evaluation figures
│   ├── metrics tables
│   ├── robustness results
│   ├── error analysis
│   └── timing results
│
├── requirements.txt
├── README.md
└── .gitignore
```

> Sample WAV recordings used for local testing are not included in the GitHub repository. The application can accept user-provided WAV recordings as input.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nithuvarshini/hydro-sentinel.git
cd hydro-sentinel
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Flask application:

```bash
python -m app.app
```

Then open:

```text
http://127.0.0.1:5000
```

in a web browser.

Upload a WAV recording through the application to perform acoustic analysis and target detection.

---

## Reproduce the Experiments

The research pipeline can be executed using the following modules.

### Generate / prepare the datasets

```bash
python -m src.data_pipeline
```

### Train the models

```bash
python -m src.train
```

### Evaluate the models

```bash
python -m src.evaluate
```

### Generate EDA results

```bash
python -m src.eda
```

### Run 5-seed robustness evaluation

```bash
python -m src.robustness_check
```

The generated evaluation artifacts are stored in the `results/` directory.

---

## Computational Performance

The measured inference time for the evaluated models was in the millisecond range per 1-second analysis clip in the test environment.

For the Deep DLR model, the measured inference time was approximately **24.1 ms per 1-second clip**, corresponding to an RTF of approximately **0.024**.

These measurements are specific to the tested environment and should not be interpreted as benchmarks for all hardware platforms.

---

## Technologies Used

* Python
* NumPy
* SciPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Flask
* HTML / CSS / JavaScript

---

## Research Reference

The project builds on:

**W. Bao, Q. Ren, W. Wang, M. Huang, and Z. Xiao, "A dual-label-reversed ensemble transfer learning strategy for underwater target detection," Applied Acoustics, vol. 235, 110701, 2025.**

The project implements the DLR transfer-learning concept and extends the experimental setup with a deeper MLP architecture, combined acoustic features, robustness evaluation, and a web-based application.

---

## Limitations

The current project has several limitations:

* The training and evaluation datasets are physics-informed synthetic data rather than a large real-world hydrophone dataset.
* The application has been evaluated primarily on the project's generated/test data.
* The model currently performs binary target/non-target detection rather than identifying specific vessel classes.
* Performance on real underwater recordings may differ because of environmental conditions, sensor characteristics, propagation effects, and background noise.
* The computational measurements depend on the hardware and software environment used for testing.

---

## Future Work

Possible extensions include:

* Evaluation using larger real-world underwater acoustic datasets.
* Multi-class vessel identification.
* CNN or transformer-based acoustic representations.
* Domain adaptation across different hydrophone locations.
* Online/streaming inference for continuous recordings.
* Improved calibration of prediction confidence.
* Deployment on edge hardware for field testing.
* Integration with cloud-based monitoring and storage.

---

## Project

**HYDRO-SENTINEL PRO**

Advanced Machine Learning Project
Underwater Acoustic Target Detection & Monitoring
