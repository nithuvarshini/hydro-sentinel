"""
Central configuration for the Underwater Ship Acoustic Detection project.

All randomness is seeded for reproducibility. Change SAMPLE_RATE / DURATION
here if you swap in real audio (e.g. ShipsEar / DeepShip .wav files) later.
"""
import os

SEED = 42

# --- Signal parameters -------------------------------------------------
SAMPLE_RATE = 4000          # Hz. Kept modest (vs. 17,067 Hz in the source
                             # paper) since ship engine/propeller harmonics
                             # of interest sit below 500 Hz; this keeps the
                             # project runnable on a single CPU core.
DURATION = 1.0               # seconds per clip (matches the source paper's
                             # 1-second segmentation convention)
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# --- Dataset sizes (mirrors the case-study paper's 2000 / 200 split) ---
N_SOURCE_PER_CLASS = 800     # source domain, large "pretraining" set
N_TARGET_PER_CLASS = 100     # target domain, small "fine-tuning" set (200 total)
N_TEST_PER_CLASS = 250       # held-out evaluation set, drawn from target distribution

# --- Spectrogram feature parameters ------------------------------------
N_FREQ_BINS = 32
N_TIME_BINS = 32

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

CLASS_NAMES = ["No Ship (Ambient)", "Ship Present"]
