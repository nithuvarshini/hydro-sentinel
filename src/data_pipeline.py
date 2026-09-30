"""
Builds the three domain splits used throughout the project:

  * SOURCE  -- large "pretraining" domain (2 x N_SOURCE_PER_CLASS clips)
  * TARGET  -- small "fine-tuning" domain, the low-label regime we care
               about (2 x N_TARGET_PER_CLASS clips, default 200 total)
  * TEST    -- held-out evaluation set drawn from the TARGET distribution
               (2 x N_TEST_PER_CLASS clips), never seen during training

Run directly to (re)generate everything:
    python -m src.data_pipeline
"""
import os
import numpy as np

from . import config
from .signal_generator import generate_dataset
from .features import extract_all_features


def build_and_save():
    print("Generating SOURCE domain waveforms...")
    src_wave, src_y = generate_dataset(config.N_SOURCE_PER_CLASS, "source", seed=config.SEED)

    print("Generating TARGET domain waveforms (small, low-label regime)...")
    tgt_wave, tgt_y = generate_dataset(config.N_TARGET_PER_CLASS, "target", seed=config.SEED + 1)

    print("Generating TEST domain waveforms (held-out)...")
    test_wave, test_y = generate_dataset(config.N_TEST_PER_CLASS, "target", seed=config.SEED + 2)

    print("Extracting features (handcrafted + spectrogram)...")
    src_hc, src_sp, src_comb = extract_all_features(src_wave)
    tgt_hc, tgt_sp, tgt_comb = extract_all_features(tgt_wave)
    test_hc, test_sp, test_comb = extract_all_features(test_wave)

    np.savez_compressed(
        os.path.join(config.DATA_DIR, "source_domain.npz"),
        waveforms=src_wave, y=src_y, handcrafted=src_hc, spectrogram=src_sp, combined=src_comb,
    )
    np.savez_compressed(
        os.path.join(config.DATA_DIR, "target_domain.npz"),
        waveforms=tgt_wave, y=tgt_y, handcrafted=tgt_hc, spectrogram=tgt_sp, combined=tgt_comb,
    )
    np.savez_compressed(
        os.path.join(config.DATA_DIR, "test_domain.npz"),
        waveforms=test_wave, y=test_y, handcrafted=test_hc, spectrogram=test_sp, combined=test_comb,
    )

    print(f"Saved to {config.DATA_DIR}")
    print(f"  SOURCE : {len(src_y)} clips  (waveform shape {src_wave.shape})")
    print(f"  TARGET : {len(tgt_y)} clips")
    print(f"  TEST   : {len(test_y)} clips")
    print(f"  handcrafted feature dim = {src_hc.shape[1]}, spectrogram feature dim = {src_sp.shape[1]}")


def load_domain(name):
    path = os.path.join(config.DATA_DIR, f"{name}_domain.npz")
    d = np.load(path)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    build_and_save()
