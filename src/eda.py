"""
Exploratory Data Analysis.

Generates:
  results/fig_eda_waveforms.png       -- example waveforms per class/domain
  results/fig_eda_spectrograms.png    -- example spectrograms per class/domain
  results/fig_eda_class_balance.png   -- class balance bar chart
  results/fig_eda_feature_dists.png   -- handcrafted feature distributions by class
  results/eda_summary.csv             -- summary statistics table

Run:
    python -m src.eda
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sps

from . import config
from .data_pipeline import load_domain
from .features import HANDCRAFTED_NAMES


def plot_waveforms():
    src = load_domain("source")
    tgt = load_domain("target")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    t = np.arange(config.N_SAMPLES) / config.SAMPLE_RATE
    panels = [
        (src, 0, "Source domain -- No Ship (ambient)"),
        (src, 1, "Source domain -- Ship present"),
        (tgt, 0, "Target domain -- No Ship (ambient, harder SNR)"),
        (tgt, 1, "Target domain -- Ship present (harder SNR)"),
    ]
    for ax, (dom, label, title) in zip(axes.flat, panels):
        idx = np.where(dom["y"] == label)[0][0]
        ax.plot(t, dom["waveforms"][idx], linewidth=0.5, color="#065A82")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-1.05, 1.05)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Amplitude")
    fig.suptitle("Example Waveforms by Domain and Class", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "fig_eda_waveforms.png"), dpi=160)
    plt.close(fig)


def plot_spectrograms():
    src = load_domain("source")
    tgt = load_domain("target")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    panels = [
        (src, 0, "Source -- No Ship"), (src, 1, "Source -- Ship present"),
        (tgt, 0, "Target -- No Ship"), (tgt, 1, "Target -- Ship present"),
    ]
    for ax, (dom, label, title) in zip(axes.flat, panels):
        idx = np.where(dom["y"] == label)[0][0]
        f, t, Sxx = sps.spectrogram(dom["waveforms"][idx], fs=config.SAMPLE_RATE, nperseg=128, noverlap=96)
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto", cmap="magma")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        fig.colorbar(im, ax=ax, label="dB")
    fig.suptitle("Example Spectrograms by Domain and Class", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "fig_eda_spectrograms.png"), dpi=160)
    plt.close(fig)


def plot_class_balance():
    src = load_domain("source"); tgt = load_domain("target"); test = load_domain("test")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    domains = ["Source\n(1600 clips)", "Target\n(200 clips)", "Test\n(500 clips)"]
    no_ship = [np.sum(src["y"] == 0), np.sum(tgt["y"] == 0), np.sum(test["y"] == 0)]
    ship = [np.sum(src["y"] == 1), np.sum(tgt["y"] == 1), np.sum(test["y"] == 1)]
    x = np.arange(3); width = 0.35
    ax.bar(x - width / 2, no_ship, width, label="No Ship", color="#94A3B8")
    ax.bar(x + width / 2, ship, width, label="Ship Present", color="#065A82")
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylabel("Number of clips")
    ax.set_title("Class Balance Across Domains (perfectly balanced by design)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "fig_eda_class_balance.png"), dpi=160)
    plt.close(fig)


def plot_feature_distributions():
    tgt = load_domain("target")
    hc, y = tgt["handcrafted"], tgt["y"]
    feats_to_plot = ["rms", "spectral_centroid", "spectral_flatness", "zcr", "crest_factor", "spectral_rolloff"]
    idxs = [HANDCRAFTED_NAMES.index(f) for f in feats_to_plot]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, name, i in zip(axes.flat, feats_to_plot, idxs):
        ax.hist(hc[y == 0, i], bins=20, alpha=0.6, label="No Ship", color="#94A3B8")
        ax.hist(hc[y == 1, i], bins=20, alpha=0.6, label="Ship", color="#065A82")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Handcrafted Feature Distributions by Class (Target Domain, n=200)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "fig_eda_feature_dists.png"), dpi=160)
    plt.close(fig)


def summary_table():
    rows = []
    for name in ["source", "target", "test"]:
        d = load_domain(name)
        rows.append(dict(
            domain=name, n_clips=len(d["y"]),
            n_no_ship=int(np.sum(d["y"] == 0)), n_ship=int(np.sum(d["y"] == 1)),
            duration_sec=config.DURATION, sample_rate_hz=config.SAMPLE_RATE,
            handcrafted_dim=d["handcrafted"].shape[1],
            spectrogram_dim=d["spectrogram"].shape[1],
            missing_values=0,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.RESULTS_DIR, "eda_summary.csv"), index=False)
    print(df)
    return df


def main():
    print("Generating EDA figures...")
    plot_waveforms()
    plot_spectrograms()
    plot_class_balance()
    plot_feature_distributions()
    summary_table()
    print(f"Saved EDA outputs to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
