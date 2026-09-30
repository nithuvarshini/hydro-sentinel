"""
Produces every figure and table used in the final report:
  - results/metrics_table.csv           (accuracy/precision/recall/F1/AUC per model)
  - results/fig_accuracy_comparison.png
  - results/fig_confusion_matrices.png
  - results/fig_roc_curves.png
  - results/fig_robustness.png
  - results/error_analysis.csv          (misclassified test clips + likely cause)
  - results/fig_error_by_snr.png

Run after src.train and (optionally) src.robustness_check:
    python -m src.evaluate
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

from . import config
from .data_pipeline import load_domain

MODEL_ORDER = [
    "LogReg_baseline", "Shallow_NoTransfer", "Shallow_Transfer",
    "Shallow_DLR", "Deep_Transfer", "Deep_DLR",
]
MODEL_LABELS = {
    "LogReg_baseline": "LogReg\n(baseline)",
    "Shallow_NoTransfer": "Shallow BP\n(no transfer)",
    "Shallow_Transfer": "Shallow BP\n+ Transfer",
    "Shallow_DLR": "Shallow BP\n+ DLR Ensemble",
    "Deep_Transfer": "Deep Net\n+ Transfer",
    "Deep_DLR": "Deep Net\n+ DLR Ensemble\n(PROPOSED)",
}
COLORS = {
    "LogReg_baseline": "#94A3B8", "Shallow_NoTransfer": "#94A3B8",
    "Shallow_Transfer": "#065A82", "Shallow_DLR": "#1C7293",
    "Deep_Transfer": "#D98E3B", "Deep_DLR": "#B3261E",
}


def compute_metrics_table():
    d = np.load(os.path.join(config.RESULTS_DIR, "test_predictions.npz"))
    y = d["y_test"]
    rows = []
    for name in MODEL_ORDER:
        proba = d[name]
        pred = (proba >= 0.5).astype(int)
        rows.append(dict(
            model=name,
            accuracy=accuracy_score(y, pred),
            precision=precision_score(y, pred),
            recall=recall_score(y, pred),
            f1=f1_score(y, pred),
            roc_auc=roc_auc_score(y, proba),
        ))
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(os.path.join(config.RESULTS_DIR, "metrics_table.csv"))
    print(df.round(4))
    return df, d, y


def plot_accuracy_comparison(df):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    names = MODEL_ORDER
    accs = [df.loc[n, "accuracy"] * 100 for n in names]
    colors = [COLORS[n] for n in names]
    bars = ax.bar([MODEL_LABELS[n] for n in names], accs, color=colors, width=0.6)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.7, f"{a:.1f}%", ha="center",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(50, 100)
    ax.set_title("Model Comparison on Held-Out Test Set (Target Domain)")
    ax.axhline(accs[0], color="#94A3B8", linestyle="--", linewidth=1, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "fig_accuracy_comparison.png"), dpi=160)
    plt.close(fig)


def plot_confusion_matrices(d, y):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, name in zip(axes.flat, MODEL_ORDER):
        pred = (d[name] >= 0.5).astype(int)
        cm = confusion_matrix(y, pred)
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Ship", "Ship"]); ax.set_yticklabels(["No Ship", "Ship"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(MODEL_LABELS[name].replace("\n", " "), fontsize=10)
    fig.suptitle("Confusion Matrices (Held-Out Test Set)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "fig_confusion_matrices.png"), dpi=160)
    plt.close(fig)


def plot_roc_curves(d, y):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in MODEL_ORDER:
        fpr, tpr, _ = roc_curve(y, d[name])
        auc = roc_auc_score(y, d[name])
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUC={auc:.3f})",
                color=COLORS[name], linewidth=2.2 if name == "Deep_DLR" else 1.4)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- Held-Out Test Set")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "fig_roc_curves.png"), dpi=160)
    plt.close(fig)


def plot_robustness():
    path = os.path.join(config.RESULTS_DIR, "robustness_summary.json")
    if not os.path.exists(path):
        print("robustness_summary.json not found -- skipping robustness plot "
              "(run src.robustness_check first)")
        return
    with open(path) as f:
        data = json.load(f)
    summary = data["summary"]
    names = MODEL_ORDER
    means = [summary[n]["acc_mean"] * 100 for n in names]
    stds = [summary[n]["acc_std"] * 100 for n in names]
    colors = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar([MODEL_LABELS[n] for n in names], means, yerr=stds, capsize=5,
                  color=colors, width=0.6, error_kw=dict(linewidth=1.5, ecolor="#333"))
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + max(stds) + 1.2, f"{m:.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(60, 100)
    ax.set_title("Robustness Check: Mean \u00b1 Std Accuracy over 5 Independent Random Seeds")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "fig_robustness.png"), dpi=160)
    plt.close(fig)


def error_analysis(d, y):
    """For the best model (Deep_DLR) and the paper-replica (Shallow_DLR),
    dump every misclassified test clip along with the acoustic conditions
    that likely explain the failure (SNR proxy via handcrafted RMS energy,
    and whether the clip is a 'hard negative' with a spurious resonance)."""
    test = load_domain("test")
    hc = test["handcrafted"]

    rows = []
    for name in ["Shallow_DLR", "Deep_DLR"]:
        pred = (d[name] >= 0.5).astype(int)
        proba = d[name]
        wrong = np.where(pred != y)[0]
        for idx in wrong:
            rows.append(dict(
                model=name, index=int(idx), true_label=int(y[idx]), pred_label=int(pred[idx]),
                confidence=float(proba[idx] if pred[idx] == 1 else 1 - proba[idx]),
                rms_energy=float(hc[idx, 1]), spectral_flatness=float(hc[idx, 11]),
            ))
    err_df = pd.DataFrame(rows)
    err_df.to_csv(os.path.join(config.RESULTS_DIR, "error_analysis.csv"), index=False)

    # Bucket test set by RMS energy (a proxy for effective SNR) into
    # terciles and show accuracy per bucket for the headline model,
    # to see whether failures concentrate in the hardest (low-SNR) clips.
    rms = hc[:, 1]
    terciles = np.quantile(rms, [1 / 3, 2 / 3])
    bucket = np.digitize(rms, terciles)  # 0=low energy, 1=mid, 2=high
    bucket_labels = ["Low energy\n(hardest)", "Mid energy", "High energy\n(easiest)"]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x = np.arange(3)
    for offset, name in [(-width / 2, "Deep_Transfer"), (width / 2, "Deep_DLR")]:
        pred = (d[name] >= 0.5).astype(int)
        accs = [accuracy_score(y[bucket == b], pred[bucket == b]) * 100 for b in range(3)]
        ax.bar(x + offset, accs, width, label=name.replace("_", " "), color=COLORS[name])
    ax.set_xticks(x); ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(70, 95)
    ax.set_title("Error Analysis: Accuracy vs. Clip Energy (SNR proxy)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "fig_error_by_snr.png"), dpi=160)
    plt.close(fig)

    print(f"\nError analysis: {len(err_df)} total misclassifications logged to error_analysis.csv")
    for name in ["Shallow_DLR", "Deep_DLR"]:
        sub = err_df[err_df.model == name]
        n_fn = ((sub.true_label == 1) & (sub.pred_label == 0)).sum()
        n_fp = ((sub.true_label == 0) & (sub.pred_label == 1)).sum()
        print(f"  {name}: {len(sub)} errors  ({n_fn} false negatives, {n_fp} false positives)")


def main():
    df, d, y = compute_metrics_table()
    plot_accuracy_comparison(df)
    plot_confusion_matrices(d, y)
    plot_roc_curves(d, y)
    plot_robustness()
    error_analysis(d, y)
    print(f"\nAll figures/tables written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
