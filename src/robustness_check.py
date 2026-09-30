"""
Repeats the full train/evaluate pipeline across several random seeds for
the TARGET and TEST domain draws (the small, 200-sample regime is the
noisiest part of the pipeline) to check that the headline result --
"Deep + DLR Ensemble beats every other configuration" -- is not a fluke
of one lucky data split.

The SOURCE domain (large, 1600 clips) is regenerated once per seed too,
so each run is a fully independent trial, not just a resample.

Run:
    python -m src.robustness_check
"""
import json
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from . import config
from .signal_generator import generate_dataset
from .features import extract_all_features
from .models import (
    LogisticRegressionBaseline, make_shallow_bp, make_deep_net,
    TransferModel, DLREnsemble,
)

SEEDS = [1, 2, 3, 4, 5]


def run_one_seed(seed):
    src_wave, src_y = generate_dataset(config.N_SOURCE_PER_CLASS, "source", seed=seed * 100)
    tgt_wave, tgt_y = generate_dataset(config.N_TARGET_PER_CLASS, "target", seed=seed * 100 + 1)
    test_wave, test_y = generate_dataset(config.N_TEST_PER_CLASS, "target", seed=seed * 100 + 2)

    src_hc, _, src_comb = extract_all_features(src_wave)
    tgt_hc, _, tgt_comb = extract_all_features(tgt_wave)
    test_hc, _, test_comb = extract_all_features(test_wave)

    out = {}

    lr = LogisticRegressionBaseline(seed=seed).fit(tgt_hc, tgt_y)
    out["LogReg_baseline"] = lr.predict_proba(test_hc)[:, 1]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(tgt_hc)
    bp200 = make_shallow_bp(seed=seed, max_iter=400)
    bp200.fit(scaler.transform(tgt_hc), tgt_y)
    out["Shallow_NoTransfer"] = bp200.predict_proba(scaler.transform(test_hc))[:, 1]

    st = TransferModel(make_shallow_bp, seed=seed, pretrain_iter=250, finetune_iter=450)
    st.pretrain(src_hc, src_y).finetune(tgt_hc, tgt_y)
    out["Shallow_Transfer"] = st.predict_proba(test_hc)[:, 1]

    sd = DLREnsemble(make_shallow_bp, seed=seed, pretrain_iter=250, finetune_iter=450)
    sd.pretrain(src_hc, src_y).finetune(tgt_hc, tgt_y)
    out["Shallow_DLR"] = sd.predict_proba(test_hc)[:, 1]

    dt = TransferModel(make_deep_net, seed=seed, pretrain_iter=250, finetune_iter=450)
    dt.pretrain(src_comb, src_y).finetune(tgt_comb, tgt_y)
    out["Deep_Transfer"] = dt.predict_proba(test_comb)[:, 1]

    dd = DLREnsemble(make_deep_net, seed=seed, pretrain_iter=250, finetune_iter=450)
    dd.pretrain(src_comb, src_y).finetune(tgt_comb, tgt_y)
    out["Deep_DLR"] = dd.predict_proba(test_comb)[:, 1]

    metrics = {}
    for k, proba in out.items():
        pred = (proba >= 0.5).astype(int)
        metrics[k] = dict(acc=accuracy_score(test_y, pred), f1=f1_score(test_y, pred))
    return metrics


def main():
    all_runs = {}
    for seed in SEEDS:
        print(f"--- seed {seed} ---")
        m = run_one_seed(seed)
        for k, v in m.items():
            print(f"  {k:22s} acc={v['acc']:.4f}  f1={v['f1']:.4f}")
        all_runs[seed] = m

    model_names = list(next(iter(all_runs.values())).keys())
    summary = {}
    for name in model_names:
        accs = [all_runs[s][name]["acc"] for s in SEEDS]
        f1s = [all_runs[s][name]["f1"] for s in SEEDS]
        summary[name] = dict(
            acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
            f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
        )

    print("\n=== SUMMARY (mean ± std over {} seeds) ===".format(len(SEEDS)))
    for name, s in summary.items():
        print(f"{name:22s} acc={s['acc_mean']:.4f} ± {s['acc_std']:.4f}   f1={s['f1_mean']:.4f} ± {s['f1_std']:.4f}")

    with open(f"{config.RESULTS_DIR}/robustness_summary.json", "w") as f:
        json.dump({"per_seed": all_runs, "summary": summary}, f, indent=2)
    print(f"\nSaved to {config.RESULTS_DIR}/robustness_summary.json")


if __name__ == "__main__":
    main()
