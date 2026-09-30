"""
Trains all six models in the comparison and saves predictions + fitted
models to disk for evaluate.py to consume.

Run:
    python -m src.train
"""
import os
import json
import time
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from . import config
from .data_pipeline import load_domain
from .models import (
    LogisticRegressionBaseline, make_shallow_bp, make_deep_net,
    TransferModel, DLREnsemble,
)


def main():
    print("Loading data...")
    src = load_domain("source")
    tgt = load_domain("target")
    test = load_domain("test")

    X_src_hc, y_src = src["handcrafted"], src["y"]
    X_tgt_hc, y_tgt = tgt["handcrafted"], tgt["y"]
    X_test_hc, y_test = test["handcrafted"], test["y"]

    X_src_comb = src["combined"]
    X_tgt_comb = tgt["combined"]
    X_test_comb = test["combined"]

    results = {}
    predictions = {}
    timings = {}

    # ------------------------------------------------------------------
    # 1) Simple baseline: Logistic Regression, target-only, no transfer
    # ------------------------------------------------------------------
    print("\n[1/6] Logistic Regression baseline (target-only)...")
    t0 = time.time()
    lr = LogisticRegressionBaseline().fit(X_tgt_hc, y_tgt)
    timings["LogReg_baseline"] = time.time() - t0
    predictions["LogReg_baseline"] = lr.predict_proba(X_test_hc)[:, 1]
    joblib.dump(lr, os.path.join(config.MODELS_DIR, "logreg_baseline.joblib"))

    # ------------------------------------------------------------------
    # 2) Shallow BP, target-only, no transfer (paper's "BP_200")
    # ------------------------------------------------------------------
    print("[2/6] Shallow BP net, target-only (paper's BP_200 replica)...")
    t0 = time.time()
    shallow_notransfer = TransferModel(make_shallow_bp, pretrain_iter=1, finetune_iter=300)
    # "pretrain" on a single dummy pass isn't meaningful here; train from
    # scratch directly on target instead, matching the paper's BP_200 setup
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_tgt_hc)
    bp200 = make_shallow_bp(max_iter=400)
    bp200.fit(scaler.transform(X_tgt_hc), y_tgt)
    timings["Shallow_NoTransfer"] = time.time() - t0
    predictions["Shallow_NoTransfer"] = bp200.predict_proba(scaler.transform(X_test_hc))[:, 1]
    joblib.dump({"model": bp200, "scaler": scaler},
                os.path.join(config.MODELS_DIR, "shallow_notransfer.joblib"))

    # ------------------------------------------------------------------
    # 3) Shallow BP + Transfer (paper's "Transfer_Original")
    # ------------------------------------------------------------------
    print("[3/6] Shallow BP + Transfer (paper's Transfer_Original replica)...")
    t0 = time.time()
    shallow_transfer = TransferModel(make_shallow_bp, pretrain_iter=250, finetune_iter=450)
    shallow_transfer.pretrain(X_src_hc, y_src)
    shallow_transfer.finetune(X_tgt_hc, y_tgt)
    timings["Shallow_Transfer"] = time.time() - t0
    predictions["Shallow_Transfer"] = shallow_transfer.predict_proba(X_test_hc)[:, 1]
    joblib.dump(shallow_transfer, os.path.join(config.MODELS_DIR, "shallow_transfer.joblib"))

    # ------------------------------------------------------------------
    # 4) Shallow BP + DLR Ensemble (paper's proposed method, replicated)
    # ------------------------------------------------------------------
    print("[4/6] Shallow BP + DLR Ensemble (paper's proposed method)...")
    t0 = time.time()
    shallow_dlr = DLREnsemble(make_shallow_bp, pretrain_iter=250, finetune_iter=450)
    shallow_dlr.pretrain(X_src_hc, y_src)
    shallow_dlr.finetune(X_tgt_hc, y_tgt)
    timings["Shallow_DLR"] = time.time() - t0
    predictions["Shallow_DLR"] = shallow_dlr.predict_proba(X_test_hc)[:, 1]
    joblib.dump(shallow_dlr, os.path.join(config.MODELS_DIR, "shallow_dlr.joblib"))

    # ------------------------------------------------------------------
    # 5) Deep Net + Transfer (ablation: does depth alone help?)
    # ------------------------------------------------------------------
    print("[5/6] Deep Net + Transfer (depth-only ablation)...")
    t0 = time.time()
    deep_transfer = TransferModel(make_deep_net, pretrain_iter=250, finetune_iter=450)
    deep_transfer.pretrain(X_src_comb, y_src)
    deep_transfer.finetune(X_tgt_comb, y_tgt)
    timings["Deep_Transfer"] = time.time() - t0
    predictions["Deep_Transfer"] = deep_transfer.predict_proba(X_test_comb)[:, 1]
    joblib.dump(deep_transfer, os.path.join(config.MODELS_DIR, "deep_transfer.joblib"))

    # ------------------------------------------------------------------
    # 6) Deep Net + DLR Ensemble -- OUR PROPOSED FIX
    # ------------------------------------------------------------------
    print("[6/6] Deep Net + DLR Ensemble (OUR proposed fix)...")
    t0 = time.time()
    deep_dlr = DLREnsemble(make_deep_net, pretrain_iter=250, finetune_iter=450)
    deep_dlr.pretrain(X_src_comb, y_src)
    deep_dlr.finetune(X_tgt_comb, y_tgt)
    timings["Deep_DLR"] = time.time() - t0
    predictions["Deep_DLR"] = deep_dlr.predict_proba(X_test_comb)[:, 1]
    joblib.dump(deep_dlr, os.path.join(config.MODELS_DIR, "deep_dlr.joblib"))

    # ------------------------------------------------------------------
    # Save everything evaluate.py needs
    # ------------------------------------------------------------------
    np.savez_compressed(
        os.path.join(config.RESULTS_DIR, "test_predictions.npz"),
        y_test=y_test,
        **{k: v for k, v in predictions.items()},
    )
    with open(os.path.join(config.RESULTS_DIR, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)

    print("\nAll models trained. Predictions saved to results/test_predictions.npz")
    print("Training times (s):", json.dumps({k: round(v, 2) for k, v in timings.items()}, indent=2))


if __name__ == "__main__":
    main()
