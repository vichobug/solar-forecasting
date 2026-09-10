"""Ensemble the tuned GBM (aggregated summary-stat features) with the CNN
(raw sequences) to see whether combining the two beats either alone.

Rationale: GBM reads engineered mean/std/min/max/last features per attribute;
CNN reads raw (60, 24) temporal shape directly. They make different kinds of
errors, so averaging their probabilities is a cheap way to test whether that
diversity helps -- no retraining tricks, just combine two already-working
models.

Both models are evaluated on the same held-out test partitions (1-4; test
partition 0 is reserved as the CNN's validation split, matching
train_deep.py's leakage-free validation scheme) for a fair comparison.
"""
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from data import load_partitions
from features import aggregate_timeseries
from metrics import heidke_skill_score, true_skill_statistic
from train_deep import (
    N_PARTITIONS,
    check_memory,
    evaluate,
    iter_partitions_float32,
    load_train_full,
    load_val_partition,
    make_loader,
    set_seed,
    train_one_fold,
)

GBM_N_ITER = 40
GBM_RANDOM_STATE = 42
GBM_PARAM_DISTRIBUTIONS = {
    "learning_rate": loguniform(1e-2, 3e-1),
    "max_leaf_nodes": randint(15, 128),
    "max_depth": randint(3, 12),
    "min_samples_leaf": randint(10, 200),
    "l2_regularization": loguniform(1e-4, 1e1),
    "max_iter": randint(100, 500),
    "class_weight": [None, "balanced"],
}


def train_gbm():
    print("Loading and featurizing train partitions (with CV groups) for GBM...")
    X_parts, y_parts = load_partitions(split="train", num_partitions=N_PARTITIONS)
    X_feats, ys, groups = [], [], []
    for i, (X_raw, y) in enumerate(zip(X_parts, y_parts)):
        X_feat, _ = aggregate_timeseries(X_raw)
        X_feats.append(X_feat)
        ys.append(y)
        groups.append(np.full(len(y), i))
    del X_parts, y_parts
    X_train = np.concatenate(X_feats, axis=0)
    y_train = np.concatenate(ys, axis=0)
    groups = np.concatenate(groups, axis=0)
    del X_feats, ys
    gc.collect()
    print(f"GBM train features: {X_train.shape}")

    tss_scorer = make_scorer(true_skill_statistic)
    cv = GroupKFold(n_splits=N_PARTITIONS)
    search = RandomizedSearchCV(
        estimator=HistGradientBoostingClassifier(random_state=GBM_RANDOM_STATE),
        param_distributions=GBM_PARAM_DISTRIBUTIONS,
        n_iter=GBM_N_ITER,
        scoring=tss_scorer,
        cv=cv,
        n_jobs=3,
        random_state=GBM_RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)
    print(f"Best GBM CV TSS: {search.best_score_:.4f}")
    print(f"Best GBM params: {search.best_params_}")
    del X_train, y_train, groups
    gc.collect()
    return search.best_estimator_


@torch.no_grad()
def cnn_proba(model, X, y, device):
    model.eval()
    loader = make_loader(X, y, batch_size=128, shuffle=False)
    probs = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def best_threshold_by_tss(y_true, y_proba):
    thresholds = np.linspace(0.05, 0.95, 19)
    scores = [true_skill_statistic(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    check_memory("start")

    print("\n=== Training GBM ===")
    gbm_model = train_gbm()
    check_memory("after GBM training")

    print("\n=== Training CNN ===")
    X_tr, y_tr = load_train_full()
    n_attrs = X_tr.shape[-1]
    print(f"CNN train shape: {X_tr.shape}")
    check_memory("after CNN train load")

    X_val, y_val = load_val_partition()
    print(f"Val: {X_val.shape} (natural distribution, test partition 0)")

    cnn_model, best_val_tss = train_one_fold("cnn", X_tr, y_tr, X_val, y_val, n_attrs, device)
    print(f"Best CNN val TSS: {best_val_tss:.4f}")
    del X_tr, y_tr
    gc.collect()
    check_memory("before threshold selection")

    # Pick the ensemble threshold on the same validation partition used for
    # CNN early stopping, so nothing here has seen the final test partitions.
    X_val_feat, _ = aggregate_timeseries(X_val)
    gbm_val_proba = gbm_model.predict_proba(X_val_feat)[:, 1]
    cnn_val_proba = cnn_proba(cnn_model, X_val, y_val, device)
    ensemble_val_proba = (gbm_val_proba + cnn_val_proba) / 2
    threshold = best_threshold_by_tss(y_val, ensemble_val_proba)
    print(f"\nEnsemble TSS-optimal threshold (chosen on val partition): {threshold:.2f}")
    del X_val, X_val_feat, gbm_val_proba, cnn_val_proba, ensemble_val_proba
    gc.collect()

    print("\n=== Evaluating on test partitions 1-4 (streamed) ===")
    gbm_probs, cnn_probs, targets = [], [], []
    for X, y in iter_partitions_float32("test", N_PARTITIONS, skip_indices={0}):
        X_feat, _ = aggregate_timeseries(X)
        gbm_probs.append(gbm_model.predict_proba(X_feat)[:, 1])
        cnn_probs.append(cnn_proba(cnn_model, X, y, device))
        targets.append(y)
        del X, X_feat
        gc.collect()
    gbm_probs = np.concatenate(gbm_probs)
    cnn_probs = np.concatenate(cnn_probs)
    y_test = np.concatenate(targets)
    ensemble_probs = (gbm_probs + cnn_probs) / 2

    evaluate("GBM alone (threshold=0.5)", y_test, (gbm_probs >= 0.5).astype(int))
    evaluate("CNN alone (threshold=0.5)", y_test, (cnn_probs >= 0.5).astype(int))
    evaluate("Ensemble (threshold=0.5)", y_test, (ensemble_probs >= 0.5).astype(int))
    evaluate(
        f"Ensemble (threshold={threshold:.2f})",
        y_test,
        (ensemble_probs >= threshold).astype(int),
    )


if __name__ == "__main__":
    main()
