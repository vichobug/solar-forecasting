import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from data import load_partitions
from features import aggregate_timeseries
from metrics import heidke_skill_score, true_skill_statistic

N_PARTITIONS = 5
N_ITER = 40
RANDOM_STATE = 42

PARAM_DISTRIBUTIONS = {
    "learning_rate": loguniform(1e-2, 3e-1),
    "max_leaf_nodes": randint(15, 128),
    "max_depth": randint(3, 12),
    "min_samples_leaf": randint(10, 200),
    "l2_regularization": loguniform(1e-4, 1e1),
    "max_iter": randint(100, 500),
    "class_weight": [None, "balanced"],
}


def load_train_with_groups():
    """Load train partitions, featurize each, and tag rows with their partition id.

    Partitions correspond to distinct time periods in SWAN-SF, so we use them as
    CV groups (leave-one-partition-out) instead of a random split to avoid mixing
    temporally adjacent samples across train/validation folds.
    """
    X_parts, y_parts = load_partitions(split="train", num_partitions=N_PARTITIONS)
    X_feats, ys, groups = [], [], []
    for i, (X_raw, y) in enumerate(zip(X_parts, y_parts)):
        X_feat, columns = aggregate_timeseries(X_raw)
        X_feats.append(X_feat)
        ys.append(y)
        groups.append(np.full(len(y), i))
    X = np.concatenate(X_feats, axis=0)
    y = np.concatenate(ys, axis=0)
    groups = np.concatenate(groups, axis=0)
    return X, y, groups, columns


def load_test():
    X_parts, y_parts = load_partitions(split="test", num_partitions=N_PARTITIONS)
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    X_feat, _ = aggregate_timeseries(X)
    return X_feat, y


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["No Flare", "Flare"]))
    print(f"TSS: {true_skill_statistic(y_true, y_pred):.4f}")
    print(f"HSS: {heidke_skill_score(y_true, y_pred):.4f}")


def best_threshold_by_tss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 19)
    scores = [true_skill_statistic(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def main():
    print("Loading and featurizing train partitions (with CV groups)...")
    X_train, y_train, groups, columns = load_train_with_groups()
    print(f"Train features: {X_train.shape}, groups: {np.unique(groups)}")

    tss_scorer = make_scorer(true_skill_statistic)
    cv = GroupKFold(n_splits=N_PARTITIONS)

    search = RandomizedSearchCV(
        estimator=HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring=tss_scorer,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)

    print(f"\nBest CV TSS: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    best_model = search.best_estimator_

    print("\nLoading and featurizing test set...")
    X_test, y_test = load_test()

    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    evaluate("Tuned HistGradientBoosting (threshold=0.5)", y_test, (y_proba_test >= 0.5).astype(int))

    threshold = best_threshold_by_tss(y_train, best_model.predict_proba(X_train)[:, 1])
    print(f"\nTSS-optimal threshold (chosen on train): {threshold:.2f}")
    evaluate(
        f"Tuned HistGradientBoosting (threshold={threshold:.2f})",
        y_test,
        (y_proba_test >= threshold).astype(int),
    )


if __name__ == "__main__":
    main()
