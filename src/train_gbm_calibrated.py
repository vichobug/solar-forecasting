import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from data import load_partitions
from features import aggregate_timeseries
from metrics import heidke_skill_score, true_skill_statistic

RANDOM_STATE = 42

# Best hyperparameters found via leave-one-partition-out search (train_gbm_tuned.py).
BEST_PARAMS = dict(
    class_weight="balanced",
    l2_regularization=0.0023988103285642534,
    learning_rate=0.027392514012725373,
    max_depth=3,
    max_iter=444,
    max_leaf_nodes=85,
    min_samples_leaf=18,
    random_state=RANDOM_STATE,
)


def load_and_featurize(split: str):
    X_parts, y_parts = load_partitions(split=split, num_partitions=5)
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    X_feat, columns = aggregate_timeseries(X)
    return X_feat, y, columns


def correct_prior(p: np.ndarray, pi_train: float, pi_target: float) -> np.ndarray:
    """Rescale P(flare|x) estimated under the train class prior to the target prior.

    Standard label-shift / prior-correction formula (Saerens et al., 2002):
    since the model was trained on resampled ~50/50 data, its raw probabilities
    systematically overstate P(flare) relative to the true ~2% deployment rate.
    """
    ratio_pos = pi_target / pi_train
    ratio_neg = (1 - pi_target) / (1 - pi_train)
    numerator = p * ratio_pos
    denominator = numerator + (1 - p) * ratio_neg
    return numerator / denominator


def best_threshold_by_hss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [heidke_skill_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["No Flare", "Flare"]))
    print(f"TSS: {true_skill_statistic(y_true, y_pred):.4f}")
    print(f"HSS: {heidke_skill_score(y_true, y_pred):.4f}")


def main():
    print("Loading and featurizing data...")
    X_train, y_train, _ = load_and_featurize("train")
    X_test, y_test, _ = load_and_featurize("test")

    pi_train = y_train.mean()
    print(f"Train flare prior: {pi_train:.4f}")

    # Carve a dev slice out of test to estimate the deployment prior and pick a
    # threshold, leaving the rest untouched for final evaluation.
    X_dev, X_final, y_dev, y_final = train_test_split(
        X_test, y_test, test_size=0.7, stratify=y_test, random_state=RANDOM_STATE
    )
    pi_dev = y_dev.mean()
    print(f"Dev (held-out slice of test) flare prior: {pi_dev:.4f}  (n={len(y_dev)})")
    print(f"Final held-out test size: {len(y_final)}")

    model = HistGradientBoostingClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train)

    p_final_raw = model.predict_proba(X_final)[:, 1]
    evaluate("Uncorrected, threshold=0.5 (baseline)", y_final, (p_final_raw >= 0.5).astype(int))

    p_dev_raw = model.predict_proba(X_dev)[:, 1]
    p_dev_corrected = correct_prior(p_dev_raw, pi_train, pi_dev)
    threshold = best_threshold_by_hss(y_dev, p_dev_corrected)
    print(f"\nPrior-corrected, HSS-optimal threshold (chosen on dev): {threshold:.2f}")

    p_final_corrected = correct_prior(p_final_raw, pi_train, pi_dev)
    evaluate(
        f"Prior-corrected, threshold={threshold:.2f}",
        y_final,
        (p_final_corrected >= threshold).astype(int),
    )


if __name__ == "__main__":
    main()
