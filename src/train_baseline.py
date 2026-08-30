import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from data import load_partitions
from features import aggregate_timeseries
from metrics import heidke_skill_score, true_skill_statistic


def load_and_featurize(split: str):
    X_parts, y_parts = load_partitions(split=split, num_partitions=5)
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    X_feat, columns = aggregate_timeseries(X)
    return X_feat, y, columns


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["No Flare", "Flare"]))
    print(f"TSS: {true_skill_statistic(y_true, y_pred):.4f}")
    print(f"HSS: {heidke_skill_score(y_true, y_pred):.4f}")


def main():
    print("Loading and featurizing data...")
    X_train, y_train, columns = load_and_featurize("train")
    X_test, y_test, _ = load_and_featurize("test")
    print(f"Train features: {X_train.shape}  Test features: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    evaluate("Logistic Regression", y_test, logreg.predict(X_test_scaled))

    hgb = HistGradientBoostingClassifier(random_state=42)
    hgb.fit(X_train, y_train)
    evaluate("HistGradientBoosting", y_test, hgb.predict(X_test))


if __name__ == "__main__":
    main()
