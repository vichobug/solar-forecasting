"""Train a 1D-CNN or LSTM directly on raw SWAN-SF time series (Phase 2).

Unlike train_baseline.py / train_gbm_*.py, this operates on the full
(n_samples, timesteps, attrs) sequence instead of aggregated summary stats,
so the model can learn temporal dynamics itself.
"""
import argparse
import gc
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import DATA_DIR
from metrics import heidke_skill_score, true_skill_statistic
from models_dl import CNN1D, LSTMClassifier

RANDOM_STATE = 42
N_PARTITIONS = 5
BATCH_SIZE = 128
MAX_EPOCHS = 60
PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Below this much *available* RAM, refuse to start the next heavy step rather
# than risk exhausting memory and crashing the whole machine (this has
# actually rebooted an 8GB M1 before -- do not remove this guard).
MIN_AVAILABLE_GB = 1.5


def set_seed(seed: int = RANDOM_STATE):
    torch.manual_seed(seed)
    np.random.seed(seed)


def check_memory(step: str):
    vm = psutil.virtual_memory()
    available_gb = vm.available / 1e9
    print(f"  [mem check @ {step}] available={available_gb:.2f} GB ({100 - vm.percent:.0f}% free)")
    if available_gb < MIN_AVAILABLE_GB:
        raise MemoryError(
            f"Only {available_gb:.2f} GB available before '{step}' "
            f"(need >= {MIN_AVAILABLE_GB} GB). Aborting instead of risking a crash -- "
            "close other apps and retry."
        )


def _partition_paths(split: str, i: int):
    x_suffix, y_suffix = (
        ("RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl",
         "Labels_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl")
        if split == "train"
        else ("LSBZM-Norm_FPCKNN-impute.pkl", "Labels_LSBZM-Norm_FPCKNN-impute.pkl")
    )
    d = DATA_DIR / split
    return d / f"Partition{i+1}_{x_suffix}", d / f"Partition{i+1}_{y_suffix}"


def iter_partitions_float32(split: str, num_partitions: int = N_PARTITIONS, skip_indices=()):
    """Yield (X, y) per partition, cast to float32, without ever holding more
    than one partition's float64 copy in memory at a time."""
    for i in range(num_partitions):
        if i in skip_indices:
            continue
        x_path, y_path = _partition_paths(split, i)
        with open(x_path, "rb") as f:
            X64 = pickle.load(f)
        X = X64.astype(np.float32)
        del X64
        with open(y_path, "rb") as f:
            y64 = pickle.load(f)
        y = y64.astype(np.float32)
        del y64
        gc.collect()
        yield X, y


def load_train_full():
    """Load and concatenate all raw train partitions (no aggregation).

    Train partitions are small (~1GB total as float64) so it's safe to
    accumulate all of them, but each is still cast to float32 and freed of
    its float64 copy immediately on load.
    """
    X_list, y_list = [], []
    for X, y in iter_partitions_float32("train", N_PARTITIONS):
        X_list.append(X)
        y_list.append(y)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    del X_list, y_list
    gc.collect()
    return X, y


def load_val_partition():
    """Load a single natural-distribution test partition to use as validation.

    Train partitions went through RUS-Tomek-TimeGAN resampling, so validating
    on a held-out *train* partition leaks synthetic near-duplicates across
    folds and inflates val TSS (observed: val 0.90 vs test 0.48 on the CNN).
    Validating on a slice of the untouched test set avoids that leakage and
    reflects the real, imbalanced distribution the model is actually scored
    on. The remaining test partitions stay held out for the final streamed
    evaluation.
    """
    x_path, y_path = _partition_paths("test", 0)
    with open(x_path, "rb") as f:
        X = pickle.load(f).astype(np.float32)
    with open(y_path, "rb") as f:
        y = pickle.load(f).astype(np.float32)
    gc.collect()
    return X, y


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def build_model(name: str, n_attrs: int) -> torch.nn.Module:
    if name == "cnn":
        return CNN1D(n_attrs=n_attrs)
    if name == "lstm":
        return LSTMClassifier(n_attrs=n_attrs)
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(targets)


@torch.no_grad()
def predict_proba_streaming(model, device, split: str, num_partitions: int = N_PARTITIONS, skip_indices=()):
    """Same as predict_proba, but streams partitions from disk one at a time
    instead of materializing the full (multi-GB) test set in memory."""
    model.eval()
    probs, targets = [], []
    for X, y in iter_partitions_float32(split, num_partitions, skip_indices=skip_indices):
        loader = make_loader(X, y, BATCH_SIZE, shuffle=False)
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(yb.numpy())
        del X, y, loader
        gc.collect()
    return np.concatenate(probs), np.concatenate(targets)


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    from sklearn.metrics import classification_report

    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["No Flare", "Flare"]))
    print(f"TSS: {true_skill_statistic(y_true, y_pred):.4f}")
    print(f"HSS: {heidke_skill_score(y_true, y_pred):.4f}")


def best_threshold_by_tss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 19)
    scores = [true_skill_statistic(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def train_one_fold(model_name, X_train, y_train, X_val, y_val, n_attrs, device):
    model = build_model(model_name, n_attrs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    best_tss = -np.inf
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_proba, val_true = predict_proba(model, val_loader, device)
        val_tss = true_skill_statistic(val_true, (val_proba >= 0.5).astype(int))
        print(f"  epoch {epoch:2d}: val TSS = {val_tss:.4f}", flush=True)

        if val_tss > best_tss:
            best_tss = val_tss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_tss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "lstm"], default="cnn")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    check_memory("start")

    print("Loading raw train partitions...")
    X_tr, y_tr = load_train_full()
    n_attrs = X_tr.shape[-1]
    print(f"Train shape: {X_tr.shape}")
    check_memory("after train load")

    # Validate on a natural-distribution test partition rather than a
    # held-out train partition: train partitions are TimeGAN-resampled, so
    # a train/train split leaks synthetic near-duplicates across folds and
    # inflates val TSS. Test partition 0 is excluded from the final
    # streamed test evaluation below to keep it unseen until this point.
    X_val, y_val = load_val_partition()
    print(f"Val: {X_val.shape} (natural distribution, test partition 0)")

    print(f"\nTraining {args.model.upper()}...")
    model, best_val_tss = train_one_fold(args.model, X_tr, y_tr, X_val, y_val, n_attrs, device)
    print(f"Best validation TSS: {best_val_tss:.4f}")

    check_memory("before test eval")
    print("\nEvaluating on test set (streamed partition-by-partition)...")
    test_proba, test_true = predict_proba_streaming(model, device, split="test", skip_indices={0})

    evaluate(f"{args.model.upper()} (threshold=0.5)", test_true, (test_proba >= 0.5).astype(int))

    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)
    val_proba, val_true = predict_proba(model, val_loader, device)
    threshold = best_threshold_by_tss(val_true, val_proba)
    print(f"\nTSS-optimal threshold (chosen on val partition): {threshold:.2f}")
    evaluate(
        f"{args.model.upper()} (threshold={threshold:.2f})",
        test_true,
        (test_proba >= threshold).astype(int),
    )


if __name__ == "__main__":
    main()
