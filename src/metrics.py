import numpy as np
from sklearn.metrics import confusion_matrix


def true_skill_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TSS = recall - false positive rate. Range [-1, 1], 0 = no skill."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return recall - fpr


def heidke_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """HSS2, as commonly used in solar flare forecasting. Range (-inf, 1], 0 = no skill."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return numerator / denominator if denominator else 0.0
