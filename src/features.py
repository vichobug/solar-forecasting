import numpy as np

from data import ATTRIBUTES

AGG_FUNCS = ["mean", "std", "min", "max", "last"]


def aggregate_timeseries(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Collapse (n_samples, n_timesteps, n_attrs) into tabular (n_samples, n_features).

    For each attribute, computes mean/std/min/max over the window plus the final
    timestep value (closest to the forecast point).
    """
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    minimum = X.min(axis=1)
    maximum = X.max(axis=1)
    last = X[:, -1, :]

    features = np.concatenate([mean, std, minimum, maximum, last], axis=1)
    columns = [f"{attr}_{agg}" for agg in AGG_FUNCS for attr in ATTRIBUTES]
    return features, columns
