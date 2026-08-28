import pickle
from pathlib import Path

ATTRIBUTES = [
    'R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP',
    'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSX', 'EPSY', 'EPSZ', 'MEANSHR', 'SHRGT45',
    'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD',
    'MEANALP', 'TOTFX'
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Preprocessed_SWANSF"


def load_partitions(data_dir: Path = DATA_DIR, split: str = "train", num_partitions: int = 5):
    data_dir = Path(data_dir) / split
    X, y = [], []

    if split == "train":
        x_suffix = "RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl"
        y_suffix = "Labels_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl"
    else:
        x_suffix = "LSBZM-Norm_FPCKNN-impute.pkl"
        y_suffix = "Labels_LSBZM-Norm_FPCKNN-impute.pkl"

    for i in range(num_partitions):
        x_path = data_dir / f"Partition{i+1}_{x_suffix}"
        y_path = data_dir / f"Partition{i+1}_{y_suffix}"
        with open(x_path, "rb") as f:
            X.append(pickle.load(f))
        with open(y_path, "rb") as f:
            y.append(pickle.load(f))

    return X, y
