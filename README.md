# Solar Flare Forecasting Project — Context

## Goal
Genuine research-style exploration of solar flare prediction using the SWAN-SF dataset, as an ongoing background project alongside coursework. Ambition: approach or beat published SWAN-SF benchmarks (see below). No fixed endpoint.

## Domain Background

### The physical phenomenon
The Sun's surface has **active regions** — areas of intensely tangled, concentrated magnetic field, visible as sunspots. When the field gets twisted/sheared past a breaking point, it can snap into a simpler configuration, releasing a burst of energy: a **solar flare**.

### Why it matters
Large flares (and associated coronal mass ejections) can disrupt satellites, GPS, radio communications, and power grids. Forecasting "will this active region flare in the next N hours" has real operational value.

### Flare classification
Ranked by X-ray brightness: A, B, C, M, X (each ~10x more energetic than the last). Forecasting typically targets catching **M** and **X** class flares — rare but consequential, which is why flare datasets are heavily class-imbalanced.

### Key terms
- **Magnetic shear**: misalignment of field lines across the polarity boundary — indicates stored energy, like a twisted rope.
- **Polarity Inversion Line (PIL)**: the line separating positive/negative magnetic polarity — the "fault line" flares tend to originate near.
- **Total unsigned flux**: overall magnetic field strength present, regardless of direction — more flux = more available energy.
- **Observation / prediction window**: watch a region's parameters for a period (e.g. 12h), then check if a flare occurred in the following period (e.g. next 24h). Mirrors real operational monitoring.
- **SHARP**: Spaceweather HMI Active Region Patch — the cropped, tracked magnetogram patch per active region that numeric parameters are computed from.

### The pipeline, conceptually
```
Sun's surface → SDO magnetogram images → SHARP patches (per active region)
→ 24 numeric parameters per timestep → sequence over an observation window
→ [MODEL] → probability of a flare in the next N hours
```
This is a **multivariate time series classification** problem: given the recent time-evolution of 24 magnetic parameters, predict whether the region is about to flare.

## Dataset — SWAN-SF

Multivariate time series covering 4,000+ active regions across 9+ years of SDO data, 51 flare-predictive parameters (24 commonly used), 10,000+ flare reports, split into 5 non-overlapping time-segmented partitions (~May 2010–Aug 2018) for reproducible train/test splits.

### Access
- **Raw source**: Harvard Dataverse — doi:10.7910/DVN/EBCFKM. Ground-truth CSVs but needs preprocessing (imputation, normalization, imbalance handling).
- **Preprocessed shortcut** (recommended to start): [samresume/Cleaned-SWANSF-Dataset](https://github.com/samresume/Cleaned-SWANSF-Dataset) on GitHub. Actual `.pkl` files are hosted on Google Drive — link is in that repo's `download.txt`. Download manually and place under `data/Preprocessed_SWANSF/{train,test}/`.

### Data structure
- Shape per partition: `(num_samples, num_timestamps, 24 attributes)`
- Attribute order:
```python
ATTRIBUTES = [
    'R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP',
    'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSX', 'EPSY', 'EPSZ', 'MEANSHR', 'SHRGT45',
    'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD',
    'MEANALP', 'TOTFX'
]
```
- Labels: binary 0/1 per sample, separate 1D vector per partition.
- Train partitions: imputation (FPCKNN) + normalization (LSBZM) + sampling techniques (TimeGAN, Tomek Links, Random Under-sampling) applied to address class imbalance and Class C elimination.
- Test partitions: only imputation + normalization applied — left "natural" for honest evaluation.

### File naming
```
train: Partition{i}_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl
       Partition{i}_Labels_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl
test:  Partition{i}_LSBZM-Norm_FPCKNN-impute.pkl
       Partition{i}_Labels_LSBZM-Norm_FPCKNN-impute.pkl
```

### Loader
```python
import pickle
import numpy as np
from pathlib import Path

ATTRIBUTES = [
    'R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP',
    'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSX', 'EPSY', 'EPSZ', 'MEANSHR', 'SHRGT45',
    'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD',
    'MEANALP', 'TOTFX'
]

def load_partitions(data_dir: str, split: str = "train", num_partitions: int = 5):
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
```

## Evaluation metrics
Accuracy is close to meaningless given extreme class imbalance. Use:
- **TSS (True Skill Statistic)**: true positive rate minus false positive rate. Range -1 to 1. Field-standard primary metric — robust to imbalance.
- **HSS (Heidke Skill Score)**: improvement over chance prediction.

## Published benchmarks (binary large-flare prediction)

| Approach | TSS | Notes |
|---|---|---|
| Classical ML baselines (SVM, LSTM, MiniRocket, robust sampling study) | ~0.51 | HSS ~0.38 — solid mid-tier reference point |
| CONTREX (contrastive representation learning, 2024) | ~0.71 | 0.73 accuracy, 0.85 ROC AUC |
| Lightweight tuned CNN (2026) | ~0.86 | Current high-water mark found; very recent |

Progression (0.51 → 0.71 → 0.86) comes mostly from representation/architecture choices (contrastive learning, tuned CNNs, attention-based fusion), not more data — a good sign for a solo effort.

## Project phases

### Phase 1 — Data + naive baseline
- Set up environment, pull preprocessed SWAN-SF partitions.
- Load and inspect one partition: shapes, label distribution, what a single instance looks like.
- Define task: binary large-flare-vs-not (standard benchmark framing).
- Build simple baseline: statistical summary features (mean/std/max/last per parameter) + Logistic Regression or Random Forest.
- Evaluate with TSS/HSS. Target: any real number, even modest (~0.2–0.3 TSS typical for naive baselines).

### Phase 2 — Match established baseline
- Implement LSTM or 1D-CNN properly on raw time series (not just summary stats).
- Handle class imbalance carefully — sampling strategy matters as much as architecture.
- Target: TSS 0.45–0.55, matching published classical baselines.
- Overlaps with coursework: RNN weeks and backprop-through-sequences reinforce this directly.

### Phase 3 — Ongoing, research-y (no fixed endpoint)
- Contrastive representation learning (as in CONTREX).
- Attention-based temporal fusion.
- Careful feature selection (one study found only 6 of 24 features meaningfully help).
- Ties into course's Transformer/attention weeks and personal AI engineering roadmap's ML Foundations/Capstone pillar.

## Personal context for continuity
- Coming from a background with DS 4400 (MLE, KDE, sampling, information theory, PCA, LDA, clustering, logistic regression) — probability/stats fluency is solid; main gap is multivariable chain rule / Jacobians for backprop mechanics.
- Currently taking a deep learning course covering backprop, CNNs, RNNs, Transformers, posttraining, interpretability (NDIF), and diffusion models — this project is meant to reinforce course material as it's covered, particularly RNN/attention weeks.
- Prefers routing learning through real projects rather than toy exercises.
- Also has a comparable multivariate time-series classification project at work (turbine downtime event classification, TF-IDF + Logistic Regression baseline) — useful as a structural analogy when reasoning about this project.
