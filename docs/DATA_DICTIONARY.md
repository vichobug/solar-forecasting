# Data Dictionary — Preprocessed SWAN-SF

Source: SWAN-SF (Space Weather ANalytics for Solar Flares) benchmark dataset, preprocessed
into pickled NumPy arrays under `data/Preprocessed_SWANSF/`. Loaded via
`src/data.py:load_partitions`.

## Structure

- **Unit of observation**: one multivariate time series per active-region sample.
- **Shape**: `(n_samples, 60, 24)` for features, `(n_samples,)` for labels.
  - 60 timesteps: sequential magnetogram observations leading up to the forecast point
    (12-minute cadence in the original SHARP data).
  - 24 columns: SHARP parameters (see table below).
- **Partitions**: 5 partitions per split (`Partition1` ... `Partition5`), stored as
  separate pickle files and concatenated by `load_partitions`.
- **Splits**:
  | Split | File suffix | Class balance |
  |---|---|---|
  | `train` | `RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl` | Resampled to ~50/50 (RUS + Tomek links + TimeGAN oversampling) |
  | `test` | `LSBZM-Norm_FPCKNN-impute.pkl` | Natural/imbalanced (~1-3% positive) |

  Both splits share the same preprocessing: `LSBZM-Norm` (per-attribute normalization),
  `FPCKNN-impute` (KNN-based imputation of missing values). Train additionally excludes
  C-class flares (`WithoutC`) and is class-balanced.

## Label

| Column | Type | Meaning |
|---|---|---|
| `y` | binary (0/1) | `1` = a flare event follows the observation window; `0` = no flare |

## Features (SHARP parameters)

All values are normalized (LSBZM-Norm), so per-attribute range is ~[0, 1] rather than
physical units.

| Attribute | Description |
|---|---|
| `USFLUX` | Total unsigned magnetic flux |
| `TOTUSJH` | Total unsigned current helicity |
| `TOTUSJZ` | Total unsigned vertical current |
| `TOTBSQ` | Total magnitude of Lorentz force |
| `TOTPOT` | Total photospheric magnetic free energy density |
| `MEANPOT` | Mean photospheric magnetic free energy density |
| `SAVNCPP` | Sum of the modulus of the net current per polarity |
| `ABSNJZH` | Absolute value of the net current helicity |
| `MEANJZH` | Mean current helicity (vertical-current contribution) |
| `MEANJZD` | Mean vertical current density |
| `MEANALP` | Mean characteristic twist parameter (alpha) |
| `MEANSHR` | Mean shear angle |
| `SHRGT45` | Fraction of pixels with shear angle > 45 degrees |
| `MEANGAM` | Mean angle of the magnetic field from radial (inclination) |
| `MEANGBT` | Mean gradient of the total magnetic field |
| `MEANGBZ` | Mean gradient of the vertical magnetic field |
| `MEANGBH` | Mean gradient of the horizontal magnetic field |
| `TOTFX` | Sum of the x-component of the Lorentz force |
| `TOTFY` | Sum of the y-component of the Lorentz force |
| `TOTFZ` | Sum of the z-component of the Lorentz force |
| `EPSX` | Sum of the x-component of the normalized Lorentz force |
| `EPSY` | Sum of the y-component of the normalized Lorentz force |
| `EPSZ` | Sum of the z-component of the normalized Lorentz force |
| `R_VALUE` | Sum of unsigned flux near strong-gradient polarity inversion lines |

These are the standard SHARP keywords derived from SDO/HMI vector magnetograms
(Bobra & Couvidat 2015), commonly used across solar flare forecasting literature.

## Notes / caveats

- Highly correlated feature groups exist (e.g. `TOTUSJZ`/`USFLUX`, `SHRGT45`/`MEANGAM`) —
  see `notebooks/data_exploration.ipynb` for the full correlation matrix.
- Train/test are not a random split of the same pool: train is resampled and excludes
  C-class flares, while test keeps the natural label distribution. Don't compare raw
  class counts between them as if they were i.i.d. samples of the same population.
