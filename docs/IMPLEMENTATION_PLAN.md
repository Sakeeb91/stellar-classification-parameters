# Implementation Plan: Stellar Classification and Atmospheric Parameters

## Expert Role

**Role**: ML Engineer / Astrophysics Domain Specialist

**Rationale**: This project requires deep understanding of spectroscopic data processing, multi-survey data fusion, and regression/classification techniques for stellar parameters. The role combines ML expertise with domain knowledge of stellar atmospheres, spectral features, and survey-specific systematics.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STELLAR CLASSIFICATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   APOGEE     │    │    GALAH     │    │   LAMOST     │                   │
│  │   DR17       │    │    DR3       │    │   DR7        │                   │
│  │  (allStar)   │    │  (Summary)   │    │  (Catalog)   │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                            │
│         └─────────────┬─────┴─────────────┬─────┘                            │
│                       ▼                   ▼                                  │
│              ┌────────────────────────────────────┐                         │
│              │       DATA INGESTION LAYER         │                         │
│              │  • Survey-specific parsers         │                         │
│              │  • Quality flag filtering          │                         │
│              │  • Cross-match by coordinates      │                         │
│              └────────────────┬───────────────────┘                         │
│                               ▼                                              │
│              ┌────────────────────────────────────┐                         │
│              │      PREPROCESSING LAYER           │                         │
│              │  • Missing value handling          │                         │
│              │  • Outlier detection               │                         │
│              │  • Feature normalization           │                         │
│              │  • Survey bias correction          │                         │
│              └────────────────┬───────────────────┘                         │
│                               ▼                                              │
│              ┌────────────────────────────────────┐                         │
│              │      FEATURE ENGINEERING           │                         │
│              │  • Spectral indices                │                         │
│              │  • Color-magnitude features        │                         │
│              │  • Abundance ratios                │                         │
│              └────────────────┬───────────────────┘                         │
│                               ▼                                              │
│         ┌─────────────────────┴─────────────────────┐                       │
│         ▼                                           ▼                        │
│  ┌──────────────────┐                    ┌──────────────────┐               │
│  │  CLASSIFICATION  │                    │   REGRESSION     │               │
│  │     MODULE       │                    │     MODULE       │               │
│  │                  │                    │                  │               │
│  │  • Stellar type  │                    │  • Teff          │               │
│  │  • Luminosity    │                    │  • log g         │               │
│  │    class         │                    │  • [Fe/H]        │               │
│  │  • Evolutionary  │                    │  • [alpha/Fe]    │               │
│  │    stage         │                    │                  │               │
│  └────────┬─────────┘                    └────────┬─────────┘               │
│           │                                       │                          │
│           └─────────────────┬─────────────────────┘                          │
│                             ▼                                                │
│              ┌────────────────────────────────────┐                         │
│              │       EVALUATION & OUTPUT          │                         │
│              │  • Cross-survey validation         │                         │
│              │  • Uncertainty quantification      │                         │
│              │  • Results export                  │                         │
│              └────────────────────────────────────┘                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Selection

| Component | Choice | Rationale | Tradeoff | Fallback |
|-----------|--------|-----------|----------|----------|
| **Language** | Python 3.10+ | Industry standard for ML/astro | Slower than C++ | None needed |
| **Data Loading** | astropy, pandas | Native FITS support, tabular data | Memory for large files | Dask for >RAM |
| **ML Framework** | scikit-learn | Simple API, good baselines | Limited deep learning | XGBoost upgrade |
| **Regression** | Random Forest, XGBoost | Handles non-linear, missing data | Requires tuning | Ridge regression |
| **Classification** | Random Forest | Interpretable, robust | May underfit complex | Gradient boosting |
| **Visualization** | matplotlib, seaborn | Standard, extensive docs | Basic interactivity | Plotly |
| **Testing** | pytest | Simple, widely used | None significant | unittest |

### Key Libraries
```
astropy>=5.0        # FITS file handling, coordinates
pandas>=2.0         # Data manipulation
numpy>=1.24         # Numerical operations
scikit-learn>=1.3   # ML algorithms
xgboost>=2.0        # Gradient boosting
matplotlib>=3.7     # Visualization
seaborn>=0.12       # Statistical plots
pytest>=7.0         # Testing
```

---

## Phased Implementation Plan

### Phase 1: Data Foundation
**Scope**: Download APOGEE allStar file, create data loading infrastructure

**Deliverables**:
- `src/data/apogee_loader.py` - APOGEE data parser
- `src/data/quality_filters.py` - Quality flag filtering
- `data/raw/` - Downloaded allStar FITS file
- `notebooks/01_data_exploration.ipynb` - Initial EDA

**Files to Create**:
| File | Purpose | Lines (est) |
|------|---------|-------------|
| `src/__init__.py` | Package init | 1 |
| `src/data/__init__.py` | Subpackage init | 1 |
| `src/data/apogee_loader.py` | Load APOGEE data | 80-100 |
| `src/data/quality_filters.py` | Filter bad data | 50-70 |
| `configs/data_config.yaml` | Data paths, parameters | 30 |

**Technical Challenges**:
- FITS file can be >1GB; need chunked loading strategy
- Quality flags are bitmasks requiring careful interpretation
- Missing values encoded as -9999 or NaN inconsistently

**Verification**:
```python
# Test: Load data successfully
def test_apogee_loader():
    df = load_apogee_allstar("data/raw/allStar-dr17-synspec_rev1.fits")
    assert len(df) > 600000
    assert "TEFF" in df.columns
    assert "LOGG" in df.columns
```

**Definition of Done**:
- [ ] APOGEE allStar file downloaded
- [ ] Loader returns DataFrame with correct columns
- [ ] Quality filters remove flagged stars
- [ ] EDA notebook shows data distributions

---

### Phase 2: Preprocessing Pipeline
**Scope**: Clean data, handle missing values, normalize features

**Deliverables**:
- `src/data/preprocessor.py` - Data cleaning pipeline
- `src/data/feature_selector.py` - Feature selection utilities
- `data/processed/apogee_clean.parquet` - Cleaned dataset

**Files to Create**:
| File | Purpose | Lines (est) |
|------|---------|-------------|
| `src/data/preprocessor.py` | Cleaning pipeline | 120-150 |
| `src/data/feature_selector.py` | Select relevant features | 60-80 |
| `tests/test_preprocessor.py` | Preprocessing tests | 80 |

**Technical Challenges**:
- Deciding imputation strategy for missing abundances
- Outlier detection without removing rare stellar types
- Feature correlation may cause multicollinearity

**Verification**:
```python
def test_no_missing_values():
    df = preprocess_apogee(raw_df)
    assert df.isnull().sum().sum() == 0

def test_feature_ranges():
    df = preprocess_apogee(raw_df)
    assert df["TEFF"].between(3000, 8000).all()
```

**Definition of Done**:
- [ ] No missing values in output
- [ ] Features normalized to comparable scales
- [ ] Outliers flagged but preserved
- [ ] Parquet file saved for fast loading

---

### Phase 3: Baseline Classification Model
**Scope**: Train stellar type classifier using spectroscopic features

**Deliverables**:
- `src/models/classifier.py` - Classification model class
- `src/evaluation/metrics.py` - Evaluation utilities
- `models/stellar_classifier_v1.joblib` - Trained model

**Files to Create**:
| File | Purpose | Lines (est) |
|------|---------|-------------|
| `src/models/__init__.py` | Subpackage init | 1 |
| `src/models/classifier.py` | Classifier implementation | 100-130 |
| `src/evaluation/__init__.py` | Subpackage init | 1 |
| `src/evaluation/metrics.py` | Custom metrics | 60-80 |
| `tests/test_classifier.py` | Model tests | 70 |

**Technical Challenges**:
- Class imbalance (many more dwarfs than giants)
- Defining meaningful stellar type labels from continuous parameters
- Cross-validation strategy for astronomical data (spatial correlation)

**Verification**:
```python
def test_classifier_accuracy():
    clf = StellarClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    assert accuracy > 0.85  # Baseline threshold
```

**Definition of Done**:
- [ ] Classifier achieves >85% accuracy
- [ ] Confusion matrix shows reasonable class separation
- [ ] Model saved and loadable
- [ ] Feature importance extracted

---

### Phase 4: Parameter Regression Models
**Scope**: Predict Teff, log g, [Fe/H], [alpha/Fe] from features

**Deliverables**:
- `src/models/regressor.py` - Multi-output regression
- `src/evaluation/regression_metrics.py` - MAE, scatter metrics
- `models/parameter_regressor_v1.joblib` - Trained models

**Files to Create**:
| File | Purpose | Lines (est) |
|------|---------|-------------|
| `src/models/regressor.py` | Regression models | 130-160 |
| `src/evaluation/regression_metrics.py` | Regression metrics | 50-70 |
| `notebooks/03_regression_analysis.ipynb` | Analysis notebook | N/A |

**Technical Challenges**:
- Different parameters have different scales and error characteristics
- Correlation between parameters (Teff-log g relation)
- Heteroscedastic errors (uncertainty varies with stellar type)

**Verification**:
```python
def test_teff_prediction():
    reg = ParameterRegressor(target="TEFF")
    reg.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, reg.predict(X_test))
    assert mae < 100  # Within 100K for Teff
```

**Definition of Done**:
- [ ] Teff MAE < 100 K
- [ ] log g MAE < 0.2 dex
- [ ] [Fe/H] MAE < 0.1 dex
- [ ] Residual analysis shows no systematic bias

---

### Phase 5: Cross-Survey Validation
**Scope**: Validate on GALAH overlap, assess systematic differences

**Deliverables**:
- `src/data/galah_loader.py` - GALAH data parser
- `src/data/crossmatch.py` - Coordinate cross-matching
- `notebooks/04_cross_validation.ipynb` - Cross-survey analysis

**Files to Create**:
| File | Purpose | Lines (est) |
|------|---------|-------------|
| `src/data/galah_loader.py` | Load GALAH data | 70-90 |
| `src/data/crossmatch.py` | Cross-match utilities | 80-100 |
| `src/evaluation/cross_survey.py` | Cross-survey metrics | 60-80 |

**Technical Challenges**:
- Different wavelength coverage (IR vs optical)
- Systematic offsets between survey pipelines
- Limited overlap sample size

**Verification**:
```python
def test_cross_survey_consistency():
    # Compare APOGEE predictions to GALAH labels
    overlap = crossmatch_surveys(apogee_df, galah_df)
    teff_diff = overlap["TEFF_apogee"] - overlap["TEFF_galah"]
    assert abs(teff_diff.median()) < 50  # Systematic < 50K
```

**Definition of Done**:
- [ ] Cross-match identifies >10,000 common stars
- [ ] Systematic offsets quantified
- [ ] Calibration applied if needed
- [ ] Results documented

---

### Phase 6: Production Pipeline and Documentation
**Scope**: End-to-end pipeline, CLI interface, comprehensive docs

**Deliverables**:
- `src/pipeline.py` - Full inference pipeline
- `scripts/run_pipeline.py` - CLI entry point
- `docs/` - API documentation

**Files to Create**:
| File | Purpose | Lines (est) |
|------|---------|-------------|
| `src/pipeline.py` | End-to-end pipeline | 100-130 |
| `scripts/run_pipeline.py` | CLI interface | 50-70 |
| `requirements.txt` | Dependencies | 15 |

**Definition of Done**:
- [ ] Single command runs full pipeline
- [ ] Results reproducible with seed
- [ ] Documentation complete
- [ ] All tests passing

---

## Risk Assessment

| Risk | Likelihood | Impact | Early Warning | Mitigation |
|------|------------|--------|---------------|------------|
| FITS file download fails | Medium | 🔴 High | Network timeout | Use SDSS CasJobs for subset |
| Memory overflow | High | 🟡 Medium | Swap usage | Use chunked loading, Dask |
| Poor model performance | Medium | 🔴 High | Val loss plateau | Simpler model, more features |
| Survey systematics | High | 🟡 Medium | Cross-val scatter | Document as limitation |
| Missing rare types | Medium | 🟡 Medium | Class imbalance | Stratified sampling |

---

## Testing Strategy

### Test Framework: pytest

### First Three Tests

**Test 1: Data Loading**
```python
# tests/test_data_loading.py
def test_apogee_loader_returns_dataframe():
    """Verify APOGEE loader returns valid DataFrame."""
    from src.data.apogee_loader import load_apogee_allstar
    df = load_apogee_allstar("data/raw/allStar-dr17-synspec_rev1.fits")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    required_cols = ["APOGEE_ID", "TEFF", "LOGG", "FE_H"]
    for col in required_cols:
        assert col in df.columns
```

**Test 2: Quality Filtering**
```python
# tests/test_quality_filters.py
def test_quality_filter_removes_bad_data():
    """Verify quality filter removes flagged stars."""
    from src.data.quality_filters import apply_quality_cuts
    raw_df = pd.DataFrame({
        "TEFF": [4500, -9999, 5000],
        "ASPCAPFLAG": [0, 1, 0]
    })
    filtered = apply_quality_cuts(raw_df)
    assert len(filtered) == 2
    assert -9999 not in filtered["TEFF"].values
```

**Test 3: Model Prediction Shape**
```python
# tests/test_model.py
def test_classifier_prediction_shape():
    """Verify classifier returns correct shape."""
    from src.models.classifier import StellarClassifier
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, 100)
    clf = StellarClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)
```

---

## First Concrete Task

### File to Create: `src/data/apogee_loader.py`

### Function Signature
```python
def load_apogee_allstar(filepath: str, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Load APOGEE allStar FITS file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the allStar FITS file
    columns : list[str] | None
        Specific columns to load. If None, loads default set.

    Returns
    -------
    pd.DataFrame
        DataFrame with stellar parameters and metadata
    """
```

### Starter Code
```python
"""APOGEE DR17 data loader."""

from pathlib import Path
import pandas as pd
from astropy.io import fits
from astropy.table import Table


# Default columns to extract (reduces memory usage)
DEFAULT_COLUMNS = [
    "APOGEE_ID",
    "RA", "DEC",
    "TEFF", "TEFF_ERR",
    "LOGG", "LOGG_ERR",
    "FE_H", "FE_H_ERR",
    "ALPHA_M", "ALPHA_M_ERR",
    "ASPCAPFLAG", "STARFLAG",
    "SNR",
    "J", "H", "K"
]


def load_apogee_allstar(
    filepath: str,
    columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Load APOGEE allStar FITS file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the allStar FITS file
    columns : list[str] | None
        Specific columns to load. If None, loads default set.

    Returns
    -------
    pd.DataFrame
        DataFrame with stellar parameters and metadata

    Raises
    ------
    FileNotFoundError
        If the FITS file doesn't exist
    ValueError
        If requested columns don't exist in file
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"FITS file not found: {filepath}")

    cols_to_load = columns or DEFAULT_COLUMNS

    # Use astropy Table for efficient FITS reading
    with fits.open(filepath, memmap=True) as hdul:
        # allStar data is in extension 1
        table = Table.read(hdul[1])

        # Validate requested columns exist
        available = set(table.colnames)
        requested = set(cols_to_load)
        missing = requested - available
        if missing:
            raise ValueError(f"Columns not found: {missing}")

        # Convert to DataFrame with selected columns
        df = table[cols_to_load].to_pandas()

    return df


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        df = load_apogee_allstar(sys.argv[1])
        print(f"Loaded {len(df)} stars")
        print(df.head())
```

### Verification Command
```bash
python -c "from src.data.apogee_loader import load_apogee_allstar; print('Import successful')"
```

### First Commit Message
```
feat(data): Add APOGEE allStar data loader

- Implement load_apogee_allstar() function
- Support selective column loading for memory efficiency
- Use astropy Table for robust FITS parsing
- Add input validation and error handling
```

---

## Learning Notes for Junior Developer

### Concepts to Understand Before Coding

1. **FITS Files**: Flexible Image Transport System - standard format in astronomy. Unlike CSV, can contain multiple data tables ("extensions"). Use `astropy.io.fits` to read.

2. **Stellar Parameters**:
   - `Teff` (Effective Temperature): Surface temperature in Kelvin (3000-8000K typical)
   - `log g` (Surface Gravity): Logarithm of gravity in cgs units (0-5 typical)
   - `[Fe/H]` (Metallicity): Iron abundance relative to Sun (-2 to +0.5 typical)
   - `[alpha/Fe]` (Alpha Enhancement): Ratio of alpha elements to iron

3. **Quality Flags**: Bitmask integers where each bit indicates a specific issue. Use bitwise AND (`&`) to check specific flags.

4. **Memory Mapping**: `memmap=True` reads file without loading entirely into RAM - essential for large files.

### Resources
- [APOGEE DR17 Documentation](https://www.sdss.org/dr17/irspec/)
- [Astropy FITS Tutorial](https://docs.astropy.org/en/stable/io/fits/)
- [Stellar Classification Wikipedia](https://en.wikipedia.org/wiki/Stellar_classification)
