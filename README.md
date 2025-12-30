# Stellar Classification and Atmospheric Parameters

Machine learning system for classifying stars and predicting atmospheric parameters from spectroscopic survey data (APOGEE, GALAH, LAMOST, Gaia).

## Overview

This project implements a multi-survey stellar classification pipeline that:
- Classifies stellar types from spectroscopic features
- Predicts atmospheric parameters (Teff, log g, [Fe/H], [alpha/Fe])
- Enables cross-survey calibration using The Cannon-style approaches
- Handles large-scale spectroscopic datasets efficiently

## Data Sources

| Survey | Scale | Resolution | Access |
|--------|-------|------------|--------|
| APOGEE DR17 | ~650,000 stars | High (IR) | skyserver.sdss.org |
| GALAH DR3 | ~600,000 stars | High (Optical) | galah-survey.org |
| LAMOST DR7 | Millions | Medium | lamost.org |
| Gaia DR3 RVS | Billions | Lower | gea.esac.esa.int/archive |

## Project Structure

```
stellar-classification-parameters/
├── data/
│   ├── raw/                 # Downloaded survey data
│   ├── processed/           # Cleaned and merged datasets
│   └── features/            # Extracted spectral features
├── src/
│   ├── data/                # Data loading and preprocessing
│   ├── features/            # Feature extraction pipelines
│   ├── models/              # Classification and regression models
│   └── evaluation/          # Metrics and validation
├── notebooks/               # Exploratory analysis
├── tests/                   # Unit and integration tests
├── docs/                    # Documentation and implementation plan
└── configs/                 # Configuration files
```

## Technical Stack

- **Language**: Python 3.10+
- **ML Framework**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy, astropy
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Sakeeb91/stellar-classification-parameters.git
cd stellar-classification-parameters

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Implementation Status

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for detailed implementation phases and progress tracking.

## License

MIT License

## Author

Sakeeb Rahman (rahman.sakeeb@gmail.com)
