from pathlib import Path
from numpy import mean, std

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data\\processed\\2025-10-11_16-13\\"
CLEAN_DATA = DATA_DIR / "clean.pkl"
NOISY_DATA = DATA_DIR / "noisy.pkl"
DATA_METADATA = DATA_DIR / "metadata.pkl"
METADATA_READABLE = DATA_DIR / "metadata.txt"


MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data statistics
DATA_STATS = {
    'mean' : mean,
    'std' : std,
}