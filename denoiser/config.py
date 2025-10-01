from pathlib import Path

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data\\processed\\2025-05-2_11-45\\"
TEST_DATA = DATA_DIR / "test"
TRAIN_DATA = DATA_DIR / "train"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"