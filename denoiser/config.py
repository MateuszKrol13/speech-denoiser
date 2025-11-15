from pathlib import Path
from numpy import mean, std

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]  # project root directory
DATA_ROOT = PROJ_ROOT / "data"
DATA_DIR = PROJ_ROOT / "data\\processed\\2025-10-18_20-11\\"
TRAIN_DATA = PROJ_ROOT / "data\\raw\\commonvoice\\train\\clips"
VAL_DATA = PROJ_ROOT / "data\\raw\\commonvoice\\validation\\clips"
TEST_DATA = PROJ_ROOT / "data\\raw\\commonvoice\\test\\clips"

CLIPS_DURATIONS = PROJ_ROOT / "data\\raw\\cv-corpus-20.0-delta-2024-12-06\\en\\clip_durations.tsv"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# denoiser parameters
SNR_LEVEL = 0  # dB
SAMPLING_FREQ = 8000  # Hz
FRAMES_LENGTH = 8  # Count of frames used to reconstruct clean signal
FEATURES_COUNT = 129  # spectrogram bin count
NOISE = 'PinkNoise'

# Data statistics
DATA_STATS = {
    'mean' : mean,
    'std' : std,
    'min' : min,
    'max' : max,
}