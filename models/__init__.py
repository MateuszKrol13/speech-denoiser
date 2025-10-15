from models.custom_models import cnn_ced
from models.custom_callbacks import LearningRateStopping
from models.custom_dataloaders import SequenceLoader, MemmapLoader

def get_models():
    import os
    from denoiser.config import MODELS_DIR
    return [model for model in os.listdir(MODELS_DIR) if model.endswith('.keras')]