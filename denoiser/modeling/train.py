from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
import numpy as np
import pickle

from denoiser.config import DATA_DIR, REPORTS_DIR, MODELS_DIR
from models import SequenceLoader
from models.custom_models import cnn_ced
from models.custom_callbacks import LearningRateStopping, LearningRateLogger

if __name__ == "__main__":

    # Load data
    with open(DATA_DIR / 'clean.pkl', "rb") as f:
        y = pickle.load(f)
    with open(DATA_DIR / 'noisy.pkl', "rb") as f:
        x = pickle.load(f)

    # dirty dataload workaround to see if everything works
    train = SequenceLoader(x_noisy=x, y_clean=y, window_size=8, batch_size=128, shuffle=True)
    validate = SequenceLoader(x_noisy=x, y_clean=y, window_size=8, batch_size=128, shuffle=True)
    # around 5% used for validation
    split_idx = len(train.indexes) // 20
    train.indexes, validate.indexes = train.indexes[split_idx:], train.indexes[:split_idx]


    # Model
    model = cnn_ced(model_name="CNN-CED-2000", show_model=False)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-4),
        LearningRateLogger(),
        TensorBoard(log_dir=REPORTS_DIR / model.name),
        LearningRateStopping(min_lr=1e-4),
    ]

    result = model.fit(x=train, validation_data=validate, epochs=120, verbose=1, callbacks=callbacks)
    model.save(f"{MODELS_DIR / model.name}.keras")