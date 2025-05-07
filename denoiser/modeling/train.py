from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
import h5py
import numpy as np

from denoiser.config import TRAIN_DATA, REPORTS_DIR, MODELS_DIR
from models.custom_models import cnn_ced
from models.custom_callbacks import LearningRateStopping, LearningRateLogger

if __name__ == "__main__":

    # Load data
    f = h5py.File(name=TRAIN_DATA / 'source.mat', mode='r')
    f2 = h5py.File(name=TRAIN_DATA / 'target.mat', mode='r')
    train_x = np.transpose(np.array(f.get('sourceTrain')), axes=[0, 2, 1])
    train_y = np.array(f2.get('targetTrain'))

    # Model
    model = cnn_ced(show_model=False)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-4),
        LearningRateLogger(),
        TensorBoard(log_dir=REPORTS_DIR / model.name),
        #LearningRateStopping(min_lr=1e-4)
    ]

    result = model.fit(train_x, train_y, validation_split=0.05, batch_size=128, epochs=1, verbose=1, callbacks=callbacks)
    model.save(f"{MODELS_DIR / model.name}.keras")