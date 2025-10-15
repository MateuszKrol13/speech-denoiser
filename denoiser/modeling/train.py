import os.path

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import numpy as np
import pickle

from denoiser.config import DATA_DIR, REPORTS_DIR, MODELS_DIR, DATA_METADATA
from models import SequenceLoader, MemmapLoader
from models.custom_models import cnn_ced
from models.custom_callbacks import LearningRateStopping

BATCH_SIZE = 1024

if __name__ == "__main__":
    model_name = input("Please provide model name: ")
    print("Opening dataset...")

    with open(DATA_DIR / 'metadata.pkl', "rb") as f:
        metadata_=pickle.load(f)

    x = np.memmap(DATA_DIR / "noisy.npy", dtype='float32', mode='r', shape=metadata_["noisy_mmap_shape"])
    y = np.memmap(DATA_DIR / "clean.npy", dtype='float32', mode='r', shape=metadata_["clean_mmap_shape"])

    # split indexes between train and validate data
    assert x.shape[:-1] == y.shape  # sanity check
    index_list = [idx for idx in range(x.shape[0])]
    np.random.shuffle(index_list)
    train_indexes, validate_indexes = index_list[len(index_list) // 20:], index_list[:len(index_list) // 20] # Around 5%

    # Dataloaders
    print("Preparing dataloaders...")
    train_loader = MemmapLoader(
        x_noisy=x,
        y_clean=y,
        idx_list=train_indexes,
        batch_size=BATCH_SIZE
    )
    validate_loader = MemmapLoader(
        x_noisy=x,
        y_clean=y,
        idx_list=validate_indexes,
        batch_size=BATCH_SIZE
    )

    # Model
    print("Opening model...")
    model = cnn_ced(model_name=model_name, show_model=False)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-4),
        TensorBoard(log_dir=REPORTS_DIR / model.name, update_freq=1000),
        LearningRateStopping(min_lr=1e-4)
    ]

    result = None
    try:
        result = model.fit(x=train_loader, validation_data=validate_loader, epochs=120, verbose=1, callbacks=callbacks)

    except KeyboardInterrupt:
        print("User terminated model fitting, saving model...")

    finally:
        model.save(f"{MODELS_DIR / model.name}.keras")
        if result is not None:
            with open(os.path.join(REPORTS_DIR, model.name, 'history.history'), 'wb') as f:
                pickle.dump(result, f)

    model.save(f"{MODELS_DIR / model.name}.keras")