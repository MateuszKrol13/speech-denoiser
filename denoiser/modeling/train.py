import os.path

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import pickle

from denoiser.config import DATA_DIR, REPORTS_DIR, MODELS_DIR, DATA_METADATA, TRAIN_DATA, VAL_DATA
from models import SequenceLoader
from models.custom_models import cnn_ced
from models.custom_callbacks import LearningRateStopping

BATCH_SIZE = 128

if __name__ == "__main__":
    model_name = input("Please provide model name: ")

    # Train
    print("Loading training data from files...")
    with open(TRAIN_DATA / 'clean.pkl', "rb") as f:
        y = pickle.load(f)
    with open(TRAIN_DATA / 'noisy.pkl', "rb") as f:
        x = pickle.load(f)
    with open(TRAIN_DATA / 'metadata.pkl', "rb") as f:
        metadata = pickle.load(f)
    train = SequenceLoader(x_noisy=x, y_clean=y, window_size=8, batch_size=BATCH_SIZE, shuffle=True)

    # Validate
    print("Loading validation data from files...")
    with open(VAL_DATA / 'clean.pkl', "rb") as f:
        y = pickle.load(f)
    with open(VAL_DATA / 'noisy.pkl', "rb") as f:
        x = pickle.load(f)
    validate = SequenceLoader(x_noisy=x, y_clean=y, window_size=8, batch_size=BATCH_SIZE, shuffle=True)

    print("Loading model...")
    # Model
    model = cnn_ced(model_name=model_name, show_model=False)
    model.metadata = metadata

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-4),
        TensorBoard(log_dir=REPORTS_DIR / model.name),
        LearningRateStopping(min_lr=1e-4)
    ]

    result = None
    try:
        result = model.fit(x=train, validation_data=validate, epochs=120, verbose=1, callbacks=callbacks)

    except KeyboardInterrupt:
        print("User terminated model fitting, saving model...")

    finally:
        model.save(f"{MODELS_DIR / model.name}.keras")
        if result is not None:
            with open(os.path.join(REPORTS_DIR, model.name, 'history.history'), 'wb') as f:
                pickle.dump(result, f)

    model.save(f"{MODELS_DIR / model.name}.keras")