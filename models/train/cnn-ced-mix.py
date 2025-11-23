from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard

from denoiser.config import REPORTS_DIR, MODELS_DIR, TRAIN_DATA, VAL_DATA, FRAMES_LENGTH
from denoiser.data import Dataset, FeatureType, SignalType
from denoiser.preprocessor import z_norm
from models import SequenceLoader, cnn_ced
from models.custom_callbacks import LearningRateStopping

if __name__ == "__main__":
    model_name = input("Please provide model name: ")
    BATCH_SIZE = 512

    print("Loading training dataset...")
    train_ds = Dataset(path=TRAIN_DATA).load().normalize_sources()
    x_train, _ = train_ds.extract_feature(feature=FeatureType.MAGNITUDE, signal=SignalType.SOURCE, normalise=z_norm)
    y_train, _ = train_ds.extract_feature(feature=FeatureType.MAGNITUDE, signal=SignalType.TARGET, normalise=z_norm)
    del train_ds

    print("Loading validation dataset...")
    validate_ds = Dataset(path=VAL_DATA).load().normalize_sources()
    x_val, _ = validate_ds.extract_feature(feature=FeatureType.MAGNITUDE, signal=SignalType.SOURCE, normalise=z_norm)
    y_val, _ = validate_ds.extract_feature(feature=FeatureType.MAGNITUDE, signal=SignalType.TARGET, normalise=z_norm)
    del validate_ds

    train = SequenceLoader(
        x_noisy=x_train,
        y_clean=y_train,
        window_size=FRAMES_LENGTH,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validate = SequenceLoader(
        x_noisy=x_val,
        y_clean=y_val,
        window_size=FRAMES_LENGTH,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("Loading model...")
    model = cnn_ced(model_name=model_name, show_model=False)
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-4),
        TensorBoard(log_dir=REPORTS_DIR / model.name),
        LearningRateStopping(min_lr=1e-4)
    ]

    try:
        result = model.fit(x=train, validation_data=validate, epochs=120, verbose=1, callbacks=callbacks)

    except KeyboardInterrupt:
        print("User terminated model fitting, saving model...")

    finally:
        model.save(f"{MODELS_DIR / model.name}.keras")