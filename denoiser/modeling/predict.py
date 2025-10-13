import keras
import numpy as np
from numpy.typing import NDArray
from typing import Union
import os

from denoiser.config import MODELS_DIR, FRAMES_LENGTH

def predict(
        model: Union[keras.models.Model, keras.models.Sequential, str],
        audio_path: str
) -> Union[list[NDArray[np.float32]], None]:

    if not os.path.exists(audio_path):
        raise FileNotFoundError("Specified directory does not exist!")

    if isinstance(model, (keras.models.Model, keras.models.Sequential)):
        loaded_model = model
    else:
        if not os.path.exists(MODELS_DIR / model):
            raise FileNotFoundError("Specified model not found! Check trained model list...")
        loaded_model = keras.models.load_model(MODELS_DIR / model)

    spec_paths = [os.path.join(audio_path, audio_file) for audio_file in os.listdir(audio_path) if audio_file.endswith('noisy_mag.npy')]
    predictions = []
    for file_path in spec_paths:
        with open(file_path, "rb") as f:
            spec_magnitude = np.load(f).T  # data shape needs to be (..., 129) and is saved the other way around
        windowed = np.lib.stride_tricks.sliding_window_view(spec_magnitude, window_shape=FRAMES_LENGTH, axis=0)

        predictions.append(loaded_model.predict(windowed, verbose="auto", steps=None, callbacks=None))

    return predictions if predictions else None

if __name__ == "__main__":
    pass
    #predict(model="CNN-CED-norm-test.py", audio_path="")
