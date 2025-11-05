import keras
import numpy as np
from numpy.typing import NDArray
from typing import Union, Any
import os
import pickle

from denoiser.config import MODELS_DIR, FRAMES_LENGTH, TRAIN_DATA

def predict(file_path, model: Union[keras.models.Model, keras.models.Sequential], metadata_: dict[str, Any]):
    """Passes specified file through provided keras model, performing all the necessary preprocessing and postprocessing
    steps

    Parameters:
        file_path (str): Path to numpy array containing noisy magnitude spectrogram data
        model (Model | Sequential): Loaded into memory keras model
        metadata_ (dict): Dictionary containing all the data and statistics necessary to perform preprocessing and
            postprocessing for neural network input and output

    Returns:
        (2D-ndarray): network denoised and denormalised spectrogram magnitude data

    Notes:
        * to perform directory-wide prediction, use directory_predict function
    """
    with open(file_path, "rb") as f:
        spec_magnitude = np.load(f).T  # data shape needs to be (..., 129) and is saved the other way around
    spec_normalised = (spec_magnitude - metadata_["noisy_mean"]) / metadata_["noisy_std"]
    windowed = np.lib.stride_tricks.sliding_window_view(spec_normalised, window_shape=FRAMES_LENGTH, axis=0)

    pred = model.predict(windowed, verbose="auto", steps=None, callbacks=None)
    pred_norm = (np.squeeze(pred).T * metadata_["clean_std"]) + metadata_["clean_mean"]  # (..., 129, 1) -> (129, ...)

    return pred_norm

def predict_directory(
        model: Union[keras.models.Model, keras.models.Sequential, str],
        audio_path: str
) -> Union[list[NDArray[np.float32]], None]:
    """Loads keras model from provided path - or passed directly into the function - and for each noisy spectrogram
    array data in provided directory model prediction is derived. For single file prediction use predict() function

    Parameters:
        model (Model | Sequential | str): Keras model or path to keras model to be used for prediction.
        audio_path (str): directory path containing noisy spectrogram data in _noisy.npy files

    Returns:
        (list of 2D-ndarraus): List of network denoised and denormalised spectrogram magnitude data for each noisy array
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError("Specified directory does not exist!")

    with open(TRAIN_DATA / "metadata.pkl", "rb") as f:
        metadata_ = pickle.load(f)

    if isinstance(model, (keras.models.Model, keras.models.Sequential)):
        loaded_model = model
    else:
        if not os.path.exists(MODELS_DIR / model):
            raise FileNotFoundError("Specified model not found! Check trained model list...")
        loaded_model = keras.models.load_model(MODELS_DIR / model)

    spec_paths = [os.path.join(audio_path, audio_file) for audio_file in os.listdir(audio_path) if audio_file.endswith('noisy_mag.npy')]
    predictions = [predict(pth, loaded_model, metadata_=metadata_) for pth in spec_paths]

    return predictions if predictions else None

if __name__ == "__main__":
    pass