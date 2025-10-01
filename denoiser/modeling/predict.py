from tensorflow.keras.models import load_model
from scipy.io import loadmat

from denoiser.config import MODELS_DIR, TEST_DATA

if __name__ == "__main__":
    input_data = r"F:\Magisterka\speech-denoiser\data\processed\2025-05-2_11-45\test\cleanrecs\inference\spectrogramMag.mat"

    model = load_model("speech-denoiser//models//CNN-CED-1200.keras")
    model.predict(input_data, batch_size=None, verbose="auto", steps=None, callbacks=None)
