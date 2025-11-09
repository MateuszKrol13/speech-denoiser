import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

from denoiser.data import Dataset, FeatureType, SignalType
from denoiser.metrics import pesq, stoi, si_sdr
from denoiser.config import TRAIN_DATA, TEST_DATA, FRAMES_LENGTH, MODELS_DIR
from denoiser import get_models

if __name__ == "__main__":
    saved_models = get_models()
    print("Running evaluation for cnn-ced type model\nAvailable models:")
    for i, model in enumerate(saved_models):
        print(f"{i}.) {model}")
    selected_model = int(input("Pick model for evaluation (int) "))
    loaded_model = load_model(filepath=os.path.join(MODELS_DIR, saved_models[selected_model]))

    stoi_avg = []
    pesq_avg = []
    sdr_avg = []

    clean_ds = Dataset(TRAIN_DATA).load()
    _, clean_stats = clean_ds.extract_feature(FeatureType.MAGNITUDE, SignalType.TARGET)
    _, noisy_stats = clean_ds.extract_feature(FeatureType.MAGNITUDE, SignalType.SOURCE)
    del clean_ds  # save memory

    test_ds = Dataset(TEST_DATA).load()  # this keep for inference
    for data in test_ds:
        clean, noisy = data

        # transpose, normalise and window input feature
        x = noisy.get_magnitude().T
        x_norm = (x - noisy_stats["mean"]) * noisy_stats["std"]
        x_windowed = np.lib.stride_tricks.sliding_window_view(x_norm, window_shape=FRAMES_LENGTH, axis=0)

        # model prediction
        y_pred = np.squeeze(loaded_model.predict(x_windowed, verbose="auto", steps=None, callbacks=None)).T
        pred_denorm = np.where(y_pred > 0, y_pred * clean_stats["std"] + clean_stats["mean"], y_pred)

        # reconstruct signal
        phase = noisy.get_phase()[:, :(-FRAMES_LENGTH + 1)]
        reconstructed_spec = pred_denorm * np.exp(1j * phase)
        reconstructed_sig = librosa.istft(reconstructed_spec, n_fft=256, hop_length=64, win_length=256, window="hamming")

        # calculate metrics
        stoi_avg.append(stoi(x=clean.waveform[:len(reconstructed_sig)], y=reconstructed_sig, fs_sig=8000))
        pesq_avg.append(pesq(ref=clean.waveform[:len(reconstructed_sig)], deg=reconstructed_sig, mode='nb', fs=8000))
        sdr_avg.append(si_sdr(y=clean.waveform[:len(reconstructed_sig)], x=reconstructed_sig))