import os
import librosa
import numpy as np
from pystoi import stoi
from pesq import pesq

from predict import predict

def si_sdr(y, x):
    alpha = np.dot(y, x) / (np.linalg.norm(y) ** 2)
    frac = (np.linalg.norm(alpha * y) ** 2) /  (np.linalg.norm(alpha * y - x) ** 2)
    return 10 * np.log10(frac)


def evaluate(model, data_path, metadata_):
    stoi_avg = []
    pesq_avg = []
    sdr_avg = []

    # recording names : common_voice_en_{number}_...
    recording_names = list(set(
        ["_".join(file.split('_')[0:4]) for file in os.listdir(data_path)]  # remove duplicates; list -> set -> list
    ))

    for rec_name in recording_names:
        prediction = predict(file_path=os.path.join(data_path, rec_name + "_noisy_mag.npy"), model=model, metadata_=metadata_)
        phase = np.load(os.path.join(data_path, rec_name + "_phase.npy"))[:, :-7]  # cause of moving window
        reconstructed_spectrogram = prediction * np.exp(1j * phase)
        reconstructed_sig = librosa.istft(reconstructed_spectrogram, n_fft=256, hop_length=64, win_length=256, window="hamming")

        clean_audio, _ = librosa.load(os.path.join(data_path, rec_name + "_clean.wav"), sr=8000)
        stoi_avg.append(stoi(x=clean_audio[:len(reconstructed_sig)], y=reconstructed_sig, fs_sig=8000))
        pesq_avg.append(pesq(ref=clean_audio[:len(reconstructed_sig)], deg=reconstructed_sig, fs=8000))
        sdr_avg.append(si_sdr(y=clean_audio[:len(reconstructed_sig)], x=reconstructed_sig))

    return stoi_avg, pesq_avg


if __name__ == "__main__":
    pass