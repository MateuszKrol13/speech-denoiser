"""
Module for preparing data as input for Neural Network.

Output data will a 2D ndarray(shape=(features, window_count)) that contains concatenated frames of short-time fourier
transform. Each index of noisy data corrseponds to each index of clean data.
"""
import json
import os
import numpy as np
from time import perf_counter
import librosa
from datetime import datetime
import pickle
from typing import Union

from denoiser.config import DATA_STATS, RAW_DATA, SNR_LEVEL, SAMPLING_FREQ


def generate_pink_noise(n_samples:int, sample_rate:int):
    white_noise = np.random.randn(n_samples)
    white_fft = np.fft.rfft(white_noise)
    freqs = np.fft.rfftfreq(n_samples, d=1 / sample_rate)
    scale = np.zeros_like(freqs)
    scale[1:] = 1 / np.sqrt(freqs[1:])
    pink_fft = white_fft * scale
    pink_noise = np.fft.irfft(pink_fft, n_samples)
    return pink_noise

def apply_noise(clear_signal, noise_signal, *, snr_level):
    if not clear_signal.shape == noise_signal.shape:
        raise ValueError("Signal shape does not match provided noise shape")

    # SNR = 10*log[BASE10](P_S / P_N)
    # P_N = P_S / ( 10 ** (0.1 * SNR) )
    # desired P_N = P_noise * alpha
    # desired S_N = S_noise * sqrt(alpha)
    signal_power = np.sum(clear_signal ** 2, axis=0) / clear_signal.shape[0]
    scaled_noise = noise_signal * np.sqrt(signal_power / (10 ** (snr_level / 10)))

    return scaled_noise + clear_signal

def normalize_audio(signal):
    """Normalises audio to [-1;1] value range. Used to limit the amplitude of audio after applying noise.
    :param signal: Audio signal (1D-ndarray)
    :return: same audio signal normalised.
    """
    scaling_factor = max(abs(signal))
    if scaling_factor == 0:  # empty recordings
        return signal
    else:
        return signal / max(abs(signal))

def prepare_data(
        include_phase: bool=True,
        path: str=None,
        rec_no: int=None,
        snr_level: Union[int, float]=SNR_LEVEL,
) -> None:
    """Converts directory containing mp3 recordings into list of ndarrays containing specrogram data of noisy and clear
    recordings and saves them as pickle file if path is not specified.

    :param include_phase: saves phase from noisy spectrogram
    :param path: if path is specified, provided path is used instead of paths specified in denoiser.config. Processed
    data is then saved back to the same directory with corresponding name of .mp3 file. Please do note, that if feature
    extraction fails, there will be a shift in file name, so it's not error safe. Do watch the terminal for errors when
    preparing data for inference. For preparing training dataset this is not an issue.
    :param rec_no: specifies the number of recordings used inside provided (or set in denoiser.config) directory.
    if set to None, every .mp3 file inside directory is used.
    :param snr_level: sets signal-to-noise rato in dB value for noisy recording. Regardless of specified SNR, noisy
    audio is limited to [-1, 1] range. Defaults to value specified in denoiser.config.
    :return: Saves data as bytestream of list of spectrogram data (2D-ndarray) or if path is specified, then each .mp3
    file will get corresponding .npy file containing noisy and clean audio spectrogram magnitude and phase data of noisy
    recording, used in audio reconstruction
    """

    # load audio and prepare data containers
    if path is None:
        audio_files = [os.path.join(RAW_DATA, f) for f in os.listdir(RAW_DATA) if f.endswith(".mp3")]
    else:
        audio_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".mp3")]
    clean_data_magnitude = []
    noisy_data_magnitude = []
    phase_data = []  # Phase data from noisy recording can be used to deconstruct spectrogram into audio

    # Process loop
    rec_no = min(rec_no, len(audio_files)) if rec_no is not None else len(audio_files)  # if not set, default to dir len
    print(f"Loading and processing {rec_no} recordings...")
    timer_start = perf_counter()
    for recording_no in range(rec_no):
        # Progress bar
        remaining_time = perf_counter() * (rec_no - recording_no + 1) / (
                    recording_no + 1)  # Estimate remaining by measuring elapsed
        elapsed = perf_counter() - timer_start
        print(
            f"Loading completed in {(recording_no / rec_no) * 100:.0f}%"
            f"\t\telapsed ({elapsed // 60:.0f}min, {elapsed % 60:.0f}s)"
            f"\t\testimated remaining ({remaining_time // 60:.0f}min, {remaining_time % 60:.0f}s)",
            end="\r"
        )

        # Some audios have issues with encoding
        try:
            # get clean and noisy audio
            clean_signal, sr = librosa.load(audio_files[recording_no])
            downsampled_signal = librosa.resample(clean_signal, orig_sr=sr, target_sr=SAMPLING_FREQ, res_type='scipy')
            pink_noise = generate_pink_noise(downsampled_signal.shape[0], SAMPLING_FREQ)  # called each time, cause we want "random" noise
            noisy_signal = normalize_audio(apply_noise(downsampled_signal, pink_noise, snr_level=snr_level))

            # derive spectrograms
            clean_spectrogram = librosa.stft(downsampled_signal, n_fft=256, hop_length=64, win_length=256, window="hamming")
            noisy_spectrogram = librosa.stft(noisy_signal, n_fft=256, hop_length=64, win_length=256, window="hamming")

            # eight noisy consecutive frames are used to reconstruct first frame
            assert clean_spectrogram.shape == noisy_spectrogram.shape
            clean_data_magnitude.append(abs(clean_spectrogram).astype('float32'))
            noisy_data_magnitude.append(abs(noisy_spectrogram).astype('float32'))

            if include_phase:
                phase_data.append(np.angle(noisy_spectrogram).astype('float32'))

        except ValueError as e:
            print("Concatenation failed, terminating dataset generation!")
            raise e

        except AssertionError as e:
            print("Mismatch in spectrogram shapes, something went wrong... termintaing dataset generation!")
            # mismatch in shapes must be caught
            raise e

        except Exception as e:
            print("Exception happened, continuing...", e, end='\n')

    print(f"\nProcessing finished!\nNormalising data...")
    savefile = os.path.join("../data/processed\\", datetime.now().strftime("%Y-%m-%d_%H-%M")) if path is None else path
    metadata_ = {}

    if path is None:  # Dataset preparation
        os.mkdir(savefile)
        clean_data, noisy_data = os.path.join(savefile, "clean.pkl"), os.path.join(savefile, "noisy.pkl")

        for arr, save in zip((clean_data_magnitude, noisy_data_magnitude), (clean_data, noisy_data)):
            data_type = os.path.basename(save).split('.')[0]
            print(f"Processing {data_type} data...", end="\t")
            flatten_data = np.concatenate([a.flatten() for a in arr])

            for stat, func in DATA_STATS.items():
                stat_name = "_".join([data_type, stat])
                metadata_[stat_name] = func(flatten_data)

            mean, std = metadata_[data_type + "_mean"], metadata_[data_type + "_std"]
            norm_arr = [(a - mean) / std for a in arr]

            print("saving...", end="\t")
            with open(save, "wb") as file:
                pickle.dump(norm_arr, file)
            print("Done!")
    else:
        for f_idx in range(len(audio_files)):
            file_path = audio_files[f_idx].split('.')[0]  # path with file name, but no extension
            for arr, save_suffix in zip(
                    (clean_data_magnitude[f_idx], noisy_data_magnitude[f_idx], phase_data[f_idx]),
                    ("_clean_mag.npy", "_noisy_mag.npy", "_phase.npy")
            ):
                with open(file_path + save_suffix, "wb") as fs:
                    np.save(fs, arr)

    print("Saving metadata...", end="\n")
    with open(os.path.join(savefile, "metadata.pkl"), 'wb') as file:  # binary form as json lib does not like numpy types
        pickle.dump(metadata_, file)

    # readable form
    with open(os.path.join(savefile, "metadata.txt"), 'w') as readable_file:
        json.dump({k:float(v) for k, v in metadata_.items()}, readable_file)

    print("Data processing finished!")

if "__main__" == __name__:
    pass
    #prepare_data(include_phase=False, rec_no=100)