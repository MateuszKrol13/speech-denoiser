"""
Module for preparing data as input for Neural Network.

Output data will a 2D ndarray(shape=(features, window_count)) that contains concatenated frames of short-time fourier
transform. Each index of noisy data corrseponds to each index of clean data.
"""
SNR_LEVEL = 0
TARGET_FREQ = 8000
FRAMES_LENGTH = 8
NOISE = 'PinkNoise'
OUTPUT_DATA = 'Spectrogram absolute values'

TRAIN_SPLIT = 0.97
TEST_SPLIT = 1 - TRAIN_SPLIT
REC_NUMBER = 1000  # To be deleted later on.

import os
import numpy as np
from time import perf_counter, sleep
import librosa

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
    scaling_factor = max(abs(signal))
    if scaling_factor == 0:  # empty recordings
        return signal
    else:
        return signal / max(abs(signal))

if "__main__" == __name__:

    DIRECTORY = "..\\data\\raw\\cv-corpus-20.0-delta-2024-12-06\\en\\clips"
    # load audio and prepare data containers
    audio_files = [os.path.join(DIRECTORY, f) for f in os.listdir(DIRECTORY) if f.endswith(".mp3")]
    clear_input = []
    noisy_input = []

    # Process loop
    print(f"Loading and rocessing {REC_NUMBER} recordings...")
    timer_start = perf_counter()
    for recording_no in range(REC_NUMBER):
        # Progress bar
        remaining_time = perf_counter() * (REC_NUMBER - recording_no + 1) / (
                    recording_no + 1)  # Estimate remaining by measuring elapsed
        elapsed = perf_counter() - timer_start
        print(
            f"Loading completed in {(recording_no / REC_NUMBER) * 100:.0f}%"
            f"\t\telapsed ({elapsed // 60:.0f}min, {elapsed % 60:.0f}s)"
            f"\t\testimated remaining ({remaining_time // 60:.0f}min, {remaining_time % 60:.0f}s)",
            end="\r"
        )

        # Some audios have issues with encoding
        try:
            # get clean and noisy audio
            clean_signal, sr = librosa.load(audio_files[recording_no])
            downsampled_signal = librosa.resample(clean_signal, orig_sr=sr, target_sr=TARGET_FREQ, res_type='scipy')
            pink_noise = generate_pink_noise(downsampled_signal.shape[0], TARGET_FREQ)  # called each time, cause we want "random" noise
            noisy_signal = normalize_audio(apply_noise(downsampled_signal, pink_noise, snr_level=SNR_LEVEL))

            # derive spectrograms
            clean_spectrogram = librosa.stft(downsampled_signal, n_fft=256, hop_length=64, win_length=256, window="hamming")
            noisy_spectrogram = librosa.stft(noisy_signal, n_fft=256, hop_length=64, win_length=256, window="hamming")

            # eight noisy consecutive frames are used to reconstruct first frame
            assert clean_spectrogram.shape == noisy_spectrogram.shape
            clear_input.append(abs(clean_spectrogram).astype('float32'))
            noisy_input.append(abs(noisy_spectrogram).astype('float32'))

        except ValueError as e:
            print("Concatenation failed, terminating dataset generation!")
            raise e

        except AssertionError as e:
            print("Mismatch in spectrogram shapes, something went wrong... termintaing dataset generation!")
            # mismatch in shapes must be caught
            raise e

        except Exception as e:
            pass

    clean_array = np.concatenate(clear_input, axis=1)
    noisy_array = np.concatenate(noisy_input, axis=1)
    print("Processing finished! Saving dataset...")

    from datetime import datetime
    savefile = os.path.join("..\\data\\processed\\", datetime.now().strftime("%Y-%m-%d_%H-%M"))
    clean_data, noisy_data = os.path.join(savefile, "clean.npy"), os.path.join(savefile, "noisy.npy")
    os.mkdir(savefile)

    for arr, save in zip((clean_array.T, noisy_array.T), (clean_data, noisy_data)):
        with open(save, "wb") as f:
            np.save(f, arr)

    print("Done!")