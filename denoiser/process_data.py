"""
Module for preparing data as input for Neural Network.

Output data will a 2D ndarray(shape=(features, window_count)) that contains concatenated frames of short-time fourier
transform. Each index of noisy data corrseponds to each index of clean data.
"""
import json
import os
import random

import numpy as np
from time import perf_counter
import librosa
from datetime import datetime
import pickle
from typing import Union
import pandas as pd
from scipy.io import wavfile

from denoiser.config import (DATA_STATS, SNR_LEVEL, SAMPLING_FREQ, CLIPS_DURATIONS, DATA_ROOT, FRAMES_LENGTH,
                             FEATURES_COUNT, TEST_DATA_RAW, VALIDATE_DATA_RAW, TRAIN_DATA_RAW)


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
    """
    Applies noise signal at specified SNR_db level

    Parameters:
        clear_signal (1D-ndarray) : Clear audio signal
        noise_signal (1D-ndarray) : Noisy audio signal of any noise type
        snr_level (int | float) : Signal to noise ration in decibels

    Raises:
        ValueError: If input signals are not equal in length(size)

    Returns:
        1D-ndarray : Noisy signal scaled to desired SNR_db level
    """
    if not clear_signal.shape == noise_signal.shape:
        raise ValueError("Signal shape does not match provided noise shape")

    """
    Calculating scaling factor for noise signal
    
    SNR_db = 10log[BASE10](P_S / P_N)       where P_S is signal power and P_N is properly scaled noise power.
    P_N = P_S / 10^(SNR_db /10)             P_N is power of properly scaled noise signal
    RMS_S = sqrt( sum( sample^2 for sample in S ) / len(S) )
    
    Signal power is proportionate to the square of its Root Mean Square, since both clear signal and noisy signal are
    the same in length, we can ditch dividing by sample count.
    
    let:
    'n' be noise to be scaled
    'N' be properly scaled noise
    'S' be clear, not noisy signal
    
    assume that:
    len(n) == len(N) == len(S)
    
    with that:
    P_N ~ RMS_N ^ 2
    P_N = 1 / len(N) * sum(sample^2 for sample in N)
    P_N = 1 / len(n) * sum( (factor * sample)^2 for sample in n)
    
    with N = factor * n:
    P_N = 1 / len(N) * sum( (factor * sample)^2 for sample in n)
    P_N = factor^2 / len(N) * sum(sample^2 for sample in n)
    factor = sqrt( P_N * len(n) / sum(sample^2 for sample in n) )
    factor = sqrt(P_N * len(n)) / sqrt(sum(sample^2 for sample in n))
    
    substituting P_N from SNR_db
    factor = sqrt( P_S / 10^(SNR_db / 10) * len(n)) / sqrt(sum(sample^2 for sample in n))
    
    with P_S ~ RMS_S ^ 2 = 1 / len(S) * sum( sample^2 for sample in S)
    factor = sqrt( 1 / 10^(SNR_db / 10) ) * sqrt ( sum(sample^2 for sample in S) / sum(sample^2 for sample in n) )
    """
    signal_sum = np.sum(np.square(clear_signal))
    noise_sum = np.sum(np.square(noise_signal))
    if signal_sum == 0:
        return noise_signal

    snr_factor = np.sqrt(1 / 10 ** (snr_level / 10))
    noise_factor = snr_factor * np.sqrt(signal_sum / noise_sum)
    scaled_noise = noise_factor * noise_signal

    return scaled_noise + clear_signal

def normalize_audio(signal):
    """Normalises audio to [-1;1] value range. Used to limit the amplitude of audio after applying noise.

    Parameters:
        signal (1D-ndarray) : Audio signal

    Returns:
        (1D-ndarray) same audio signal but limited to [-1, 1] in value.
    """
    scaling_factor = max(abs(signal))
    if scaling_factor == 0:  # empty recordings
        return signal
    else:
        return signal / max(abs(signal))

def derive_features(
        audio_paths: list[str],
        snr_level: Union[int, float]=SNR_LEVEL,
        include_phase: bool=False,
        include_noisy: bool=False
):
    clean_data_magnitude = []
    noisy_data_magnitude = []
    phase_data = []  # Phase data from noisy recording can be used to deconstruct spectrogram into audio
    noisy_recs = []

    # Process loop
    rec_no = len(audio_paths)
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
            clean_signal, sr = librosa.load(audio_paths[recording_no])
            downsampled_signal = librosa.resample(clean_signal, orig_sr=sr, target_sr=SAMPLING_FREQ, res_type='scipy')
            pink_noise = generate_pink_noise(downsampled_signal.shape[0], SAMPLING_FREQ)  # called each time, cause we want "random" noise
            noisy_signal = apply_noise(downsampled_signal, pink_noise, snr_level=snr_level)
            if include_noisy:
                noisy_recs.append(noisy_signal)

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
            print("\nConcatenation failed, terminating dataset generation!")
            raise e

        except AssertionError as e:
            print("\nMismatch in spectrogram shapes, something went wrong... termintaing dataset generation!")
            # mismatch in shapes must be caught
            raise e

        except Exception as e:
            print(f"\nException happened at {recording_no}, file_name={audio_paths[recording_no]}, continuing...\n", e, end='\n')
    print("\nProcessing finished!")
    return (
        clean_data_magnitude,
        noisy_data_magnitude,
        phase_data if include_phase else None,
        noisy_recs if include_noisy else None
    )

def get_feature_stats(data: list[np.ndarray], funcs: dict[str, callable]=DATA_STATS) -> dict[str, float]:
    """
    Gets features global values, namely mean, standard deviation, sample count and data shape

    Paramteres:
        data (list[2D -ndarray]) : list of ndarrays to get global stats from
        funcs (dict[str, callable]): specifies the number of recordings to be used for feature extraction. If set to
            None, every .mp3 file inside dataset directory are used and converted.

    Returns:
        (dict[str, float]) : dictionary containg numeric value output for each of statistic function
    """
    data_len = sum([arr.shape[1] - FRAMES_LENGTH + 1 for arr in data])  # input data will be windowed
    data_shape = (FEATURES_COUNT, data_len)
    stats = {"sample_count": data_len, "data_shape": data_shape}

    flatten_data = np.concatenate([a.flatten() for a in data])
    for stat, func in funcs.items():
        stats[stat] = func(flatten_data)

    return stats

def save_metadata(metadata: dict, path: str, readable: bool=True):
    with open(os.path.join(path, "metadata.pkl"), 'wb') as file:  # binary form as json lib does not like numpy types
        pickle.dump(metadata, file)

    # readable form
    if readable:
        with open(os.path.join(path, "metadata.txt"), 'w') as readable_file:
            json.dump({k:(float(v) if isinstance(v, np.float32) else v) for k, v in metadata.items()}, readable_file)


def prepare_dataset(
        data_pth: str=None,
        rec_no: int=None,
        snr_level: Union[int, float]=SNR_LEVEL,
        data_notes: str=None,
) -> None:
    """Converts dataset directory meeting criteria of having .tsv file describing audio recordings, with said recordings
    as .wav files stored inside nested directory 'clips'. For each audio recording a 2D-ndarray of spectrogram magnitude
     data is derived, these arrays are then saved as pickled list od ndarrays, as they are mismatched in length.

    Parameters:
        data_pth (str): provides the data directory, which meets following criteria: recordings metadata is stored in
            .tsv file, audio recordings are located in 'clips' directory, each must have file extension of .wav
        rec_no (int): specifies the number of recordings to be used for feature extraction. If set to None, every .wav
            file inside dataset directory are used and converted.
        snr_level (int | float): sets signal-to-noise rato in dB value for noisy recording. Regardless of specified SNR,
            noisy audio is limited to [-1, 1] range. Defaults to value specified in denoiser.config.

    Returns:
        (None): Saves data as bytedata of list of spectrogram noisy and clear data (2D-ndarray) into processed data
            directory.

    Notes:
        * Normalising noisy signal has no influence on derived features, as magnitude information
    """
    if data_notes is None:
        raise TypeError("Data notes field must be filled in, if you don't want to provide notes, pass empty string!")

    if data_pth is None:
        raise ValueError("Please provide path to dataset!")

    metadata_ = {"desc" : data_notes}

    tsv_file = [file for file in os.listdir(data_pth) if file.endswith('.tsv')][0]
    clips_data = pd.read_csv(os.path.join(data_pth, tsv_file), sep='\t')

    # get specified number of recordings, or all recordings if number not specified
    audio_list = clips_data["path"][0:rec_no].to_list() if rec_no else clips_data["path"].to_list()
    audio_files = [os.path.join(data_pth, "clips", file + ".wav") for file in audio_list]

    clean_data_magnitude, noisy_data_magnitude, _, _ = derive_features(audio_paths=audio_files, snr_level=snr_level)

    print("Creating dataset directory...")
    savefile = os.path.join(DATA_ROOT, "processed", datetime.now().strftime("%Y-%m-%d_%H-%M"))
    clean_data, noisy_data = os.path.join(savefile, "clean.pkl"), os.path.join(savefile, "noisy.pkl")
    os.mkdir(savefile)

    print("Normalising and saving data...")
    for arr, save in zip((clean_data_magnitude, noisy_data_magnitude), (clean_data, noisy_data)):
        data_type = os.path.basename(save).split('.')[0]
        print(f"Processing {data_type} data...", end="\t")

        stats = get_feature_stats(arr)
        metadata_.update(
            {data_type + "_" + key: val for key, val in stats.items()}
        )

        print("normalising...", end="\t")
        norm_arr = [(a - stats["mean"]) / stats["std"] for a in arr]

        print("saving...", end="\t")
        with open(save, "wb") as file:
            pickle.dump(norm_arr, file)
        print("Done!")

    print("Saving metadata...")
    save_metadata(metadata_, savefile, readable=True)
    print("Data processing finished!")

def process_data(
        path: str=None,
        snr_level: Union[int, float]=SNR_LEVEL,
):
    audio_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".mp3")]
    _, noisy_data_magnitude, phase_data, noisy_recs = derive_features(
        audio_paths=audio_files,
        snr_level=snr_level,
        include_phase=True,
        include_noisy=True
    )

    print("Saving processed data!")
    for f_idx in range(len(audio_files)):
        file_path = audio_files[f_idx].split('.')[0]  # path with file name, but no extension
        for arr, save_suffix in zip(
                (noisy_data_magnitude[f_idx], phase_data[f_idx])
                , ("_noisy_mag.npy", "_phase.npy")
        ):
            with open(file_path + save_suffix, "wb") as fs:
                np.save(fs, arr)

            # Win11 does not support float encoding...
            rec = np.iinfo(np.int16).max * noisy_recs[f_idx]
            wavfile.write(file_path + ".wav", rate=SAMPLING_FREQ, data=rec.astype(np.int16))

    print("Data processing finished!")


def __legacy_prepare_data(
        include_phase: bool=True,
        save_noisy: bool=False,
        path: str=None,
        rec_no: int=None,
        snr_level: Union[int, float]=SNR_LEVEL,
        data_notes: str=None,
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
    if data_notes is None:
        raise TypeError("Data notes field must be filled in, if you don't want to provide notes, pass empty string!")
    metadata_ = {"desc" : data_notes}

    # load audio and prepare data containers
    if path is None:
        clips_data = pd.read_csv(CLIPS_DURATIONS, sep='\t')

        # get specified number of recordings, or all recordings if number not specified
        audio_list = clips_data["clip"][0:rec_no].to_list() if rec_no else clips_data["clip"].to_list()
        audio_files = [os.path.join(RAW_DATA, file) for file in audio_list]
        audio_times = clips_data["duration[ms]"][:rec_no] if rec_no else clips_data["duration[ms]"]
        audio_time = int(audio_times.sum()) // (1000 * 60)  # np.int64 is not JSON serializable
        metadata_.update({"hours": audio_time // 60, "minutes": audio_time % 60})

    else:  # data for inference
        audio_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".mp3")]

    clean_data_magnitude = []
    noisy_data_magnitude = []
    phase_data = []  # Phase data from noisy recording can be used to deconstruct spectrogram into audio
    noisy_recs = []

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
            if save_noisy:
                noisy_recs.append(noisy_signal)

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
            print("\nConcatenation failed, terminating dataset generation!")
            raise e

        except AssertionError as e:
            print("\nMismatch in spectrogram shapes, something went wrong... termintaing dataset generation!")
            # mismatch in shapes must be caught
            raise e

        except Exception as e:
            print("\nException happened, continuing...", e, end='\n')

    print(f"\nProcessing finished!\nNormalising and saving data...")
    savefile = os.path.join(DATA_ROOT, "processed", datetime.now().strftime("%Y-%m-%d_%H-%M")) if path is None else path
    metadata_["sample_count"] = sum([a.shape[1] - FRAMES_LENGTH + 1 for a in clean_data_magnitude])

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

        data_shape = len(clean_data_magnitude), FEATURES_COUNT, sum([a.shape[1] for a in clean_data_magnitude])
        metadata_["arrays_shape"] = data_shape

    else:  # Inference data preparation
        for f_idx in range(len(audio_files)):
            file_path = audio_files[f_idx].split('.')[0]  # path with file name, but no extension
            for arr, save_suffix in zip(
                    (noisy_data_magnitude[f_idx], phase_data[f_idx]),
                    ("_noisy_mag.npy", "_phase.npy")
            ):
                with open(file_path + save_suffix, "wb") as fs:
                    np.save(fs, arr)

            if save_noisy:
                wavfile.write(file_path + ".wav", rate=SAMPLING_FREQ, data=noisy_recs[f_idx])

    print("Saving metadata...", end="\n")
    with open(os.path.join(savefile, "metadata.pkl"), 'wb') as file:  # binary form as json lib does not like numpy types
        pickle.dump(metadata_, file)

    # readable form
    with open(os.path.join(savefile, "metadata.txt"), 'w') as readable_file:
        json.dump({k:(float(v) if isinstance(v, np.float32) else v) for k, v in metadata_.items()}, readable_file)

    print("Data processing finished!")

if "__main__" == __name__:
    pass