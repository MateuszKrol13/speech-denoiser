"""
Module of commonly used functions for signal processing used across whole project
"""
import numpy as np
from denoiser.config import DATA_STATS, FRAMES_LENGTH, FEATURES_COUNT
from denoiser.data import FeatureSet

def z_normalize(features_list, stats):
    return [(arr - stats["mean"]) / stats["std"] for arr in features_list]

def min_max_normalize(features_list, stats, ubound: int, lbound:int):
    return[((arr - stats["min"]) * (ubound - lbound) / (stats["max"] - stats["min"])) + lbound for arr in features_list ]

def generate_pink_noise(n_samples:int, sample_rate:int):
    """
    Generates pink noise at a random power level. Must be rescaled later on.
    """
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