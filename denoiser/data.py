from enum import Enum
from typing import Optional, Union
from collections import namedtuple
import numpy as np
import librosa
import os
import pickle

from denoiser.config import SAMPLING_FREQ, SNR_LEVEL
from denoiser import apply_noise, generate_pink_noise, get_feature_stats

FeatureSet = namedtuple("FeatureSet", ["features","metadata"])

class SignalType(Enum):
    SOURCE = "source"
    TARGET = "target"

class FeatureType(Enum):
    SPECTROGRAM = "spectrogram"
    MAGNITUDE = "magnitude"
    PHASE = "phase"
    WAVEFORM = "waveform"

class Signal(object):
    spectrogram = None

    def __init__(self, signal, sample_rate):
        self.waveform = signal
        self.freq = sample_rate

    def __len__(self):
        return len(self.waveform)

    def _calculate_spectrogram(self):
        self.spectrogram = librosa.stft(self.waveform, n_fft=256, hop_length=64, win_length=256, window="hamming")

    def get_spectrogram(self):
        if self.spectrogram is None:
            self._calculate_spectrogram()
        return self.spectrogram

    def get_magnitude(self):
        if self.spectrogram is None:
            self._calculate_spectrogram()
        return np.abs(self.spectrogram).astype('float32')

    def get_phase(self):
        if self.spectrogram is None:
            self._calculate_spectrogram()
        return np.angle(self.spectrogram).astype('float32')

    def get_waveform(self):
        return self.waveform

    @classmethod
    def from_file(cls, abs_path, sample_rate: int=SAMPLING_FREQ):
        sig = cls(*librosa.load(abs_path, sr=sample_rate))
        sig._calculate_spectrogram()
        return sig


class Data(object):
    target = None
    source = None

    def __init__(self, path):
        self.rec_name = path

    def __iter__(self):
        return iter((self.target, self.source))

    def load_signal(self, root_path: Optional[str] = ""):
        self.target = Signal(*librosa.load(os.path.join(root_path, self.rec_name), sr=SAMPLING_FREQ))

    def load_features(self):
        raise NotImplementedError("Each signal have features derived upon construction")

    def derive_source(self):
        noise_sig = generate_pink_noise(n_samples=len(self.target), sample_rate=SAMPLING_FREQ)
        noisy_sig = apply_noise(self.target.waveform, noise_sig, snr_level=SNR_LEVEL)
        self.source = Signal(signal=noisy_sig, sample_rate=SAMPLING_FREQ)

    def set_source(self, source: Signal):
        self.source = source

    @classmethod
    def from_signals(cls, *, tar_sig, src_sig):
        data = cls(path=None)
        data.target, data.source = tar_sig, src_sig
        return data


class Dataset(object):

    def __init__(self, path):
        self.path = path
        self._data = [Data(recording) for recording in os.listdir(path) if recording.endswith(".wav")]
        self._loaded = False
        self._metadata = dict()

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def load(self, target_only: bool=False):
        for data in self._data:
            data.load_signal(self.path)
            if not target_only:
                data.derive_source()
        self._loaded = True
        return self

    def extract_feature(self, feature: FeatureType, signal: SignalType, normalise: Union[callable, None]=None):
        features_list = []
        for data_elem in self:
            sig = getattr(data_elem, signal.value)  # get source or target
            features_list.append(getattr(sig, "get_" + feature.value)())  # get correct feature
        stats = get_feature_stats(features_list)
        if normalise is not None:
            features_list = normalise(features_list)

        return FeatureSet(features=features_list, metadata=stats)

    def to_pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        out = cls(path)
        out._data = data
        return out
