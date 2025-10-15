from tensorflow import keras
import numpy as np
from typing import Any
from numpy.typing import DTypeLike

class SequenceLoader(keras.utils.Sequence):
    def __init__(
            self,
            x_noisy: list[np.ndarray[np.float32]],
            y_clean: list[np.ndarray[np.float32]],
            window_size: int,
            batch_size: int,
            shuffle: bool=True,
    ):
        self.x = x_noisy
        self.y = y_clean
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = self.__set_indexes()
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __set_indexes(self):
        # check dims -> outer list
        if len(self.x) != len(self.y):
            raise IndexError(f"Length of x={len(self.x)} does not match len of y={len(self.y)}")
        # inner spectrogram dims
        for elem in range(len(self.x)):
            if self.x[elem].shape != self.y[elem].shape:
                raise IndexError(
                    f"Spectrogram shape of noisy data x={self.x[elem].shape} does not correspond to the shape"
                    f" of clean data y={self.y[elem].shape}"
                )

        # (rec_count, 129, var_length) for both data
        # last axis is for frames, we need to limit the last axis by window length - 1 for noisy data
        from itertools import product, chain
        frames_for_each_idx = [
            list(product([idx], range(0, x.shape[-1] - self.window_size + 1))) for idx, x in enumerate(self.x)
        ]
        return list(chain.from_iterable(frames_for_each_idx))  # unpacked list of lists

    def __getitem__(self, index):
        # idx[0] -> spectrogram_no, idx[1] -> frame_no
        batch_indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]
        y_batch = np.stack(
            [self.y[spectrogram_idx][:, frame_idx] for spectrogram_idx, frame_idx in batch_indexes],
            axis=0
        )
        x_batch = np.stack(
            [self.x[spectrogram_idx][:, frame_idx:frame_idx+self.window_size] for spectrogram_idx, frame_idx in batch_indexes],
            axis=0
        )
        return x_batch, y_batch

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

class MemmapLoader(keras.utils.Sequence):
    def __init__(
            self,
            x_noisy: np.memmap[Any, np.float32],
            y_clean: np.memmap[Any, np.float32],
            batch_size: int,
            idx_list: list[int]=None,
            shuffle: bool=True,
    ):
        """

        :param x_noisy:
        :param y_clean:
        :param batch_size:
        :param idx_list: specifies which indexes can be accessed by
        :param shuffle:
        """
        self.x = x_noisy
        self.y = y_clean
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = [idx for idx in range(self.x.shape[0])]
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]
        return self.x[batch_indexes, :, :], self.y[batch_indexes, :]

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
