from numpy.typing import NDArray
from numpy import squeeze

class ShapeError(Exception):
    pass

def validate_clear(arr: NDArray, unstacked_arr: list[NDArray]) -> None:
    # Verify dimension axes
    if arr.shape[1] != 129:
        raise ShapeError(f"NDarray shape is {arr.shape} while expected the second axis shape to be 129!")

    # Assert that first two elements are handled correctly
    first_unstacked = unstacked_arr[0].T[:-7, :]  # all but last seven frames are used in training
    second_unstacked = unstacked_arr[1].T[:-7, :]
    first_end_idx = first_unstacked.shape[0]
    second_end_idx = second_unstacked.shape[0] + first_end_idx
    first, second = arr[0:first_end_idx, :], arr[first_end_idx:second_end_idx, :]

    if not ((first == first_unstacked).all() or (second == second_unstacked).all()):
        raise ValueError(f"First two arrays do not match as expected. Stacking procedure failed...")

    # Check dtype
    if not all([arr.dtype == inner_arr.dtype for inner_arr in unstacked_arr]):
        raise TypeError("Data types between jagged list and stacked array are mismatched.")


def validate_noisy(arr: NDArray, unstacked_arr: list[NDArray]) -> None:
    # Verify dimension axes
    if arr.shape[1] != 129:
        raise ShapeError(f"NDarray shape is {arr.shape} while expected the second axis shape to be 129!")

    # Assert that first two elements are handled correctly
    first_unstacked = unstacked_arr[0].T[:-7, :]  # all but last seven frames are used in training
    second_unstacked = unstacked_arr[1].T[:-7, :]
    first_end_idx = first_unstacked.shape[0]
    second_end_idx = second_unstacked.shape[0] + first_end_idx
    # check first frame of window only
    first, second = squeeze(arr[0:first_end_idx, :, 0]), squeeze(arr[first_end_idx:second_end_idx, :, 0])

    if not ((first == first_unstacked).all() or (second == second_unstacked).all()):
        raise ValueError(f"First two arrays do not match as expected. Stacking procedure failed...")

    # Check dtype
    if not all([arr.dtype == inner_arr.dtype for inner_arr in unstacked_arr]):
        raise TypeError("Data types between jagged list and stacked array are mismatched.")

    # TODO: check other frames in window