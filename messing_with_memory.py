import numpy as np
import ctypes

arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)
array_size = arr.size
pointer_to_array, read_only = arr.__array_interface__["data"]


def save_expected(arr, name):
    np.save(f"expected_{name}.npy", arr)


def compare_to_expected(expected_name, ptr, size):
    expected = np.load(f"expected_{expected_name}.npy")
    ptr = ctypes.cast(pointer_to_array, ctypes.POINTER(ctypes.c_double))
    assert np.all(expected == np.array(ptr[:size]))


save_expected(arr, '')
# arr[2] += 1
compare_to_expected('', pointer_to_array, array_size)
