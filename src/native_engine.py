import ctypes
import os
import numpy as np
import torch

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), "libcollatz.so")
try:
    _lib = ctypes.CDLL(lib_path)
    
    # Define argument types
    _lib.generate_hard_batch.argtypes = [
        ctypes.c_int, # count
        ctypes.c_int, # max_len
        ctypes.POINTER(ctypes.c_uint64), # out_nums (high, low)
        ctypes.POINTER(ctypes.c_int32),  # out_stops
        ctypes.POINTER(ctypes.c_int8)    # out_parity
    ]
except OSError:
    print("Warning: libcollatz.so not found. Native engine disabled.")
    _lib = None

def generate_hard_batch_native(batch_size, max_len=500):
    if _lib is None:
        raise RuntimeError("Native engine not available")
        
    # Allocate output arrays
    # Numbers are 128-bit, passed as 2x uint64
    out_nums = np.zeros(batch_size * 2, dtype=np.uint64)
    out_stops = np.zeros(batch_size, dtype=np.int32)
    out_parity = np.zeros(batch_size * max_len, dtype=np.int8)
    
    # Call C++ function
    _lib.generate_hard_batch(
        batch_size,
        max_len,
        out_nums.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        out_stops.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_parity.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    )
    
    # Reconstruct Python integers from high/low pairs
    # This is the slow part in Python, but unavoidable if we want PyTorch tensors of numbers
    # But wait, PyTorch doesn't support int128. We convert to float for the model anyway.
    # Let's do the conversion efficiently.
    
    # Reshape parity
    parity_vectors = out_parity.reshape(batch_size, max_len)
    
    # Numbers: (high << 64) | low
    # We can do this vectorized?
    high = out_nums[0::2].astype(object)
    low = out_nums[1::2].astype(object)
    numbers = (high << 64) | low
    
    return numbers, out_stops, parity_vectors
