import ctypes
import os
import sys

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), "src", "libloop_searcher.so")
try:
    _loop_lib = ctypes.CDLL(lib_path)
    
    # Define argument types
    _loop_lib.parallel_loop_search.argtypes = [
        ctypes.c_int,     # num_threads
        ctypes.c_uint64,  # start_high
        ctypes.c_uint64,  # start_low
        ctypes.c_uint64   # numbers_per_thread
    ]
    _loop_lib.parallel_loop_search.restype = None

    print(f"Successfully loaded library from {lib_path}")
    
    # Run a small search
    print("Running small search...")
    # Start at 2^68
    start_number = 1 << 68
    start_high = start_number >> 64
    start_low = start_number & 0xFFFFFFFFFFFFFFFF
    
    _loop_lib.parallel_loop_search(2, start_high, start_low, 1000)
    print("Search completed successfully.")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
