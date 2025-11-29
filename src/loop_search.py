import ctypes
import os
import threading

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), "libloop_searcher.so")
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

    # Define argument types for negative search
    _loop_lib.parallel_loop_search_negative.argtypes = [
        ctypes.c_int,     # num_threads
        ctypes.c_uint64,  # start_high
        ctypes.c_uint64,  # start_low
        ctypes.c_uint64   # numbers_per_thread
    ]
    _loop_lib.parallel_loop_search_negative.restype = None

except OSError:
    print("Warning: libloop_searcher.so not found. Loop searcher disabled.")
    _loop_lib = None

def start_background_loop_search(num_threads=20, start_number=None, numbers_per_thread=1000000):
    """
    Start a background thread that searches for Collatz loops.
    
    Args:
        num_threads: Number of parallel CPU threads to use
        start_number: Starting number (default: 2^68)
        numbers_per_thread: How many numbers each thread checks
    """
    if _loop_lib is None:
        print("Loop searcher not available.")
        return
    
    if start_number is None:
        # Default: start at 2^68
        start_number = 1 << 68
    
    # Split 128-bit number into high/low 64-bit parts
    start_high = start_number >> 64
    start_low = start_number & 0xFFFFFFFFFFFFFFFF
    
    def search_worker():
        print(f"🔍 Background loop search starting with {num_threads} threads...")
        _loop_lib.parallel_loop_search(num_threads, start_high, start_low, numbers_per_thread)
    
    search_thread = threading.Thread(target=search_worker, daemon=True)
    search_thread.start()
    print(f"✅ Background loop search launched!")

def start_background_negative_loop_search(num_threads=20, start_number=None, numbers_per_thread=1000000):
    """
    Start a background thread that searches for NEGATIVE Collatz loops.
    
    Args:
        num_threads: Number of parallel CPU threads to use
        start_number: Starting magnitude (default: 10)
        numbers_per_thread: How many numbers each thread checks
    """
    if _loop_lib is None:
        print("Loop searcher not available.")
        return
    
    if start_number is None:
        start_number = 10
    
    # Split 128-bit number into high/low 64-bit parts
    start_high = start_number >> 64
    start_low = start_number & 0xFFFFFFFFFFFFFFFF
    
    def search_worker():
        print(f"🔍 Background NEGATIVE loop search starting with {num_threads} threads...")
        _loop_lib.parallel_loop_search_negative(num_threads, start_high, start_low, numbers_per_thread)
    
    search_thread = threading.Thread(target=search_worker, daemon=True)
    search_thread.start()
    print(f"✅ Background NEGATIVE loop search launched!")
