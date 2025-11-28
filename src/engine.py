import numpy as np
from numba import jit, prange
import random

@jit(nopython=True)
def next_collatz(n):
    if n % 2 == 0:
        return n // 2
    else:
        return 3 * n + 1

@jit(nopython=True)
def detect_cycle(n, max_steps=10000):
    """
    Floyd's Tortoise and Hare algorithm to detect cycles.
    Returns:
        0: Converged to 1
        1: Cycle detected (other than 4-2-1 if we filter)
        2: Max steps reached (possible divergence or very long cycle)
    """
    tortoise = n
    hare = n
    steps = 0
    
    while steps < max_steps:
        if tortoise == 1 or hare == 1:
            return 0 # Converged
            
        # Tortoise moves 1 step
        if tortoise % 2 == 0:
            tortoise //= 2
        else:
            tortoise = 3 * tortoise + 1
            
        # Hare moves 2 steps
        if hare % 2 == 0:
            hare //= 2
        else:
            hare = 3 * hare + 1
            
        if hare == 1:
            return 0
            
        if hare % 2 == 0:
            hare //= 2
        else:
            hare = 3 * hare + 1
            
        if tortoise == hare:
            # Cycle detected!
            # Check if it's the trivial 1-2-4 loop (which we shouldn't reach if we stop at 1, but just in case)
            if tortoise == 1 or tortoise == 2 or tortoise == 4:
                return 0
            return 1 # Non-trivial cycle found!
            
        steps += 1
        
    return 2 # Limit reached

@jit(nopython=True)
def get_stopping_time(n):
    steps = 0
    while n > 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
        if steps > 20000: # Increased limit
            return steps
    return steps

@jit(nopython=True)
def get_parity_vector(n, max_len=1000):
    vector = np.full(max_len, -1, dtype=np.int8)
    idx = 0
    while n > 1 and idx < max_len:
        if n % 2 == 0:
            vector[idx] = 0
            n = n // 2
        else:
            vector[idx] = 1
            n = 3 * n + 1
        idx += 1
    return vector, idx

@jit(nopython=True, parallel=True)
def generate_batch_data(start, end, max_len=500):
    count = end - start
    numbers = np.arange(start, end, dtype=np.int64)
    stopping_times = np.zeros(count, dtype=np.int32)
    parity_vectors = np.zeros((count, max_len), dtype=np.int8)
    
    for i in prange(count):
        n = numbers[i]
        stopping_times[i] = get_stopping_time(n)
        vec, _ = get_parity_vector(n, max_len)
        parity_vectors[i] = vec
        
    return numbers, stopping_times, parity_vectors

def generate_hard_candidates(batch_size, max_len=500):
    # Use Python objects for arbitrary precision
    numbers = np.zeros(batch_size, dtype=np.object_)
    base_2_68 = (1 << 68)
    
    for i in range(batch_size):
        strategy = random.random()
        if strategy < 0.3:
            offset = random.randint(1, 10**15)
            n = base_2_68 + offset
        elif strategy < 0.6:
            offset = random.randint(1, 10**15)
            n = base_2_68 + offset
            if n % 4 != 3:
                n += (3 - (n % 4))
        elif strategy < 0.8:
            length = random.randint(70, 120)
            n = 0
            for _ in range(length):
                n = (n << 1) | (1 if random.random() < 0.8 else 0)
            if n < base_2_68:
                n += base_2_68
        else:
            n = base_2_68 + random.randint(1, 10000000)
        numbers[i] = n

    stopping_times = np.zeros(batch_size, dtype=np.int32)
    parity_vectors = np.zeros((batch_size, max_len), dtype=np.int8)
    
    for i in range(batch_size):
        n = int(numbers[i])
        
        # Check for cycles explicitly for these hard candidates
        # We can't use Numba for these huge numbers, so we use Python
        # Floyd's algo in Python for huge int
        tortoise = n
        hare = n
        steps = 0
        limit = 5000 # Quick check
        
        # We just generate sequence for model
        curr = n
        vec = np.full(max_len, -1, dtype=np.int8)
        s_time = 0
        
        while curr > 1 and s_time < max_len:
            if curr % 2 == 0:
                vec[s_time] = 0
                curr //= 2
            else:
                vec[s_time] = 1
                curr = 3 * curr + 1
            s_time += 1
            
        # If not stopped, continue counting
        while curr > 1 and s_time < 20000:
             if curr % 2 == 0:
                curr //= 2
             else:
                curr = 3 * curr + 1
             s_time += 1
             
        stopping_times[i] = s_time
        parity_vectors[i] = vec
        
    return numbers, stopping_times, parity_vectors
