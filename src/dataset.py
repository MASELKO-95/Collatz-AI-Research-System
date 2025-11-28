import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
from .engine import generate_batch_data
from .native_engine import generate_hard_batch_native
import random

class CollatzIterableDataset(IterableDataset):
    def __init__(self, start_n=10, batch_size=1024, max_len=500, hard_mode_prob=0.5):
        super(CollatzIterableDataset).__init__()
        self.start_n = start_n
        self.batch_size = batch_size
        self.max_len = max_len
        self.hard_mode_prob = hard_mode_prob
        
    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Single process
            current_n = self.start_n
        else:
            # Multi-process
            # Assign each worker a huge block to avoid overlap in "normal" generation
            per_worker = 1000000000 # 1 billion
            current_n = self.start_n + worker_info.id * per_worker
            
        while True:
            # Decide strategy for this chunk
            if random.random() < self.hard_mode_prob:
                # Generate Hard Candidates (Native C++ Speed)
                try:
                    numbers, stopping_times, parity_vectors = generate_hard_batch_native(
                        self.batch_size, self.max_len
                    )
                except Exception as e:
                    # Fallback if native fails
                    print(f"Native engine failed: {e}")
                    from .engine import generate_hard_candidates
                    numbers, stopping_times, parity_vectors = generate_hard_candidates(
                        self.batch_size, self.max_len
                    )
            else:
                # Generate Normal Sequence (Numba optimized)
                end_n = current_n + self.batch_size
                numbers, stopping_times, parity_vectors = generate_batch_data(
                    current_n, end_n, self.max_len
                )
                current_n = end_n
            
            for i in range(len(numbers)):
                p_vec = parity_vectors[i].copy()
                p_vec[p_vec == -1] = 2
                
                yield {
                    "number": numbers[i],
                    "stopping_time": stopping_times[i],
                    "parity_vector": torch.tensor(p_vec, dtype=torch.long)
                }

def collate_fn(batch):
    numbers = torch.tensor([item['number'] for item in batch], dtype=torch.float32).unsqueeze(1)
    stopping_times = torch.tensor([item['stopping_time'] for item in batch], dtype=torch.float32).unsqueeze(1)
    parity_vectors = torch.stack([item['parity_vector'] for item in batch])
    
    return numbers, parity_vectors, stopping_times
