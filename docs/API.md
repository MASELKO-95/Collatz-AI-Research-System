# Collatz AI API Documentation

This document provides detailed API documentation for the Collatz AI Research System modules.

## Table of Contents

- [Model Architecture](#model-architecture)
- [Data Generation Engine](#data-generation-engine)
- [Dataset and DataLoader](#dataset-and-dataloader)
- [Native C++ Engine](#native-c-engine)
- [Training Module](#training-module)
- [Analysis Tools](#analysis-tools)

---

## Model Architecture

### `model.py`

#### `PositionalEncoding`

Adds positional information to sequence embeddings using sinusoidal functions.

**Parameters:**
- `d_model` (int): Dimension of the model embeddings
- `max_len` (int, default=5000): Maximum sequence length

**Methods:**
- `forward(x)`: Adds positional encoding to input tensor
  - **Args:** `x` (Tensor): Input tensor of shape `[seq_len, batch, d_model]`
  - **Returns:** Tensor with positional encoding added

#### `CollatzTransformer`

Transformer-based model for Collatz sequence prediction.

**Parameters:**
- `d_model` (int, default=128): Model dimension
- `nhead` (int, default=4): Number of attention heads
- `num_layers` (int, default=4): Number of transformer encoder layers
- `max_len` (int, default=500): Maximum sequence length

**Architecture:**
- Embedding layer: Maps parity values (0, 1, 2) to `d_model` dimensions
- Positional encoding: Adds position information
- Transformer encoder: `num_layers` layers with `nhead` attention heads
- Dual prediction heads:
  - Stopping time head: Predicts total steps to reach 1
  - Next step head: Predicts next parity bit in sequence

**Methods:**
- `forward(src, src_key_padding_mask=None)`: Forward pass
  - **Args:**
    - `src` (Tensor): Parity vectors of shape `[batch, seq_len]`
    - `src_key_padding_mask` (Tensor, optional): Padding mask `[batch, seq_len]`
  - **Returns:** Tuple of `(stopping_time_pred, next_step_logits)`
    - `stopping_time_pred`: Shape `[batch, 1]`
    - `next_step_logits`: Shape `[batch, seq_len, 3]`

**Example:**
```python
from model import CollatzTransformer
import torch

model = CollatzTransformer(d_model=128, nhead=4, num_layers=4)
src = torch.randint(0, 2, (32, 100))  # Batch of 32, seq_len 100
stopping_time, next_step = model(src)
```

---

## Data Generation Engine

### `engine.py`

Numba-optimized functions for generating Collatz sequence data.

#### `next_collatz(n)`

Computes the next number in the Collatz sequence.

**Args:**
- `n` (int): Current number

**Returns:**
- int: Next number (n/2 if even, 3n+1 if odd)

#### `get_stopping_time(n)`

Calculates the stopping time for a number.

**Args:**
- `n` (int): Starting number

**Returns:**
- int: Number of steps to reach 1

**Example:**
```python
from engine import get_stopping_time

time = get_stopping_time(27)  # Returns 111
```

#### `get_parity_vector(n, max_len=1000)`

Generates the parity vector for a Collatz sequence.

**Args:**
- `n` (int): Starting number
- `max_len` (int): Maximum vector length

**Returns:**
- Tuple of `(vector, actual_length)`
  - `vector`: NumPy array of shape `[max_len]` with values 0 (even), 1 (odd), -1 (padding)
  - `actual_length`: Actual sequence length before padding

#### `generate_batch_data(start, end, max_len=500)`

Generates batch data for a range of numbers (parallelized with Numba).

**Args:**
- `start` (int): Start of range (inclusive)
- `end` (int): End of range (exclusive)
- `max_len` (int): Maximum sequence length

**Returns:**
- Tuple of `(numbers, stopping_times, parity_vectors)`
  - `numbers`: Array of shape `[batch_size]`
  - `stopping_times`: Array of shape `[batch_size]`
  - `parity_vectors`: Array of shape `[batch_size, max_len]`

#### `generate_hard_candidates(batch_size, max_len=500)`

Generates challenging candidates (n > 2^68) using various strategies.

**Args:**
- `batch_size` (int): Number of candidates to generate
- `max_len` (int): Maximum sequence length

**Returns:**
- Same as `generate_batch_data`

---

## Dataset and DataLoader

### `dataset.py`

#### `CollatzIterableDataset`

PyTorch IterableDataset for streaming Collatz sequence data.

**Parameters:**
- `start_range` (int): Start of number range
- `end_range` (int): End of number range
- `batch_size` (int): Batch size
- `max_len` (int, default=500): Maximum sequence length
- `use_native` (bool, default=True): Use C++ engine if available
- `hard_mode_ratio` (float, default=0.5): Ratio of hard candidates

**Methods:**
- `__iter__()`: Returns iterator yielding batches

**Example:**
```python
from dataset import CollatzIterableDataset
from torch.utils.data import DataLoader

dataset = CollatzIterableDataset(
    start_range=1000,
    end_range=100000,
    batch_size=512,
    max_len=500
)

dataloader = DataLoader(dataset, batch_size=None, num_workers=4)

for batch in dataloader:
    parity_vectors = batch["parity_vector"]
    stopping_times = batch["stopping_time"]
    # Training code here
```

#### `collate_fn(batch)`

Custom collate function for batching sequences of different lengths.

**Args:**
- `batch` (list): List of samples

**Returns:**
- dict: Batched data with padding

---

## Native C++ Engine

### `native_engine.py`

Python bindings for the C++ data generation engine.

#### `is_native_available()`

Checks if the native C++ library is available.

**Returns:**
- bool: True if available, False otherwise

#### `get_stopping_time_native(n)`

C++ implementation of stopping time calculation.

**Args:**
- `n` (int): Starting number

**Returns:**
- int: Stopping time

#### `generate_batch_native(start, end, max_len)`

C++ implementation of batch data generation (20-30% faster than Python).

**Args:**
- `start` (int): Start of range
- `end` (int): End of range
- `max_len` (int): Maximum sequence length

**Returns:**
- Tuple of `(numbers, stopping_times, parity_vectors)`

---

## Training Module

### `train.py`

Main training script with curriculum learning and mixed precision.

**Key Functions:**

#### `train()`

Main training loop with the following features:
- Mixed precision training (AMP)
- Cosine annealing learning rate schedule
- Gradient clipping
- Automatic checkpointing
- Discord integration for monitoring
- Interactive commands (stop, status)

**Configuration:**
- Modify parameters in `config.yaml` (recommended) or directly in `train.py`

---

## Analysis Tools

### `analyze.py`

Tools for analyzing model predictions and visualizing results.

#### `analyze_range(model, start, end, device)`

Analyzes model predictions for a range of numbers.

**Args:**
- `model`: Trained CollatzTransformer model
- `start` (int): Start of range
- `end` (int): End of range
- `device`: PyTorch device

**Returns:**
- dict: Analysis results including errors, anomalies, and statistics

**Example:**
```python
from analyze import analyze_range
import torch

model = CollatzTransformer()
model.load_state_dict(torch.load("checkpoint.pt"))
device = torch.device("cuda")

results = analyze_range(model, 1, 10000, device)
print(f"Mean error: {results['mean_error']}")
```

---

## Type Annotations

All modules include type hints for better IDE support and static analysis. Use `mypy` for type checking:

```bash
mypy src/ --ignore-missing-imports
```

## Further Reading

- [Tutorials](tutorials/) - Step-by-step guides
- [Examples](examples/) - Code examples
- [README.md](../README.md) - Project overview
