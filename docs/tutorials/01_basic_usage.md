# Tutorial 1: Basic Usage

This tutorial will guide you through the basic usage of the Collatz AI Research System.

## Prerequisites

- Python 3.11 or higher
- NVIDIA GPU with CUDA support (recommended)
- 16GB+ RAM

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/collatz-ai.git
cd collatz-ai
```

### Step 2: Run Setup Script

The `run.sh` script handles everything:

```bash
chmod +x run.sh
./run.sh
```

This will:
1. Create a Python virtual environment
2. Install all dependencies
3. Compile C++ modules
4. Start training

### Step 3: Verify Installation

Check that C++ modules compiled successfully:

```bash
ls src/*.so
```

You should see:
- `libcollatz.so` - Data generation engine
- `libloop_searcher.so` - Loop detection engine

## Basic Training

### Starting Training

Simply run:

```bash
./run.sh
```

You'll see output like:

```
Step 100/1000000 | Loss: 2.3456 | Stop: 1.2345 | Seq: 1.1111 | Time: 27.3s
```

### Interactive Commands

While training, you can type:

- `stop` - Save checkpoint and exit gracefully
- `status` - Display current training progress
- `Ctrl+C` - Graceful shutdown

### Understanding the Output

- **Step**: Current training step
- **Loss**: Combined loss (lower is better)
- **Stop**: Stopping time prediction loss
- **Seq**: Sequence prediction loss
- **Time**: Time per 100 steps

## Using the Model Programmatically

### Loading a Trained Model

```python
import torch
from src.model import CollatzTransformer

# Initialize model
model = CollatzTransformer(d_model=128, nhead=4, num_layers=4)

# Load checkpoint
checkpoint = torch.load("checkpoints/checkpoint_step_100000.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Making Predictions

```python
from src.engine import get_parity_vector
import torch

# Generate parity vector for a number
n = 27
parity_vec, length = get_parity_vector(n, max_len=500)

# Convert to tensor
parity_tensor = torch.tensor(parity_vec, dtype=torch.long).unsqueeze(0)
parity_tensor = parity_tensor.to(device)

# Make prediction
with torch.no_grad():
    stopping_time_pred, next_step_logits = model(parity_tensor)

print(f"Predicted stopping time: {stopping_time_pred.item():.2f}")
print(f"Actual stopping time: {length}")
```

## Generating Data

### Using the Python Engine

```python
from src.engine import generate_batch_data

# Generate data for numbers 1000-1100
numbers, stopping_times, parity_vectors = generate_batch_data(
    start=1000,
    end=1100,
    max_len=500
)

print(f"Generated {len(numbers)} sequences")
print(f"Stopping times: {stopping_times[:10]}")
```

### Using the C++ Engine (Faster)

```python
from src.native_engine import generate_batch_native, is_native_available

if is_native_available():
    numbers, stopping_times, parity_vectors = generate_batch_native(
        start=1000,
        end=1100,
        max_len=500
    )
    print("Using native C++ engine (20-30% faster)")
else:
    print("Native engine not available, using Python")
```

## Analyzing Results

### Analyzing a Range of Numbers

```python
from src.analyze import analyze_range

results = analyze_range(
    model=model,
    start=1,
    end=10000,
    device=device
)

print(f"Mean error: {results['mean_error']:.4f}")
print(f"Max error: {results['max_error']}")
print(f"Anomalies found: {len(results['anomalies'])}")
```

## Checkpoints

Checkpoints are automatically saved to `checkpoints/` every 10,000 steps.

### Checkpoint Contents

Each checkpoint contains:
- `model_state_dict` - Model weights
- `optimizer_state_dict` - Optimizer state
- `scheduler_state_dict` - Learning rate scheduler state
- `step` - Current training step
- `loss` - Current loss value

### Resuming from Checkpoint

Training automatically resumes from the latest checkpoint if available.

To start fresh:

```bash
rm -rf checkpoints/*
./run.sh
```

## Next Steps

- [Tutorial 2: Custom Training](02_custom_training.md) - Customize training parameters
- [Tutorial 3: Analyzing Results](03_analyzing_results.md) - Deep dive into analysis tools
- [API Documentation](../API.md) - Full API reference

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in `config.yaml`:

```yaml
training:
  batch_size: 256  # Reduce from 512
```

### C++ Compilation Errors

Ensure you have g++ installed:

```bash
sudo apt-get install build-essential g++
```

### Import Errors

Make sure you're in the virtual environment:

```bash
source venv/bin/activate
```
