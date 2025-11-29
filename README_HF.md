---
language: en
license: gpl-3.0
tags:
- mathematics
- collatz-conjecture
- transformer
- sequence-prediction
- number-theory
datasets:
- synthetic (Collatz sequences)
metrics:
- mae
- accuracy
library_name: pytorch
pipeline_tag: tabular-regression
---

# Collatz AI - Transformer for Collatz Conjecture Analysis

## Model Description

This is a **Transformer-based neural network** trained to predict stopping times and sequence patterns in the [Collatz Conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture).

**Architecture:**
- 4-layer Transformer Encoder
- 128-dimensional embeddings
- 4 attention heads
- Dual output heads (regression + classification)

**Training:**
- 120,305 steps
- Mixed Precision (AMP)
- Cosine Annealing LR
- Curriculum Learning (50% "Hard Mode" data with n > 2^68)

## Performance

| Metric | Value |
|--------|-------|
| Stopping Time MAE | 2.3 steps |
| Log-Space Error | 0.0003 |
| Sequence Accuracy | 70.3% |
| Numbers Analyzed | 120M+ |

## Intended Use

### Primary Use Cases

1. **Mathematical Research**: Analyze Collatz sequence patterns
2. **Anomaly Detection**: Identify numbers with unusual stopping times
3. **Educational**: Demonstrate AI applications in number theory

### Out-of-Scope Use

- This model does NOT prove or disprove the Collatz Conjecture
- Not suitable for production systems requiring guaranteed accuracy
- Not designed for other mathematical conjectures

## How to Use

### Installation

```bash
pip install torch transformers
```

### Inference

```python
import torch
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained("MASELKO-95/collatz-ai")
model.eval()

# Example: Predict stopping time for number 27
# First, generate Collatz sequence (parity vector)
def collatz_sequence(n, max_len=500):
    seq = []
    while n > 1 and len(seq) < max_len:
        if n % 2 == 0:
            seq.append(0)
            n = n // 2
        else:
            seq.append(1)
            n = 3 * n + 1
    # Pad to max_len
    seq += [2] * (max_len - len(seq))
    return seq

parity = collatz_sequence(27)
src = torch.tensor([parity], dtype=torch.long)
src_key_padding_mask = (src == 2)

with torch.no_grad():
    stopping_pred, next_step_logits = model(src, src_key_padding_mask=src_key_padding_mask)
    
# Convert from log-space
stopping_time = torch.expm1(stopping_pred).item()
print(f"Predicted stopping time: {stopping_time:.0f}")
# Actual: 111 steps
```

## Training Data

**Synthetic Data Generation:**
- **Normal Mode** (50%): Sequential numbers 10 to 10^9
- **Hard Mode** (50%): Numbers > 2^68, special patterns (n ≡ 3 mod 4, dense binary)

**Data Augmentation:**
- Random sampling from large number space
- Curriculum learning (easy → hard)
- Multi-worker parallel generation (C++ engine)

## Training Procedure

### Hyperparameters

```python
BATCH_SIZE = 512
LEARNING_RATE = 1e-4 (Cosine Annealing → 1e-6)
OPTIMIZER = AdamW
LOSS = Huber Loss (stopping) + CrossEntropy (sequence)
GRADIENT_CLIPPING = 1.0
MIXED_PRECISION = True (AMP)
```

### Hardware

- **GPU**: NVIDIA RTX 3070 Ti (8GB VRAM)
- **CPU**: AMD Ryzen 5900X (24 threads)
- **Training Time**: ~52 hours (120k steps)

## Evaluation Results

### Stopping Time Prediction

| Number Range | MAE | Median Error |
|--------------|-----|--------------|
| 1-1,000 | 1.2 | 0.3 |
| 1,000-10,000 | 3.5 | 1.1 |
| 10,000-100,000 | 8.7 | 2.8 |
| >100,000 | 15.2 | 5.4 |

### Known Hard Numbers

| Number | Actual | Predicted | Error |
|--------|--------|-----------|-------|
| 27 | 111 | 109 | 2 |
| 703 | 170 | 168 | 2 |
| 1161 | 181 | 183 | 2 |

## Limitations

1. **Sequence Plateau**: Accuracy stuck at ~70%, may need larger model
2. **Large Number Variance**: Higher error for n > 10^9
3. **No Cycle Detection**: Model predicts patterns, doesn't prove convergence
4. **Computational Cost**: Inference requires full sequence generation

## Ethical Considerations

- This model is for **research purposes only**
- Results should be verified mathematically
- Not a substitute for rigorous mathematical proof
- Open-source (GPL v3) to encourage collaboration

## Citation

```bibtex
@misc{collatz-ai-2025,
  author = {MASELKO-95},
  title = {Collatz AI: Transformer-based Analysis of the Collatz Conjecture},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/MASELKO-95/collatz-ai}}
}
```

## Model Card Authors

MASELKO-95

## Model Card Contact

- GitHub: [Collatz-AI-Research-System](https://github.com/MASELKO-95/Collatz-AI-Research-System)
- Issues: [GitHub Issues](https://github.com/MASELKO-95/Collatz-AI-Research-System/issues)

---

**License**: GNU General Public License v3.0

**Last Updated**: 2025-11-28
