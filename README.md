# Collatz Conjecture AI Research System

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

> **Advanced AI system for analyzing the Collatz Conjecture using Deep Learning and parallel brute-force search**

## 🎯 Overview

This project combines **AI-guided pattern recognition** with **parallel brute-force search** to investigate the [Collatz Conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture), one of mathematics' most famous unsolved problems.

### Key Features

- 🧠 **Transformer-based Neural Network** for sequence prediction
- 🔍 **Multi-threaded C++ Loop Searcher** (Floyd's Cycle Detection)
- ⚡ **Native C++ Data Engine** for maximum performance
- 📊 **Real-time Discord Integration** for monitoring
- 🎓 **Curriculum Learning** with "Hard Mode" candidates (n > 2^68)
- 🔬 **Advanced Optimizations**: Mixed Precision (AMP), Cosine Annealing LR, Gradient Clipping

## 🚀 Quick Start

### Prerequisites

- **GPU**: NVIDIA GPU with 6GB+ VRAM (tested on RTX 3070 Ti)
- **CPU**: Multi-core processor (tested on Ryzen 5900X)
- **RAM**: 16GB+ recommended
- **OS**: Linux (tested on Arch Linux)
- **Python**: 3.13+
- **CUDA**: 12.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/collatz-ai.git
cd collatz-ai

# Run setup (creates venv, installs dependencies, compiles C++ modules)
chmod +x run.sh
./run.sh
```

### Usage

```bash
# Start training
./run.sh

# Interactive commands during training:
# - Type 'stop' to save and exit
# - Type 'status' for current progress
# - Ctrl+C for graceful shutdown
```

## 🏗️ Architecture

### AI Model (GPU)

```
Input: Parity Vector [0, 1, 0, 1, 1, ...]
  ↓
Embedding Layer (3 → 128d)
  ↓
Positional Encoding
  ↓
Transformer Encoder (4 layers, 4 heads)
  ↓
Dual Heads:
  ├─ Stopping Time Prediction (Regression, Log-Space)
  └─ Next Step Prediction (Classification)
```

**Specifications:**
- Model Size: 128d, 4 layers, 4 attention heads
- Batch Size: 512
- Optimizer: AdamW with Cosine Annealing
- Loss: Huber Loss (stopping time) + CrossEntropy (sequence)

### Loop Searcher (CPU)

```cpp
// Parallel brute-force search using Floyd's algorithm
// 22 threads × 1M numbers = 22M candidates per run
// Target: n > 2^68, n ≡ 3 (mod 4)
```

**Features:**
- Multi-threaded C++ implementation
- 128-bit integer support (`__int128`)
- Detects non-trivial cycles
- Runs in background during training

## 📊 Performance

### Training Metrics (100k steps)

| Metric | Value |
|--------|-------|
| Final Loss | 0.3698 |
| Stopping Time Error | 0.0003 (log-space) |
| Sequence Accuracy | ~70% |
| Training Speed | ~27s / 100 steps |
| GPU Utilization | ~90% (7.2GB / 8GB) |
| CPU Utilization | ~85% (20 workers) |

### Loop Search Results

- **Numbers Checked**: 22,000,000 per run
- **Range**: [2^68, 2^68 + 22M]
- **Non-trivial Cycles Found**: 0 (as expected)

## 🛠️ Technical Details

### Optimizations

1. **Mixed Precision Training (AMP)**
   - Reduces VRAM usage by ~40%
   - Increases training speed by ~30%

2. **Native C++ Engine**
   - 20-30% faster data generation
   - Supports numbers > 2^64 (128-bit)

3. **Curriculum Learning**
   - 50% "Normal" data (sequential numbers)
   - 50% "Hard" data (n > 2^68, special patterns)

4. **Learning Rate Scheduling**
   - Cosine Annealing: 1e-4 → 1e-6
   - Smooth convergence, prevents oscillation

### File Structure

```
collatz_ai/
├── src/
│   ├── train.py              # Main training script
│   ├── model.py              # Transformer architecture
│   ├── engine.py             # Numba-optimized data generation
│   ├── dataset.py            # PyTorch Dataset/DataLoader
│   ├── analyze.py            # Model analysis & visualization
│   ├── discord_bot.py        # Discord webhook integration
│   ├── collatz_core.cpp      # C++ data engine
│   ├── loop_searcher.cpp     # C++ parallel loop searcher
│   ├── native_engine.py      # Python bindings (ctypes)
│   └── loop_search.py        # Loop searcher wrapper
├── checkpoints/              # Model checkpoints (auto-saved)
├── requirements.txt          # Python dependencies
├── run.sh                    # Setup & run script
└── README.md                 # This file
```

## 📈 Results & Insights

### What the Model Learned

1. **Stopping Time Prediction**: Near-perfect accuracy (99.97%)
2. **Parity Patterns**: Strong recognition of even/odd sequences
3. **Anomaly Detection**: Identifies numbers with unusual stopping times

### Known Anomalies

The model struggles with these numbers (unusually short stopping times):

```
Number: 1249, Actual: 176, Predicted: 233, Error: 57
Number: 1695, Actual: 179, Predicted: 236, Error: 57
Number: 1742, Actual: 179, Predicted: 235, Error: 56
```

## 🔬 Research Applications

### For Mathematicians

- Analyze stopping time distributions
- Identify exceptional numbers
- Visualize sequence embeddings (PCA)

### For ML Researchers

- Benchmark for sequence prediction
- Study curriculum learning effects
- Explore transformer behavior on mathematical sequences

### For Collatz Enthusiasts

- Automated large-scale verification
- Pattern discovery in high ranges (> 2^68)
- Real-time progress monitoring via Discord

## 🚧 Future Work

- [ ] Distributed training across multiple GPUs
- [ ] Larger model (256d, 6 layers) from scratch
- [ ] GPU-accelerated loop detection
- [ ] Extended search range (2^100 - 2^120)
- [ ] Hybrid LSTM+Transformer architecture

## 📝 Configuration

Edit `src/train.py` to customize:

```python
BATCH_SIZE = 512          # Adjust for your VRAM
NUM_WORKERS = 20          # CPU threads for data loading
STEPS = 1000000           # Training duration
D_MODEL = 128             # Model dimension
NUM_LAYERS = 4            # Transformer layers
NHEAD = 4                 # Attention heads
```

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Model architecture improvements
- Faster loop detection algorithms
- Better anomaly detection
- Visualization enhancements

## 📄 License

**GNU General Public License v3.0**

This project is licensed under GPL v3, which means:

✅ **You CAN:**
- Use for any purpose (personal, commercial, research)
- Modify and improve the code
- Distribute original or modified versions
- Sell modified versions

⚠️ **You MUST:**
- Share source code of any modifications
- Use the same GPL v3 license
- State significant changes
- Include copyright and license notices

🎯 **Mission:** Help humanity solve the Collatz Conjecture through open collaboration!

See [LICENSE](LICENSE) file for full details.

## 🙏 Acknowledgments

- **Collatz Conjecture**: Lothar Collatz (1937)
- **PyTorch Team**: For the amazing framework
- **Numba Team**: For JIT compilation magic
- **Community**: For mathematical insights

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/collatz-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/collatz-ai/discussions)

---

**⚠️ Disclaimer**: This is a research project. Finding a non-trivial cycle would disprove the Collatz Conjecture, which is highly unlikely but mathematically possible.

**🎯 Goal**: Advance our understanding of the Collatz Conjecture through AI-guided analysis and exhaustive verification.

---

*Made with ❤️ for mathematics and machine learning*
