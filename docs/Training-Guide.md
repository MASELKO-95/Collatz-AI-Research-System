# Training Guide

## Quick Start

```bash
# Start training
./run.sh

# Interactive commands:
# - Type 'stop' to save and exit
# - Type 'status' for current progress
```

## Configuration

Edit `src/train.py` to customize training:

```python
# Hardware Settings
BATCH_SIZE = 512          # Adjust for your VRAM (256-1024)
NUM_WORKERS = 20          # CPU threads for data loading
PREFETCH_FACTOR = 4       # Prefetch batches per worker

# Model Architecture
D_MODEL = 128             # Model dimension (128, 256, 512)
NUM_LAYERS = 4            # Transformer layers (4, 6, 8)
NHEAD = 4                 # Attention heads (4, 8, 16)

# Training Parameters
LEARNING_RATE = 1e-4      # Initial learning rate
STEPS = 1000000           # Total training steps
SAVE_EVERY = 1000         # Checkpoint frequency
```

## Training Process

### 1. Data Generation

**Two modes:**
- **Normal Mode** (50%): Sequential numbers
- **Hard Mode** (50%): Numbers > 2^68, special patterns

**Generation happens in parallel across 20 CPU workers**

### 2. Forward Pass

```python
# Input: Parity vector [0, 1, 0, 1, ...]
# Output: (stopping_time_pred, next_step_logits)

with torch.amp.autocast('cuda'):
    stopping_pred, next_step_logits = model(src, src_key_padding_mask)
```

### 3. Loss Calculation

```python
# Stopping time (log-space Huber loss)
log_stopping_times = torch.log1p(stopping_times)
loss_stopping = criterion_stopping(stopping_pred, log_stopping_times)

# Sequence (cross-entropy)
loss_next_step = criterion_next_step(next_step_logits, target_seq)

# Total
loss = loss_stopping + loss_next_step
```

### 4. Backward Pass (AMP)

```python
# Scale loss for mixed precision
scaler.scale(loss).backward()

# Gradient clipping
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Optimizer step
scaler.step(optimizer)
scaler.update()
scheduler.step()
```

## Optimizations

### Mixed Precision Training (AMP)

**Benefits:**
- 40% VRAM reduction
- 30% speed increase
- No accuracy loss

**Implementation:**
```python
scaler = torch.amp.GradScaler('cuda')

with torch.amp.autocast('cuda'):
    # Forward pass
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Clipping

**Prevents exploding gradients:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Learning Rate Scheduling

**Cosine Annealing:**
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=STEPS, 
    eta_min=1e-6
)
```

## Monitoring

### Console Output

```
Step 100000 | Loss: 0.3698 (Stop: 0.0003, Seq: 0.3695) | LR: 4.13e-05 | Time: 26.83s
```

### Discord Alerts

Configure webhook in `src/discord_bot.py`:
```python
DISCORD_WEBHOOK_URL = "your_webhook_url_here"
```

**Alerts sent for:**
- Training start/stop
- Anomalies detected
- Non-trivial cycles found
- Every 500 steps

### Checkpoints

**Saved every 1000 steps:**
```
checkpoints/model_step_100000.pth
```

**Contains:**
- Model weights
- Optimizer state
- Scheduler state
- Scaler state (AMP)
- Current step

## Resuming Training

Training automatically resumes from the latest checkpoint:

```bash
./run.sh
# Output: "Loading checkpoint: checkpoints/model_step_100000.pth"
# Output: "Resumed from step 100000"
```

## Troubleshooting

### Out of Memory (OOM)

**Solution 1: Reduce batch size**
```python
BATCH_SIZE = 256  # or 384
```

**Solution 2: Reduce workers**
```python
NUM_WORKERS = 12  # instead of 20
```

**Solution 3: Enable memory optimization**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Slow Training

**Check CPU utilization:**
```bash
htop  # Should see ~85% usage
```

**Increase workers if CPU < 80%:**
```python
NUM_WORKERS = 24  # if you have more cores
```

### Loss Not Decreasing

**Possible causes:**
1. Learning rate too high → Reduce to 5e-5
2. Gradient explosion → Check gradient norms
3. Data quality → Verify Hard Mode generation

## Advanced Techniques

### Distributed Training (Multi-GPU)

```python
# Coming soon!
# Will support DDP across multiple GPUs
```

### Custom Loss Weights

```python
# Adjust importance of each loss component
loss = 2.0 * loss_stopping + 1.0 * loss_next_step
```

### Early Stopping

```python
# Add to training loop
if loss < 0.30:
    print("Target loss reached!")
    break
```

---

**Next**: [Loop Searcher](Loop-Searcher)
