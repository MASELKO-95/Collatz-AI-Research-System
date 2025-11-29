import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import CollatzIterableDataset, collate_fn
from .model import CollatzTransformer
from .discord_bot import send_status_update, send_anomaly_alert, send_message
from .analyze import analyze_range
from .engine import generate_hard_candidates
import time
import os
import glob
import threading
import sys
import numpy as np

# Global control flags
STOP_REQUESTED = False

def command_listener():
    global STOP_REQUESTED
    print("Command listener started. Type 'stop' to save and exit, 'status' for current info.")
    while not STOP_REQUESTED:
        try:
            cmd = input().strip().lower()
            if cmd == 'stop':
                STOP_REQUESTED = True
                print("Stopping requested... finishing current batch.")
            elif cmd == 'status':
                print("Training is running...")
        except EOFError:
            break
        except Exception:
            pass

def save_result_txt(step, loss, loss_stop, loss_seq, path="last_result.txt"):
    with open(path, "w") as f:
        f.write(f"Step: {step}\n")
        f.write(f"Total Loss: {loss:.6f}\n")
        f.write(f"Stopping Time Loss: {loss_stop:.6f}\n")
        f.write(f"Sequence Loss: {loss_seq:.6f}\n")
        f.write(f"Timestamp: {time.ctime()}\n")

def train():
    global STOP_REQUESTED
    
    # Maximum Optimization Settings (22 threads, 29GB RAM available)
    BATCH_SIZE = 512 # Stable batch size for 8GB VRAM (768 causes OOM)
    NUM_WORKERS = 20  # Utilize 22 threads (20 workers + 2 for main/system)
    PREFETCH_FACTOR = 4
    
    # Model Parameters (Keep compatible with existing checkpoints)
    D_MODEL = 128
    NUM_LAYERS = 4
    NHEAD = 4
    
    LEARNING_RATE = 1e-4
    STEPS = 1000000  # 1 million steps - practically unlimited training
    PRINT_EVERY = 100
    SAVE_EVERY = 1000
    ANOMALY_CHECK_EVERY = 500
    ANOMALY_THRESHOLD = 5000.0
    CHECKPOINT_DIR = "checkpoints"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    print(f"Model: D_MODEL={D_MODEL}, LAYERS={NUM_LAYERS}, HEADS={NHEAD}")
    send_message(f"🚀 MAXIMUM OPTIMIZATION MODE\nBatch: {BATCH_SIZE} | Model: {D_MODEL}d, {NUM_LAYERS}L, {NHEAD}H")
    
    # Start command listener
    cmd_thread = threading.Thread(target=command_listener, daemon=True)
    cmd_thread.start()
    
    # Start background loop searcher (CPU-based brute force)
    try:
        from .loop_search import start_background_loop_search
        start_background_loop_search(num_threads=22, numbers_per_thread=1000000)  # 22M total
    except Exception as e:
        print(f"Could not start loop searcher: {e}")
    
    # Model
    model = CollatzTransformer(d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS).to(DEVICE)
    
    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision Scaler
    
    # Learning Rate Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-6)
    
    criterion_stopping = nn.HuberLoss(delta=1.0)
    criterion_next_step = nn.CrossEntropyLoss(ignore_index=2)

    # Resume from checkpoint
    start_step = 0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_step = checkpoint["step"]
            print(f"Resumed from step {start_step}")
        else:
            model.load_state_dict(checkpoint)
            
    torch.cuda.empty_cache()
    model.train()

    # Data - Initialize AFTER loading checkpoint to use correct start_n
    # Calculate start_n based on steps to avoid "stuck" training on restart
    # Each step consumes BATCH_SIZE numbers (roughly, excluding hard mode)
    current_start_n = 10 + (start_step * BATCH_SIZE)
    print(f"Resuming data generation from number: {current_start_n}")
    
    # Hard mode probability handled inside dataset workers now
    dataset = CollatzIterableDataset(start_n=current_start_n, batch_size=BATCH_SIZE, hard_mode_prob=0.5)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=True
    )
    data_iter = iter(dataloader)
    
    step = start_step
    start_time = time.time()
    
    try:
        while step < STEPS:
            if STOP_REQUESTED:
                break
                
            step += 1
            
            try:
                batch = next(data_iter)
                numbers, parity_vectors, stopping_times = batch
                numbers = numbers.to(DEVICE)
                parity_vectors = parity_vectors.to(DEVICE)
                stopping_times = stopping_times.to(DEVICE)
            except StopIteration:
                data_iter = iter(dataloader)
                continue
            
            optimizer.zero_grad()
            
            # Forward with AMP
            with torch.amp.autocast('cuda'):
                src = parity_vectors[:, :-1]
                target_seq = parity_vectors[:, 1:]
                src_key_padding_mask = (src == 2)
                
                stopping_pred, next_step_logits = model(src, src_key_padding_mask=src_key_padding_mask)
                
                # Loss
                log_stopping_times = torch.log1p(stopping_times)
                loss_stopping = criterion_stopping(stopping_pred, log_stopping_times)
                
                loss_next_step = criterion_next_step(
                    next_step_logits.reshape(-1, 3), 
                    target_seq.reshape(-1)
                )
                
                loss = loss_stopping + loss_next_step
            
            # Backward with Scaler
            scaler.scale(loss).backward()
            
            # Gradient Clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Update learning rate
            
            if step % PRINT_EVERY == 0:
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                print(f"Step {step} | Loss: {loss.item():.4f} (Stop: {loss_stopping.item():.4f}, Seq: {loss_next_step.item():.4f}) | LR: {current_lr:.2e} | Time: {elapsed:.2f}s")
                start_time = time.time()
                
                save_result_txt(step, loss.item(), loss_stopping.item(), loss_next_step.item())
                
                if step % (PRINT_EVERY * 5) == 0:
                    send_status_update(step, loss.item(), loss_stopping.item(), loss_next_step.item())
            
            if step % ANOMALY_CHECK_EVERY == 0:
                with torch.no_grad():
                    # Check for high error in log space
                    stop_pred_val = stopping_pred.squeeze(1)
                    stop_actual_log = log_stopping_times.squeeze(1)
                    errors = torch.abs(stop_pred_val - stop_actual_log)
                    max_error, max_idx = torch.max(errors, dim=0)
                    
                    # Threshold in log space! 
                    # log(1000) ~ 6.9, log(10000) ~ 9.2. Error of 2.0 is huge order of magnitude.
                    LOG_THRESHOLD = 3.0 
                    
                    if max_error.item() > LOG_THRESHOLD:
                        print(f"\n🚨 ANOMALY DETECTED! LogError: {max_error.item():.2f}")
                        
                        anom_num = numbers[max_idx].item()
                        anom_actual = torch.expm1(stop_actual_log[max_idx]).item()
                        anom_pred = torch.expm1(stop_pred_val[max_idx]).item()
                        
                        send_anomaly_alert(anom_num, anom_actual, anom_pred, max_error.item())
                        
                        # Check if it's a loop candidate (Stopping time > 20000 or similar)
                        if anom_actual > 19000:
                             send_message(f"⚠️ **POSSIBLE LOOP/DIVERGENCE** found: {anom_num} took > 19000 steps!")
                        
                        # Optional: Don't stop training every time, just log
                        # STOP_REQUESTED = True 

            if step % SAVE_EVERY == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_step_{step}.pth")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss.item(),
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
    
    print("Saving final state...")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_step_{step}_final.pth")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss.item() if 'loss' in locals() else 0.0,
    }, checkpoint_path)
    
    if 'loss' in locals():
        save_result_txt(step, loss.item(), loss_stopping.item(), loss_next_step.item())
        
    print(f"Saved final checkpoint to {checkpoint_path}")
    print("Training finished.")

if __name__ == "__main__":
    train()
