import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .engine import generate_batch_data
from .model import CollatzTransformer
from .discord_bot import send_analysis_report, send_anomaly_alert
import os

def load_model(path, device):
    model = CollatzTransformer(d_model=128, nhead=4, num_layers=4)
    checkpoint = torch.load(path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def analyze_range(model, start, end, device, report_to_discord=False):
    print(f"Analyzing range {start} to {end}...")
    numbers, stopping_times, parity_vectors = generate_batch_data(start, end, max_len=500)
    
    # Prepare data
    parity_vectors[parity_vectors == -1] = 2
    src = torch.tensor(parity_vectors, dtype=torch.long).to(device)
    
    with torch.no_grad():
        src_key_padding_mask = (src == 2)
        stopping_pred, next_step_logits = model(src, src_key_padding_mask=src_key_padding_mask)
        
        stopping_pred = stopping_pred.squeeze(1).cpu().numpy()
        # Inverse transform: expm1
        stopping_pred = np.expm1(stopping_pred)
        stopping_error = np.abs(stopping_pred - stopping_times)
        
    # Find top anomalies
    top_k = 5
    anomaly_indices = np.argsort(stopping_error)[-top_k:][::-1]
    
    report_lines = []
    print("\nTop Anomalies (Unexpected Stopping Time):")
    for idx in anomaly_indices:
        n = numbers[idx]
        actual = stopping_times[idx]
        pred = stopping_pred[idx]
        err = stopping_error[idx]
        line = f"Number: {n}, Actual: {actual}, Pred: {pred:.2f}, Error: {err:.2f}"
        print(line)
        report_lines.append(line)
        
    if report_to_discord:
        # Visualize embeddings first to attach
        visualize_embeddings(model, start, end, device, save_path="analysis_embeddings.png")
        
        summary = "I analyzed the range {} to {}.\n\n**Top Anomalies:**\n".format(start, end)
        summary += "\n".join(report_lines)
        summary += "\n\n**Thought Process:**\nThe numbers with high error are likely 'outliers' in the learned embedding space. I have attached a visualization of the embeddings colored by stopping time."
        
        send_analysis_report(summary, image_paths=["analysis_embeddings.png"])

    return numbers, stopping_times, stopping_pred, stopping_error

def visualize_embeddings(model, start, end, device, save_path="embeddings.png"):
    print(f"Generating embeddings for {start} to {end}...")
    numbers, stopping_times, parity_vectors = generate_batch_data(start, end, max_len=500)
    parity_vectors[parity_vectors == -1] = 2
    src = torch.tensor(parity_vectors, dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Model expects [batch, seq_len] (batch_first=True)
        
        # src is [batch, seq_len]
        # No transpose needed!
        src_key_padding_mask = (src == 2)
        
        x = model.embedding(src) * np.sqrt(128) # [batch, seq, d_model]
        x = model.pos_encoder(x)
        output = model.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        pooled = output.mean(dim=1) # [batch, d_model] (dim=1 is seq)
        
        embeddings = pooled.cpu().numpy()
        
    # PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    
    # 1. Embeddings Plot
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=stopping_times, cmap='viridis', alpha=0.6)
    plt.colorbar(sc, label='Stopping Time')
    plt.title(f"Embeddings PCA ({start}-{end})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    # 2. Stopping Time Distribution
    plt.subplot(2, 2, 2)
    plt.hist(stopping_times, bins=30, color='skyblue', edgecolor='black')
    plt.title("Stopping Time Distribution")
    plt.xlabel("Stopping Time")
    plt.ylabel("Count")
    
    # 3. Stopping Time vs Number
    plt.subplot(2, 2, 3)
    plt.scatter(numbers, stopping_times, alpha=0.5, s=10, c='orange')
    plt.title("Stopping Time vs Number")
    plt.xlabel("Number")
    plt.ylabel("Stopping Time")
    
    # 4. Parity Sequence Length (approx) vs Stopping Time
    # We can approximate sequence length by non-padding count
    seq_lens = np.sum(parity_vectors != 2, axis=1)
    plt.subplot(2, 2, 4)
    plt.scatter(seq_lens, stopping_times, alpha=0.5, s=10, c='green')
    plt.title("Seq Length vs Stopping Time")
    plt.xlabel("Sequence Length")
    plt.ylabel("Stopping Time")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoints = [f for f in os.listdir("checkpoints") if f.endswith(".pth")]
        if not checkpoints:
            print("No checkpoints found.")
        else:
            # Sort by modification time to get the absolute latest
            latest = max([os.path.join("checkpoints", f) for f in checkpoints], key=os.path.getctime)
            print(f"Loading {latest}...")
            
            model = load_model(latest, DEVICE)
            
            # Parse step from filename to determine training frontier
            step = 0
            try:
                basename = os.path.basename(latest)
                name = os.path.splitext(basename)[0]
                parts = name.split('_')
                if 'step' in parts:
                    idx = parts.index('step')
                    if idx + 1 < len(parts):
                        step = int(parts[idx+1])
            except ValueError:
                pass
            
            # Calculate frontier
            # Assuming BATCH_SIZE=512 from train.py
            BATCH_SIZE = 512
            start_n = 10 + step * BATCH_SIZE
            end_n = start_n + 2000 # Analyze next 2000 numbers
            
            print(f"Training is at step {step}. Analyzing frontier range: {start_n} to {end_n}")
            
            analyze_range(model, start_n, end_n, DEVICE, report_to_discord=True)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
