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
        # Model expects [batch, seq_len], but internally transposes to [seq_len, batch]
        # We need to manually replicate the forward pass correctly
        
        # src is [batch, seq_len]
        src_t = src.transpose(0, 1) # [seq_len, batch]
        src_key_padding_mask = (src == 2)
        
        x = model.embedding(src_t) * np.sqrt(128)
        x = model.pos_encoder(x)
        output = model.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        pooled = output.mean(dim=0) # [batch, d_model]
        
        embeddings = pooled.cpu().numpy()
        
    # PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=stopping_times, cmap='viridis', alpha=0.6)
    plt.colorbar(sc, label='Stopping Time')
    plt.title(f"Collatz Embeddings ({start}-{end})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
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
            
            analyze_range(model, 1000, 2000, DEVICE, report_to_discord=True)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
