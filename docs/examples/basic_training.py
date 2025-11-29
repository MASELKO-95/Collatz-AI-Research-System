"""
Basic training example for Collatz AI.

This script demonstrates how to train the model with custom parameters.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from model import CollatzTransformer
from dataset import CollatzIterableDataset


def main():
    # Configuration
    BATCH_SIZE = 256
    D_MODEL = 128
    NUM_LAYERS = 4
    NHEAD = 4
    MAX_LEN = 500
    LEARNING_RATE = 1e-4
    STEPS = 1000

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = CollatzTransformer(
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, max_len=MAX_LEN
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize dataset
    dataset = CollatzIterableDataset(
        start_range=1000,
        end_range=1000000,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        hard_mode_ratio=0.5,
    )

    dataloader = DataLoader(dataset, batch_size=None, num_workers=4)

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion_stop = nn.HuberLoss()
    criterion_seq = nn.CrossEntropyLoss(ignore_index=2)

    # Training loop
    model.train()
    for step, batch in enumerate(dataloader):
        if step >= STEPS:
            break

        # Move to device
        parity_vectors = batch["parity_vector"].to(device)
        stopping_times = batch["stopping_time"].to(device)

        # Forward pass
        stopping_time_pred, next_step_logits = model(parity_vectors)

        # Compute losses
        loss_stop = criterion_stop(stopping_time_pred.squeeze(), stopping_times)

        # For sequence loss, shift targets
        targets = parity_vectors[:, 1:].contiguous()
        logits = next_step_logits[:, :-1, :].contiguous()
        loss_seq = criterion_seq(
            logits.view(-1, 3),
            targets.view(-1)
        )

        loss = loss_stop + loss_seq

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if (step + 1) % 100 == 0:
            print(
                f"Step {step+1}/{STEPS} | "
                f"Loss: {loss.item():.4f} | "
                f"Stop: {loss_stop.item():.4f} | "
                f"Seq: {loss_seq.item():.4f}"
            )

    # Save model
    save_path = Path(__file__).parent.parent / "checkpoints" / "example_model.pt"
    save_path.parent.mkdir(exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": STEPS,
        },
        save_path,
    )

    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
