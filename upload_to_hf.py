#!/usr/bin/env python3
"""
Upload Collatz AI model to Hugging Face Hub
"""

import torch
from huggingface_hub import HfApi, create_repo, upload_file
import os
import sys

# Configuration
MODEL_NAME = "collatz-ai"
USERNAME = "MASELKO-95"  # Change to your HF username
REPO_ID = f"{USERNAME}/{MODEL_NAME}"

# Files to upload
CHECKPOINT_PATH = "checkpoints/model_step_120305_final.pth"
MODEL_CARD_PATH = "README_HF.md"

def upload_to_huggingface(token):
    """
    Upload model to Hugging Face Hub
    
    Args:
        token: Hugging Face API token
    """
    print(f"🚀 Uploading to Hugging Face: {REPO_ID}")
    
    # Initialize API
    api = HfApi()
    
    # Create repository (if doesn't exist)
    try:
        create_repo(
            repo_id=REPO_ID,
            token=token,
            repo_type="model",
            exist_ok=True
        )
        print(f"✅ Repository created/verified: {REPO_ID}")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")
    
    # Upload checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📦 Uploading checkpoint: {CHECKPOINT_PATH}")
        upload_file(
            path_or_fileobj=CHECKPOINT_PATH,
            path_in_repo="pytorch_model.bin",
            repo_id=REPO_ID,
            token=token
        )
        print("✅ Checkpoint uploaded")
    else:
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return False
    
    # Upload model card (README)
    if os.path.exists(MODEL_CARD_PATH):
        print(f"📝 Uploading model card: {MODEL_CARD_PATH}")
        upload_file(
            path_or_fileobj=MODEL_CARD_PATH,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            token=token
        )
        print("✅ Model card uploaded")
    else:
        print(f"⚠️  Model card not found: {MODEL_CARD_PATH}")
    
    # Upload model architecture
    print("📦 Uploading model.py...")
    upload_file(
        path_or_fileobj="src/model.py",
        path_in_repo="model.py",
        repo_id=REPO_ID,
        token=token
    )
    print("✅ Model architecture uploaded")
    
    # Upload config
    print("📦 Creating config.json...")
    config = {
        "model_type": "collatz-transformer",
        "d_model": 128,
        "num_layers": 4,
        "nhead": 4,
        "dim_feedforward": 512,
        "max_len": 500,
        "vocab_size": 3,
        "architectures": ["CollatzTransformer"],
        "torch_dtype": "float32"
    }
    
    import json
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    upload_file(
        path_or_fileobj="config.json",
        path_in_repo="config.json",
        repo_id=REPO_ID,
        token=token
    )
    print("✅ Config uploaded")
    
    print(f"\n🎉 Model successfully uploaded to: https://huggingface.co/{REPO_ID}")
    return True

def main():
    """Main upload function"""
    print("=" * 60)
    print("Collatz AI - Hugging Face Upload")
    print("=" * 60)
    
    # Check if token is provided
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        print("\n📝 Please provide your Hugging Face token:")
        print("   Get it from: https://huggingface.co/settings/tokens")
        print("\nUsage:")
        print(f"   python {sys.argv[0]} YOUR_HF_TOKEN")
        print("\nOr set environment variable:")
        print("   export HF_TOKEN=your_token_here")
        print(f"   python {sys.argv[0]}")
        
        # Try environment variable
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("\n❌ No token provided. Exiting.")
            return
    
    # Verify checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ Checkpoint not found: {CHECKPOINT_PATH}")
        print("   Please train the model first or specify correct path.")
        return
    
    # Upload
    success = upload_to_huggingface(token)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Upload complete!")
        print("=" * 60)
        print(f"\n🔗 View your model: https://huggingface.co/{REPO_ID}")
        print("\n📥 Download with:")
        print(f'   from transformers import AutoModel')
        print(f'   model = AutoModel.from_pretrained("{REPO_ID}")')

if __name__ == "__main__":
    main()
