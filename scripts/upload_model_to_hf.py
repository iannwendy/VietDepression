"""
Upload fine-tuned PhoBERT model to HuggingFace Hub.

Usage:
    python scripts/upload_model_to_hf.py

Before running:
    1. Run: hf auth login (with Write token)
    2. Edit HF_REPO_ID below to your repo
"""

import os
import json
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_DIR / "web_demo" / "backend"

MODEL_DIR = BACKEND_DIR / "models" / "phobert_seed_42" / "best_model"
TOKENIZER_DIR = BACKEND_DIR / "models" / "phobert_base_local"

# CHANGE THIS to your HuggingFace repo (e.g., "yourusername/depression-phobert-v2")
HF_REPO_ID = "iannwendy/depression-phobert-round6v2-seed42"

def upload_to_huggingface():
    """Upload the fine-tuned model to HuggingFace Hub."""
    from safetensors.torch import load_file
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

    print(f"Loading tokenizer from: {TOKENIZER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR),
        local_files_only=True,
        use_fast=False,
    )
    print("✓ Tokenizer loaded")

    print(f"Loading model config from: {MODEL_DIR}")
    config = AutoConfig.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_config(config)
    print("✓ Model config loaded")

    print(f"Loading model weights from: {MODEL_DIR}")
    state_dict = load_file(str(MODEL_DIR / "model.safetensors"))
    model.load_state_dict(state_dict, strict=False)
    print("✓ Model weights loaded")

    # Update config with proper model info
    model.config.model_type = "xlm-roberta"
    model.config.name_or_path = HF_REPO_ID
    model.config.push_to_hub = True

    # Create temp directory for upload
    temp_dir = Path("/tmp/model_to_upload")
    temp_dir.mkdir(exist_ok=True)

    print(f"Saving model to: {temp_dir}")
    tokenizer.save_pretrained(temp_dir)
    model.save_pretrained(temp_dir)
    print("✓ Model saved locally")

    print(f"\nUploading to: https://huggingface.co/{HF_REPO_ID}")
    print("This may take a few minutes...")

    # Push to Hub
    tokenizer.push_to_hub(HF_REPO_ID)
    model.push_to_hub(HF_REPO_ID)
    print(f"\n✅ Successfully uploaded!")
    print(f"   Model URL: https://huggingface.co/{HF_REPO_ID}")

    print("\nTo use in your backend, update phobert_engine.py:")
    print(f'   HF_REPO_ID = "{HF_REPO_ID}"')

if __name__ == "__main__":
    upload_to_huggingface()
