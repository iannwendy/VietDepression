"""Round 5 Active Learning: Retrain PhoBERT with multiple seeds.

Trains PhoBERT on the updated final_dataset with multiple random seeds
for robust evaluation.

Usage:
    .venv/bin/python scripts/train_phobert_round5_multiseed.py
"""

from __future__ import annotations

# Set offline mode BEFORE importing transformers
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = MODEL_DIR / "round5_predictions"
RESULTS_DIR = PROJECT_DIR / "results"

TRAIN_FILE = DATA_DIR / "labeled" / "final_train.csv"
VAL_FILE = DATA_DIR / "labeled" / "final_val.csv"
TEST_FILE = DATA_DIR / "labeled" / "final_test.csv"

MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
SEEDS = [42, 123, 2024]

# ── Set seed ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# ── Custom Dataset ─────────────────────────────────────────────────────────────
class DepressionDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ── Training function ──────────────────────────────────────────────────────────
def train_model(seed: int, train_texts, train_labels, val_texts, val_labels):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"TRAINING WITH SEED {seed}")
    print(f"{'='*60}")

    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)

    # Create datasets
    train_dataset = DepressionDataset(train_texts, train_labels)
    val_dataset = DepressionDataset(val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.06), num_training_steps=total_steps
    )

    # Class weights
    n_neg = (train_labels == 0).sum()
    n_pos = (train_labels == 1).sum()
    pos_weight = n_neg / n_pos
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device))

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Eval
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        # Calculate F1
        f1 = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            seed_dir = OUTPUT_DIR / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = seed_dir / "best_model"
            model.save_pretrained(str(checkpoint_path))

    print(f"Best F1 for seed {seed}: {best_f1:.4f}")
    return best_f1, seed_dir

# ── Evaluation function ────────────────────────────────────────────────────────
def evaluate_model(model_dir: Path, test_texts, test_labels, split_name: str = "test"):
    print(f"\n--- Evaluating {split_name} set ---")

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir / "best_model"))
    model.to(device)
    model.eval()

    test_dataset = DepressionDataset(test_texts)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs[:, 1].cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(test_labels, all_preds)
    f1_macro = f1_score(test_labels, all_preds, average="macro")
    f1_depression = f1_score(test_labels, all_preds, average="binary", pos_label=1)
    precision = precision_score(test_labels, all_preds, average="macro")
    recall = recall_score(test_labels, all_preds, average="macro")

    print(f"{split_name.capitalize()} Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (dep):   {f1_depression:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_depression": f1_depression,
        "precision": precision,
        "recall": recall,
        "predictions": all_preds,
        "probabilities": all_probs
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR / f"round5_eval_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("ROUND 5 PHOBERT TRAINING WITH MULTIPLE SEEDS")
    print("="*60)

    # Load data
    print(f"\nLoading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    train_texts = train_df["comment_text"].values
    train_labels = train_df["label"].values
    val_texts = val_df["comment_text"].values
    val_labels = val_df["label"].values
    test_texts = test_df["comment_text"].values
    test_labels = test_df["label"].values

    # Train with each seed
    all_results = {}
    best_seed = None
    best_f1 = 0.0

    for seed in SEEDS:
        f1, model_dir = train_model(seed, train_texts, train_labels, val_texts, val_labels)
        all_results[seed] = {"model_dir": model_dir, "val_f1": f1}

        if f1 > best_f1:
            best_f1 = f1
            best_seed = seed

    # Evaluate best model on test set
    print(f"\n{'='*60}")
    print(f"BEST MODEL (seed={best_seed}, val_f1={best_f1:.4f})")
    print(f"{'='*60}")

    best_model_dir = all_results[best_seed]["model_dir"]
    test_results = evaluate_model(best_model_dir, test_texts, test_labels, "test")

    # Save results
    results = {
        "timestamp": timestamp,
        "best_seed": best_seed,
        "best_val_f1": best_f1,
        "seeds": SEEDS,
        "test_results": {k: v for k, v in test_results.items() if k not in ["predictions", "probabilities"]},
        "per_seed_val_f1": {str(k): v["val_f1"] for k, v in all_results.items()}
    }

    import json
    results_file = results_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Models saved to: {OUTPUT_DIR}/seed_*/best_model")

    return results

if __name__ == "__main__":
    main()
