"""Round 5 Active Learning: Retrain PhoBERT and predict on remaining samples.

This script (no datasets library needed):
1. Trains PhoBERT on current final_dataset (post Round 4)
2. Predicts on remaining ~120K unlabeled samples
3. Saves predictions for next selection step

Usage:
    .venv/bin/python scripts/retrain_phobert_for_round5.py
"""

from __future__ import annotations

# Set offline mode BEFORE importing transformers
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path

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
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = MODEL_DIR / "round5_predictions"
PREDICTIONS_FILE = DATA_DIR / "phobert_remaining_predictions.csv"

TRAIN_FILE = DATA_DIR / "labeled" / "final_train.csv"
VAL_FILE = DATA_DIR / "labeled" / "final_val.csv"
MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
SEED = 42

# ── Set seed ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# ── Load data ────────────────────────────────────────────────────────────────
print(f"Loading training data...")
train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

print(f"Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")

train_texts = train_df["comment_text"].values
train_labels = train_df["label"].values
val_texts = val_df["comment_text"].values
val_labels = val_df["label"].values

# Class weights
n_neg = (train_labels == 0).sum()
n_pos = (train_labels == 1).sum()
pos_weight = n_neg / n_pos
print(f"Class weights: neg={n_neg}, pos={n_pos}, pos_weight={pos_weight:.4f}")

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
)
model.to(device)

# ── Training ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING PHOBERT FOR ROUND 5")
print("="*60)

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

# Loss with class weights
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device))

best_f1 = 0.0

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

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
    print(f"Train Loss: {avg_loss:.4f}")

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
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    print(f"Val Accuracy: {acc:.4f} | Val F1: {f1:.4f}")

    # Save best model
    if f1 > best_f1:
        best_f1 = f1
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = OUTPUT_DIR / "best_model"
        model.save_pretrained(str(checkpoint_path))
        print(f"  -> Saved best model (F1: {best_f1:.4f})")

print(f"\nTraining complete! Best F1: {best_f1:.4f}")

# ── Load best model for prediction ──────────────────────────────────────────────
print(f"\nLoading best model for prediction...")
model = AutoModelForSequenceClassification.from_pretrained(str(OUTPUT_DIR / "best_model"))
model.to(device)
model.eval()

# ── Predict on remaining samples ──────────────────────────────────────────────
print("\n" + "="*60)
print("PREDICTING ON REMAINING SAMPLES")
print("="*60)

print(f"Loading remaining predictions from {PREDICTIONS_FILE}...")
remaining_df = pd.read_csv(PREDICTIONS_FILE)
print(f"Remaining samples: {len(remaining_df):,}")

# Create dataset for prediction
remaining_texts = remaining_df["comment_text"].values
remaining_dataset = DepressionDataset(remaining_texts)
remaining_loader = DataLoader(remaining_dataset, batch_size=32, shuffle=False)

print("Running inference...")
all_probs = []

with torch.no_grad():
    for i, batch in enumerate(remaining_loader):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(remaining_loader)} batches")

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Probability of depression (class 1)
        all_probs.extend(probs[:, 1].cpu().numpy())

remaining_df["phobert_prob"] = all_probs
remaining_df["phobert_label"] = remaining_df["phobert_prob"].apply(
    lambda x: "depression" if x >= 0.5 else "normal"
)
remaining_df["probability"] = remaining_df["phobert_prob"].apply(
    lambda x: x if x >= 0.5 else 1 - x
)

# Save predictions
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_file = OUTPUT_DIR / "round5_phobert_predictions.csv"
remaining_df.to_csv(output_file, index=False)
print(f"\nPredictions saved to: {output_file}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PREDICTION SUMMARY")
print("="*60)

print("\nPhoBERT Label Distribution:")
print(remaining_df["phobert_label"].value_counts())

print("\nProbability Distribution:")
for threshold in [0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
    count = (remaining_df["probability"] >= threshold).sum()
    print(f"  P >= {threshold:.2f}: {count:,} samples")

# Find samples closest to 0.5
remaining_df["distance_from_05"] = abs(remaining_df["probability"] - 0.5)
hardest = remaining_df.sort_values("distance_from_05").head(10)
print("\nTop 10 Hardest Samples (closest to 0.5):")
for _, row in hardest.iterrows():
    text = row["comment_text"][:50] if len(row["comment_text"]) > 50 else row["comment_text"]
    print(f"  P={row['probability']:.4f} | {row['phobert_label']:10s} | {text}...")

print("\n" + "="*60)
print("NEXT STEP: Run scripts/prepare_round5_active_learning.py")
print("="*60)
