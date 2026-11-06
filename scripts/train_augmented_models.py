"""Train PhoBERT on augmented dataset using local model.

Usage:
  .venv/bin/python scripts/train_augmented_models.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Config
PHOBERT_BASE = MODEL_DIR / "phobert_base_local"  # Local PhoBERT base
EPOCHS = 5
BATCH_SIZE = 16
MAX_LEN = 128
LR = 2e-5
SEED = 42


def set_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    """Load augmented dataset."""
    train = pd.read_csv(DATA_DIR / "labeled" / "final_train.csv", dtype=str).fillna("")
    val = pd.read_csv(DATA_DIR / "labeled" / "final_val.csv", dtype=str).fillna("")
    test = pd.read_csv(DATA_DIR / "labeled" / "final_test.csv", dtype=str).fillna("")
    cross = pd.read_csv(PROJECT_DIR / "data_unified" / "cross_domain_test.csv", dtype=str).fillna("")

    for df in [train, val, test, cross]:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
        df = df[df["comment_text"].str.strip().ne("")]

    return train, val, test, cross


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_model(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
            preds.extend(batch_preds)
            labels.extend(batch["label"].tolist())

    return labels, preds


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_depression": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def main():
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    train_df, val_df, test_df, cross_df = load_data()
    logger.info(f"Data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, cross={len(cross_df)}")
    logger.info(f"Train labels: {train_df['label'].value_counts().to_dict()}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(PHOBERT_BASE), use_fast=False)

    # Create datasets
    train_dataset = TextDataset(
        train_df["comment_text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        MAX_LEN
    )
    val_dataset = TextDataset(
        val_df["comment_text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        MAX_LEN
    )
    test_dataset = TextDataset(
        test_df["comment_text"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
        MAX_LEN
    )
    cross_texts = cross_df["comment_text"].tolist()
    cross_labels_list = cross_df["label"].tolist()

    # Create eval dataset class inline
    class EvalDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            encoding = self.tokenizer(text, truncation=True, padding="max_length",
                                       max_length=self.max_len, return_tensors="pt")
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    cross_dataset = EvalDataset(cross_texts, cross_labels_list, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    cross_loader = DataLoader(cross_dataset, batch_size=BATCH_SIZE)

    # Load model
    logger.info("Loading PhoBERT base...")
    model = AutoModelForSequenceClassification.from_pretrained(str(PHOBERT_BASE), num_labels=2)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # Training
    best_val_f1 = 0
    best_model_state = None

    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)

        # Validate
        val_labels, val_preds = eval_model(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_preds)

        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_model_state = model.state_dict().copy()

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluate
    logger.info("Evaluating on test sets...")
    test_labels, test_preds = eval_model(model, test_loader, device)
    cross_labels, cross_preds = eval_model(model, cross_loader, device)

    test_metrics = compute_metrics(test_labels, test_preds)
    cross_metrics = compute_metrics(cross_labels, cross_preds)

    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS (Augmented Dataset)")
    logger.info(f"{'='*50}")
    logger.info(f"In-domain F1:     {test_metrics['f1_macro']:.4f}")
    logger.info(f"Cross-domain F1:  {cross_metrics['f1_macro']:.4f}")
    logger.info(f"Accuracy (in):    {test_metrics['accuracy']:.4f}")
    logger.info(f"F1 Depression:    {test_metrics['f1_depression']:.4f}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        "timestamp": timestamp,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_samples": len(train_df),
        "test": test_metrics,
        "cross_domain": cross_metrics,
        "comparison": {
            "model": "PhoBERT (augmented)",
            "in_domain_f1": test_metrics["f1_macro"],
            "cross_domain_f1": cross_metrics["f1_macro"],
            "previous_in_domain_f1": 0.8681,
            "previous_cross_domain_f1": 0.3727,
        }
    }

    output_file = MODEL_DIR / f"augmented_results_{timestamp}.json"
    output_file.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved: {output_file}")

    return results


if __name__ == "__main__":
    main()
