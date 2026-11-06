"""End-to-end Round 5 evaluation on the repaired dataset.

This script is the canonical evaluator for the Round 5 paper. It:
  1. Loads the repaired final_{train,val,test}.csv (7,336 / 241 / 242).
  2. Loads the VSMEC cross-domain test set (3,084).
  3. Loads all Round 5 models from models/round5_retrained/:
     - PhoBERT (3 seeds)
     - BiLSTM (3 seeds)
     - TF-IDF + Logistic Regression
     - TF-IDF + LinearSVC
  4. Evaluates each model on in-domain and cross-domain.
  5. Computes PhoBERT majority-vote from the 3 seeds.
  6. Saves a complete evaluation_results.json containing both splits.

Usage:
    PYTHONPATH="$PWD" .venv/bin/python scripts/run_final_round5_evaluation.py
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
UNIFIED_DIR = PROJECT_DIR / "data_unified"

TRAIN_FILE = DATA_DIR / "labeled" / "final_train.csv"
VAL_FILE = DATA_DIR / "labeled" / "final_val.csv"
TEST_FILE = DATA_DIR / "labeled" / "final_test.csv"
VSMEC_FILE = UNIFIED_DIR / "cross_domain_test.csv"

RETRAINED_DIR = MODEL_DIR / "round5_retrained"
PHOBERT_DIR = RETRAINED_DIR  # alias

MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 128
SEEDS = [42, 123, 2024]

OUTPUT_DIR = RESULTS_DIR / f"round5_final_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")


# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_depression": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ── Tokenizer ────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)


class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def predict_phobert(model_dir: Path, texts):
    """Return (predictions, probabilities) for a PhoBERT model."""
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    ds = TextDataset(texts)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    preds, probs = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs_pos = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds.extend((probs_pos >= 0.5).astype(int).tolist())
            probs.extend(probs_pos.tolist())

    del model
    return np.array(preds), np.array(probs)


# ── BiLSTM ───────────────────────────────────────────────────────────────────
class Vocabulary:
    """Token vocabulary used by BiLSTM checkpoints. Must be defined at module
    level so torch.load can unpickle it (matches training script)."""

    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, texts):
        freq = {}
        for text in texts:
            for word in text.split():
                freq[word] = freq.get(word, 0) + 1
        idx = 2
        for word, count in freq.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def text_to_sequence(self, text):
        return [self.stoi.get(w, 1) for w in text.split()]


class BiLSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.dropout(hidden)
        return self.fc(output)


class BiLSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = self.vocab.text_to_sequence(str(self.texts[idx]))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def predict_bilstm(model_path: Path, eval_texts):
    """Load BiLSTM model and predict on eval_texts. Uses vocab from checkpoint.

    Note: ``weights_only=False`` is required because the checkpoint embeds a
    custom ``Vocabulary`` object (built in-project during training); the file
    is a trusted artifact produced by our own ``retrain_all_models_round5.py``.
    """
    from torch.nn.utils.rnn import pad_sequence

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]

    model = BiLSTMClassifier(len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    ds = BiLSTMDataset(eval_texts, [0] * len(eval_texts), vocab)

    def collate(batch):
        seqs, _ = zip(*batch)
        return pad_sequence(seqs, batch_first=True, padding_value=0)

    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)

    preds = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch.to(device))
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())

    del model
    return np.array(preds)


# ── TF-IDF ───────────────────────────────────────────────────────────────────
def predict_tfidf(model_path: Path, texts):
    """Predict using a TF-IDF pipeline."""
    pipeline = joblib.load(model_path)
    preds = pipeline.predict(texts)
    return np.array(preds)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ROUND 5 FINAL EVALUATION (END-TO-END)")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    vsmec_df = pd.read_csv(VSMEC_FILE)

    test_texts = test_df["comment_text"].values
    test_labels = test_df["label"].values
    vsmec_texts = (
        vsmec_df["comment_text"].values
        if "comment_text" in vsmec_df.columns
        else vsmec_df["text"].values
    )
    vsmec_labels = vsmec_df["label"].values

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"VSMEC: {len(vsmec_df):,} | Test labels: {dict(test_df['label'].value_counts())}")
    print(f"VSMEC labels: {dict(vsmec_df['label'].value_counts())}")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_info": {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "vsmec_size": len(vsmec_df),
            "train_label_dist": dict(train_df["label"].value_counts()),
            "test_label_dist": dict(test_df["label"].value_counts()),
            "vsmec_label_dist": dict(vsmec_df["label"].value_counts()),
        },
        "in_domain": {},
        "cross_domain": {},
    }

    # ── 1. PhoBERT (3 seeds) ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("1. PHOBERT (3 seeds)")
    print("=" * 70)

    phobert_preds_test = {}
    phobert_preds_vsmec = {}

    for seed in SEEDS:
        model_dir = PHOBERT_DIR / f"phobert_seed_{seed}" / "best_model"
        if not model_dir.exists():
            print(f"  Seed {seed}: model not found at {model_dir}, skipping")
            continue

        print(f"\n  Seed {seed}...")
        # In-domain
        preds_test, _ = predict_phobert(model_dir, test_texts)
        metrics_test = compute_metrics(test_labels, preds_test)
        all_results["in_domain"][f"phobert_seed{seed}"] = metrics_test
        phobert_preds_test[seed] = preds_test
        print(f"    In-domain:  Acc={metrics_test['accuracy']:.4f}, F1={metrics_test['f1_macro']:.4f}")

        # Cross-domain
        preds_vsmec, _ = predict_phobert(model_dir, vsmec_texts)
        metrics_vsmec = compute_metrics(vsmec_labels, preds_vsmec)
        all_results["cross_domain"][f"phobert_seed{seed}"] = metrics_vsmec
        phobert_preds_vsmec[seed] = preds_vsmec
        print(f"    Cross-domain: Acc={metrics_vsmec['accuracy']:.4f}, F1={metrics_vsmec['f1_macro']:.4f}")

    # PhoBERT avg via majority vote (over predictions)
    if phobert_preds_test:
        seeds = sorted(phobert_preds_test.keys())
        # In-domain majority vote
        stacked_test = np.stack([phobert_preds_test[s] for s in seeds], axis=0)
        avg_pred_test = (np.mean(stacked_test, axis=0) >= 0.5).astype(int)
        phobert_avg_test = compute_metrics(test_labels, avg_pred_test)
        all_results["in_domain"]["phobert_avg"] = phobert_avg_test
        print(f"\n  PhoBERT avg vote (in-domain):   Acc={phobert_avg_test['accuracy']:.4f}, "
              f"F1={phobert_avg_test['f1_macro']:.4f}, F1-Dep={phobert_avg_test['f1_depression']:.4f}")

        # Cross-domain majority vote
        stacked_vsmec = np.stack([phobert_preds_vsmec[s] for s in seeds], axis=0)
        avg_pred_vsmec = (np.mean(stacked_vsmec, axis=0) >= 0.5).astype(int)
        phobert_avg_vsmec = compute_metrics(vsmec_labels, avg_pred_vsmec)
        all_results["cross_domain"]["phobert_avg"] = phobert_avg_vsmec
        print(f"  PhoBERT avg vote (cross-domain): Acc={phobert_avg_vsmec['accuracy']:.4f}, "
              f"F1={phobert_avg_vsmec['f1_macro']:.4f}, F1-Dep={phobert_avg_vsmec['f1_depression']:.4f}")

    # ── 2. BiLSTM (3 seeds) ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. BILSTM (3 seeds)")
    print("=" * 70)

    bilstm_preds_test = {}
    bilstm_preds_vsmec = {}

    for seed in SEEDS:
        model_path = PHOBERT_DIR / f"bilstm_seed_{seed}" / "best_model.pt"
        if not model_path.exists():
            print(f"  Seed {seed}: model not found at {model_path}, skipping")
            continue

        print(f"\n  Seed {seed}...")
        preds_test = predict_bilstm(model_path, test_texts)
        metrics_test = compute_metrics(test_labels, preds_test)
        all_results["in_domain"][f"bilstm_seed{seed}"] = metrics_test
        bilstm_preds_test[seed] = preds_test
        print(f"    In-domain:  Acc={metrics_test['accuracy']:.4f}, F1={metrics_test['f1_macro']:.4f}")

        preds_vsmec = predict_bilstm(model_path, vsmec_texts)
        metrics_vsmec = compute_metrics(vsmec_labels, preds_vsmec)
        all_results["cross_domain"][f"bilstm_seed{seed}"] = metrics_vsmec
        bilstm_preds_vsmec[seed] = preds_vsmec
        print(f"    Cross-domain: Acc={metrics_vsmec['accuracy']:.4f}, F1={metrics_vsmec['f1_macro']:.4f}")

    if bilstm_preds_test:
        seeds = sorted(bilstm_preds_test.keys())
        stacked_test = np.stack([bilstm_preds_test[s] for s in seeds], axis=0)
        avg_pred_test = (np.mean(stacked_test, axis=0) >= 0.5).astype(int)
        bilstm_avg_test = compute_metrics(test_labels, avg_pred_test)
        all_results["in_domain"]["bilstm_avg"] = bilstm_avg_test
        print(f"\n  BiLSTM avg vote (in-domain):   Acc={bilstm_avg_test['accuracy']:.4f}, "
              f"F1={bilstm_avg_test['f1_macro']:.4f}")

        stacked_vsmec = np.stack([bilstm_preds_vsmec[s] for s in seeds], axis=0)
        avg_pred_vsmec = (np.mean(stacked_vsmec, axis=0) >= 0.5).astype(int)
        bilstm_avg_vsmec = compute_metrics(vsmec_labels, avg_pred_vsmec)
        all_results["cross_domain"]["bilstm_avg"] = bilstm_avg_vsmec
        print(f"  BiLSTM avg vote (cross-domain): Acc={bilstm_avg_vsmec['accuracy']:.4f}, "
              f"F1={bilstm_avg_vsmec['f1_macro']:.4f}")

    # ── 3. TF-IDF + LogReg ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. TF-IDF + LOGISTIC REGRESSION")
    print("=" * 70)
    tfidf_logreg_path = PHOBERT_DIR / "tfidf_logreg_round5.joblib"
    if tfidf_logreg_path.exists():
        preds_test = predict_tfidf(tfidf_logreg_path, test_texts)
        metrics_test = compute_metrics(test_labels, preds_test)
        all_results["in_domain"]["tfidf_logreg"] = metrics_test
        print(f"  In-domain:  Acc={metrics_test['accuracy']:.4f}, F1={metrics_test['f1_macro']:.4f}")

        preds_vsmec = predict_tfidf(tfidf_logreg_path, vsmec_texts)
        metrics_vsmec = compute_metrics(vsmec_labels, preds_vsmec)
        all_results["cross_domain"]["tfidf_logreg"] = metrics_vsmec
        print(f"  Cross-domain: Acc={metrics_vsmec['accuracy']:.4f}, F1={metrics_vsmec['f1_macro']:.4f}")
    else:
        print(f"  Model not found: {tfidf_logreg_path}")

    # ── 4. TF-IDF + LinearSVC ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. TF-IDF + LINEARSVC")
    print("=" * 70)
    tfidf_svc_path = PHOBERT_DIR / "tfidf_linearsvc_round5.joblib"
    if tfidf_svc_path.exists():
        preds_test = predict_tfidf(tfidf_svc_path, test_texts)
        metrics_test = compute_metrics(test_labels, preds_test)
        all_results["in_domain"]["tfidf_svc"] = metrics_test
        print(f"  In-domain:  Acc={metrics_test['accuracy']:.4f}, F1={metrics_test['f1_macro']:.4f}")

        preds_vsmec = predict_tfidf(tfidf_svc_path, vsmec_texts)
        metrics_vsmec = compute_metrics(vsmec_labels, preds_vsmec)
        all_results["cross_domain"]["tfidf_svc"] = metrics_vsmec
        print(f"  Cross-domain: Acc={metrics_vsmec['accuracy']:.4f}, F1={metrics_vsmec['f1_macro']:.4f}")
    else:
        print(f"  Model not found: {tfidf_svc_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nIn-Domain ({len(test_df)} samples):")
    print(f"  {'Model':<25} | {'Acc':>7} | {'F1-M':>7} | {'F1-D':>7}")
    print("  " + "-" * 55)
    for model_name, metrics in all_results["in_domain"].items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            print(f"  {model_name:<25} | {metrics['accuracy']:>7.4f} | "
                  f"{metrics['f1_macro']:>7.4f} | {metrics['f1_depression']:>7.4f}")

    print(f"\nCross-Domain ({len(vsmec_df)} samples):")
    print(f"  {'Model':<25} | {'Acc':>7} | {'F1-M':>7} | {'F1-D':>7}")
    print("  " + "-" * 55)
    for model_name, metrics in all_results["cross_domain"].items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            print(f"  {model_name:<25} | {metrics['accuracy']:>7.4f} | "
                  f"{metrics['f1_macro']:>7.4f} | {metrics['f1_depression']:>7.4f}")

    # ── Save results ────────────────────────────────────────────────────────
    results_file = OUTPUT_DIR / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {results_file}")
    return all_results


if __name__ == "__main__":
    main()
