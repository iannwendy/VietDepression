"""Complete, Reproducible Round 6 v2 Model Training Pipeline.

Addresses Reviewer 1 (Comments 1, 2, 4, 5) and Reviewer 2 (Major Comments 1, 3, 4):
1. Explicitly documents and logs exact split class counts, hardware, software
   versions, and all model hyperparameters (including LogReg L2 regularization
   C=1.0, solver='lbfgs', class_weight='balanced').
2. Trains BOTH BiLSTM (unweighted CrossEntropyLoss) and BiLSTM (class-weighted
   CrossEntropyLoss, w_1 = N_neg / N_pos) across 3 seeds (42, 123, 2024) to
   reconcile the weighted vs. unweighted comparison.
3. Explicitly separates and reports all three multi-seed aggregation rules:
   - `mean_std_across_seeds`: mean +/- SD of per-seed metrics across seeds (42, 123, 2024)
   - `majority_vote`: hard majority vote of binary predictions across 3 seeds
   - `soft_vote_prob_avg`: average predicted positive probability across 3 seeds
4. Retains the exact 2x2 confusion matrix `[[TN, FP], [FN, TP]]`, ROC-AUC, and
   PR-AUC for every single model and ensemble (never stripping confusion matrices).
5. Evaluates on In-Domain Test (n=1,371), Cross-Domain UIT-VSMEC Full
   (Sadness+Fear vs. Enjoyment, n=3,084), and Cross-Domain UIT-VSMEC
   Sadness-Only (Sadness vs. Enjoyment, n=2,390).

Usage:
    PYTHONPATH="$PWD" .venv/bin/python scripts/retrain_all_models_round6.py
"""

from __future__ import annotations

import json
import os
import platform
import random
import re
import shutil
import sys
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import transformers
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# Prefer local HuggingFace cache if already downloaded, without crashing if online needed
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
if (HF_CACHE / "models--vinai--phobert-base").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

TRAIN_FILE = DATA_DIR / "labeled" / "final_train.csv"
VAL_FILE = DATA_DIR / "labeled" / "final_val.csv"
TEST_FILE = DATA_DIR / "labeled" / "final_test.csv"
VSMEC_FILE = PROJECT_DIR / "data_unified" / "cross_domain_test.csv"

MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
SEEDS = [42, 123, 2024]

OUTPUT_DIR = MODEL_DIR / "round6_v2_retrained"
LEGACY_OUTPUT_DIR = MODEL_DIR / "round6_retrained"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper Functions ──────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(
            precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "precision_depression": float(
            precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
        ),
        "recall_depression": float(
            recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
        ),
        "f1_depression": float(
            f1_score(
                y_true_arr, y_pred_arr, average="binary", pos_label=1, zero_division=0
            )
        ),
        "f1_normal": float(
            f1_score(
                y_true_arr, y_pred_arr, average="binary", pos_label=0, zero_division=0
            )
        ),
        "predicted_positives": int(tp + fp),
        "predicted_negatives": int(tn + fn),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    if y_prob is not None and len(np.unique(y_true_arr)) == 2:
        probs_arr = np.asarray(y_prob, dtype=float)
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, probs_arr))
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, probs_arr))
    return metrics


def summarize_seed_metrics(per_seed_metrics: list[dict]) -> dict:
    """Compute mean and sample standard deviation (ddof=1) across seeds."""
    keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "precision_depression",
        "recall_depression",
        "f1_depression",
        "f1_normal",
        "roc_auc",
        "pr_auc",
    ]
    summary: dict = {"n_seeds": len(per_seed_metrics), "seeds": SEEDS}
    for k in keys:
        vals = [m[k] for m in per_seed_metrics if k in m and m[k] is not None]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    # Keep top-level aliases so legacy readers still find `accuracy`, `f1_macro`, `f1_depression`
    summary["accuracy"] = summary.get("accuracy_mean", 0.0)
    summary["f1_macro"] = summary.get("f1_macro_mean", 0.0)
    summary["f1_depression"] = summary.get("f1_depression_mean", 0.0)
    return summary


# ── PhoBERT Setup ────────────────────────────────────────────────────────────
print("Loading PhoBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)


class PhoBERTDataset(Dataset):
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
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── BiLSTM Setup ────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, texts):
        freq = {}
        for text in texts:
            for word in str(text).split():
                freq[word] = freq.get(word, 0) + 1
        idx = 2
        for word, count in freq.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def text_to_sequence(self, text):
        seq = [self.stoi.get(w, 1) for w in str(text).split()]
        return seq if seq else [1]


class BiLSTMClassifier(nn.Module):
    def __init__(
        self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

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
        return torch.tensor(seq, dtype=torch.long), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, torch.tensor(labels), lengths


# ── Training Functions ───────────────────────────────────────────────────────
def train_phobert(seed, train_texts, train_labels, val_texts, val_labels):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"TRAINING PHOBERT - SEED {seed}")
    print(f"{'='*60}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)

    train_dataset = PhoBERTDataset(train_texts, train_labels)
    val_dataset = PhoBERTDataset(val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.06),
        num_training_steps=total_steps,
    )

    n_neg = int((train_labels == 0).sum())
    n_pos = int((train_labels == 1).sum())
    pos_weight = n_neg / n_pos
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    )

    best_f1 = 0.0
    seed_dir = OUTPUT_DIR / f"phobert_seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = criterion(outputs.logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        f1 = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.4f}, Val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(str(seed_dir / "best_model"))
            tokenizer.save_pretrained(str(seed_dir / "best_model"))

    print(f"  Best Val F1 for seed {seed}: {best_f1:.4f}")
    return best_f1, seed_dir


def train_bilstm(
    seed,
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    use_class_weights: bool = False,
):
    set_seed(seed)
    variant_tag = "bilstm_weighted" if use_class_weights else "bilstm"
    print(f"\n{'='*60}")
    print(f"TRAINING {variant_tag.upper()} - SEED {seed}")
    print(f"{'='*60}")

    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocab(train_texts)
    print(f"  Vocabulary size: {len(vocab)}")

    train_dataset = BiLSTMDataset(train_texts, train_labels, vocab)
    val_dataset = BiLSTMDataset(val_texts, val_labels, vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
    )

    model = BiLSTMClassifier(
        len(vocab), embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=0.001)
    if use_class_weights:
        n_neg = int((train_labels == 0).sum())
        n_pos = int((train_labels == 1).sum())
        pos_weight = n_neg / n_pos
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    seed_dir = OUTPUT_DIR / f"{variant_tag}_seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for texts, labels, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(texts.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for texts, labels, _ in val_loader:
                outputs = model(texts.to(device))
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        f1 = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.4f}, Val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "use_class_weights": use_class_weights,
                    "seed": seed,
                },
                seed_dir / "best_model.pt",
            )

    print(f"  Best Val F1 for {variant_tag} seed {seed}: {best_f1:.4f}")
    return best_f1, seed_dir


def evaluate_phobert(model_dir, texts, labels, split_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir / "best_model")
    )
    model.to(device)
    model.eval()

    dataset = PhoBERTDataset(texts)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    del model
    preds_arr = np.array(all_preds, dtype=int)
    probs_arr = np.array(all_probs, dtype=float)
    metrics = compute_metrics(labels, preds_arr, probs_arr)
    print(
        f"  {split_name} - PhoBERT: Acc={metrics['accuracy']:.4f}, "
        f"F1-M={metrics['f1_macro']:.4f}, ROC-AUC={metrics.get('roc_auc', 0):.4f}"
    )
    return metrics, preds_arr, probs_arr


def evaluate_bilstm(model_path, texts, labels, split_name):
    eval_device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=eval_device, weights_only=False)
    vocab = checkpoint["vocab"]

    model = BiLSTMClassifier(
        len(vocab), embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(eval_device)
    model.eval()

    dataset = BiLSTMDataset(texts, labels, vocab)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for texts_batch, _, _ in loader:
            outputs = model(texts_batch.to(eval_device))
            probs = torch.nn.functional.softmax(outputs, dim=-1)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    del model
    preds_arr = np.array(all_preds, dtype=int)
    probs_arr = np.array(all_probs, dtype=float)
    metrics = compute_metrics(labels, preds_arr, probs_arr)
    print(
        f"  {split_name} - BiLSTM: Acc={metrics['accuracy']:.4f}, "
        f"F1-M={metrics['f1_macro']:.4f}, ROC-AUC={metrics.get('roc_auc', 0):.4f}"
    )
    return metrics, preds_arr, probs_arr


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR / f"round6_retrained_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("ROUND 6 v2 COMPLETE REPRODUCIBLE RETRAINING")
    print("=" * 75)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_texts = train_df["comment_text"].fillna("").astype(str).values
    train_labels = train_df["label"].astype(int).values
    val_texts = val_df["comment_text"].fillna("").astype(str).values
    val_labels = val_df["label"].astype(int).values
    test_texts = test_df["comment_text"].fillna("").astype(str).values
    test_labels = test_df["label"].astype(int).values

    print(
        f"Train: {len(train_df):,} (Normal={(train_labels==0).sum():,}, Depression={(train_labels==1).sum():,})"
    )
    print(
        f"Val:   {len(val_df):,} (Normal={(val_labels==0).sum():,}, Depression={(val_labels==1).sum():,})"
    )
    print(
        f"Test:  {len(test_df):,} (Normal={(test_labels==0).sum():,}, Depression={(test_labels==1).sum():,})"
    )

    # Load cross-domain VSMEC (Full & Sadness-Only)
    vsmec_df = pd.read_csv(VSMEC_FILE)
    vsmec_texts = (
        vsmec_df["comment_text"].fillna("").astype(str).values
        if "comment_text" in vsmec_df.columns
        else vsmec_df["text"].fillna("").astype(str).values
    )
    vsmec_labels = vsmec_df["label"].astype(int).values

    sadness_mask = vsmec_df["original_label"].isin(["Sadness", "Enjoyment"]).to_numpy()
    vsmec_sad_texts = vsmec_texts[sadness_mask]
    vsmec_sad_labels = vsmec_labels[sadness_mask]
    print(
        f"Cross-domain VSMEC (Full): {len(vsmec_df):,} | (Sadness-Only): {len(vsmec_sad_labels):,}"
    )

    n_neg_train = int((train_labels == 0).sum())
    n_pos_train = int((train_labels == 1).sum())
    pos_class_weight = round(n_neg_train / n_pos_train, 4)

    all_results = {
        "timestamp": timestamp,
        "protocol_version": "round6_v2_audited",
        "hardware_and_environment": {
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "processor": platform.processor(),
            "device": str(device),
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "sklearn_version": sklearn.__version__,
            "seeds": SEEDS,
        },
        "hyperparameters": {
            "tfidf_logreg": {
                "max_features": 10000,
                "ngram_range": [1, 2],
                "min_df": 2,
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42,
            },
            "tfidf_svc": {
                "max_features": 10000,
                "ngram_range": [1, 2],
                "min_df": 2,
                "C": 1.0,
                "loss": "squared_hinge",
                "class_weight": "balanced",
                "max_iter": 2000,
                "random_state": 42,
            },
            "bilstm_unweighted": {
                "vocab_freq_threshold": 2,
                "embed_dim": 128,
                "hidden_dim": 128,
                "num_layers": 2,
                "bidirectional": True,
                "dropout": 0.3,
                "batch_size": 64,
                "lr": 1e-3,
                "optimizer": "AdamW",
                "epochs": EPOCHS,
                "loss": "CrossEntropyLoss(unweighted)",
                "seeds": SEEDS,
            },
            "bilstm_weighted": {
                "vocab_freq_threshold": 2,
                "embed_dim": 128,
                "hidden_dim": 128,
                "num_layers": 2,
                "bidirectional": True,
                "dropout": 0.3,
                "batch_size": 64,
                "lr": 1e-3,
                "optimizer": "AdamW",
                "epochs": EPOCHS,
                "loss": f"CrossEntropyLoss(weight=[1.0, {pos_class_weight}])",
                "seeds": SEEDS,
            },
            "phobert": {
                "pretrained_model": MODEL_NAME,
                "max_length": MAX_LEN,
                "batch_size": BATCH_SIZE,
                "eval_batch_size": 32,
                "lr": 2e-5,
                "weight_decay": 0.01,
                "warmup_ratio": 0.06,
                "grad_clip_norm": 1.0,
                "epochs": EPOCHS,
                "loss": f"CrossEntropyLoss(weight=[1.0, {pos_class_weight}])",
                "checkpoint_selection": "best validation macro-F1 on final_val.csv",
                "seeds": SEEDS,
            },
        },
        "dataset_info": {
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "vsmec_full_size": int(len(vsmec_df)),
            "vsmec_sadness_only_size": int(len(vsmec_sad_labels)),
            "train_label_dist": {
                "0": int((train_labels == 0).sum()),
                "1": int((train_labels == 1).sum()),
            },
            "val_label_dist": {
                "0": int((val_labels == 0).sum()),
                "1": int((val_labels == 1).sum()),
            },
            "test_label_dist": {
                "0": int((test_labels == 0).sum()),
                "1": int((test_labels == 1).sum()),
            },
            "vsmec_full_label_dist": {
                "0": int((vsmec_labels == 0).sum()),
                "1": int((vsmec_labels == 1).sum()),
            },
        },
        "in_domain": {},
        "cross_domain": {},
        "cross_domain_sadness_only": {},
    }

    # ── 0. Trivial All-Negative Reference Baseline ───────────────────────────
    all_results_in_neg = compute_metrics(
        test_labels, np.zeros_like(test_labels), np.zeros_like(test_labels, dtype=float)
    )
    all_results["in_domain"]["all_negative_baseline"] = all_results_in_neg
    all_results["cross_domain"]["all_negative_baseline"] = compute_metrics(
        vsmec_labels,
        np.zeros_like(vsmec_labels),
        np.zeros_like(vsmec_labels, dtype=float),
    )

    # ── 1. TF-IDF + Logistic Regression ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. TRAINING TF-IDF + LOGISTIC REGRESSION")
    print("=" * 60)

    tfidf_logreg_path = OUTPUT_DIR / "tfidf_logreg_round6_v2.joblib"
    tfidf_logreg_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    tfidf_logreg_pipeline.fit(train_texts, train_labels)
    joblib.dump(tfidf_logreg_pipeline, tfidf_logreg_path)
    joblib.dump(tfidf_logreg_pipeline, OUTPUT_DIR / "tfidf_logreg_round6.joblib")

    logreg_preds = tfidf_logreg_pipeline.predict(test_texts)
    logreg_probs = tfidf_logreg_pipeline.predict_proba(test_texts)[:, 1]
    logreg_metrics = compute_metrics(test_labels, logreg_preds, logreg_probs)

    # 5-fold stratified CV on train set to quantify variance for classical model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logreg_cv_f1s = []
    for tr_idx, va_idx in skf.split(train_texts, train_labels):
        cv_pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        cv_pipe.fit(train_texts[tr_idx], train_labels[tr_idx])
        logreg_cv_f1s.append(
            float(
                f1_score(
                    train_labels[va_idx],
                    cv_pipe.predict(train_texts[va_idx]),
                    average="macro",
                )
            )
        )
    logreg_metrics["cv5_f1_macro_mean"] = float(np.mean(logreg_cv_f1s))
    logreg_metrics["cv5_f1_macro_std"] = float(np.std(logreg_cv_f1s, ddof=1))

    print(
        f"TF-IDF + LogReg (In-Domain): Acc={logreg_metrics['accuracy']:.4f}, "
        f"F1-M={logreg_metrics['f1_macro']:.4f}, ROC-AUC={logreg_metrics['roc_auc']:.4f} "
        f"(5-fold CV F1={logreg_metrics['cv5_f1_macro_mean']:.4f}±{logreg_metrics['cv5_f1_macro_std']:.4f})"
    )
    all_results["in_domain"]["tfidf_logreg"] = logreg_metrics

    logreg_vsmec_preds = tfidf_logreg_pipeline.predict(vsmec_texts)
    logreg_vsmec_probs = tfidf_logreg_pipeline.predict_proba(vsmec_texts)[:, 1]
    all_results["cross_domain"]["tfidf_logreg"] = compute_metrics(
        vsmec_labels, logreg_vsmec_preds, logreg_vsmec_probs
    )
    all_results["cross_domain_sadness_only"]["tfidf_logreg"] = compute_metrics(
        vsmec_sad_labels,
        logreg_vsmec_preds[sadness_mask],
        logreg_vsmec_probs[sadness_mask],
    )

    # ── 2. TF-IDF + LinearSVC ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. TRAINING TF-IDF + LINEARSVC")
    print("=" * 60)

    tfidf_svc_path = OUTPUT_DIR / "tfidf_linearsvc_round6_v2.joblib"
    tfidf_svc_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2),
            ),
            (
                "clf",
                LinearSVC(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    tfidf_svc_pipeline.fit(train_texts, train_labels)
    joblib.dump(tfidf_svc_pipeline, tfidf_svc_path)
    joblib.dump(tfidf_svc_pipeline, OUTPUT_DIR / "tfidf_linearsvc_round6.joblib")

    svc_preds = tfidf_svc_pipeline.predict(test_texts)
    svc_scores = expit(tfidf_svc_pipeline.decision_function(test_texts))
    svc_metrics = compute_metrics(test_labels, svc_preds, svc_scores)

    svc_cv_f1s = []
    for tr_idx, va_idx in skf.split(train_texts, train_labels):
        cv_pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2),
                ),
                (
                    "clf",
                    LinearSVC(
                        C=1.0,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        cv_pipe.fit(train_texts[tr_idx], train_labels[tr_idx])
        svc_cv_f1s.append(
            float(
                f1_score(
                    train_labels[va_idx],
                    cv_pipe.predict(train_texts[va_idx]),
                    average="macro",
                )
            )
        )
    svc_metrics["cv5_f1_macro_mean"] = float(np.mean(svc_cv_f1s))
    svc_metrics["cv5_f1_macro_std"] = float(np.std(svc_cv_f1s, ddof=1))

    print(
        f"TF-IDF + LinearSVC (In-Domain): Acc={svc_metrics['accuracy']:.4f}, "
        f"F1-M={svc_metrics['f1_macro']:.4f}, ROC-AUC={svc_metrics['roc_auc']:.4f} "
        f"(5-fold CV F1={svc_metrics['cv5_f1_macro_mean']:.4f}±{svc_metrics['cv5_f1_macro_std']:.4f})"
    )
    all_results["in_domain"]["tfidf_svc"] = svc_metrics

    svc_vsmec_preds = tfidf_svc_pipeline.predict(vsmec_texts)
    svc_vsmec_scores = expit(tfidf_svc_pipeline.decision_function(vsmec_texts))
    all_results["cross_domain"]["tfidf_svc"] = compute_metrics(
        vsmec_labels, svc_vsmec_preds, svc_vsmec_scores
    )
    all_results["cross_domain_sadness_only"]["tfidf_svc"] = compute_metrics(
        vsmec_sad_labels,
        svc_vsmec_preds[sadness_mask],
        svc_vsmec_scores[sadness_mask],
    )

    # ── 3A & 3B. BiLSTM (Unweighted & Class-Weighted across 3 seeds) ─────────
    for use_weights, prefix in [
        (False, "bilstm"),
        (True, "bilstm_weighted"),
    ]:
        print("\n" + "=" * 60)
        print(f"3. TRAINING {prefix.upper()} (3 SEEDS: {SEEDS})")
        print("=" * 60)

        in_metrics_list, cross_metrics_list, sad_metrics_list = [], [], []
        test_preds_seeds, test_probs_seeds = [], []
        vsmec_preds_seeds, vsmec_probs_seeds = [], []

        for seed in SEEDS:
            val_f1, model_dir = train_bilstm(
                seed,
                train_texts,
                train_labels,
                val_texts,
                val_labels,
                use_class_weights=use_weights,
            )
            ckpt_path = model_dir / "best_model.pt"
            m_in, p_in, pr_in = evaluate_bilstm(
                ckpt_path, test_texts, test_labels, f"Test ({prefix} s={seed})"
            )
            m_in["seed"] = seed
            m_in["val_f1_macro"] = float(val_f1)
            all_results["in_domain"][f"{prefix}_seed{seed}"] = m_in
            in_metrics_list.append(m_in)
            test_preds_seeds.append(p_in)
            test_probs_seeds.append(pr_in)

            m_cr, p_cr, pr_cr = evaluate_bilstm(
                ckpt_path, vsmec_texts, vsmec_labels, f"VSMEC ({prefix} s={seed})"
            )
            m_cr["seed"] = seed
            all_results["cross_domain"][f"{prefix}_seed{seed}"] = m_cr
            cross_metrics_list.append(m_cr)
            vsmec_preds_seeds.append(p_cr)
            vsmec_probs_seeds.append(pr_cr)

            m_sad = compute_metrics(
                vsmec_sad_labels, p_cr[sadness_mask], pr_cr[sadness_mask]
            )
            all_results["cross_domain_sadness_only"][f"{prefix}_seed{seed}"] = m_sad
            sad_metrics_list.append(m_sad)

        # Aggregations: (1) mean ± std across seeds, (2) hard majority vote, (3) soft prob avg
        mean_in = summarize_seed_metrics(in_metrics_list)
        mean_cr = summarize_seed_metrics(cross_metrics_list)
        mean_sad = summarize_seed_metrics(sad_metrics_list)

        maj_pred_in = (np.mean(np.stack(test_preds_seeds, axis=0), axis=0) >= 0.5).astype(int)
        soft_prob_in = np.mean(np.stack(test_probs_seeds, axis=0), axis=0)
        soft_pred_in = (soft_prob_in >= 0.5).astype(int)

        maj_pred_cr = (np.mean(np.stack(vsmec_preds_seeds, axis=0), axis=0) >= 0.5).astype(int)
        soft_prob_cr = np.mean(np.stack(vsmec_probs_seeds, axis=0), axis=0)
        soft_pred_cr = (soft_prob_cr >= 0.5).astype(int)

        all_results["in_domain"][f"{prefix}_mean_std"] = mean_in
        all_results["in_domain"][f"{prefix}_majority_vote"] = compute_metrics(
            test_labels, maj_pred_in, soft_prob_in
        )
        all_results["in_domain"][f"{prefix}_soft_vote"] = compute_metrics(
            test_labels, soft_pred_in, soft_prob_in
        )
        # Keep `{prefix}_avg` pointing to majority_vote (with mean_of_seeds recorded inside)
        avg_entry_in = dict(all_results["in_domain"][f"{prefix}_majority_vote"])
        avg_entry_in["aggregation_rule"] = "majority_vote_over_3_seeds"
        avg_entry_in["mean_f1_macro_across_seeds"] = mean_in["f1_macro_mean"]
        avg_entry_in["std_f1_macro_across_seeds"] = mean_in["f1_macro_std"]
        all_results["in_domain"][f"{prefix}_avg"] = avg_entry_in

        all_results["cross_domain"][f"{prefix}_mean_std"] = mean_cr
        all_results["cross_domain"][f"{prefix}_majority_vote"] = compute_metrics(
            vsmec_labels, maj_pred_cr, soft_prob_cr
        )
        all_results["cross_domain"][f"{prefix}_soft_vote"] = compute_metrics(
            vsmec_labels, soft_pred_cr, soft_prob_cr
        )
        avg_entry_cr = dict(all_results["cross_domain"][f"{prefix}_majority_vote"])
        avg_entry_cr["aggregation_rule"] = "majority_vote_over_3_seeds"
        avg_entry_cr["mean_f1_macro_across_seeds"] = mean_cr["f1_macro_mean"]
        avg_entry_cr["std_f1_macro_across_seeds"] = mean_cr["f1_macro_std"]
        all_results["cross_domain"][f"{prefix}_avg"] = avg_entry_cr

        all_results["cross_domain_sadness_only"][f"{prefix}_mean_std"] = mean_sad
        all_results["cross_domain_sadness_only"][f"{prefix}_majority_vote"] = compute_metrics(
            vsmec_sad_labels, maj_pred_cr[sadness_mask], soft_prob_cr[sadness_mask]
        )

    # ── 4. PhoBERT (3 seeds) ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"4. TRAINING PHOBERT (3 SEEDS: {SEEDS})")
    print("=" * 60)

    phobert_in_list, phobert_cr_list, phobert_sad_list = [], [], []
    phobert_test_preds, phobert_test_probs = [], []
    phobert_vsmec_preds, phobert_vsmec_probs = [], []

    for seed in SEEDS:
        val_f1, model_dir = train_phobert(
            seed, train_texts, train_labels, val_texts, val_labels
        )
        m_in, p_in, pr_in = evaluate_phobert(
            model_dir, test_texts, test_labels, f"Test (seed={seed})"
        )
        m_in["seed"] = seed
        m_in["val_f1_macro"] = float(val_f1)
        all_results["in_domain"][f"phobert_seed{seed}"] = m_in
        phobert_in_list.append(m_in)
        phobert_test_preds.append(p_in)
        phobert_test_probs.append(pr_in)

        m_cr, p_cr, pr_cr = evaluate_phobert(
            model_dir, vsmec_texts, vsmec_labels, f"VSMEC (seed={seed})"
        )
        m_cr["seed"] = seed
        all_results["cross_domain"][f"phobert_seed{seed}"] = m_cr
        phobert_cr_list.append(m_cr)
        phobert_vsmec_preds.append(p_cr)
        phobert_vsmec_probs.append(pr_cr)

        m_sad = compute_metrics(
            vsmec_sad_labels, p_cr[sadness_mask], pr_cr[sadness_mask]
        )
        all_results["cross_domain_sadness_only"][f"phobert_seed{seed}"] = m_sad
        phobert_sad_list.append(m_sad)

    # PhoBERT aggregations: (1) mean ± std across seeds, (2) hard majority vote, (3) soft prob avg
    pho_mean_in = summarize_seed_metrics(phobert_in_list)
    pho_mean_cr = summarize_seed_metrics(phobert_cr_list)
    pho_mean_sad = summarize_seed_metrics(phobert_sad_list)

    pho_maj_pred_in = (
        np.mean(np.stack(phobert_test_preds, axis=0), axis=0) >= 0.5
    ).astype(int)
    pho_soft_prob_in = np.mean(np.stack(phobert_test_probs, axis=0), axis=0)
    pho_soft_pred_in = (pho_soft_prob_in >= 0.5).astype(int)

    pho_maj_pred_cr = (
        np.mean(np.stack(phobert_vsmec_preds, axis=0), axis=0) >= 0.5
    ).astype(int)
    pho_soft_prob_cr = np.mean(np.stack(phobert_vsmec_probs, axis=0), axis=0)
    pho_soft_pred_cr = (pho_soft_prob_cr >= 0.5).astype(int)

    all_results["in_domain"]["phobert_mean_std"] = pho_mean_in
    all_results["in_domain"]["phobert_majority_vote"] = compute_metrics(
        test_labels, pho_maj_pred_in, pho_soft_prob_in
    )
    all_results["in_domain"]["phobert_soft_vote"] = compute_metrics(
        test_labels, pho_soft_pred_in, pho_soft_prob_in
    )
    pho_avg_in = dict(all_results["in_domain"]["phobert_soft_vote"])
    pho_avg_in["aggregation_rule"] = "soft_vote_prob_avg_over_3_seeds"
    pho_avg_in["majority_vote_f1_macro"] = all_results["in_domain"][
        "phobert_majority_vote"
    ]["f1_macro"]
    pho_avg_in["majority_vote_confusion_matrix"] = all_results["in_domain"][
        "phobert_majority_vote"
    ]["confusion_matrix"]
    pho_avg_in["mean_f1_macro_across_seeds"] = pho_mean_in["f1_macro_mean"]
    pho_avg_in["std_f1_macro_across_seeds"] = pho_mean_in["f1_macro_std"]
    all_results["in_domain"]["phobert_avg"] = pho_avg_in

    all_results["cross_domain"]["phobert_mean_std"] = pho_mean_cr
    all_results["cross_domain"]["phobert_majority_vote"] = compute_metrics(
        vsmec_labels, pho_maj_pred_cr, pho_soft_prob_cr
    )
    all_results["cross_domain"]["phobert_soft_vote"] = compute_metrics(
        vsmec_labels, pho_soft_pred_cr, pho_soft_prob_cr
    )
    pho_avg_cr = dict(all_results["cross_domain"]["phobert_soft_vote"])
    pho_avg_cr["aggregation_rule"] = "soft_vote_prob_avg_over_3_seeds"
    pho_avg_cr["majority_vote_f1_macro"] = all_results["cross_domain"][
        "phobert_majority_vote"
    ]["f1_macro"]
    pho_avg_cr["majority_vote_confusion_matrix"] = all_results["cross_domain"][
        "phobert_majority_vote"
    ]["confusion_matrix"]
    pho_avg_cr["mean_f1_macro_across_seeds"] = pho_mean_cr["f1_macro_mean"]
    pho_avg_cr["std_f1_macro_across_seeds"] = pho_mean_cr["f1_macro_std"]
    all_results["cross_domain"]["phobert_avg"] = pho_avg_cr

    all_results["cross_domain_sadness_only"]["phobert_mean_std"] = pho_mean_sad
    all_results["cross_domain_sadness_only"]["phobert_majority_vote"] = compute_metrics(
        vsmec_sad_labels, pho_maj_pred_cr[sadness_mask], pho_soft_prob_cr[sadness_mask]
    )
    all_results["cross_domain_sadness_only"]["phobert_soft_vote"] = compute_metrics(
        vsmec_sad_labels, pho_soft_pred_cr[sadness_mask], pho_soft_prob_cr[sadness_mask]
    )

    # ── Summary Table ────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("IN-DOMAIN TEST RESULTS SUMMARY (n=1,371)")
    print("=" * 85)
    print(
        f"{'Model':<28} | {'Acc':>7} | {'Prec-M':>7} | {'Rec-M':>7} | {'F1-M':>7} | {'F1-D':>7} | {'ROC-AUC':>7}"
    )
    print("-" * 85)

    for model_name, metrics in all_results["in_domain"].items():
        if isinstance(metrics, dict) and "accuracy" in metrics and not model_name.endswith("_mean_std"):
            acc = metrics.get("accuracy", 0)
            prec = metrics.get("precision_macro", 0)
            rec = metrics.get("recall_macro", 0)
            f1_m = metrics.get("f1_macro", 0)
            f1_d = metrics.get("f1_depression", 0)
            roc = metrics.get("roc_auc", 0)
            print(
                f"{model_name:<28} | {acc:>7.4f} | {prec:>7.4f} | {rec:>7.4f} | {f1_m:>7.4f} | {f1_d:>7.4f} | {roc:>7.4f}"
            )

    # Save results
    results_file = results_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    canonical_eval = OUTPUT_DIR / "evaluation_results.json"
    with open(canonical_eval, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Ensure `models/round6_retrained` symlink points to `models/round6_v2_retrained`
    if not LEGACY_OUTPUT_DIR.exists():
        try:
            LEGACY_OUTPUT_DIR.symlink_to(OUTPUT_DIR.name)
        except Exception:
            pass

    print(f"\n{'='*75}")
    print(f"RESULTS SAVED TO: {results_file}")
    print(f"CANONICAL RESULTS: {canonical_eval}")
    print(f"MODELS SAVED TO: {OUTPUT_DIR}")
    print(f"{'='*75}")

    return all_results


if __name__ == "__main__":
    main()
