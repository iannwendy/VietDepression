"""Complete Round 5 Retraining with Fixed Data.

Retrains all models on the corrected datasets:
1. TF-IDF + Logistic Regression
2. TF-IDF + LinearSVC
3. BiLSTM (3 seeds)
4. PhoBERT (3 seeds)

Usage:
    .venv/bin/python scripts/retrain_all_models_round5.py
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
from datetime import datetime
import json
import random
import re
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

TRAIN_FILE = DATA_DIR / "labeled" / "final_train.csv"
VAL_FILE = DATA_DIR / "labeled" / "final_val.csv"
TEST_FILE = DATA_DIR / "labeled" / "final_test.csv"

MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
SEEDS = [42, 123, 2024]

OUTPUT_DIR = MODEL_DIR / "round5_retrained"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helper Functions ──────────────────────────────────────────────────────────
def set_seed(seed: int):
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
            text, padding="max_length", truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
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

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
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
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

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
        optimizer, num_warmup_steps=int(total_steps * 0.06), num_training_steps=total_steps
    )

    n_neg = (train_labels == 0).sum()
    n_pos = (train_labels == 1).sum()
    pos_weight = n_neg / n_pos
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    )

    best_f1 = 0.0
    seed_dir = OUTPUT_DIR / f"phobert_seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
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
                    attention_mask=batch["attention_mask"].to(device)
                )
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        f1 = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(str(seed_dir / "best_model"))

    print(f"  Best F1 for seed {seed}: {best_f1:.4f}")
    return best_f1, seed_dir

def train_bilstm(seed, train_texts, train_labels, val_texts, val_labels):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"TRAINING BILSTM - SEED {seed}")
    print(f"{'='*60}")

    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocab(train_texts)
    print(f"  Vocabulary size: {len(vocab)}")

    train_dataset = BiLSTMDataset(train_texts, train_labels, vocab)
    val_dataset = BiLSTMDataset(val_texts, val_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMClassifier(len(vocab), embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    seed_dir = OUTPUT_DIR / f"bilstm_seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
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
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
            }, seed_dir / "best_model.pt")

    print(f"  Best F1 for seed {seed}: {best_f1:.4f}")
    return best_f1, seed_dir

def evaluate_phobert(model_dir, texts, labels, split_name):
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir / "best_model"))
    model.to(device)
    model.eval()

    dataset = PhoBERTDataset(texts)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1].cpu().numpy())

    metrics = compute_metrics(labels, all_preds)
    print(f"\n{split_name} - PhoBERT: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
    return metrics, all_preds, all_probs

def evaluate_bilstm(model_path, texts, labels, vocab, split_name):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]

    model = BiLSTMClassifier(len(vocab), embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = BiLSTMDataset(texts, labels, vocab)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    with torch.no_grad():
        for texts_batch, _, _ in loader:
            outputs = model(texts_batch.to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)

    metrics = compute_metrics(labels, all_preds)
    print(f"{split_name} - BiLSTM: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
    return metrics, all_preds

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR / f"round5_retrained_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("ROUND 5 COMPLETE RETRAINING WITH FIXED DATA")
    print("="*70)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Train label dist: {dict(train_df['label'].value_counts())}")

    train_texts = train_df["comment_text"].values
    train_labels = train_df["label"].values
    val_texts = val_df["comment_text"].values
    val_labels = val_df["label"].values
    test_texts = test_df["comment_text"].values
    test_labels = test_df["label"].values

    all_results = {
        "timestamp": timestamp,
        "dataset_info": {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_label_dist": dict(train_df["label"].value_counts()),
        },
        "in_domain": {},
        "cross_domain": {},
    }

    # ── Load cross-domain VSMEC ────────────────────────────────────────────
    vsmec_path = PROJECT_DIR / "data_unified" / "cross_domain_test.csv"
    vsmec_df = pd.read_csv(vsmec_path)
    vsmec_texts = (
        vsmec_df["comment_text"].values
        if "comment_text" in vsmec_df.columns
        else vsmec_df["text"].values
    )
    vsmec_labels = vsmec_df["label"].values
    print(f"\nCross-domain VSMEC: {len(vsmec_df):,} samples")

    # ── 1. TF-IDF + Logistic Regression ──────────────────────────────────────
    print("\n" + "="*60)
    print("1. TRAINING TF-IDF + LOGISTIC REGRESSION")
    print("="*60)

    tfidf_logreg_path = OUTPUT_DIR / "tfidf_logreg_round5.joblib"
    tfidf_logreg_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    tfidf_logreg_pipeline.fit(train_texts, train_labels)

    logreg_preds = tfidf_logreg_pipeline.predict(test_texts)
    logreg_metrics = compute_metrics(test_labels, logreg_preds)

    joblib.dump(tfidf_logreg_pipeline, tfidf_logreg_path)
    print(f"TF-IDF + LogReg: Acc={logreg_metrics['accuracy']:.4f}, F1={logreg_metrics['f1_macro']:.4f}")
    all_results["in_domain"]["tfidf_logreg"] = logreg_metrics

    # Cross-domain TF-IDF + LogReg
    logreg_vsmec_preds = tfidf_logreg_pipeline.predict(vsmec_texts)
    all_results["cross_domain"]["tfidf_logreg"] = compute_metrics(vsmec_labels, logreg_vsmec_preds)

    # ── 2. TF-IDF + LinearSVC ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("2. TRAINING TF-IDF + LINEARSVC")
    print("="*60)

    tfidf_svc_path = OUTPUT_DIR / "tfidf_linearsvc_round5.joblib"
    tfidf_svc_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)),
        ("clf", LinearSVC(max_iter=2000, class_weight="balanced", random_state=42))
    ])
    tfidf_svc_pipeline.fit(train_texts, train_labels)

    svc_preds = tfidf_svc_pipeline.predict(test_texts)
    svc_metrics = compute_metrics(test_labels, svc_preds)

    joblib.dump(tfidf_svc_pipeline, tfidf_svc_path)
    print(f"TF-IDF + LinearSVC: Acc={svc_metrics['accuracy']:.4f}, F1={svc_metrics['f1_macro']:.4f}")
    all_results["in_domain"]["tfidf_svc"] = svc_metrics

    # Cross-domain TF-IDF + LinearSVC
    svc_vsmec_preds = tfidf_svc_pipeline.predict(vsmec_texts)
    all_results["cross_domain"]["tfidf_svc"] = compute_metrics(vsmec_labels, svc_vsmec_preds)

    # ── 3. BiLSTM (3 seeds) ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("3. TRAINING BILSTM (3 SEEDS)")
    print("="*60)

    bilstm_results = []
    for seed in SEEDS:
        f1, model_dir = train_bilstm(seed, train_texts, train_labels, val_texts, val_labels)
        metrics, preds = evaluate_bilstm(model_dir / "best_model.pt", test_texts, test_labels, None, "Test")
        # Cross-domain eval for BiLSTM
        _, vsmec_preds = evaluate_bilstm(model_dir / "best_model.pt", vsmec_texts, vsmec_labels, None, "VSMEC")
        vsmec_metrics = compute_metrics(vsmec_labels, vsmec_preds)
        all_results["cross_domain"][f"bilstm_seed{seed}"] = {
            k: v for k, v in vsmec_metrics.items() if k != "confusion_matrix"
        }
        metrics["seed"] = seed
        bilstm_results.append({"seed": seed, "val_f1": f1, "test_metrics": metrics})
        all_results["in_domain"][f"bilstm_seed{seed}"] = {
            k: v for k, v in metrics.items() if k != "confusion_matrix"
        }

    bilstm_avg = {
        "accuracy": np.mean([r["test_metrics"]["accuracy"] for r in bilstm_results]),
        "f1_macro": np.mean([r["test_metrics"]["f1_macro"] for r in bilstm_results]),
        "f1_depression": np.mean([r["test_metrics"]["f1_depression"] for r in bilstm_results]),
    }
    bilstm_cross_avg = {
        "accuracy": np.mean([all_results["cross_domain"][f"bilstm_seed{s}"]["accuracy"] for s in SEEDS]),
        "f1_macro": np.mean([all_results["cross_domain"][f"bilstm_seed{s}"]["f1_macro"] for s in SEEDS]),
        "f1_depression": np.mean([all_results["cross_domain"][f"bilstm_seed{s}"]["f1_depression"] for s in SEEDS]),
    }
    print(f"\nBiLSTM Average (in-domain):  Acc={bilstm_avg['accuracy']:.4f}, F1={bilstm_avg['f1_macro']:.4f}")
    print(f"BiLSTM Average (cross-domain): Acc={bilstm_cross_avg['accuracy']:.4f}, F1={bilstm_cross_avg['f1_macro']:.4f}")
    all_results["in_domain"]["bilstm_avg"] = bilstm_avg
    all_results["cross_domain"]["bilstm_avg"] = bilstm_cross_avg

    # ── 4. PhoBERT (3 seeds) ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("4. TRAINING PHOBERT (3 SEEDS)")
    print("="*60)

    phobert_results = []
    for seed in SEEDS:
        f1, model_dir = train_phobert(seed, train_texts, train_labels, val_texts, val_labels)
        metrics, preds, probs = evaluate_phobert(model_dir, test_texts, test_labels, "Test")
        # Cross-domain eval for PhoBERT
        vsmec_metrics, vsmec_preds, vsmec_probs = evaluate_phobert(model_dir, vsmec_texts, vsmec_labels, "VSMEC")
        all_results["cross_domain"][f"phobert_seed{seed}"] = {
            k: v for k, v in vsmec_metrics.items() if k not in ["confusion_matrix", "predictions", "probabilities"]
        }
        metrics["seed"] = seed
        phobert_results.append({
            "seed": seed, "val_f1": f1,
            "test_metrics": metrics, "predictions": preds, "probabilities": probs,
            "vsmec_probs": vsmec_probs,
        })
        all_results["in_domain"][f"phobert_seed{seed}"] = {
            k: v for k, v in metrics.items() if k not in ["confusion_matrix", "predictions", "probabilities"]
        }

    # Average PhoBERT predictions (in-domain)
    all_probs = np.array([r["probabilities"] for r in phobert_results])
    avg_probs = np.mean(all_probs, axis=0)
    avg_preds = (avg_probs >= 0.5).astype(int)

    phobert_avg_metrics = compute_metrics(test_labels, avg_preds)
    print(f"\nPhoBERT Average (in-domain): Acc={phobert_avg_metrics['accuracy']:.4f}, F1={phobert_avg_metrics['f1_macro']:.4f}")
    all_results["in_domain"]["phobert_avg"] = {
        k: v for k, v in phobert_avg_metrics.items() if k != "confusion_matrix"
    }

    # Average PhoBERT predictions (cross-domain)
    vsmec_all_probs = np.array([r["vsmec_probs"] for r in phobert_results])
    vsmec_avg_probs = np.mean(vsmec_all_probs, axis=0)
    vsmec_avg_preds = (vsmec_avg_probs >= 0.5).astype(int)
    phobert_vsmec_avg = compute_metrics(vsmec_labels, vsmec_avg_preds)
    print(f"PhoBERT Average (cross-domain): Acc={phobert_vsmec_avg['accuracy']:.4f}, F1={phobert_vsmec_avg['f1_macro']:.4f}")
    all_results["cross_domain"]["phobert_avg"] = {
        k: v for k, v in phobert_vsmec_avg.items() if k != "confusion_matrix"
    }

    # ── 5. Cross-domain evaluation (VSMEC) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("5. CROSS-DOMAIN EVALUATION (VSMEC, n=3,084)")
    print("=" * 70)

    # TF-IDF cross-domain
    logreg_vsmec_preds = tfidf_logreg_pipeline.predict(vsmec_texts)
    logreg_vsmec_metrics = compute_metrics(vsmec_labels, logreg_vsmec_preds)
    all_results["cross_domain"]["tfidf_logreg"] = {
        k: v for k, v in logreg_vsmec_metrics.items() if k != "confusion_matrix"
    }
    print(f"TF-IDF + LogReg on VSMEC: Acc={logreg_vsmec_metrics['accuracy']:.4f}, "
          f"F1={logreg_vsmec_metrics['f1_macro']:.4f}")

    svc_vsmec_preds = tfidf_svc_pipeline.predict(vsmec_texts)
    svc_vsmec_metrics = compute_metrics(vsmec_labels, svc_vsmec_preds)
    all_results["cross_domain"]["tfidf_svc"] = {
        k: v for k, v in svc_vsmec_metrics.items() if k != "confusion_matrix"
    }
    print(f"TF-IDF + LinearSVC on VSMEC: Acc={svc_vsmec_metrics['accuracy']:.4f}, "
          f"F1={svc_vsmec_metrics['f1_macro']:.4f}")

    # BiLSTM cross-domain (3 seeds + avg)
    bilstm_cross_per_seed = {}
    for r in bilstm_results:
        seed = r["seed"]
        model_path = OUTPUT_DIR / f"bilstm_seed_{seed}" / "best_model.pt"
        _, preds_vsmec = evaluate_bilstm(model_path, vsmec_texts, None, "VSMEC")
        bilstm_cross_per_seed[seed] = preds_vsmec
        vsmec_metrics = compute_metrics(vsmec_labels, preds_vsmec)
        all_results["cross_domain"][f"bilstm_seed{seed}"] = {
            k: v for k, v in vsmec_metrics.items() if k != "confusion_matrix"
        }

    # BiLSTM avg cross-domain
    bilstm_avg_vsmec_preds = np.array(
        [bilstm_cross_per_seed[r["seed"]] for r in bilstm_results]
    )
    bilstm_avg_vsmec = (
        np.mean(bilstm_avg_vsmec_preds, axis=0) >= 0.5
    ).astype(int)
    bilstm_avg_vsmec_metrics = compute_metrics(vsmec_labels, bilstm_avg_vsmec)
    all_results["cross_domain"]["bilstm_avg"] = {
        k: v for k, v in bilstm_avg_vsmec_metrics.items() if k != "confusion_matrix"
    }
    print(f"BiLSTM avg on VSMEC: Acc={bilstm_avg_vsmec_metrics['accuracy']:.4f}, "
          f"F1={bilstm_avg_vsmec_metrics['f1_macro']:.4f}")

    # PhoBERT cross-domain (3 seeds + avg)
    phobert_cross_per_seed = {}

    # ── Summary Table ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("IN-DOMAIN TEST RESULTS SUMMARY")
    print("="*70)
    print(f"{'Model':<25} | {'Acc':>7} | {'Prec-M':>7} | {'Rec-M':>7} | {'F1-M':>7} | {'F1-D':>7}")
    print("-"*70)

    for model_name, metrics in all_results["in_domain"].items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            acc = metrics.get('accuracy', 0)
            prec = metrics.get('precision_macro', metrics.get('precision', 0))
            rec = metrics.get('recall_macro', metrics.get('recall', 0))
            f1_m = metrics.get('f1_macro', 0)
            f1_d = metrics.get('f1_depression', 0)
            print(f"{model_name:<25} | {acc:>7.4f} | {prec:>7.4f} | {rec:>7.4f} | {f1_m:>7.4f} | {f1_d:>7.4f}")

    # Save results to two locations for redundancy:
    #   1. results/round5_retrained_<timestamp>/ (historical/audit trail)
    #   2. models/round5_retrained/evaluation_results.json (canonical, fixed)
    results_file = results_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    canonical_eval = OUTPUT_DIR / "evaluation_results.json"
    with open(canonical_eval, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED TO: {results_file}")
    print(f"CANONICAL RESULTS: {canonical_eval}")
    print(f"MODELS SAVED TO: {OUTPUT_DIR}")
    print(f"{'='*70}")

    return all_results

if __name__ == "__main__":
    main()
