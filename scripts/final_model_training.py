"""Final comprehensive model training for Round 5 submission.

Trains and evaluates all models on Round 5 dataset (6,080 samples):
1. PhoBERT (3 seeds) - already trained
2. TF-IDF + LogReg
3. TF-IDF + LinearSVC
4. BiLSTM (random + PhoBERT-frozen)
5. PhoBERT + BERTopic

Usage:
    .venv/bin/python scripts/final_model_training.py
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
from datetime import datetime
import json
import pickle
import random

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / f"final_round5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "labeled" / "final_train.csv"
VAL_FILE = DATA_DIR / "labeled" / "final_val.csv"
TEST_FILE = DATA_DIR / "labeled" / "final_test.csv"
VSMEC_FILE = PROJECT_DIR / "data_unified" / "cross_domain_test.csv"
BERTOPIC_DIR = MODEL_DIR / "bertopic"
PHOBERT_DIR = MODEL_DIR / "round5_predictions"

MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 128
SEEDS = [42, 123, 2024]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ── Metric computation ────────────────────────────────────────────────────────
def compute_all_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_depression": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

# ── PhoBERT Dataset ───────────────────────────────────────────────────────────
class PhoBertDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ── BiLSTM Model ──────────────────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        output = self.dropout(hidden)
        return self.fc(output)

# ── PhoBERT Training ───────────────────────────────────────────────────────────
def train_phobert(seed, train_texts, train_labels, val_texts, val_labels):
    print(f"\n{'='*60}")
    print(f"Training PhoBERT - Seed {seed}")
    print(f"{'='*60}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)

    train_dataset = PhoBertDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = PhoBertDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.06), num_training_steps=total_steps
    )

    n_neg, n_pos = (train_labels == 0).sum(), (train_labels == 1).sum()
    pos_weight = n_neg / n_pos
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device))

    best_f1 = 0.0
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            loss = criterion(outputs.logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            seed_dir = PHOBERT_DIR / f"seed_{seed}" / "best_model"
            seed_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(seed_dir))

    print(f"  Best Val F1: {best_f1:.4f}")
    return best_f1

# ── PhoBERT Prediction ─────────────────────────────────────────────────────────
def get_phobert_predictions(model_dir, texts, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    dataset = PhoBertDataset(texts, None, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)

# ── TF-IDF Training ───────────────────────────────────────────────────────────
def train_tfidf(train_texts, train_labels, val_texts, val_labels, model_name="tfidf"):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    if model_name == "tfidf_logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    else:  # tfidf_svc
        clf = LinearSVC(max_iter=2000, class_weight="balanced", random_state=42)

    clf.fit(X_train, train_labels)
    val_preds = clf.predict(X_val)
    val_f1 = f1_score(val_labels, val_preds, average="macro")
    print(f"  Val F1: {val_f1:.4f}")

    # Save pipeline
    pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    output_path = MODEL_DIR / f"{model_name}_round5.joblib"
    joblib.dump(pipeline, output_path)
    print(f"  Saved: {output_path}")

    return val_f1, pipeline

# ── BiLSTM Training ───────────────────────────────────────────────────────────
def train_bilstm(seed, train_texts, train_labels, val_texts, val_labels, embedding_type="random"):
    print(f"\n{'='*60}")
    print(f"Training BiLSTM ({embedding_type}) - Seed {seed}")
    print(f"{'='*60}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    # Build vocabulary
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for word in str(text).split():
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    # Prepare data
    def encode_texts(texts, max_len=128):
        encoded = []
        for text in texts:
            words = str(text).split()[:max_len]
            ids = [word2idx.get(w, word2idx["<UNK>"]) for w in words]
            ids += [0] * (max_len - len(ids))
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    train_X = encode_texts(train_texts)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    val_X = encode_texts(val_texts)
    val_y = torch.tensor(val_labels, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model
    vocab_size = len(word2idx)
    model = BiLSTMClassifier(vocab_size=vocab_size, embed_dim=128, hidden_dim=128)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(10):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())

        f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"  Epoch {epoch+1}: Val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            bilstm_dir = MODEL_DIR / "bilstm_round5" / embedding_type / f"seed_{seed}"
            bilstm_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "word2idx": word2idx,
                "config": {"vocab_size": vocab_size, "embed_dim": 128, "hidden_dim": 128}
            }, bilstm_dir / "model.pt")

    print(f"  Best Val F1: {best_f1:.4f}")
    return best_f1

# ── PhoBERT + BERTopic ─────────────────────────────────────────────────────────
def train_phobert_bertopic():
    print(f"\n{'='*60}")
    print("Training PhoBERT + BERTopic")
    print(f"{'='*60}")

    # Load BERTopic
    with open(BERTOPIC_DIR / "bertopic_model.pkl", "rb") as f:
        topic_model = pickle.load(f)

    from sentence_transformers import SentenceTransformer
    topic_embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(PHOBERT_DIR / "seed_42" / "best_model"))
    model.to(device)
    model.eval()

    # Extract features
    def get_phobert_embeddings(texts):
        dataset = PhoBertDataset(texts, None, tokenizer, MAX_LEN)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        embeddings = []
        with torch.no_grad():
            for batch in loader:
                outputs = model(**{k: v.to(device) for k, v in batch.items() if k != "labels"})
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_emb)
        return np.vstack(embeddings)

    def get_topic_features(texts):
        embeddings = topic_embedder.encode(texts, show_progress_bar=True, batch_size=64)
        topics, probs = topic_model.transform(texts, embeddings=embeddings)
        probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1) if probs is not None else np.zeros(len(texts))
        return np.column_stack([topics, probs_arr])

    # Get all text embeddings
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    fit_texts = train_df["comment_text"].tolist() + val_df["comment_text"].tolist()
    fit_labels = train_df["label"].astype(int).tolist() + val_df["label"].astype(int).tolist()

    print("  Extracting PhoBERT embeddings...")
    phobert_emb = get_phobert_embeddings(fit_texts)
    print("  Extracting BERTopic features...")
    topic_feat = get_topic_features(fit_texts)
    X = np.hstack([phobert_emb, topic_feat])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(X, fit_labels)

    # Save
    output = {
        "scaler": scaler,
        "clf": clf,
        "phobert_dir": str(PHOBERT_DIR / "seed_42" / "best_model"),
        "bertopic_path": str(BERTOPIC_DIR / "bertopic_model.pkl")
    }
    output_path = MODEL_DIR / "phobert_bertopic_round5.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    print(f"  Saved: {output_path}")

    return output_path

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("FINAL COMPREHENSIVE MODEL TRAINING - ROUND 5 SUBMISSION")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # Handle NaN values
    train_df["comment_text"] = train_df["comment_text"].fillna("")
    val_df["comment_text"] = val_df["comment_text"].fillna("")
    test_df["comment_text"] = test_df["comment_text"].fillna("")

    train_texts = train_df["comment_text"].astype(str).values
    train_labels = train_df["label"].values
    val_texts = val_df["comment_text"].astype(str).values
    val_labels = val_df["label"].values
    test_texts = test_df["comment_text"].astype(str).values
    test_labels = test_df["label"].values

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Class distribution - Train: N={sum(train_labels==0)}, D={sum(train_labels==1)}")

    # Load tokenizer for PhoBERT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_normal": int(sum(train_labels == 0)),
            "train_depression": int(sum(train_labels == 1))
        },
        "in_domain": {},
        "cross_domain": {}
    }

    # ── 1. PhoBERT (already trained, just evaluate) ──────────────────────────
    print("\n" + "=" * 70)
    print("1. PHOBERT EVALUATION")
    print("=" * 70)

    phobert_results = []
    for seed in SEEDS:
        model_dir = PHOBERT_DIR / f"seed_{seed}" / "best_model"
        if model_dir.exists():
            print(f"\n  Seed {seed}:")
            preds = get_phobert_predictions(model_dir, test_texts, tokenizer)
            metrics = compute_all_metrics(test_labels, preds)
            phobert_results.append({"seed": seed, "predictions": preds, **metrics})
            print(f"    In-Domain: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

    # Average prediction
    avg_pred = (np.mean([r["predictions"] for r in phobert_results], axis=0) >= 0.5).astype(int)
    avg_metrics = compute_all_metrics(test_labels, avg_pred)
    avg_metrics["seeds"] = SEEDS
    avg_metrics["per_seed"] = {str(r["seed"]): {k: v for k, v in r.items() if k not in ["seed", "predictions", "confusion_matrix"]} for r in phobert_results}
    all_results["in_domain"]["phobert_avg"] = avg_metrics

    # ── 2. TF-IDF Models ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. TF-IDF TRAINING")
    print("=" * 70)

    # Train TF-IDF + LogReg
    val_f1_logreg, pipeline_logreg = train_tfidf(train_texts, train_labels, val_texts, val_labels, "tfidf_logreg")
    test_preds = pipeline_logreg.predict(test_texts)
    metrics = compute_all_metrics(test_labels, test_preds)
    all_results["in_domain"]["tfidf_logreg"] = metrics

    # Train TF-IDF + SVC
    val_f1_svc, pipeline_svc = train_tfidf(train_texts, train_labels, val_texts, val_labels, "tfidf_svc")
    test_preds = pipeline_svc.predict(test_texts)
    metrics = compute_all_metrics(test_labels, test_preds)
    all_results["in_domain"]["tfidf_svc"] = metrics

    # ── 3. BiLSTM Training ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. BILSTM TRAINING")
    print("=" * 70)

    for embedding_type in ["random"]:
        for seed in SEEDS:
            val_f1 = train_bilstm(seed, train_texts, train_labels, val_texts, val_labels, embedding_type)

    # ── 4. PhoBERT + BERTopic ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. PHOBERT + BERTOPIC")
    print("=" * 70)
    print("  (Skipping - use scripts/rerun_phobert_bertopic.py instead)")
    # Note: PhoBERT + BERTopic requires specific setup, run separately

    # ── Cross-domain Evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("5. CROSS-DOMAIN EVALUATION (VSMEC)")
    print("=" * 70)

    if VSMEC_FILE.exists():
        vsmec_df = pd.read_csv(VSMEC_FILE)
        vsmec_texts = vsmec_df["text"].values if "text" in vsmec_df.columns else vsmec_df["comment_text"].values
        vsmec_labels = vsmec_df["label"].values

        print(f"VSMEC: {len(vsmec_df)} samples")

        # PhoBERT cross-domain
        for seed in SEEDS:
            model_dir = PHOBERT_DIR / f"seed_{seed}" / "best_model"
            preds = get_phobert_predictions(model_dir, vsmec_texts, tokenizer)
            metrics = compute_all_metrics(vsmec_labels, preds)
            all_results["cross_domain"][f"phobert_seed{seed}"] = metrics
            print(f"  PhoBERT seed {seed}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<25} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1-M':>7} | {'F1-W':>7} | {'F1-D':>7}")
    print("-" * 80)

    for model_name, metrics in all_results["in_domain"].items():
        print(f"{model_name:<25} | {metrics['accuracy']:>7.4f} | {metrics['precision_macro']:>7.4f} | "
              f"{metrics['recall_macro']:>7.4f} | {metrics['f1_macro']:>7.4f} | {metrics['f1_weighted']:>7.4f} | {metrics['f1_depression']:>7.4f}")

    # ── Save Results ──────────────────────────────────────────────────────────
    results_file = OUTPUT_DIR / "final_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {results_file}")

    return all_results

if __name__ == "__main__":
    main()
