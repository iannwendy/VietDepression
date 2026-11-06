"""Phase 3 — Train 5 comparison models, evaluate on in-domain + cross-domain test sets.

Models:
  1. TF-IDF + SVM (baseline)
  2. BiLSTM (deep learning)
  3. PhoBERT final (transformer)
  4. BERTopic-only (topic classifier)
  5. PhoBERT + BERTopic (proposed model)

All train on data/final_train.csv, evaluate on:
  - In-domain: data/final_test.csv
  - Cross-domain: data_unified/cross_domain_test.csv
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

SEED = 42
FINAL_TRAIN = PROJECT_DIR / "data" / "final_train.csv"
FINAL_VAL = PROJECT_DIR / "data" / "final_val.csv"
FINAL_TEST = PROJECT_DIR / "data" / "final_test.csv"
CROSS_DOMAIN_TEST = PROJECT_DIR / "data_unified" / "cross_domain_test.csv"
BERTOPIC_MODEL_FILE = PROJECT_DIR / "models" / "bertopic" / "bertopic_model.pkl"
REPORT_FILE = PROJECT_DIR / "docs" / "phase3_comparison_report.json"
MODEL_DIR = PROJECT_DIR / "models" / "phase3"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"comment_text": str}).fillna("")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
    df = df[df["comment_text"].str.strip().ne("")].copy()
    return df


def load_cross_domain(path: Path = CROSS_DOMAIN_TEST) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"comment_text": str}).fillna("")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
    df = df[df["comment_text"].str.strip().ne("")].copy()
    return df


def compute_metrics(y_true, y_pred) -> dict:
    labels = [0, 1]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_depression": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_model(model, test_df: pd.DataFrame, cross_df: pd.DataFrame, model_name: str) -> dict:
    """Evaluate on both in-domain and cross-domain test sets."""
    results = {"model": model_name}

    # In-domain
    y_pred_in = model.predict(test_df["comment_text"])
    y_true_in = test_df["label"].astype(int)
    results["in_domain"] = compute_metrics(y_true_in, y_pred_in)

    # Cross-domain
    y_pred_cross = model.predict(cross_df["comment_text"])
    y_true_cross = cross_df["label"].astype(int)
    results["cross_domain"] = compute_metrics(y_true_cross, y_pred_cross)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Model 1: TF-IDF + SVM
# ═══════════════════════════════════════════════════════════════════════

def train_tfidf_svm(train_df, val_df):
    print("\n" + "=" * 50)
    print("MODEL 1: TF-IDF + SVM")
    print("=" * 50)

    pipeline = Pipeline(steps=[
        ("features", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=80000, lowercase=True)),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                max_features=80000, lowercase=True)),
        ])),
        ("classifier", LinearSVC(
            max_iter=2000, class_weight="balanced",
            random_state=SEED, dual=False)),
    ])

    pipeline.fit(train_df["comment_text"], train_df["label"].astype(int))
    print(f"  Trained on {len(train_df)} samples")

    return pipeline


# ═══════════════════════════════════════════════════════════════════════
# Model 2: BiLSTM
# ═══════════════════════════════════════════════════════════════════════

class BiLSTMClassifier:
    """BiLSTM for Vietnamese depression detection."""

    def __init__(self, vocab_size=30000, embed_dim=256, hidden_dim=128,
                 num_layers=2, dropout=0.3, max_len=128, epochs=10,
                 batch_size=32, lr=0.001):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.model = None
        self._device = None

    def _build_vocab(self, texts):
        from collections import Counter
        counter = Counter()
        for text in texts:
            for token in text.lower().split():
                counter[token] += 1
        for word, _ in counter.most_common(self.vocab_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def _encode(self, texts):
        data = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, text in enumerate(texts):
            tokens = text.lower().split()[:self.max_len]
            for j, token in enumerate(tokens):
                data[i, j] = self.word2idx.get(token, 1)  # <UNK> = 1
        return data

    def fit(self, texts, labels, val_texts=None, val_labels=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._build_vocab(texts)
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"  BiLSTM device: {self._device}")
        print(f"  Vocab size: {len(self.word2idx)}")

        X = torch.tensor(self._encode(texts), dtype=torch.long)
        y = torch.tensor(np.array(labels, dtype=np.int64), dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = nn.ModuleDict({
            "embedding": nn.Embedding(len(self.word2idx), self.embed_dim, padding_idx=0),
            "lstm": nn.LSTM(self.embed_dim, self.hidden_dim, self.num_layers,
                            batch_first=True, bidirectional=True, dropout=self.dropout),
            "classifier": nn.Sequential(
                nn.Linear(self.hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(64, 2),
            ),
        }).to(self._device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self._device), batch_y.to(self._device)
                optimizer.zero_grad()
                emb = self.model["embedding"](batch_x)
                lstm_out, _ = self.model["lstm"](emb)
                pooled = lstm_out[:, -1, :]  # Take last hidden state
                logits = self.model["classifier"](pooled)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(loader), 1)
            if (epoch + 1) % 3 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} loss={avg_loss:.4f}")

        print(f"  BiLSTM trained: {self.epochs} epochs")

    def predict(self, texts):
        import torch

        self.model.eval()
        X = torch.tensor(self._encode(texts), dtype=torch.long)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i+self.batch_size].to(self._device)
                emb = self.model["embedding"](batch)
                lstm_out, _ = self.model["lstm"](emb)
                pooled = lstm_out[:, -1, :]
                logits = self.model["classifier"](pooled)
                batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
                preds.extend(batch_preds)
        return preds


def train_bilstm(train_df, val_df):
    print("\n" + "=" * 50)
    print("MODEL 2: BiLSTM")
    print("=" * 50)

    model = BiLSTMClassifier(
        vocab_size=15000, embed_dim=128, hidden_dim=128,
        num_layers=2, dropout=0.5, max_len=100, epochs=8,
        batch_size=32, lr=0.001,
    )

    model.fit(
        train_df["comment_text"].tolist(),
        train_df["label"].astype(int).tolist(),
    )

    return model


# ═══════════════════════════════════════════════════════════════════════
# Model 3: PhoBERT final
# ═══════════════════════════════════════════════════════════════════════

class PhoBERTFinalWrapper:
    """Wrapper that matches sklearn-style predict interface."""

    def __init__(self, model_dir):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from yt_depression_crawler.modeling.phobert.phobert_utils import (
            get_device, prepare_many_texts,
        )

        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()
        self._prepare = prepare_many_texts
        self._batch_size = 32

    def predict(self, texts):
        import torch
        from torch.utils.data import DataLoader
        from yt_depression_crawler.core.config import PHOBERT_MAX_LENGTH
        from yt_depression_crawler.modeling.phobert.phobert_utils import PhoBertDataset

        prepared = self._prepare(list(texts))
        dataset = PhoBertDataset(prepared, None, self.tokenizer, PHOBERT_MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=self._batch_size)

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                preds.extend(batch_preds)
        return preds


def train_phobert_final(train_df, val_df):
    print("\n" + "=" * 50)
    print("MODEL 3: PhoBERT (final)")
    print("=" * 50)

    from yt_depression_crawler.modeling.phobert.phobert_train import train_phobert_first

    output_dir = MODEL_DIR / "phobert_final"

    # Use existing train_phobert_first but with final dataset
    report = train_phobert_first(
        train_file=FINAL_TRAIN,
        val_file=FINAL_VAL,
        test_file=FINAL_TEST,
        output_dir=output_dir,
        metrics_file=output_dir / "metrics.json",
    )

    print(f"  Best epoch: {report['best_epoch']} (val F1: {report['best_val_f1_macro']})")
    print(f"  Test F1 macro: {report['test']['f1_macro']}")

    return PhoBERTFinalWrapper(output_dir)


# ═══════════════════════════════════════════════════════════════════════
# Model 4: BERTopic-only
# ═══════════════════════════════════════════════════════════════════════

class BERTopicOnlyClassifier:
    """Classifier using only topic assignment as feature."""

    def __init__(self):
        self.topic_model = None
        self.clf = None
        self.scaler = StandardScaler()

    def _load_topic_model(self):
        if not BERTOPIC_MODEL_FILE.exists():
            raise FileNotFoundError(f"BERTopic model not found: {BERTOPIC_MODEL_FILE}")
        with open(BERTOPIC_MODEL_FILE, "rb") as f:
            self.topic_model = pickle.load(f)

    def _get_topic_features(self, texts):
        topics, probs = self.topic_model.transform(texts)
        # One-hot encode topic + probability as feature
        features = np.column_stack([
            np.array(topics, dtype=np.float64).reshape(-1, 1),
            np.array(probs if probs is not None else [0]*len(texts), dtype=np.float64).reshape(-1, 1),
        ])
        return features

    def fit(self, texts, labels):
        self._load_topic_model()
        X = self._get_topic_features(texts)
        X = self.scaler.fit_transform(X)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        X = self._get_topic_features(texts)
        X = self.scaler.transform(X)
        return self.clf.predict(X)


def train_bertopic_only(train_df, val_df):
    print("\n" + "=" * 50)
    print("MODEL 4: BERTopic-only")
    print("=" * 50)

    model = BERTopicOnlyClassifier()
    model.fit(
        train_df["comment_text"].tolist(),
        train_df["label"].astype(int).tolist(),
    )
    print(f"  Trained on {len(train_df)} samples with topic features")
    return model


# ═══════════════════════════════════════════════════════════════════════
# Model 5: PhoBERT + BERTopic (proposed model)
# ═══════════════════════════════════════════════════════════════════════

class PhoBertTopicClassifier:
    """Combined PhoBERT embeddings + BERTopic features → classifier."""

    def __init__(self, phobert_model_dir, topic_model_path=BERTOPIC_MODEL_FILE):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from yt_depression_crawler.modeling.phobert.phobert_utils import get_device

        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(phobert_model_dir), use_fast=False)
        self.phobert = AutoModel.from_pretrained(str(phobert_model_dir))
        self.phobert.to(self.device)
        self.phobert.eval()

        with open(topic_model_path, "rb") as f:
            self.topic_model = pickle.load(f)

        from sentence_transformers import SentenceTransformer
        self.topic_embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        self.clf = None
        self.scaler = StandardScaler()
        self._batch_size = 16

    def _get_phobert_embeddings(self, texts):
        import torch
        from torch.utils.data import DataLoader
        from yt_depression_crawler.core.config import PHOBERT_MAX_LENGTH
        from yt_depression_crawler.modeling.phobert.phobert_utils import (
            PhoBertDataset, prepare_many_texts,
        )

        prepared = prepare_many_texts(list(texts))
        dataset = PhoBertDataset(prepared, None, self.tokenizer, PHOBERT_MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=self._batch_size)

        embeddings = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.phobert(**batch)
                # CLS token embedding
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_emb)
        return np.vstack(embeddings)

    def _get_topic_features(self, texts):
        texts_list = list(texts)
        embeddings = self.topic_embedder.encode(texts_list, show_progress_bar=False, batch_size=64)
        topics, probs = self.topic_model.transform(texts_list, embeddings=embeddings)
        return np.column_stack([
            np.array(topics, dtype=np.float64).reshape(-1, 1),
            np.array(probs if probs is not None else [0]*len(texts_list), dtype=np.float64).reshape(-1, 1),
        ])

    def fit(self, texts, labels):
        print("  Extracting PhoBERT embeddings...")
        phobert_feats = self._get_phobert_embeddings(texts)
        print("  Extracting BERTopic features...")
        topic_feats = self._get_topic_features(texts)

        X = np.hstack([phobert_feats, topic_feats])
        X = self.scaler.fit_transform(X)

        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        self.clf.fit(X, labels)
        print(f"  Feature dim: {X.shape[1]} (PhoBERT: {phobert_feats.shape[1]} + Topic: {topic_feats.shape[1]})")
        return self

    def predict(self, texts):
        phobert_feats = self._get_phobert_embeddings(texts)
        topic_feats = self._get_topic_features(texts)
        X = np.hstack([phobert_feats, topic_feats])
        X = self.scaler.transform(X)
        return self.clf.predict(X)


def train_phobert_bertopic(train_df, val_df):
    print("\n" + "=" * 50)
    print("MODEL 5: PhoBERT + BERTopic (PROPOSED)")
    print("=" * 50)

    # Use PhoBERT v2 model as base
    phobert_dir = PROJECT_DIR / "models" / "phobert_second"

    model = PhoBertTopicClassifier(str(phobert_dir))
    model.fit(
        train_df["comment_text"].tolist(),
        train_df["label"].astype(int).tolist(),
    )
    return model


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PHASE 3: Train 5 Models — Comparison")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = load_split(FINAL_TRAIN)
    val_df = load_split(FINAL_VAL)
    test_df = load_split(FINAL_TEST)
    cross_df = load_cross_domain()

    print(f"\nData: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, cross_domain={len(cross_df)}")
    print(f"Train labels: {train_df['label'].value_counts().to_dict()}")
    print(f"Cross-domain labels: {cross_df['label'].value_counts().to_dict()}")

    all_results = []

    # Model 1: TF-IDF + SVM
    model1 = train_tfidf_svm(train_df, val_df)
    r1 = evaluate_model(model1, test_df, cross_df, "TF-IDF + SVM")
    all_results.append(r1)
    print(f"  In-domain F1: {r1['in_domain']['f1_macro']}, Cross-domain F1: {r1['cross_domain']['f1_macro']}")

    # Model 2: BiLSTM
    model2 = train_bilstm(train_df, val_df)
    r2 = evaluate_model(model2, test_df, cross_df, "BiLSTM")
    all_results.append(r2)
    print(f"  In-domain F1: {r2['in_domain']['f1_macro']}, Cross-domain F1: {r2['cross_domain']['f1_macro']}")

    # Model 3: PhoBERT final
    model3 = train_phobert_final(train_df, val_df)
    r3 = evaluate_model(model3, test_df, cross_df, "PhoBERT")
    all_results.append(r3)
    print(f"  In-domain F1: {r3['in_domain']['f1_macro']}, Cross-domain F1: {r3['cross_domain']['f1_macro']}")

    # Model 4: BERTopic-only
    model4 = train_bertopic_only(train_df, val_df)
    r4 = evaluate_model(model4, test_df, cross_df, "BERTopic-only")
    all_results.append(r4)
    print(f"  In-domain F1: {r4['in_domain']['f1_macro']}, Cross-domain F1: {r4['cross_domain']['f1_macro']}")

    # Model 5: PhoBERT + BERTopic (proposed)
    model5 = train_phobert_bertopic(train_df, val_df)
    r5 = evaluate_model(model5, test_df, cross_df, "PhoBERT + BERTopic")
    all_results.append(r5)
    print(f"  In-domain F1: {r5['in_domain']['f1_macro']}, Cross-domain F1: {r5['cross_domain']['f1_macro']}")

    # ── Comparison report ──
    print("\n" + "=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70)

    print(f"\n{'Model':<25s} {'In-domain F1':>14s} {'Cross-domain F1':>17s} {'F1 Depression (in)':>18s}")
    print("-" * 75)
    for r in all_results:
        name = r["model"]
        f1_in = r["in_domain"]["f1_macro"]
        f1_cross = r["cross_domain"]["f1_macro"]
        f1_dep_in = r["in_domain"]["f1_depression"]
        print(f"{name:<25s} {f1_in:>14.4f} {f1_cross:>17.4f} {f1_dep_in:>18.4f}")

    print(f"\n{'Model':<25s} {'Accuracy':>10s} {'Prec (macro)':>14s} {'Recall (macro)':>14s} {'F1 (weighted)':>14s}")
    print("-" * 75)
    for r in all_results:
        m = r["cross_domain"]
        print(f"{r['model']:<25s} {m['accuracy']:>10.4f} {m['precision_macro']:>14.4f} {m['recall_macro']:>14.4f} {m['f1_weighted']:>14.4f}")

    # Save full report
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "phase": 3,
        "dataset": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "cross_domain_rows": int(len(cross_df)),
        },
        "models": all_results,
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report: {REPORT_FILE}")

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
