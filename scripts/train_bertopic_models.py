"""Train BERTopic-only and PhoBERT + BERTopic on augmented dataset.

Usage:
  .venv/bin/python scripts/train_bertopic_models.py
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

# Disable proxy for local model loading
import os
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

# Disable proxy for requests
import requests
requests.proxies = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data():
    """Load augmented dataset."""
    train = pd.read_csv(DATA_DIR / "final_train.csv", dtype=str).fillna("")
    val = pd.read_csv(DATA_DIR / "final_val.csv", dtype=str).fillna("")
    test = pd.read_csv(DATA_DIR / "final_test.csv", dtype=str).fillna("")
    cross = pd.read_csv(PROJECT_DIR / "data_unified" / "cross_domain_test.csv", dtype=str).fillna("")

    for df in [train, val, test, cross]:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)

    return train, val, test, cross


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_depression": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


# ── BERTopic-only Model ──────────────────────────────────────────────

class BERTopicOnlyClassifier:
    """Classifier using only BERTopic topic features."""

    def __init__(self, topic_model_path):
        with open(topic_model_path, "rb") as f:
            self.topic_model = pickle.load(f)

        # Load embedding model if not present
        if self.topic_model.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.topic_model.embedding_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )

        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    def _get_topic_features(self, texts):
        topics, probs = self.topic_model.transform(list(texts))
        probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
        if probs_arr.ndim > 1:
            probs_arr = probs_arr.max(axis=-1)
        return np.column_stack([
            np.array(topics, dtype=np.float64).reshape(-1, 1),
            probs_arr.reshape(-1, 1),
        ])

    def fit(self, texts, labels):
        X = self._get_topic_features(texts)
        X = self.scaler.fit_transform(X)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        X = self._get_topic_features(texts)
        X = self.scaler.transform(X)
        return self.clf.predict(X)


def train_bertopic_only(train_texts, train_labels, test_texts, test_labels, cross_texts, cross_labels):
    """Train and evaluate BERTopic-only model."""
    logger.info("=" * 50)
    logger.info("MODEL: BERTopic-only")
    logger.info("=" * 50)

    topic_model_path = MODEL_DIR / "bertopic" / "bertopic_model.pkl"
    if not topic_model_path.exists():
        logger.error(f"BERTopic model not found: {topic_model_path}")
        return None

    model = BERTopicOnlyClassifier(topic_model_path)
    model.fit(train_texts, train_labels)

    test_preds = model.predict(test_texts)
    cross_preds = model.predict(cross_texts)

    test_metrics = compute_metrics(test_labels, test_preds)
    cross_metrics = compute_metrics(cross_labels, cross_preds)

    logger.info(f"  In-domain F1: {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Cross-domain F1: {cross_metrics['f1_macro']:.4f}")

    return {"test": test_metrics, "cross_domain": cross_metrics}


# ── PhoBERT + BERTopic Model ────────────────────────────────────────

class PhoBertTopicClassifier:
    """Combined PhoBERT embeddings + BERTopic features."""

    def __init__(self, phobert_model_dir, topic_model_path):
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

        self.topic_embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)

    def _get_phobert_embeddings(self, texts):
        import torch
        from torch.utils.data import DataLoader
        from yt_depression_crawler.core.config import PHOBERT_MAX_LENGTH
        from yt_depression_crawler.modeling.phobert.phobert_utils import PhoBertDataset, prepare_many_texts

        prepared = prepare_many_texts(list(texts))
        dataset = PhoBertDataset(prepared, None, self.tokenizer, PHOBERT_MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=16)

        embeddings = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.phobert(**batch)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_emb)

        return np.vstack(embeddings)

    def _get_topic_features(self, texts):
        texts_list = list(texts)
        embeddings = self.topic_embedder.encode(texts_list, show_progress_bar=False, batch_size=64)
        topics, probs = self.topic_model.transform(texts_list, embeddings=embeddings)
        probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
        if probs_arr.ndim > 1:
            probs_arr = probs_arr.max(axis=-1)
        return np.column_stack([
            np.array(topics, dtype=np.float64).reshape(-1, 1),
            probs_arr.reshape(-1, 1),
        ])

    def fit(self, texts, labels):
        logger.info("  Extracting PhoBERT embeddings...")
        phobert_feats = self._get_phobert_embeddings(texts)
        logger.info("  Extracting BERTopic features...")
        topic_feats = self._get_topic_features(texts)

        X = np.hstack([phobert_feats, topic_feats])
        X = self.scaler.fit_transform(X)
        logger.info(f"  Feature dim: {X.shape[1]}")
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        phobert_feats = self._get_phobert_embeddings(texts)
        topic_feats = self._get_topic_features(texts)
        X = np.hstack([phobert_feats, topic_feats])
        X = self.scaler.transform(X)
        return self.clf.predict(X)


def train_phobert_bertopic(train_texts, train_labels, test_texts, test_labels, cross_texts, cross_labels):
    """Train and evaluate PhoBERT + BERTopic model."""
    logger.info("=" * 50)
    logger.info("MODEL: PhoBERT + BERTopic")
    logger.info("=" * 50)

    # Use phobert_base_local
    phobert_dir = MODEL_DIR / "phobert_base_local"
    topic_model_path = MODEL_DIR / "bertopic" / "bertopic_model.pkl"

    model = PhoBertTopicClassifier(phobert_dir, topic_model_path)
    model.fit(train_texts, train_labels)

    logger.info("  Predicting...")
    test_preds = model.predict(test_texts)
    cross_preds = model.predict(cross_texts)

    test_metrics = compute_metrics(test_labels, test_preds)
    cross_metrics = compute_metrics(cross_labels, cross_preds)

    logger.info(f"  In-domain F1: {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Cross-domain F1: {cross_metrics['f1_macro']:.4f}")

    return {"test": test_metrics, "cross_domain": cross_metrics}


# ── Main ────────────────────────────────────────────────────────────

def main():
    logger.info("Loading data...")
    train_df, val_df, test_df, cross_df = load_data()

    # Combine train + val for fitting
    train_texts = train_df["comment_text"].tolist() + val_df["comment_text"].tolist()
    train_labels = train_df["label"].astype(int).tolist() + val_df["label"].astype(int).tolist()

    test_texts = test_df["comment_text"].tolist()
    test_labels = test_df["label"].astype(int).tolist()
    cross_texts = cross_df["comment_text"].tolist()
    cross_labels = cross_df["label"].astype(int).tolist()

    logger.info(f"Fit samples: {len(train_texts)}")
    logger.info(f"Test samples: {len(test_texts)}")
    logger.info(f"Cross-domain samples: {len(cross_texts)}")

    results = {}

    # BERTopic-only
    bertopic_result = train_bertopic_only(
        train_texts, train_labels, test_texts, test_labels, cross_texts, cross_labels
    )
    if bertopic_result:
        results["BERTopic-only"] = bertopic_result

    # PhoBERT + BERTopic
    phobert_bertopic_result = train_phobert_bertopic(
        train_texts, train_labels, test_texts, test_labels, cross_texts, cross_labels
    )
    if phobert_bertopic_result:
        results["PhoBERT + BERTopic"] = phobert_bertopic_result

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        "timestamp": timestamp,
        "models": results,
        "comparison": {
            "BERTopic-only": {
                "in_domain_f1": results.get("BERTopic-only", {}).get("test", {}).get("f1_macro", 0),
                "cross_domain_f1": results.get("BERTopic-only", {}).get("cross_domain", {}).get("f1_macro", 0),
            },
            "PhoBERT + BERTopic": {
                "in_domain_f1": results.get("PhoBERT + BERTopic", {}).get("test", {}).get("f1_macro", 0),
                "cross_domain_f1": results.get("PhoBERT + BERTopic", {}).get("cross_domain", {}).get("f1_macro", 0),
            },
        }
    }

    output_file = MODEL_DIR / f"bertopic_results_{timestamp}.json"
    output_file.write_text(json.dumps(output, indent=2))
    logger.info(f"Results saved: {output_file}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS (Augmented Dataset)")
    logger.info("=" * 60)
    for name, r in results.items():
        logger.info(f"{name}:")
        logger.info(f"  In-domain F1:     {r['test']['f1_macro']:.4f}")
        logger.info(f"  Cross-domain F1:  {r['cross_domain']['f1_macro']:.4f}")

    return results


if __name__ == "__main__":
    main()
