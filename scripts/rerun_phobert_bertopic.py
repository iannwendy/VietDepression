"""A2: Rerun PhoBERT + BERTopic on final_dataset (1,786 samples).

The previous Table 5.1 row (0.9501 / 0.3977) was trained on the pre-round-3
data (phobert_second, 1,124 train rows). This script:

  1. Loads the fine-tuned PhoBERT checkpoint trained on final_dataset
     (models/phobert_first/, 1,786 train rows).
  2. Loads the BERTopic model trained on the 316K unified corpus.
  3. Concatenates PhoBERT CLS embeddings + (topic_id, topic_prob) features.
  4. Fits a logistic regression classifier with class_weight="balanced".
  5. Evaluates on:
     - In-domain test (data/final_test.csv, 383 rows)
     - Cross-domain VSMEC test (data_unified/cross_domain_test.csv, 3,084 rows)

Output: docs/phase3_phobert_bertopic_metrics.json + console summary.
Reproduces paper §5.1 Table 5.1 last row.

Usage:
    python3 scripts/rerun_phobert_bertopic.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler


SEED = 42
FINAL_TRAIN = PROJECT_DIR / "data" / "final_train.csv"
FINAL_VAL = PROJECT_DIR / "data" / "final_val.csv"
FINAL_TEST = PROJECT_DIR / "data" / "final_test.csv"
CROSS_DOMAIN_TEST = PROJECT_DIR / "data_unified" / "cross_domain_test.csv"

# Fine-tuned on final_dataset (post-round-3, 1,786 train rows)
PHOBERT_DIR = PROJECT_DIR / "models" / "phobert_first"
BERTOPIC_MODEL_FILE = PROJECT_DIR / "models" / "bertopic" / "bertopic_model.pkl"

OUTPUT_METRICS = PROJECT_DIR / "docs" / "phase3_phobert_bertopic_metrics.json"
OUTPUT_PREDICTIONS_IN = PROJECT_DIR / "results" / "phobert_bertopic_predictions_in_domain.csv"
OUTPUT_PREDICTIONS_CROSS = PROJECT_DIR / "results" / "phobert_bertopic_predictions_cross_domain.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("phobert_bertopic")


def load_split(path: Path) -> pd.DataFrame:
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


def get_phobert_embeddings(texts: list[str], device, tokenizer, model, prepare, dataset_cls, max_len: int, batch_size: int = 16) -> np.ndarray:
    """Extract CLS-token embeddings from fine-tuned PhoBERT."""
    import torch
    from torch.utils.data import DataLoader

    prepared = prepare(list(texts))
    dataset = dataset_cls(prepared, None, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    return np.vstack(embeddings)


def get_topic_features(texts: list[str], topic_model, topic_embedder) -> np.ndarray:
    """Compute (topic_id, topic_prob) features using the trained BERTopic model."""
    embeddings = topic_embedder.encode(texts, show_progress_bar=True, batch_size=64)
    topics, probs = topic_model.transform(texts, embeddings=embeddings)

    # Some rows return probs as scalar or None — coerce to float.
    if probs is None:
        probs_arr = np.zeros(len(texts))
    else:
        probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
        # If 2D (multi-topic), take max
        if probs_arr.ndim > 1:
            probs_arr = probs_arr.max(axis=-1)

    return np.column_stack([
        np.array(topics, dtype=np.float64).reshape(-1, 1),
        probs_arr.reshape(-1, 1),
    ])


def main() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from yt_depression_crawler.modeling.phobert.phobert_utils import (
        PhoBertDataset, get_device, prepare_many_texts,
    )
    from sentence_transformers import SentenceTransformer

    OUTPUT_PREDICTIONS_IN.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("A2: PhoBERT + BERTopic rerun on final_dataset")
    logger.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────────
    train_df = load_split(FINAL_TRAIN)
    val_df = load_split(FINAL_VAL)
    test_df = load_split(FINAL_TEST)
    cross_df = load_split(CROSS_DOMAIN_TEST)
    logger.info("train=%d val=%d test=%d cross_domain=%d",
                len(train_df), len(val_df), len(test_df), len(cross_df))

    # ── Load models ──────────────────────────────────────────────────
    device = get_device()
    logger.info("Device: %s", device)

    logger.info("Loading PhoBERT from %s", PHOBERT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(PHOBERT_DIR), use_fast=False)
    phobert = AutoModel.from_pretrained(str(PHOBERT_DIR))
    phobert.to(device)
    phobert.eval()

    logger.info("Loading BERTopic model from %s", BERTOPIC_MODEL_FILE)
    with open(BERTOPIC_MODEL_FILE, "rb") as f:
        topic_model = pickle.load(f)

    logger.info("Loading topic embedder (paraphrase-multilingual-MiniLM-L12-v2)")
    topic_embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # ── Combine train+val for fitting (paper §4.3 practice) ────────
    fit_texts = train_df["comment_text"].tolist() + val_df["comment_text"].tolist()
    fit_labels = (
        train_df["label"].astype(int).tolist() + val_df["label"].astype(int).tolist()
    )
    logger.info("Fitting on combined train+val: %d samples", len(fit_texts))

    # ── Feature extraction ──────────────────────────────────────────
    logger.info("Extracting PhoBERT CLS embeddings for fit set...")
    phobert_fit = get_phobert_embeddings(
        fit_texts, device, tokenizer, phobert, prepare_many_texts, PhoBertDataset, 128
    )
    logger.info("  shape: %s", phobert_fit.shape)

    logger.info("Extracting BERTopic features for fit set...")
    topic_fit = get_topic_features(fit_texts, topic_model, topic_embedder)
    logger.info("  shape: %s", topic_fit.shape)

    X_fit = np.hstack([phobert_fit, topic_fit])
    logger.info("Combined feature shape: %s", X_fit.shape)

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit)

    # ── Train logistic regression ────────────────────────────────────
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED
    )
    clf.fit(X_fit, fit_labels)
    logger.info("LogReg fitted. Class distribution: %s",
                dict(zip(*np.unique(fit_labels, return_counts=True))))

    # ── Evaluate on in-domain test ──────────────────────────────────
    logger.info("Extracting features for in-domain test (%d rows)...", len(test_df))
    phobert_test = get_phobert_embeddings(
        test_df["comment_text"].tolist(), device, tokenizer, phobert,
        prepare_many_texts, PhoBertDataset, 128
    )
    topic_test = get_topic_features(
        test_df["comment_text"].tolist(), topic_model, topic_embedder
    )
    X_test = scaler.transform(np.hstack([phobert_test, topic_test]))
    y_pred_in = clf.predict(X_test)
    y_true_in = test_df["label"].astype(int).to_numpy()
    in_metrics = compute_metrics(y_true_in, y_pred_in)

    # Save in-domain predictions
    pd.DataFrame({
        "comment_text": test_df["comment_text"].values,
        "true_label": y_true_in,
        "pred_label": y_pred_in,
    }).to_csv(OUTPUT_PREDICTIONS_IN, index=False)
    logger.info("In-domain F1-macro: %.4f | F1-dep: %.4f | Accuracy: %.4f",
                in_metrics["f1_macro"], in_metrics["f1_depression"], in_metrics["accuracy"])

    # ── Evaluate on cross-domain VSMEC ──────────────────────────────
    logger.info("Extracting features for cross-domain VSMEC (%d rows)...", len(cross_df))
    phobert_cross = get_phobert_embeddings(
        cross_df["comment_text"].tolist(), device, tokenizer, phobert,
        prepare_many_texts, PhoBertDataset, 128
    )
    topic_cross = get_topic_features(
        cross_df["comment_text"].tolist(), topic_model, topic_embedder
    )
    X_cross = scaler.transform(np.hstack([phobert_cross, topic_cross]))
    y_pred_cross = clf.predict(X_cross)
    y_true_cross = cross_df["label"].astype(int).to_numpy()
    cross_metrics = compute_metrics(y_true_cross, y_pred_cross)

    # Save cross-domain predictions
    pd.DataFrame({
        "comment_text": cross_df["comment_text"].values,
        "true_label": y_true_cross,
        "pred_label": y_pred_cross,
    }).to_csv(OUTPUT_PREDICTIONS_CROSS, index=False)
    logger.info("Cross-domain F1-macro: %.4f | F1-dep: %.4f | Accuracy: %.4f",
                cross_metrics["f1_macro"], cross_metrics["f1_depression"], cross_metrics["accuracy"])

    # ── Summary ─────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("PhoBERT + BERTopic (post-round-3, on final_dataset)")
    logger.info("  In-domain    F1-macro: %.4f  F1-dep: %.4f  Accuracy: %.4f",
                in_metrics["f1_macro"], in_metrics["f1_depression"], in_metrics["accuracy"])
    logger.info("  Cross-domain F1-macro: %.4f  F1-dep: %.4f  Accuracy: %.4f",
                cross_metrics["f1_macro"], cross_metrics["f1_depression"], cross_metrics["accuracy"])
    logger.info("  Δ F1 (in - cross): %.4f",
                in_metrics["f1_macro"] - cross_metrics["f1_macro"])

    # ── Persist ─────────────────────────────────────────────────────
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model": "PhoBERT + BERTopic (rerun on final_dataset)",
        "phobert_checkpoint": str(PHOBERT_DIR),
        "bertopic_model": str(BERTOPIC_MODEL_FILE),
        "fit_samples": len(fit_texts),
        "in_domain": in_metrics,
        "cross_domain": cross_metrics,
        "delta_f1_in_minus_cross": round(
            in_metrics["f1_macro"] - cross_metrics["f1_macro"], 4
        ),
        "predictions_in_domain_csv": str(OUTPUT_PREDICTIONS_IN),
        "predictions_cross_domain_csv": str(OUTPUT_PREDICTIONS_CROSS),
    }
    OUTPUT_METRICS.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Metrics saved: %s", OUTPUT_METRICS)


if __name__ == "__main__":
    main()