"""Train BERTopic with Vietnamese-aware tokenization using underthesea.

This script:
1. Loads corpus_text_all
2. Pre-tokenizes with underthesea (preserves Vietnamese diacritics)
3. Trains BERTopic with pre-computed embeddings
4. Saves model, corpus_with_topics, and metrics

Usage:
    PYTHONPATH=. .venv/bin/python scripts/train_bertopic_vietnamese.py
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from underthesea import word_tokenize

# Disable proxy
import os
for _var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_var, None)

PROJECT_DIR = Path(".")
MODEL_DIR = PROJECT_DIR / "models" / "bertopic"
CORPUS_FILE = PROJECT_DIR / "data_unified" / "corpus_text_all.csv"
EMBEDDINGS_FILE = MODEL_DIR / "embeddings.npy"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def vietnamese_tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text using underthesea, preserving diacritics.

    Returns space-separated tokens with underscores for compound words.
    """
    if not text or len(str(text).strip()) < 2:
        return ""
    text = str(text).strip()
    # underthesea word_tokenize returns "word1 word2_word3 word4"
    tokenized = word_tokenize(text, format="text")
    return tokenized


def _vietnamese_count_vectorizer_tokenize(text: str) -> list[str]:
    """Tokenizer for CountVectorizer that preserves diacritics."""
    if not text:
        return []
    return re.findall(r"[a-zA-ZÀ-ỹ0-9_]+", text)


def load_corpus():
    """Load and preprocess corpus."""
    logger.info("Loading corpus...")
    df = pd.read_csv(CORPUS_FILE, dtype=str).fillna("")
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() >= 10].reset_index(drop=True)
    logger.info("Loaded %d rows", len(df))
    return df


def load_embeddings():
    """Load pre-computed embeddings."""
    logger.info("Loading embeddings from %s", EMBEDDINGS_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)
    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


def build_vietnamese_vectorizer():
    """CountVectorizer with Vietnamese diacritics preserved."""
    return CountVectorizer(
        tokenizer=_vietnamese_count_vectorizer_tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=10000,
        lowercase=False,
    )


def train_bertopic():
    """Main training pipeline."""
    import bertopic
    from bertopic.cluster import BaseCluster
    from hdbscan import HDBSCAN
    from umap import UMAP

    # 1. Load data
    df = load_corpus()
    embeddings = load_embeddings()

    # Ensure embeddings match corpus size
    if embeddings.shape[0] != len(df):
        logger.warning(
            "Embeddings size (%d) != corpus size (%d). Using first %d embeddings.",
            embeddings.shape[0], len(df), min(embeddings.shape[0], len(df))
        )
        df = df.iloc[:embeddings.shape[0]].reset_index(drop=True)

    original_count = len(df)
    texts_raw = df["text"].tolist()

    # 2. Pre-tokenize with underthesea (preserves diacritics)
    logger.info("Pre-tokenizing %d texts with underthesea...", len(texts_raw))
    texts_tokenized = []
    for t in tqdm(texts_raw, desc="Tokenizing"):
        texts_tokenized.append(vietnamese_tokenize(t))
    logger.info("Tokenization complete. Sample: '%s' -> '%s'", texts_raw[0], texts_tokenized[0])

    # 3. Build BERTopic with Vietnamese-aware components
    logger.info("Building BERTopic model...")

    umap_model = UMAP(
        n_neighbors=15,
        min_dist=0.0,
        n_components=5,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = bertopic.BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=build_vietnamese_vectorizer(),
        representation_model=None,  # Disable c-TF-IDF to preserve diacritics
        calculate_probabilities=False,
        top_n_words=10,
        verbose=True,
    )

    # 4. Fit with pre-computed embeddings
    logger.info("Fitting BERTopic on %d documents...", len(texts_tokenized))
    topics, _ = topic_model.fit_transform(texts_tokenized, embeddings=embeddings)
    logger.info("Fitted. Topics discovered: %d", len(set(topics)))

    # 5. Assign topics to corpus
    df["topic_id"] = topics
    df["topic_label"] = df["topic_id"].apply(
        lambda tid: _make_label(topic_model, tid)
    )
    df["text_tokenized"] = texts_tokenized

    # 6. Build summary
    topic_info = topic_model.get_topic_info()
    n_topics = topic_info.shape[0]
    n_outliers = int(topic_info[topic_info["Topic"] == -1]["Count"].sum())
    n_docs = int(topic_info["Count"].sum())

    summaries = []
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        words_data = topic_model.get_topic(tid)
        top_words = []
        if words_data:
            for word, score in words_data[:10]:
                # Clean underscores for display
                word_clean = word.replace("_", " ")
                top_words.append({"word": word_clean, "score": round(float(score), 4)})
        summaries.append({
            "topic_id": tid,
            "name": row["Name"],
            "document_count": int(row["Count"]),
            "percentage": round(int(row["Count"]) / n_docs * 100, 2),
            "top_words": top_words,
        })

    # 7. Save artifacts
    logger.info("Saving artifacts...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    with open(MODEL_DIR / "bertopic_model.pkl", "wb") as f:
        pickle.dump(topic_model, f)
    logger.info("Saved model to %s", MODEL_DIR / "bertopic_model.pkl")

    # Corpus with topics
    df.to_csv(MODEL_DIR / "corpus_with_topics.csv", index=False, encoding="utf-8-sig")
    logger.info("Saved corpus to %s", MODEL_DIR / "corpus_with_topics.csv")

    # Metrics
    metrics = {
        "corpus": {
            "original_rows": original_count,
            "processed_rows": len(df),
        },
        "model": {
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dim": int(embeddings.shape[1]),
            "n_topics_discovered": n_topics,
            "n_outlier_documents": n_outliers,
            "outlier_percentage": round(n_outliers / n_docs * 100, 2) if n_docs > 0 else 0,
        },
        "topic_distribution": summaries[:50],
    }

    with open(MODEL_DIR / "bertopic_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("Saved metrics to %s", MODEL_DIR / "bertopic_metrics.json")

    # 8. Print top topics
    logger.info("\n" + "=" * 60)
    logger.info("TOP TOPICS (Top 15)")
    logger.info("=" * 60)
    for t in summaries[:15]:
        words = [w["word"] for w in t["top_words"][:5]]
        logger.info(
            "Topic %3d: %-40s | %6d docs (%.1f%%)",
            t["topic_id"],
            " | ".join(words),
            t["document_count"],
            t["percentage"],
        )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Total topics: %d", n_topics)
    logger.info("Outliers: %d (%.1f%%)", n_outliers, n_outliers / n_docs * 100 if n_docs > 0 else 0)
    logger.info("=" * 60)

    return metrics


def _make_label(topic_model, topic_id: int) -> str:
    """Create human-readable label from top words."""
    if topic_id == -1:
        return "outlier"
    try:
        words = topic_model.get_topic(topic_id)
        if words:
            top_words = [w.replace("_", " ") for w, _ in words[:5]]
            return " | ".join(top_words)
        return f"topic_{topic_id}"
    except Exception:
        return f"topic_{topic_id}"


if __name__ == "__main__":
    train_bertopic()
