"""Standalone BERTopic training script - bypasses module import issues.

Run: python3 scripts/run_bertopic_standalone.py
"""
import sys
import os

# Add project to path
sys.path.insert(0, '/Users/iannwendy/Documents/crawl_yt/youtube_depression_crawler')
os.chdir('/Users/iannwendy/Documents/crawl_yt/youtube_depression_crawler')

import json
import logging
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("BERTopic")

# Paths
CORPUS_FILE = Path("data_unified/corpus_text_all.csv")
MODEL_DIR = Path("models/bertopic")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BERTOPIC_MODEL_FILE = MODEL_DIR / "bertopic_model.pkl"
BERTOPIC_EMBEDDINGS_FILE = MODEL_DIR / "embeddings.npy"
BERTOPIC_TOPICS_FILE = MODEL_DIR / "corpus_with_topics.csv"
BERTOPIC_METRICS_FILE = MODEL_DIR / "bertopic_metrics.json"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MIN_TOPIC_SIZE = 50


def load_corpus():
    """Load corpus."""
    df = pd.read_csv(CORPUS_FILE, dtype=str).fillna("")
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() >= 10].reset_index(drop=True)
    logger.info("Loaded corpus: %d rows", len(df))
    return df


def get_embeddings(texts, model):
    """Compute embeddings."""
    logger.info("Computing embeddings for %d texts...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


def train_topic_model(texts, embeddings):
    """Train BERTopic model."""
    logger.info("Training BERTopic on %d documents...", len(texts))
    
    umap_model = UMAP(
        n_neighbors=15,
        min_dist=0.0,
        n_components=5,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        top_n_words=10,
        verbose=True,
    )
    
    logger.info("Fitting BERTopic...")
    topics, _ = topic_model.fit_transform(texts, embeddings)
    logger.info("BERTopic fitted!")
    
    return topic_model, topics


def save_results(topic_model, df, embeddings, topics):
    """Save all results."""
    # Save model
    with open(BERTOPIC_MODEL_FILE, "wb") as f:
        pickle.dump(topic_model, f)
    logger.info("Saved model to %s", BERTOPIC_MODEL_FILE)
    
    # Save embeddings
    np.save(BERTOPIC_EMBEDDINGS_FILE, embeddings)
    logger.info("Saved embeddings to %s", BERTOPIC_EMBEDDINGS_FILE)
    
    # Assign topics to df
    df = df.copy()
    df["topic_id"] = topics
    df["topic_label"] = df["topic_id"].apply(
        lambda tid: _make_label(topic_model, tid)
    )
    
    # Save corpus with topics
    df.to_csv(BERTOPIC_TOPICS_FILE, index=False, encoding="utf-8-sig")
    logger.info("Saved corpus with topics to %s", BERTOPIC_TOPICS_FILE)
    
    # Save metrics
    topic_freq = topic_model.get_topic_freq()
    n_outliers = int(topic_freq[topic_freq["Topic"] == -1]["Count"].sum())
    total_docs = int(topic_freq["Count"].sum())
    
    metrics = {
        "n_documents": len(df),
        "n_topics": len(topic_freq),
        "n_outliers": n_outliers,
        "outlier_pct": round(n_outliers / total_docs * 100, 2) if total_docs > 0 else 0,
        "topic_distribution": topic_freq.head(30).to_dict(orient="records"),
    }
    
    with open(BERTOPIC_METRICS_FILE, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("Saved metrics to %s", BERTOPIC_METRICS_FILE)
    
    return metrics


def _make_label(model, topic_id):
    if topic_id == -1:
        return "outlier"
    try:
        words = model.get_topic(topic_id)
        if words:
            return " | ".join([w for w, _ in words[:5]])
        return f"topic_{topic_id}"
    except:
        return f"topic_{topic_id}"


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("BERTopic Standalone Training")
    logger.info("=" * 60)
    
    # Load corpus
    df = load_corpus()
    texts = df["text"].tolist()
    
    # Check for cached embeddings
    if BERTOPIC_EMBEDDINGS_FILE.exists():
        logger.info("Loading cached embeddings...")
        embeddings = np.load(BERTOPIC_EMBEDDINGS_FILE)
        if embeddings.shape[0] != len(texts):
            logger.warning("Size mismatch, recomputing...")
            embeddings = None
    
    if BERTOPIC_EMBEDDINGS_FILE.exists() and BERTOPIC_EMBEDDINGS_FILE.stat().st_size > 0:
        try:
            embeddings = np.load(BERTOPIC_EMBEDDINGS_FILE)
            if embeddings.shape[0] != len(texts):
                embeddings = None
        except:
            embeddings = None
    
    # Load embedding model
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Compute or load embeddings
    if "embeddings" not in dir() or embeddings is None:
        embeddings = get_embeddings(texts, embed_model)
        np.save(BERTOPIC_EMBEDDINGS_FILE, embeddings)
    
    # Train topic model
    topic_model, topics = train_topic_model(texts, embeddings)
    
    # Save results
    metrics = save_results(topic_model, df, embeddings, topics)
    
    # Print summary
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("COMPLETED in %.1f seconds", elapsed)
    logger.info("Topics discovered: %d", metrics["n_topics"])
    logger.info("Outliers: %d (%.1f%%)", metrics["n_outliers"], metrics["outlier_pct"])
    logger.info("Output: %s", BERTOPIC_TOPICS_FILE)
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("TOPIC SUMMARY (Top 20)")
    print("=" * 60)
    topic_info = topic_model.get_topic_info()
    for _, row in topic_info.head(20).iterrows():
        tid = int(row["Topic"])
        count = int(row["Count"])
        label = _make_label(topic_model, tid)
        pct = count / len(df) * 100
        print(f"Topic {tid:3d}: {label[:50]:<50} | {count:>6} ({pct:.1f}%)")
    
    return metrics


if __name__ == "__main__":
    main()
