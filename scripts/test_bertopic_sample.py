"""Quick test script for BERTopic module on a small sample.

Run this BEFORE running the full 316K corpus to verify:
1. Dependencies are installed correctly
2. Embedding model can be loaded
3. BERTopic can fit on a small sample

Usage:
    python scripts/test_bertopic_sample.py
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from yt_depression_crawler.modeling.bertopic import (
    compute_embeddings,
    get_embedding_model,
    build_topic_model,
    assign_topics_to_corpus,
    build_topic_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Use small sample for testing
SAMPLE_SIZE = 5000
CORPUS_PATH = "data_unified/corpus_text_all.csv"
OUTPUT_DIR = "models/bertopic"


def test_embedding_model():
    """Test 1: Load sentence transformer."""
    logger.info("=" * 60)
    logger.info("TEST 1: Loading embedding model...")
    t0 = time.time()

    model = get_embedding_model("paraphrase-multilingual-MiniLM-L12-v2")

    elapsed = time.time() - t0
    logger.info("✓ Embedding model loaded in %.1fs", elapsed)
    return model


def test_embeddings(model):
    """Test 2: Compute embeddings on sample."""
    logger.info("=" * 60)
    logger.info("TEST 2: Computing embeddings on %d samples...", SAMPLE_SIZE)
    t0 = time.time()

    # Load corpus
    df = pd.read_csv(CORPUS_PATH, dtype=str).fillna("")
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() >= 10].head(SAMPLE_SIZE).reset_index(drop=True)

    texts = df["text"].tolist()
    logger.info("Sample texts: %d", len(texts))

    # Compute embeddings
    embeddings = compute_embeddings(texts, model, batch_size=128)
    elapsed = time.time() - t0

    logger.info("✓ Embeddings shape: %s", embeddings.shape)
    logger.info("✓ Computed in %.1fs", elapsed)
    return df, embeddings


def test_topic_model(df, embeddings):
    """Test 3: Fit BERTopic on sample."""
    logger.info("=" * 60)
    logger.info("TEST 3: Fitting BERTopic on %d documents...", len(df))
    t0 = time.time()

    # Use larger min_topic_size for small sample
    topic_model = build_topic_model(
        embeddings=embeddings,
        texts=df["text"].tolist(),
        min_topic_size=20,  # Smaller for test
        umap_n_neighbors=5,
        hdbscan_min_cluster_size=20,
    )

    elapsed = time.time() - t0
    logger.info("✓ Topic model fitted in %.1fs", elapsed)

    # Get topic info
    topic_info = topic_model.get_topic_freq()
    logger.info("✓ Topics discovered: %d", len(topic_info))
    logger.info("\n%s", topic_info.head(10).to_string())

    return topic_model


def test_topic_assignment(topic_model, df, embeddings):
    """Test 4: Assign topics to corpus."""
    logger.info("=" * 60)
    logger.info("TEST 4: Assigning topics to corpus...")
    t0 = time.time()

    df_result = assign_topics_to_corpus(
        topic_model,
        df["text"].tolist(),
        df,
        embeddings=embeddings,  # Pass pre-computed embeddings
    )

    elapsed = time.time() - t0
    logger.info("✓ Topics assigned in %.1fs", elapsed)
    logger.info("✓ Topic IDs range: %d to %d",
                df_result["topic_id"].min(),
                df_result["topic_id"].max())

    # Show sample
    sample = df_result[["text", "topic_id", "topic_label"]].head(5)
    logger.info("\nSample results:\n%s", sample.to_string())

    return df_result


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("BERTOPIC MODULE TEST")
    logger.info("Sample size: %d documents", SAMPLE_SIZE)
    logger.info("=" * 60)

    try:
        # Test 1: Load model
        model = test_embedding_model()

        # Test 2: Compute embeddings
        df, embeddings = test_embeddings(model)

        # Test 3: Fit topic model
        topic_model = test_topic_model(df, embeddings)

        # Test 4: Assign topics
        df_result = test_topic_assignment(topic_model, df, embeddings)

        # Build summary
        summary = build_topic_summary(topic_model)
        logger.info("\n" + "=" * 60)
        logger.info("TOPIC SUMMARY (Top 10)")
        logger.info("=" * 60)
        for topic in summary[:10]:
            words = [w["word"] for w in topic["top_words"][:5]]
            logger.info("Topic %3d: %-30s | %5d docs (%.1f%%)",
                        topic["topic_id"],
                        " | ".join(words),
                        topic["document_count"],
                        topic["percentage"])

        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("Ready to run on full corpus (316K documents)")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.exception("TEST FAILED: %s", e)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
