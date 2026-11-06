"""Extract Vietnamese topic labels from corpus_with_topics.

Uses TF-IDF on each topic's documents to get meaningful keywords with diacritics.
"""

import json
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize

MODEL_DIR = "models/bertopic"


def vietnamese_tokenize(text: str) -> list[str]:
    """Tokenize with underthesea, preserving diacritics."""
    if not text:
        return []
    text = str(text).strip()
    text = word_tokenize(text, format="text")
    # Split on spaces and underscores
    tokens = re.findall(r"[a-zA-ZÀ-ỹ_]+", text)
    return [t for t in tokens if len(t) >= 2]


def extract_topic_labels(df: pd.DataFrame, min_docs: int = 50) -> dict:
    """Extract Vietnamese keywords for each topic using TF-IDF."""
    print(f"Extracting topic labels from {len(df)} documents...")

    # Filter out outliers and very small topics
    df_topics = df[df["topic_id"] != -1].copy()
    topic_counts = df_topics["topic_id"].value_counts()
    valid_topics = topic_counts[topic_counts >= min_docs].index.tolist()

    print(f"Valid topics (>= {min_docs} docs): {len(valid_topics)}")

    # TF-IDF vectorizer with Vietnamese tokenizer
    vectorizer = TfidfVectorizer(
        tokenizer=vietnamese_tokenize,
        token_pattern=None,
        max_features=5000,
        min_df=3,
        max_df=0.8,
        ngram_range=(1, 2),
    )

    topic_labels = {}
    for topic_id in sorted(valid_topics):
        topic_docs = df_topics[df_topics["topic_id"] == topic_id]["text"].tolist()
        if len(topic_docs) < min_docs:
            continue

        try:
            # Fit TF-IDF on topic documents
            tfidf_matrix = vectorizer.fit_transform(topic_docs)
            feature_names = vectorizer.get_feature_names_out()

            # Get top keywords by TF-IDF score
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-15:][::-1]
            top_keywords = [(feature_names[i], round(scores[i], 4)) for i in top_indices]

            # Format as "word1 | word2 | word3"
            words = [w for w, _ in top_keywords[:5]]
            label = " | ".join(words)

            topic_labels[int(topic_id)] = {
                "label": label,
                "keywords": [{"word": w, "score": s} for w, s in top_keywords[:10]],
                "document_count": len(topic_docs),
            }
        except Exception as e:
            print(f"  Topic {topic_id}: Error - {e}")
            continue

        if len(topic_labels) % 50 == 0:
            print(f"  Processed {len(topic_labels)} topics...")

    return topic_labels


def main():
    print("Loading corpus with topics...")
    df = pd.read_csv(f"{MODEL_DIR}/corpus_with_topics.csv", dtype=str)
    print(f"Loaded {len(df)} documents")
    print(f"Topic range: {df['topic_id'].min()} to {df['topic_id'].max()}")

    # Extract Vietnamese topic labels
    topic_labels = extract_topic_labels(df, min_docs=50)

    # Save
    output_file = f"{MODEL_DIR}/topic_labels_vietnamese.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(topic_labels, f, ensure_ascii=False, indent=2)
    print(f"\nSaved topic labels to {output_file}")

    # Print top topics
    print("\n" + "=" * 70)
    print("TOP TOPICS WITH VIETNAMESE KEYWORDS")
    print("=" * 70)

    # Sort by document count
    sorted_topics = sorted(
        topic_labels.items(),
        key=lambda x: x[1]["document_count"],
        reverse=True
    )

    for topic_id, data in sorted_topics[:20]:
        words = [w["word"] for w in data["keywords"][:5]]
        print(f"Topic {topic_id:>3}: {' | '.join(words):<50} | {data['document_count']:>6} docs")

    # Find depression-related topics
    print("\n" + "=" * 70)
    print("POTENTIAL DEPRESSION-RELATED TOPICS")
    print("=" * 70)

    depression_keywords = ["buồn", "mệt", "trầm", "chán", "mất", "ngủ", "tự", "hại", "cuộc sống", "không", "muốn", "sợ", "lo"]

    for topic_id, data in sorted_topics:
        words_text = " ".join([w["word"] for w in data["keywords"]])
        if any(kw in words_text.lower() for kw in depression_keywords):
            words = [w["word"] for w in data["keywords"][:5]]
            print(f"Topic {topic_id:>3}: {' | '.join(words):<50} | {data['document_count']:>6} docs")


if __name__ == "__main__":
    main()
