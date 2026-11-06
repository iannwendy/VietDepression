"""Regenerate BERTopic metrics JSON from trained model."""

import json
import pickle
import pandas as pd

MODEL_DIR = "models/bertopic"
MODEL_FILE = f"{MODEL_DIR}/bertopic_model.pkl"
METRICS_FILE = f"{MODEL_DIR}/bertopic_metrics.json"


def _vietnamese_tokenize(text: str):
    import re
    if not text:
        return []
    text = str(text).lower()
    return re.findall(r"[a-zA-ZÀ-ỹ0-9]+", text)


def _build_vietnamese_vectorizer():
    from sklearn.feature_extraction.text import CountVectorizer
    return CountVectorizer(
        tokenizer=_vietnamese_tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=10000,
        lowercase=False,
    )


def _make_topic_label(topic_model, topic_id: int) -> str:
    if topic_id == -1:
        return "outlier"
    try:
        words = topic_model.get_topic(topic_id)
        if words:
            top_words = [w for w, _ in words[:5]]
            return " | ".join(top_words)
        return f"topic_{topic_id}"
    except Exception:
        return f"topic_{topic_id}"


def build_topic_summary(topic_model):
    topic_info = topic_model.get_topic_info()
    summaries = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        count = int(row["Count"])
        name = row["Name"]
        words_data = topic_model.get_topic(topic_id)
        top_words = []
        if words_data:
            for word, score in words_data[:10]:
                top_words.append({"word": word, "score": round(float(score), 4)})
        summaries.append({
            "topic_id": topic_id,
            "name": name,
            "document_count": count,
            "percentage": round(count / topic_info["Count"].sum() * 100, 2),
            "top_words": top_words,
        })
    return summaries


def main():
    print("Loading BERTopic model...")
    with open(MODEL_FILE, "rb") as f:
        topic_model = pickle.load(f)

    print("Building topic summary...")
    topic_summary = build_topic_summary(topic_model)
    topic_freq = topic_model.get_topic_freq()
    n_topics = topic_freq.shape[0]
    n_outliers = int(topic_freq[topic_freq["Topic"] == -1]["Count"].sum())
    n_docs = int(topic_freq["Count"].sum())

    # Load corpus for source breakdown (just sample)
    print("Loading corpus for source breakdown...")
    df = pd.read_csv(f"{MODEL_DIR}/corpus_with_topics.csv", dtype=str, nrows=1000)
    source_breakdown = {}
    for (src, tid), cnt in df.groupby(["source", "topic_id"]).size().to_dict().items():
        source_breakdown[f"{src}_{tid}"] = int(cnt)

    metrics = {
        "corpus": {
            "original_rows": 316401,
            "processed_rows": 309806,
            "columns": ["text", "source", "source_dataset", "affect_signal", "is_holdout_text"],
        },
        "model": {
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dim": 384,
            "min_topic_size": 50,
            "n_topics_discovered": n_topics,
            "n_outlier_documents": n_outliers,
            "outlier_percentage": round(n_outliers / n_docs * 100, 2) if n_docs > 0 else 0,
        },
        "topic_distribution": topic_summary[:50],
        "source_breakdown": source_breakdown,
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nSaved metrics: {n_topics} topics discovered")
    print(f"Outliers: {n_outliers} ({n_outliers/n_docs*100:.1f}%)")
    print(f"\nTop 10 topics:")
    for t in topic_summary[:10]:
        words = [w["word"] for w in t["top_words"][:5]]
        print(f"  Topic {t['topic_id']:>3}: {' | '.join(words)} ({t['document_count']} docs, {t['percentage']:.1f}%)")


if __name__ == "__main__":
    main()
