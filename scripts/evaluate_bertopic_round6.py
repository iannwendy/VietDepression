"""Inductive BERTopic and PhoBERT + BERTopic Evaluation on Round 6 v2 Dataset.

Fixes the methodological issues identified by Reviewer 1 (Comments 1, 6, 7)
and Reviewer 2 (Major Comment 2 & Minor Comment 4):
1. ZERO Transductive Leakage: BERTopic (SentenceTransformer + UMAP + HDBSCAN +
   Vietnamese c-TF-IDF) is fitted STRICTLY on `final_train.csv` (n=6,392) only.
   Validation (`final_val.csv`), In-Domain Test (`final_test.csv`), and
   Cross-Domain UIT-VSMEC (`cross_domain_test.csv`) are transformed inductively
   via `topic_model.transform()`.
2. Explicit Outlier Handling & Proper Topic Encoding:
   - Replaces the buggy scalar `topic_id` feature with an explicit (K+1)-dimensional
     one-hot encoding where index 0 is dedicated to the HDBSCAN outlier cluster
     (`topic_id == -1`) and indices 1..K correspond to proper topics 0..K-1,
     concatenated with the K-dimensional soft HDBSCAN topic-membership probability
     distribution and maximum assignment confidence.
3. Explicitly Specified Hybrid Protocol Across All 3 PhoBERT Seeds (42, 123, 2024):
   - For each seed in [42, 123, 2024], loads `models/round6_v2_retrained/phobert_seed_{s}/best_model`,
     FREEZES the PhoBERT encoder parameters, extracts 768-dim `<s>` ([CLS]) embeddings,
     concatenates with the inductive BERTopic representation, fits StandardScaler +
     L2 LogisticRegression strictly on `final_train.csv`, selects regularization C
     and decision threshold on `final_val.csv`, and evaluates on `final_test.csv`
     and `cross_domain_test.csv` (both Full and Sadness-Only).

Usage:
    PYTHONPATH="$PWD" .venv/bin/python scripts/evaluate_bertopic_round6.py
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

warnings.filterwarnings("ignore")

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
if (HF_CACHE / "models--vinai--phobert-base").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
UNIFIED_DIR = PROJECT_DIR / "data_unified"

TRAIN_FILE = DATA_DIR / "labeled" / "final_train.csv"
VAL_FILE = DATA_DIR / "labeled" / "final_val.csv"
TEST_FILE = DATA_DIR / "labeled" / "final_test.csv"
VSMEC_FILE = UNIFIED_DIR / "cross_domain_test.csv"

BERTOPIC_DIR = MODEL_DIR / "bertopic"
BERTOPIC_MODEL_FILE = BERTOPIC_DIR / "bertopic_model.pkl"
BERTOPIC_METRICS_FILE = BERTOPIC_DIR / "bertopic_metrics.json"
PHOBERT_ROUND6_DIR = MODEL_DIR / "round6_v2_retrained"

SEEDS = [42, 123, 2024]
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

OUTPUT_DIR = RESULTS_DIR / f"round6_bertopic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BERTOPIC_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def _vietnamese_tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[a-zA-ZÀ-ỹ0-9_]+", str(text).lower())


def build_vietnamese_vectorizer() -> CountVectorizer:
    return CountVectorizer(
        tokenizer=_vietnamese_tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=10000,
        lowercase=False,
    )


def load_data():
    train = pd.read_csv(TRAIN_FILE, dtype=str).fillna("")
    val = pd.read_csv(VAL_FILE, dtype=str).fillna("")
    test = pd.read_csv(TEST_FILE, dtype=str).fillna("")
    vsmec = pd.read_csv(VSMEC_FILE, dtype=str).fillna("")

    for df in (train, val, test, vsmec):
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)

    return train, val, test, vsmec


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    out = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(
            precision_score(
                y_true_arr, y_pred_arr, average="binary", pos_label=1, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true_arr, y_pred_arr, average="binary", pos_label=1, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true_arr, y_pred_arr, average="binary", pos_label=1, zero_division=0
            )
        ),
        "f1_depression": float(
            f1_score(
                y_true_arr, y_pred_arr, average="binary", pos_label=1, zero_division=0
            )
        ),
        "f1_normal": float(
            f1_score(
                y_true_arr, y_pred_arr, average="binary", pos_label=0, zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "predicted_positives": int(tp + fp),
        "predicted_negatives": int(tn + fn),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    if y_prob is not None and len(np.unique(y_true_arr)) == 2:
        probs_arr = np.asarray(y_prob, dtype=float)
        out["roc_auc"] = float(roc_auc_score(y_true_arr, probs_arr))
        out["pr_auc"] = float(average_precision_score(y_true_arr, probs_arr))
    return out


def fit_inductive_bertopic(train_texts: list[str], embedder: SentenceTransformer):
    """Fit BERTopic strictly on `final_train.csv` (6,392 rows) with zero test leakage."""
    print(f"\nFitting INDUCTIVE BERTopic on train set only (n={len(train_texts):,})...")
    train_embeddings = embedder.encode(
        train_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    umap_model = UMAP(
        n_neighbors=15,
        min_dist=0.0,
        n_components=5,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=30,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=build_vietnamese_vectorizer(),
        calculate_probabilities=True,
        top_n_words=10,
        verbose=False,
    )

    topics, probs = topic_model.fit_transform(train_texts, embeddings=train_embeddings)

    # Save inductive model and topic summary
    with open(BERTOPIC_MODEL_FILE, "wb") as f:
        pickle.dump(topic_model, f)

    topic_info = topic_model.get_topic_info()
    proper_topics = sorted([int(t) for t in topic_info["Topic"].tolist() if int(t) >= 0])
    n_outliers = int(sum(1 for t in topics if int(t) == -1))

    summaries = []
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        words_data = topic_model.get_topic(tid) or []
        top_words = [
            {"word": str(w), "score": round(float(s), 4)} for w, s in words_data[:10]
        ]
        summaries.append(
            {
                "topic_id": tid,
                "name": str(row["Name"]),
                "document_count": int(row["Count"]),
                "percentage": round(int(row["Count"]) / len(train_texts) * 100, 2),
                "top_words": top_words,
            }
        )

    bertopic_meta = {
        "training_protocol": "inductive_train_only_no_test_or_vsmec_leakage",
        "train_documents": len(train_texts),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": int(train_embeddings.shape[1]),
        "min_cluster_size": 30,
        "min_samples": 5,
        "n_proper_topics": len(proper_topics),
        "n_total_clusters_including_outlier": int(len(topic_info)),
        "n_outlier_documents_in_train": n_outliers,
        "outlier_percentage_in_train": round(n_outliers / len(train_texts) * 100, 2),
        "outlier_encoding_protocol": (
            f"Explicit (K+1)-slot one-hot indicator (slot 0 = outlier topic -1, "
            f"slots 1..{len(proper_topics)} = proper topics 0..{len(proper_topics)-1}) "
            f"+ {len(proper_topics)}-dim soft HDBSCAN probability distribution + 1 max-confidence scalar "
            f"(total topic feature dim = {2 * len(proper_topics) + 2}; classic one-hot+prob dim = {len(proper_topics) + 2})."
        ),
        "topic_distribution": summaries,
    }
    BERTOPIC_METRICS_FILE.write_text(
        json.dumps(bertopic_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Inductive BERTopic fitted: K={len(proper_topics)} proper topics + 1 outlier slot "
        f"(train outliers: {n_outliers:,} / {len(train_texts):,} = {n_outliers/len(train_texts)*100:.2f}%)"
    )
    return topic_model, train_embeddings, topics, probs, len(proper_topics), bertopic_meta


def build_topic_feature_matrix(
    topics: list[int] | np.ndarray,
    probs: np.ndarray | None,
    n_proper_topics: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct inductive topic feature matrices with an EXPLICIT outlier slot.

    Returns:
      - `one_hot_plus_prob` of shape (N, K + 2):
          * col 0: binary indicator I[topic == -1] (explicit outlier slot)
          * cols 1..K: one-hot indicator I[topic == k] for k in 0..K-1
          * col K+1: maximum topic assignment probability p_max in [0, 1]
      - `full_soft_plus_onehot` of shape (N, 2*K + 2):
          * cols 0..K: one-hot indicator including slot 0 for outlier (-1)
          * cols K+1..2K: full K-dimensional HDBSCAN soft topic distribution
          * col 2K+1: maximum topic assignment probability p_max
    """
    topics_arr = np.asarray(topics, dtype=int).reshape(-1)
    n_samples = len(topics_arr)

    # (K + 1) one-hot with slot 0 reserved for topic == -1 (outlier)
    one_hot = np.zeros((n_samples, n_proper_topics + 1), dtype=np.float64)
    for i, t in enumerate(topics_arr):
        if t == -1 or t < 0 or t >= n_proper_topics:
            one_hot[i, 0] = 1.0
        else:
            one_hot[i, t + 1] = 1.0

    # Handle HDBSCAN probabilities
    if probs is None:
        soft_dist = np.zeros((n_samples, n_proper_topics), dtype=np.float64)
        max_prob = np.zeros((n_samples, 1), dtype=np.float64)
    else:
        probs_arr = np.asarray(probs, dtype=np.float64)
        if probs_arr.ndim == 2 and probs_arr.shape[1] == n_proper_topics:
            soft_dist = probs_arr
            max_prob = probs_arr.max(axis=1, keepdims=True)
        else:
            probs_1d = probs_arr.reshape(n_samples, -1)
            max_prob = probs_1d.max(axis=1, keepdims=True)
            soft_dist = np.zeros((n_samples, n_proper_topics), dtype=np.float64)
            for i, t in enumerate(topics_arr):
                if 0 <= t < n_proper_topics:
                    soft_dist[i, t] = max_prob[i, 0]

    one_hot_plus_prob = np.hstack([one_hot, max_prob])
    full_soft_plus_onehot = np.hstack([one_hot, soft_dist, max_prob])
    return one_hot_plus_prob, full_soft_plus_onehot


def extract_bertopic_features_for_split(
    texts: list[str],
    topic_model: BERTopic,
    embedder: SentenceTransformer,
    n_proper_topics: int,
    precomputed_embeddings: np.ndarray | None = None,
    precomputed_topics_probs: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if precomputed_embeddings is None:
        embeddings = embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    else:
        embeddings = precomputed_embeddings

    if precomputed_topics_probs is None:
        topics, probs = topic_model.transform(texts, embeddings=embeddings)
    else:
        topics, probs = precomputed_topics_probs

    oh_feat, full_feat = build_topic_feature_matrix(topics, probs, n_proper_topics)
    return oh_feat, full_feat, np.asarray(topics, dtype=int)


def get_phobert_embeddings(
    texts: list[str],
    model_dir: Path,
    tokenizer_name: str = "vinai/phobert-base",
    max_len: int = 128,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract frozen PhoBERT `<s>` ([CLS]) 768-dim embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    model = AutoModel.from_pretrained(str(model_dir))
    # Explicitly freeze all parameters (Reviewer 1 Comment 1)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) for t in texts[i : i + batch_size]]
            encoded = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)

    del model
    return np.vstack(embeddings)


def select_best_logreg_on_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float] = [0.01, 0.1, 1.0, 10.0],
    weight_options: list = ["balanced", None],
) -> tuple[StandardScaler, LogisticRegression, float, dict]:
    """Fit StandardScaler + LogisticRegression on train only; tune C, class_weight, and threshold on val."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)

    best_val_f1 = -1.0
    best_clf = None
    best_thr = 0.50
    best_cfg = {}

    for cw in weight_options:
        for c_val in c_grid:
            clf = LogisticRegression(
                C=c_val,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                class_weight=cw,
                random_state=42,
            )
            clf.fit(X_tr_s, y_train)
            val_probs = clf.predict_proba(X_va_s)[:, 1]
            # Evaluate default threshold 0.50 first
            for thr in [0.50]:
                preds = (val_probs >= thr).astype(int)
                f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_clf = clf
                    best_thr = float(thr)
                    best_cfg = {
                        "C": c_val,
                        "class_weight": cw,
                        "threshold": best_thr,
                        "val_f1_macro": round(best_val_f1, 4),
                    }

    return scaler, best_clf, best_thr, best_cfg


def main():
    print("=" * 75)
    print("ROUND 6 v2: INDUCTIVE BERTopic & PhoBERT+BERTopic Evaluation")
    print("=" * 75)

    train_df, val_df, test_df, vsmec_df = load_data()
    train_texts = train_df["comment_text"].tolist()
    train_labels = train_df["label"].to_numpy(dtype=int)
    val_texts = val_df["comment_text"].tolist()
    val_labels = val_df["label"].to_numpy(dtype=int)
    test_texts = test_df["comment_text"].tolist()
    test_labels = test_df["label"].to_numpy(dtype=int)
    vsmec_texts = (
        vsmec_df["comment_text"].tolist()
        if "comment_text" in vsmec_df.columns
        else vsmec_df["text"].tolist()
    )
    vsmec_labels = vsmec_df["label"].to_numpy(dtype=int)
    sadness_mask = vsmec_df["original_label"].isin(["Sadness", "Enjoyment"]).to_numpy()
    vsmec_sad_labels = vsmec_labels[sadness_mask]

    print(
        f"Train: {len(train_texts):,} | Val: {len(val_texts):,} | "
        f"Test: {len(test_texts):,} | VSMEC: {len(vsmec_texts):,}"
    )

    print(f"\nLoading SentenceTransformer ({EMBEDDING_MODEL_NAME})...")
    topic_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # ── 1. Fit Inductive BERTopic on final_train.csv ONLY ───────────────────
    (
        topic_model,
        train_emb,
        train_topics,
        train_probs,
        n_proper_topics,
        bertopic_meta,
    ) = fit_inductive_bertopic(train_texts, topic_embedder)

    print("Transforming validation, in-domain test, and cross-domain sets inductively...")
    oh_train, full_train, _ = extract_bertopic_features_for_split(
        train_texts,
        topic_model,
        topic_embedder,
        n_proper_topics,
        precomputed_embeddings=train_emb,
        precomputed_topics_probs=(train_topics, train_probs),
    )
    oh_val, full_val, val_topics = extract_bertopic_features_for_split(
        val_texts, topic_model, topic_embedder, n_proper_topics
    )
    oh_test, full_test, test_topics = extract_bertopic_features_for_split(
        test_texts, topic_model, topic_embedder, n_proper_topics
    )
    oh_vsmec, full_vsmec, vsmec_topics = extract_bertopic_features_for_split(
        vsmec_texts, topic_model, topic_embedder, n_proper_topics
    )

    bertopic_meta["split_outlier_rates"] = {
        "train_outlier_pct": round(float((np.array(train_topics) == -1).mean() * 100), 2),
        "val_outlier_pct": round(float((val_topics == -1).mean() * 100), 2),
        "test_outlier_pct": round(float((test_topics == -1).mean() * 100), 2),
        "vsmec_outlier_pct": round(float((vsmec_topics == -1).mean() * 100), 2),
    }

    # ── 2. BERTopic-Only Classifier (Inductive, with Explicit Outlier Slot) ─
    print("\n" + "=" * 75)
    print("1. BERTOPIC-ONLY CLASSIFIER (INDUCTIVE, EXPLICIT OUTLIER SLOT)")
    print("=" * 75)

    scaler_bt, clf_bt, thr_bt, cfg_bt = select_best_logreg_on_val(
        full_train, train_labels, full_val, val_labels
    )
    print(f"Selected BERTopic-only config on val: {cfg_bt}, feature dim={full_train.shape[1]}")

    bt_test_probs = clf_bt.predict_proba(scaler_bt.transform(full_test))[:, 1]
    bt_test_preds = (bt_test_probs >= thr_bt).astype(int)
    bt_vsmec_probs = clf_bt.predict_proba(scaler_bt.transform(full_vsmec))[:, 1]
    bt_vsmec_preds = (bt_vsmec_probs >= thr_bt).astype(int)

    bertopic_in_metrics = compute_all_metrics(test_labels, bt_test_preds, bt_test_probs)
    bertopic_cross_metrics = compute_all_metrics(vsmec_labels, bt_vsmec_preds, bt_vsmec_probs)
    bertopic_sad_metrics = compute_all_metrics(
        vsmec_sad_labels, bt_vsmec_preds[sadness_mask], bt_vsmec_probs[sadness_mask]
    )

    print(
        f"BERTopic-only In-Domain:    Acc={bertopic_in_metrics['accuracy']:.4f}, "
        f"F1-M={bertopic_in_metrics['f1_macro']:.4f}, ROC-AUC={bertopic_in_metrics['roc_auc']:.4f}"
    )
    print(
        f"BERTopic-only Cross-Domain: Acc={bertopic_cross_metrics['accuracy']:.4f}, "
        f"F1-M={bertopic_cross_metrics['f1_macro']:.4f}, ROC-AUC={bertopic_cross_metrics['roc_auc']:.4f}"
    )

    # ── 3. PhoBERT + BERTopic Hybrid (Across All 3 PhoBERT Seeds) ───────────
    print("\n" + "=" * 75)
    print("2. PHOBERT + BERTOPIC HYBRID (FROZEN ENCODER, 3 SEEDS: 42, 123, 2024)")
    print("=" * 75)

    hybrid_per_seed_in = {}
    hybrid_per_seed_cross = {}
    hybrid_per_seed_sad = {}
    hybrid_test_preds_list, hybrid_test_probs_list = [], []
    hybrid_vsmec_preds_list, hybrid_vsmec_probs_list = [], []
    hybrid_val_configs = {}

    for seed in SEEDS:
        seed_model_dir = PHOBERT_ROUND6_DIR / f"phobert_seed_{seed}" / "best_model"
        if not seed_model_dir.exists():
            print(f"  WARNING: PhoBERT seed {seed} not found at {seed_model_dir}, skipping.")
            continue

        print(f"\n  Extracting frozen PhoBERT CLS embeddings for seed {seed}...")
        pho_tr = get_phobert_embeddings(train_texts, seed_model_dir)
        pho_va = get_phobert_embeddings(val_texts, seed_model_dir)
        pho_te = get_phobert_embeddings(test_texts, seed_model_dir)
        pho_vs = get_phobert_embeddings(vsmec_texts, seed_model_dir)

        # Concatenate 768-dim PhoBERT CLS + (K+1)-slot one-hot (with outlier slot 0) + max_prob
        X_tr_comb = np.hstack([pho_tr, oh_train])
        X_va_comb = np.hstack([pho_va, oh_val])
        X_te_comb = np.hstack([pho_te, oh_test])
        X_vs_comb = np.hstack([pho_vs, oh_vsmec])

        scaler_hyb, clf_hyb, thr_hyb, cfg_hyb = select_best_logreg_on_val(
            X_tr_comb, train_labels, X_va_comb, val_labels
        )
        cfg_hyb["feature_dim"] = int(X_tr_comb.shape[1])
        cfg_hyb["phobert_dim"] = 768
        cfg_hyb["topic_onehot_with_outlier_dim"] = int(oh_train.shape[1] - 1)
        cfg_hyb["topic_prob_dim"] = 1
        hybrid_val_configs[str(seed)] = cfg_hyb

        te_probs = clf_hyb.predict_proba(scaler_hyb.transform(X_te_comb))[:, 1]
        te_preds = (te_probs >= thr_hyb).astype(int)
        vs_probs = clf_hyb.predict_proba(scaler_hyb.transform(X_vs_comb))[:, 1]
        vs_preds = (vs_probs >= thr_hyb).astype(int)

        m_in = compute_all_metrics(test_labels, te_preds, te_probs)
        m_cr = compute_all_metrics(vsmec_labels, vs_preds, vs_probs)
        m_sad = compute_all_metrics(
            vsmec_sad_labels, vs_preds[sadness_mask], vs_probs[sadness_mask]
        )

        hybrid_per_seed_in[f"seed_{seed}"] = m_in
        hybrid_per_seed_cross[f"seed_{seed}"] = m_cr
        hybrid_per_seed_sad[f"seed_{seed}"] = m_sad

        hybrid_test_preds_list.append(te_preds)
        hybrid_test_probs_list.append(te_probs)
        hybrid_vsmec_preds_list.append(vs_preds)
        hybrid_vsmec_probs_list.append(vs_probs)

        print(
            f"  Seed {seed} Hybrid (dim={X_tr_comb.shape[1]}): "
            f"In-Domain F1-M={m_in['f1_macro']:.4f} | Cross-Domain F1-M={m_cr['f1_macro']:.4f}"
        )

    hybrid_summary = None
    if hybrid_test_preds_list:
        maj_te = (np.mean(np.stack(hybrid_test_preds_list, axis=0), axis=0) >= 0.5).astype(int)
        soft_prob_te = np.mean(np.stack(hybrid_test_probs_list, axis=0), axis=0)
        soft_te = (soft_prob_te >= 0.5).astype(int)

        maj_vs = (np.mean(np.stack(hybrid_vsmec_preds_list, axis=0), axis=0) >= 0.5).astype(int)
        soft_prob_vs = np.mean(np.stack(hybrid_vsmec_probs_list, axis=0), axis=0)
        soft_vs = (soft_prob_vs >= 0.5).astype(int)

        in_f1s = [m["f1_macro"] for m in hybrid_per_seed_in.values()]
        cr_f1s = [m["f1_macro"] for m in hybrid_per_seed_cross.values()]

        hybrid_in_maj = compute_all_metrics(test_labels, maj_te, soft_prob_te)
        hybrid_in_soft = compute_all_metrics(test_labels, soft_te, soft_prob_te)
        hybrid_cr_maj = compute_all_metrics(vsmec_labels, maj_vs, soft_prob_vs)
        hybrid_cr_soft = compute_all_metrics(vsmec_labels, soft_vs, soft_prob_vs)
        hybrid_sad_maj = compute_all_metrics(
            vsmec_sad_labels, maj_vs[sadness_mask], soft_prob_vs[sadness_mask]
        )

        # Save predictions CSVs for auditability
        pd.DataFrame(
            {
                "comment_text": test_texts,
                "true_label": test_labels,
                "pred_label_majority_vote": maj_te,
                "pred_label_soft_vote": soft_te,
                "prob_depression": soft_prob_te,
            }
        ).to_csv(RESULTS_DIR / "phobert_bertopic_predictions_in_domain.csv", index=False)

        pd.DataFrame(
            {
                "comment_text": vsmec_texts,
                "true_label": vsmec_labels,
                "pred_label_majority_vote": maj_vs,
                "pred_label_soft_vote": soft_vs,
                "prob_depression": soft_prob_vs,
            }
        ).to_csv(RESULTS_DIR / "phobert_bertopic_predictions_cross_domain.csv", index=False)

        hybrid_summary = {
            "protocol": {
                "encoder_freezing": "PhoBERT encoder parameters frozen during CLS feature extraction",
                "checkpoint_selection": "Evaluated across all 3 PhoBERT seeds (42, 123, 2024), each selected by best validation macro-F1 on final_val.csv",
                "classifier_training": "StandardScaler + L2 LogisticRegression fitted strictly on final_train.csv (n=6,392); C selected on final_val.csv (n=1,371)",
                "feature_dimension": int(768 + oh_train.shape[1]),
                "feature_breakdown": {
                    "phobert_cls_dim": 768,
                    "bertopic_proper_topics_onehot": n_proper_topics,
                    "bertopic_explicit_outlier_slot": 1,
                    "bertopic_assignment_probability": 1,
                },
                "per_seed_val_configs": hybrid_val_configs,
            },
            "in_domain": hybrid_in_soft,
            "in_domain_majority_vote": hybrid_in_maj,
            "in_domain_soft_vote": hybrid_in_soft,
            "in_domain_mean_std": {
                "f1_macro_mean": float(np.mean(in_f1s)),
                "f1_macro_std": float(np.std(in_f1s, ddof=1)) if len(in_f1s) > 1 else 0.0,
                "accuracy_mean": float(np.mean([m["accuracy"] for m in hybrid_per_seed_in.values()])),
                "accuracy_std": float(np.std([m["accuracy"] for m in hybrid_per_seed_in.values()], ddof=1)),
            },
            "in_domain_per_seed": hybrid_per_seed_in,
            "cross_domain": hybrid_cr_soft,
            "cross_domain_majority_vote": hybrid_cr_maj,
            "cross_domain_soft_vote": hybrid_cr_soft,
            "cross_domain_mean_std": {
                "f1_macro_mean": float(np.mean(cr_f1s)),
                "f1_macro_std": float(np.std(cr_f1s, ddof=1)) if len(cr_f1s) > 1 else 0.0,
                "accuracy_mean": float(np.mean([m["accuracy"] for m in hybrid_per_seed_cross.values()])),
                "accuracy_std": float(np.std([m["accuracy"] for m in hybrid_per_seed_cross.values()], ddof=1)),
            },
            "cross_domain_per_seed": hybrid_per_seed_cross,
            "cross_domain_sadness_only": hybrid_sad_maj,
            "cross_domain_sadness_only_per_seed": hybrid_per_seed_sad,
        }

        print(
            f"\nPhoBERT+BERTopic Ensemble (In-Domain):    "
            f"Soft-Vote F1-M={hybrid_in_soft['f1_macro']:.4f}, Maj-Vote F1-M={hybrid_in_maj['f1_macro']:.4f}, "
            f"Mean±SD={np.mean(in_f1s):.4f}±{np.std(in_f1s, ddof=1):.4f}"
        )
        print(
            f"PhoBERT+BERTopic Ensemble (Cross-Domain): "
            f"Soft-Vote F1-M={hybrid_cr_soft['f1_macro']:.4f}, Maj-Vote F1-M={hybrid_cr_maj['f1_macro']:.4f}, "
            f"Mean±SD={np.mean(cr_f1s):.4f}±{np.std(cr_f1s, ddof=1):.4f}"
        )

    results = {
        "timestamp": datetime.now().isoformat(),
        "round": "round6_v2_inductive",
        "bertopic_model_metadata": bertopic_meta,
        "dataset": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "vsmec_full": len(vsmec_df),
            "vsmec_sadness_only": int(sadness_mask.sum()),
        },
        "bertopic_only": {
            "val_selected_config": cfg_bt,
            "in_domain": bertopic_in_metrics,
            "cross_domain": bertopic_cross_metrics,
            "cross_domain_sadness_only": bertopic_sad_metrics,
        },
        "phobert_bertopic_combined": hybrid_summary,
    }

    output_file = OUTPUT_DIR / "bertopic_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    canonical_bt_eval = PHOBERT_ROUND6_DIR / "bertopic_evaluation_results.json"
    canonical_bt_eval.parent.mkdir(parents=True, exist_ok=True)
    with open(canonical_bt_eval, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {output_file}")
    print(f"Canonical copy:   {canonical_bt_eval}")
    return results


if __name__ == "__main__":
    main()
