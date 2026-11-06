"""Comprehensive evaluation of all models for Round 5.

Evaluates:
1. PhoBERT (3 seeds) - in-domain and cross-domain
2. BiLSTM - in-domain
3. TF-IDF + LogReg - in-domain and cross-domain
4. TF-IDF + LinearSVC - in-domain and cross-domain
5. Cross-domain evaluation on VSMEC dataset for all available models

Note: ``joblib.load`` reads scikit-learn pipelines serialized from our own
``retrain_all_models_round5.py`` — these are trusted in-project artifacts.

Usage:
    .venv/bin/python scripts/evaluate_all_models_round5.py
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / f"round5_final_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILE = DATA_DIR / "labeled" / "final_test.csv"
VSMEC_FILE = DATA_DIR.parent / "data_unified" / "cross_domain_test.csv"
# Use the retrained models (trained on the repaired dataset, 7,336 train rows).
PHOBERT_DIR = MODEL_DIR / "round5_retrained"
MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ── PhoBERT Dataset ───────────────────────────────────────────────────────────
# Tokenizer is set globally before evaluation functions are called
_tokenizer = None
_MAX_LEN = MAX_LEN

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    return _tokenizer

class DepressionDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer or _get_tokenizer()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=_MAX_LEN, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ── Evaluation functions ──────────────────────────────────────────────────────
def evaluate_phobert(model_dir: Path, texts, labels, split_name: str, tokenizer=None):
    """Evaluate a PhoBERT model."""
    print(f"\nEvaluating PhoBERT from {model_dir.name} on {split_name}...")

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    dataset = DepressionDataset(texts, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1].cpu().numpy())

    return {
        "accuracy": accuracy_score(labels, all_preds),
        "f1_macro": f1_score(labels, all_preds, average="macro"),
        "f1_weighted": f1_score(labels, all_preds, average="weighted"),
        "f1_depression": f1_score(labels, all_preds, average="binary", pos_label=1),
        "precision_macro": precision_score(labels, all_preds, average="macro"),
        "recall_macro": recall_score(labels, all_preds, average="macro"),
        "confusion_matrix": confusion_matrix(labels, all_preds).tolist(),
        "predictions": all_preds,
        "probabilities": all_probs
    }

def evaluate_sklearn(preds, probs, labels, model_name: str, split_name: str):
    """Evaluate sklearn model predictions."""
    print(f"\nEvaluating {model_name} on {split_name}...")
    preds = np.array(preds)
    labels = np.array(labels)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        "f1_depression": float(f1_score(labels, preds, average="binary", pos_label=1)),
        "precision_macro": float(precision_score(labels, preds, average="macro")),
        "recall_macro": float(recall_score(labels, preds, average="macro")),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }

def load_vsmec_data():
    """Load VSMEC dataset for cross-domain evaluation."""
    if not VSMEC_FILE.exists():
        print(f"VSMEC file not found: {VSMEC_FILE}")
        return None, None, None

    df = pd.read_csv(VSMEC_FILE)
    # VSMEC has 'label' column with 0=normal, 1=depression
    texts = df["text"].values if "text" in df.columns else df["comment_text"].values
    labels = df["label"].values
    print(f"VSMEC loaded: {len(df)} samples")
    return texts, labels, df

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ROUND 5 COMPREHENSIVE EVALUATION")
    print("=" * 60)

    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(TEST_FILE)
    test_texts = test_df["comment_text"].values
    test_labels = test_df["label"].values
    print(f"Test set: {len(test_df)} samples | Normal: {(test_labels==0).sum()} | Depression: {(test_labels==1).sum()}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    # Set global tokenizer for DepressionDataset
    global _tokenizer
    _tokenizer = tokenizer

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "test_set_size": len(test_df),
        "in_domain": {},
        "cross_domain": {}
    }

    # ── 1. PhoBERT (all seeds) ────────────────────────────────────────────────
    phobert_results = []
    phobert_per_seed_preds = {}
    for seed in [42, 123, 2024]:
        # Models trained by retrain_all_models_round5.py live under phobert_seed_*/best_model/
        model_dir = PHOBERT_DIR / f"phobert_seed_{seed}" / "best_model"
        if model_dir.exists():
            result = evaluate_phobert(model_dir, test_texts, test_labels, f"test_seed{seed}")
            result["seed"] = seed
            phobert_results.append(result)
            phobert_per_seed_preds[seed] = np.array(result["predictions"])

            # Per-seed summary
            print(f"  Seed {seed}: Acc={result['accuracy']:.4f}, F1={result['f1_macro']:.4f}")

    if phobert_results:
        # Average across seeds — proper majority vote on predictions
        seeds_sorted = sorted(phobert_per_seed_preds.keys())
        stacked = np.stack([phobert_per_seed_preds[s] for s in seeds_sorted], axis=0)
        avg_preds = (np.mean(stacked, axis=0) >= 0.5).astype(int)
        avg_metrics = {
            "accuracy": float(accuracy_score(test_labels, avg_preds)),
            "f1_macro": float(f1_score(test_labels, avg_preds, average="macro")),
            "f1_depression": float(f1_score(test_labels, avg_preds, average="binary", pos_label=1)),
            "precision": float(precision_score(test_labels, avg_preds, average="macro")),
            "recall": float(recall_score(test_labels, avg_preds, average="macro")),
            "seeds": seeds_sorted,
            "per_seed": {
                str(r["seed"]): {k: v for k, v in r.items() if k not in ["predictions", "probabilities", "confusion_matrix", "seed"]}
                for r in phobert_results
            },
        }
        all_results["in_domain"]["phobert_avg"] = avg_metrics
        print(f"\nPhoBERT Average (majority vote across {len(seeds_sorted)} seeds): "
              f"Acc={avg_metrics['accuracy']:.4f}, F1={avg_metrics['f1_macro']:.4f}")

    # ── 2. BiLSTM ─────────────────────────────────────────────────────────────
    bilstm_dir = MODEL_DIR / "bilstm"
    if bilstm_dir.exists():
        # Try to load BiLSTM predictions if model exists
        try:
            from yt_depression_crawler.modeling.bilstm.evaluate import evaluate_bilstm
            bilstm_result = evaluate_bilstm(bilstm_dir, test_texts, test_labels)
            all_results["in_domain"]["bilstm"] = bilstm_result
            print(f"BiLSTM: Acc={bilstm_result['accuracy']:.4f}, F1={bilstm_result['f1_macro']:.4f}")
        except Exception as e:
            print(f"BiLSTM evaluation skipped: {e}")

    # ── 3. TF-IDF + LogReg ────────────────────────────────────────────────────
    tfidf_path = PHOBERT_DIR / "tfidf_logreg_round5.joblib"
    if tfidf_path.exists():
        try:
            # Model is a Pipeline, not a dict
            pipeline = joblib.load(tfidf_path)

            # Extract predictions
            preds = pipeline.predict(test_texts)
            probs = pipeline.predict_proba(test_texts)[:, 1]

            logreg_result = {
                "accuracy": accuracy_score(test_labels, preds),
                "f1_macro": f1_score(test_labels, preds, average="macro"),
                "f1_depression": f1_score(test_labels, preds, average="binary", pos_label=1),
                "precision": precision_score(test_labels, preds, average="macro"),
                "recall": recall_score(test_labels, preds, average="macro"),
                "confusion_matrix": confusion_matrix(test_labels, preds).tolist(),
                "predictions": preds.tolist(),
                "probabilities": probs.tolist()
            }
            all_results["in_domain"]["tfidf_logreg"] = logreg_result
            print(f"TF-IDF + LogReg: Acc={logreg_result['accuracy']:.4f}, F1={logreg_result['f1_macro']:.4f}")
        except Exception as e:
            print(f"TF-IDF LogReg evaluation skipped: {e}")

    # ── 4. TF-IDF + LinearSVC ─────────────────────────────────────────────────
    tfidf_svc_path = PHOBERT_DIR / "tfidf_linearsvc_round5.joblib"
    if tfidf_svc_path.exists():
        try:
            # Model is a Pipeline
            pipeline = joblib.load(tfidf_svc_path)

            # Extract predictions
            preds = pipeline.predict(test_texts)

            svc_result = {
                "accuracy": accuracy_score(test_labels, preds),
                "f1_macro": f1_score(test_labels, preds, average="macro"),
                "f1_depression": f1_score(test_labels, preds, average="binary", pos_label=1),
                "precision": precision_score(test_labels, preds, average="macro"),
                "recall": recall_score(test_labels, preds, average="macro"),
                "confusion_matrix": confusion_matrix(test_labels, preds).tolist(),
                "predictions": preds.tolist(),
                "probabilities": None
            }
            all_results["in_domain"]["tfidf_svc"] = svc_result
            print(f"TF-IDF + LinearSVC: Acc={svc_result['accuracy']:.4f}, F1={svc_result['f1_macro']:.4f}")
        except Exception as e:
            print(f"TF-IDF SVC evaluation skipped: {e}")

    # ── 5. Cross-domain evaluation ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CROSS-DOMAIN EVALUATION (VSMEC)")
    print("=" * 60)

    vsmec_texts, vsmec_labels, vsmec_df = load_vsmec_data()
    if vsmec_texts is not None:
        # PhoBERT cross-domain — collect predictions per seed for majority vote
        phobert_cross_preds_per_seed = {}
        for seed in [42, 123, 2024]:
            model_dir = PHOBERT_DIR / f"phobert_seed_{seed}" / "best_model"
            if model_dir.exists():
                result = evaluate_phobert(model_dir, vsmec_texts, vsmec_labels, f"vsmec_seed{seed}")
                all_results["cross_domain"][f"phobert_seed{seed}"] = result
                phobert_cross_preds_per_seed[seed] = np.array(result["predictions"])
                print(f"  PhoBERT Seed {seed} on VSMEC: Acc={result['accuracy']:.4f}, F1={result['f1_macro']:.4f}")

        # PhoBERT cross-domain majority vote
        if phobert_cross_preds_per_seed:
            seeds_sorted = sorted(phobert_cross_preds_per_seed.keys())
            stacked = np.stack([phobert_cross_preds_per_seed[s] for s in seeds_sorted], axis=0)
            avg_preds = (np.mean(stacked, axis=0) >= 0.5).astype(int)
            avg_cross = {
                "accuracy": float(accuracy_score(vsmec_labels, avg_preds)),
                "f1_macro": float(f1_score(vsmec_labels, avg_preds, average="macro")),
                "f1_depression": float(f1_score(vsmec_labels, avg_preds, average="binary", pos_label=1)),
                "precision": float(precision_score(vsmec_labels, avg_preds, average="macro")),
                "recall": float(recall_score(vsmec_labels, avg_preds, average="macro")),
                "confusion_matrix": confusion_matrix(vsmec_labels, avg_preds).tolist(),
                "seeds": seeds_sorted,
            }
            all_results["cross_domain"]["phobert_avg"] = avg_cross
            print(f"  PhoBERT avg vote on VSMEC: Acc={avg_cross['accuracy']:.4f}, F1={avg_cross['f1_macro']:.4f}")

        # TF-IDF cross-domain — both LogReg and LinearSVC
        for tfidf_name, tfidf_model_path, has_proba in [
            ("tfidf_logreg", PHOBERT_DIR / "tfidf_logreg_round5.joblib", True),
            ("tfidf_svc", PHOBERT_DIR / "tfidf_linearsvc_round5.joblib", False),
        ]:
            if tfidf_model_path.exists():
                try:
                    pipeline = joblib.load(tfidf_model_path)
                    preds = pipeline.predict(vsmec_texts)
                    probs = (
                        pipeline.predict_proba(vsmec_texts)[:, 1].tolist()
                        if has_proba
                        else None
                    )
                    metrics = {
                        "accuracy": float(accuracy_score(vsmec_labels, preds)),
                        "f1_macro": float(f1_score(vsmec_labels, preds, average="macro")),
                        "f1_depression": float(f1_score(vsmec_labels, preds, average="binary", pos_label=1)),
                        "precision": float(precision_score(vsmec_labels, preds, average="macro")),
                        "recall": float(recall_score(vsmec_labels, preds, average="macro")),
                        "confusion_matrix": confusion_matrix(vsmec_labels, preds).tolist(),
                        "predictions": preds.tolist(),
                        "probabilities": probs,
                    }
                    all_results["cross_domain"][tfidf_name] = metrics
                    print(f"  {tfidf_name} on VSMEC: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
                except Exception as e:
                    print(f"{tfidf_name} cross-domain evaluation skipped: {e}")

    # ── Save results ───────────────────────────────────────────────────────────
    results_file = OUTPUT_DIR / "evaluation_results.json"
    with open(results_file, "w") as f:
        # Remove raw predictions/probs for JSON serialization
        def clean_results(obj):
            if isinstance(obj, dict):
                return {k: clean_results(v) for k, v in obj.items() if k not in ["predictions", "probabilities"]}
            return obj
        json.dump(clean_results(all_results), f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nResults saved to: {results_file}")

    # Print summary table
    print("\nIn-Domain Results:")
    print("-" * 50)
    for model_name, results in all_results["in_domain"].items():
        if isinstance(results, dict) and "accuracy" in results:
            print(f"{model_name:20s}: Acc={results['accuracy']:.4f}, F1={results['f1_macro']:.4f}, F1(dep)={results.get('f1_depression', 0):.4f}")

    print("\nCross-Domain Results (VSMEC):")
    print("-" * 50)
    for model_name, results in all_results["cross_domain"].items():
        if isinstance(results, dict) and "accuracy" in results:
            print(f"{model_name:20s}: Acc={results['accuracy']:.4f}, F1={results['f1_macro']:.4f}")

    return all_results

if __name__ == "__main__":
    main()
