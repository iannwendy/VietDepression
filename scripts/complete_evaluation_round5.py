"""Complete evaluation for Round 5 with all 6 metrics + statistical significance.

Computes:
1. All 6 metrics: Accuracy, Precision-macro, Recall-macro, F1-macro, F1-weighted, F1-depression
2. McNemar's test for statistical significance between model pairs
3. Error analysis for Appendix C

Usage:
    .venv/bin/python scripts/complete_evaluation_round5.py
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, cohen_kappa_score
)
from scipy.stats import chi2
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / f"round5_complete_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILE = DATA_DIR / "labeled" / "final_test.csv"
VSMEC_FILE = DATA_DIR.parent / "data_unified" / "cross_domain_test.csv"
PHOBERT_DIR = MODEL_DIR / "round5_retrained"
MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ── PhoBERT Dataset ───────────────────────────────────────────────────────────
_tokenizer = None

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
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ── Metric computation ────────────────────────────────────────────────────────
def compute_all_metrics(y_true, y_pred):
    """Compute all 6 metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_depression": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

# ── McNemar's Test ───────────────────────────────────────────────────────────
def mcnemar_test(y_true, y_pred1, y_pred2):
    """McNemar's test for two classifiers."""
    # Contingency table
    b = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))  # model1 wrong, model2 right
    c = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))  # model1 right, model2 wrong

    n = b + c
    if n == 0:
        return 1.0, "N/A"

    # McNemar's chi-squared
    chi2_stat = abs(b - c) ** 2 / n if n > 0 else 0
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    return p_value, significance

# ── PhoBERT Prediction ─────────────────────────────────────────────────────────
def get_phobert_predictions(model_dir, texts):
    """Get predictions from a PhoBERT model."""
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    dataset = DepressionDataset(texts)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)

# ── Error Analysis ────────────────────────────────────────────────────────────
def analyze_errors(texts, y_true, y_pred, model_name, max_examples=10):
    """Analyze misclassified samples."""
    errors = []
    for i, (text, true, pred) in enumerate(zip(texts, y_true, y_pred)):
        if true != pred:
            errors.append({
                "index": i,
                "text": text[:200] if len(text) > 200 else text,
                "true_label": int(true),
                "pred_label": int(pred),
                "error_type": "FP" if (true == 0 and pred == 1) else "FN"
            })

    return {
        "total_errors": len(errors),
        "false_positives": sum(1 for e in errors if e["error_type"] == "FP"),
        "false_negatives": sum(1 for e in errors if e["error_type"] == "FN"),
        "sample_fp": [e for e in errors if e["error_type"] == "FP"][:max_examples],
        "sample_fn": [e for e in errors if e["error_type"] == "FN"][:max_examples],
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ROUND 5 COMPLETE EVALUATION")
    print("All 6 metrics + Statistical Significance + Error Analysis")
    print("=" * 70)

    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(TEST_FILE)
    test_texts = test_df["comment_text"].values
    test_labels = test_df["label"].values

    print(f"Test set: {len(test_df)} samples")
    print(f"  Normal (0): {(test_labels==0).sum()} ({(test_labels==0).mean()*100:.1f}%)")
    print(f"  Depression (1): {(test_labels==1).sum()} ({(test_labels==1).mean()*100:.1f}%)")

    # Initialize tokenizer
    _get_tokenizer()

    # ── Get predictions from all PhoBERT seeds ───────────────────────────────
    print("\n" + "-" * 50)
    print("Getting PhoBERT predictions...")
    phobert_predictions = {}
    for seed in [42, 123, 2024]:
        model_dir = PHOBERT_DIR / f"phobert_seed_{seed}" / "best_model"
        if model_dir.exists():
            print(f"  Seed {seed}...")
            phobert_predictions[seed] = get_phobert_predictions(model_dir, test_texts)

    # Average prediction (majority vote)
    phobert_avg_pred = (np.mean([phobert_predictions[s] for s in phobert_predictions], axis=0) >= 0.5).astype(int)

    # ── Get TF-IDF predictions ───────────────────────────────────────────────
    tfidf_svc_path = PHOBERT_DIR / "tfidf_linearsvc_round5.joblib"
    if tfidf_svc_path.exists():
        pipeline = joblib.load(tfidf_svc_path)
        tfidf_pred = pipeline.predict(test_texts)

    tfidf_logreg_path = PHOBERT_DIR / "tfidf_logreg_round5.joblib"
    if tfidf_logreg_path.exists():
        pipeline = joblib.load(tfidf_logreg_path)
        tfidf_logreg_pred = pipeline.predict(test_texts)

    # ── Compute all metrics ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("IN-DOMAIN EVALUATION (Test Set, n=912)")
    print("=" * 70)
    print(f"{'Model':<25} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1-M':>6} | {'F1-W':>6} | {'F1-D':>6}")
    print("-" * 70)

    all_results = {"in_domain": {}, "cross_domain": {}, "statistical_tests": {}, "error_analysis": {}}

    # PhoBERT per-seed
    for seed, preds in phobert_predictions.items():
        metrics = compute_all_metrics(test_labels, preds)
        all_results["in_domain"][f"phobert_seed{seed}"] = metrics
        print(f"PhoBERT (seed {seed})      | {metrics['accuracy']:.4f} | {metrics['precision_macro']:.4f} | "
              f"{metrics['recall_macro']:.4f} | {metrics['f1_macro']:.4f} | {metrics['f1_weighted']:.4f} | {metrics['f1_depression']:.4f}")

    # PhoBERT average (majority vote)
    metrics_avg = compute_all_metrics(test_labels, phobert_avg_pred)
    all_results["in_domain"]["phobert_avg"] = metrics_avg
    print(f"PhoBERT (avg vote)        | {metrics_avg['accuracy']:.4f} | {metrics_avg['precision_macro']:.4f} | "
          f"{metrics_avg['recall_macro']:.4f} | {metrics_avg['f1_macro']:.4f} | {metrics_avg['f1_weighted']:.4f} | {metrics_avg['f1_depression']:.4f}")

    # TF-IDF + SVC
    metrics_svc = compute_all_metrics(test_labels, tfidf_pred)
    all_results["in_domain"]["tfidf_svc"] = metrics_svc
    print(f"TF-IDF + LinearSVC        | {metrics_svc['accuracy']:.4f} | {metrics_svc['precision_macro']:.4f} | "
          f"{metrics_svc['recall_macro']:.4f} | {metrics_svc['f1_macro']:.4f} | {metrics_svc['f1_weighted']:.4f} | {metrics_svc['f1_depression']:.4f}")

    # TF-IDF + LogReg
    metrics_logreg = compute_all_metrics(test_labels, tfidf_logreg_pred)
    all_results["in_domain"]["tfidf_logreg"] = metrics_logreg
    print(f"TF-IDF + LogReg           | {metrics_logreg['accuracy']:.4f} | {metrics_logreg['precision_macro']:.4f} | "
          f"{metrics_logreg['recall_macro']:.4f} | {metrics_logreg['f1_macro']:.4f} | {metrics_logreg['f1_weighted']:.4f} | {metrics_logreg['f1_depression']:.4f}")

    # ── Statistical Significance (McNemar's Test) ─────────────────────────────
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (McNemar's Test)")
    print("=" * 70)

    model_pairs = [
        ("phobert_avg", phobert_avg_pred, "tfidf_svc", tfidf_pred),
        ("phobert_avg", phobert_avg_pred, "tfidf_logreg", tfidf_logreg_pred),
        ("tfidf_svc", tfidf_pred, "tfidf_logreg", tfidf_logreg_pred),
    ]

    print(f"{'Comparison':<40} | {'p-value':>10} | {'Sig.':<6}")
    print("-" * 60)

    for name1, pred1, name2, pred2 in model_pairs:
        p_val, sig = mcnemar_test(test_labels, pred1, pred2)
        all_results["statistical_tests"][f"{name1}_vs_{name2}"] = {"p_value": float(p_val), "significance": sig}
        print(f"{name1} vs {name2:<20} | {p_val:>10.4f} | {sig:<6}")

    # ── Error Analysis ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    # PhoBERT errors
    phobert_errors = analyze_errors(test_texts, test_labels, phobert_avg_pred, "PhoBERT")
    all_results["error_analysis"]["phobert"] = phobert_errors
    print(f"\nPhoBERT Errors: {phobert_errors['total_errors']} total")
    print(f"  False Positives: {phobert_errors['false_positives']}")
    print(f"  False Negatives: {phobert_errors['false_negatives']}")

    # TF-IDF + SVC errors
    svc_errors = analyze_errors(test_texts, test_labels, tfidf_pred, "TF-IDF+SVC")
    all_results["error_analysis"]["tfidf_svc"] = svc_errors
    print(f"\nTF-IDF + LinearSVC Errors: {svc_errors['total_errors']} total")
    print(f"  False Positives: {svc_errors['false_positives']}")
    print(f"  False Negatives: {svc_errors['false_negatives']}")

    # ── Cross-domain Evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN EVALUATION (VSMEC, n=3,084)")
    print("=" * 70)

    if VSMEC_FILE.exists():
        vsmec_df = pd.read_csv(VSMEC_FILE)
        vsmec_texts = vsmec_df["text"].values if "text" in vsmec_df.columns else vsmec_df["comment_text"].values
        vsmec_labels = vsmec_df["label"].values

        print(f"VSMEC: {len(vsmec_df)} samples")
        print(f"  Normal (0): {(vsmec_labels==0).sum()} | Depression (1): {(vsmec_labels==1).sum()}")
        print()
        print(f"{'Model':<25} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1-M':>6} | {'F1-W':>6} | {'F1-D':>6}")
        print("-" * 70)

        for seed, _ in phobert_predictions.items():
            model_dir = PHOBERT_DIR / f"phobert_seed_{seed}" / "best_model"
            vsmec_pred = get_phobert_predictions(model_dir, vsmec_texts)
            metrics = compute_all_metrics(vsmec_labels, vsmec_pred)
            all_results["cross_domain"][f"phobert_seed{seed}"] = metrics
            print(f"PhoBERT (seed {seed})      | {metrics['accuracy']:.4f} | {metrics['precision_macro']:.4f} | "
                  f"{metrics['recall_macro']:.4f} | {metrics['f1_macro']:.4f} | {metrics['f1_weighted']:.4f} | {metrics['f1_depression']:.4f}")

    # ── Save Results ──────────────────────────────────────────────────────────
    results_file = OUTPUT_DIR / "complete_evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Save error examples for Appendix
    errors_file = OUTPUT_DIR / "error_examples.json"
    with open(errors_file, "w", encoding="utf-8") as f:
        json.dump({
            "phobert_fp": phobert_errors["sample_fp"],
            "phobert_fn": phobert_errors["sample_fn"],
            "tfidf_svc_fp": svc_errors["sample_fp"],
            "tfidf_svc_fn": svc_errors["sample_fn"],
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(f"  Full results: {results_file}")
    print(f"  Error examples: {errors_file}")

    # Summary table for paper
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 70)
    print(f"{'Model':<20} | {'Acc':>7} | {'Prec-M':>7} | {'Rec-M':>7} | {'F1-M':>7} | {'F1-W':>7} | {'F1-D':>7}")
    print("-" * 75)
    print(f"{'PhoBERT (avg)':<20} | {metrics_avg['accuracy']:>7.4f} | {metrics_avg['precision_macro']:>7.4f} | "
          f"{metrics_avg['recall_macro']:>7.4f} | {metrics_avg['f1_macro']:>7.4f} | {metrics_avg['f1_weighted']:>7.4f} | {metrics_avg['f1_depression']:>7.4f}")
    print(f"{'TF-IDF + LinearSVC':<20} | {metrics_svc['accuracy']:>7.4f} | {metrics_svc['precision_macro']:>7.4f} | "
          f"{metrics_svc['recall_macro']:>7.4f} | {metrics_svc['f1_macro']:>7.4f} | {metrics_svc['f1_weighted']:>7.4f} | {metrics_svc['f1_depression']:>7.4f}")

    return all_results

if __name__ == "__main__":
    main()
