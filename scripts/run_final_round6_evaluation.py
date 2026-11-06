"""Canonical End-to-End Round 6 v2 Scientific Evaluation & Statistical Testing Suite.

Addresses Reviewer 1 (Comments 1, 2, 4, 5, 6, 7) and Reviewer 2 (Major Comments 1, 2, 3, 4):
1. Evaluates all models under a single unified protocol on:
   - In-Domain Test (`final_test.csv`, n=1,371: 1,100 Normal, 271 Depression)
   - Cross-Domain UIT-VSMEC Full (`cross_domain_test.csv`, n=3,084: 1,542 Enjoyment, 1,149 Sadness, 393 Fear)
   - Cross-Domain UIT-VSMEC Sadness-Only (n=2,691: 1,542 Enjoyment vs. 1,149 Sadness, excluding Fear)
2. Includes Trivial All-Negative Baseline and Lexicon Weak-Labeler Baseline
   (evaluated on all 1,371 test examples with abstentions counted, plus covered-only).
3. Explicitly reports all three multi-seed aggregations for PhoBERT, BiLSTM (unweighted),
   BiLSTM (class-weighted), and PhoBERT+BERTopic:
   - `mean_std`: Mean +/- SD across seeds (42, 123, 2024)
   - `majority_vote`: Hard majority vote over binary predictions (with 2x2 confusion matrix)
   - `soft_vote`: Soft probability averaging over 3 seeds (with 2x2 confusion matrix)
4. Computes threshold-independent metrics (ROC-AUC, PR-AUC) for all models and
   performs Threshold Calibration & Prior-Shift Analysis on UIT-VSMEC.
5. Computes non-parametric Bootstrap 95% Confidence Intervals (B=2,000) and
   pairwise statistical significance tests (Exact McNemar's test + Paired
   Bootstrap Delta-F1-macro test).

Usage:
    PYTHONPATH="$PWD" .venv/bin/python scripts/run_final_round6_evaluation.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
if (HF_CACHE / "models--vinai--phobert-base").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.vietdepression_lexicon import label_comment_detailed

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
UNIFIED_DIR = PROJECT_DIR / "data_unified"

TRAIN_FILE = DATA_DIR / "labeled" / "final_train.csv"
VAL_FILE = DATA_DIR / "labeled" / "final_val.csv"
TEST_FILE = DATA_DIR / "labeled" / "final_test.csv"
VSMEC_FILE = UNIFIED_DIR / "cross_domain_test.csv"

RETRAINED_DIR = (
    MODEL_DIR / "round6_v2_retrained"
    if (MODEL_DIR / "round6_v2_retrained").exists()
    else MODEL_DIR / "round6_retrained"
)
PHOBERT_DIR = RETRAINED_DIR

MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 128
SEEDS = [42, 123, 2024]
N_BOOTSTRAP = 2000

OUTPUT_DIR = RESULTS_DIR / f"round6_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")


# ── Helpers & Statistical Tests ──────────────────────────────────────────────
def compute_metrics(
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
        "precision_macro": float(
            precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "precision_depression": float(
            precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
        ),
        "recall_depression": float(
            recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
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
        "predicted_positives": int(tp + fp),
        "predicted_negatives": int(tn + fn),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    if y_prob is not None and len(np.unique(y_true_arr)) == 2:
        probs_arr = np.asarray(y_prob, dtype=float)
        out["roc_auc"] = float(roc_auc_score(y_true_arr, probs_arr))
        out["pr_auc"] = float(average_precision_score(y_true_arr, probs_arr))
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
) -> dict:
    """Compute non-parametric percentile bootstrap SE and 95% CI for F1-macro, F1-dep, Acc."""
    rng = np.random.default_rng(seed)
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    n = len(y_t)
    idx_matrix = rng.integers(0, n, size=(n_boot, n))

    # Vectorized 2x2 confusion matrix counts across all n_boot resamples
    t_boot = y_t[idx_matrix]
    p_boot = y_p[idx_matrix]

    tp = np.sum((t_boot == 1) & (p_boot == 1), axis=1).astype(float)
    fp = np.sum((t_boot == 0) & (p_boot == 1), axis=1).astype(float)
    fn = np.sum((t_boot == 1) & (p_boot == 0), axis=1).astype(float)
    tn = np.sum((t_boot == 0) & (p_boot == 0), axis=1).astype(float)

    f1_pos = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    f1_neg = np.where((2 * tn + fn + fp) > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    f1_mac = 0.5 * (f1_pos + f1_neg)
    acc = (tp + tn) / float(n)

    def _stat(arr: np.ndarray) -> dict:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std_error": round(float(np.std(arr, ddof=1)), 4),
            "ci95_lower": round(float(lo), 4),
            "ci95_upper": round(float(hi), 4),
        }

    return {
        "n_bootstrap": n_boot,
        "f1_macro": _stat(f1_mac),
        "f1_depression": _stat(f1_pos),
        "accuracy": _stat(acc),
    }


def paired_significance_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    name_a: str,
    name_b: str,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
) -> dict:
    """Compute Exact McNemar's test and Paired Bootstrap test for (Model A - Model B)."""
    y_t = np.asarray(y_true, dtype=int)
    pa = np.asarray(y_pred_a, dtype=int)
    pb = np.asarray(y_pred_b, dtype=int)

    correct_a = pa == y_t
    correct_b = pb == y_t

    # Discordant counts: b = A right & B wrong; c = A wrong & B right
    b = int(np.sum(correct_a & (~correct_b)))
    c = int(np.sum((~correct_a) & correct_b))
    n_disc = b + c
    if n_disc > 0:
        mcnemar_chi2_cc = float((abs(b - c) - 1.0) ** 2 / n_disc)
        mcnemar_exact_p = float(binomtest(b, n_disc, 0.5, alternative="two-sided").pvalue)
    else:
        mcnemar_chi2_cc = 0.0
        mcnemar_exact_p = 1.0

    # Paired bootstrap on F1-macro difference (F1_A - F1_B)
    rng = np.random.default_rng(seed)
    n = len(y_t)
    idx_matrix = rng.integers(0, n, size=(n_boot, n))
    t_b = y_t[idx_matrix]
    pa_b = pa[idx_matrix]
    pb_b = pb[idx_matrix]

    def _f1_mac_vec(t_m, p_m):
        tp = np.sum((t_m == 1) & (p_m == 1), axis=1).astype(float)
        fp = np.sum((t_m == 0) & (p_m == 1), axis=1).astype(float)
        fn = np.sum((t_m == 1) & (p_m == 0), axis=1).astype(float)
        tn = np.sum((t_m == 0) & (p_m == 0), axis=1).astype(float)
        f1_1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_0 = np.where((2 * tn + fn + fp) > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
        return 0.5 * (f1_1 + f1_0)

    f1_a_boot = _f1_mac_vec(t_b, pa_b)
    f1_b_boot = _f1_mac_vec(t_b, pb_b)
    diff_boot = f1_a_boot - f1_b_boot

    obs_f1_a = float(f1_score(y_t, pa, average="macro", zero_division=0))
    obs_f1_b = float(f1_score(y_t, pb, average="macro", zero_division=0))
    obs_diff = obs_f1_a - obs_f1_b
    ci_lo, ci_hi = np.percentile(diff_boot, [2.5, 97.5])

    # Two-sided paired bootstrap p-value
    p_le_0 = float(np.mean(diff_boot <= 0.0))
    p_ge_0 = float(np.mean(diff_boot >= 0.0))
    boot_p_two_sided = min(1.0, 2.0 * min(p_le_0, p_ge_0))

    return {
        "model_a": name_a,
        "model_b": name_b,
        "f1_macro_a": round(obs_f1_a, 4),
        "f1_macro_b": round(obs_f1_b, 4),
        "delta_f1_macro_a_minus_b": round(obs_diff, 4),
        "paired_bootstrap": {
            "delta_mean": round(float(np.mean(diff_boot)), 4),
            "delta_se": round(float(np.std(diff_boot, ddof=1)), 4),
            "ci95_lower": round(float(ci_lo), 4),
            "ci95_upper": round(float(ci_hi), 4),
            "p_value_two_sided": round(boot_p_two_sided, 4),
            "statistically_significant_at_0_05": bool(boot_p_two_sided < 0.05),
        },
        "mcnemar_test": {
            "a_correct_b_wrong": b,
            "a_wrong_b_correct": c,
            "discordant_total": n_disc,
            "chi2_continuity_corrected": round(mcnemar_chi2_cc, 4),
            "exact_binomial_p_value": round(mcnemar_exact_p, 4),
            "statistically_significant_at_0_05": bool(mcnemar_exact_p < 0.05),
        },
    }


def summarize_seed_metrics(per_seed_metrics: list[dict]) -> dict:
    keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "f1_depression",
        "f1_normal",
        "roc_auc",
        "pr_auc",
    ]
    summary: dict = {"n_seeds": len(per_seed_metrics), "seeds": SEEDS}
    for k in keys:
        vals = [m[k] for m in per_seed_metrics if k in m and m[k] is not None]
        if vals:
            summary[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            summary[f"{k}_std"] = (
                round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0
            )
    return summary


# ── Tokenizer & Datasets ─────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)


class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def predict_phobert(model_dir: Path, texts):
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    ds = TextDataset(texts)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    preds, probs = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs_pos = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds.extend((probs_pos >= 0.5).astype(int).tolist())
            probs.extend(probs_pos.tolist())

    del model
    return np.array(preds, dtype=int), np.array(probs, dtype=float)


# ── BiLSTM ───────────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, texts):
        freq = {}
        for text in texts:
            for word in str(text).split():
                freq[word] = freq.get(word, 0) + 1
        idx = 2
        for word, count in freq.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def text_to_sequence(self, text):
        seq = [self.stoi.get(w, 1) for w in str(text).split()]
        return seq if seq else [1]


class BiLSTMClassifier(torch.nn.Module):
    def __init__(
        self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.dropout(hidden)
        return self.fc(output)


class BiLSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = self.vocab.text_to_sequence(str(self.texts[idx]))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


def predict_bilstm(model_path: Path, eval_texts):
    eval_device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=eval_device, weights_only=False)
    vocab = checkpoint["vocab"]

    model = BiLSTMClassifier(len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(eval_device)
    model.eval()

    ds = BiLSTMDataset(eval_texts, [0] * len(eval_texts), vocab)

    def collate(batch):
        seqs, _ = zip(*batch)
        return pad_sequence(seqs, batch_first=True, padding_value=0)

    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)

    preds, probs = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch.to(eval_device))
            probs_pos = torch.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            probs.extend(probs_pos.tolist())

    del model
    return np.array(preds, dtype=int), np.array(probs, dtype=float)


def predict_tfidf(model_path: Path, texts):
    pipeline = joblib.load(model_path)
    preds = pipeline.predict(texts)
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(texts)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        probs = expit(pipeline.decision_function(texts))
    else:
        probs = preds.astype(float)
    return np.array(preds, dtype=int), np.array(probs, dtype=float)


def threshold_calibration_analysis(
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    test_labels: np.ndarray,
    test_probs: np.ndarray,
    vsmec_labels: np.ndarray,
    vsmec_probs: np.ndarray,
    vsmec_sad_labels: np.ndarray,
    vsmec_sad_probs: np.ndarray,
    train_pos_prior: float = 1262.0 / 6392.0,
) -> dict:
    """Analyze threshold shift between YouTube (19.74% pos) and UIT-VSMEC (50.0% pos)."""
    # 1. Find validation-optimal threshold on final_val.csv
    best_val_thr = 0.50
    best_val_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        f1 = float(f1_score(val_labels, (val_probs >= thr).astype(int), average="macro", zero_division=0))
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_val_thr = float(round(thr, 3))

    # 2. Prior-shift Bayes threshold (target prior = 0.50, source prior = 0.1974 -> thr = 0.1974)
    prior_thr = round(float(train_pos_prior), 4)

    sweep = {}
    for thr in [0.05, 0.10, 0.15, prior_thr, 0.25, 0.30, 0.40, 0.50, best_val_thr]:
        key = f"thr_{thr:.4f}"
        sweep[key] = {
            "threshold": thr,
            "in_domain_test": compute_metrics(test_labels, (test_probs >= thr).astype(int), test_probs),
            "vsmec_full": compute_metrics(vsmec_labels, (vsmec_probs >= thr).astype(int), vsmec_probs),
            "vsmec_sadness_only": compute_metrics(
                vsmec_sad_labels, (vsmec_sad_probs >= thr).astype(int), vsmec_sad_probs
            ),
        }

    return {
        "default_threshold_0_50": sweep["thr_0.5000"],
        "val_optimal_threshold": {
            "threshold": best_val_thr,
            "val_f1_macro": round(best_val_f1, 4),
            **sweep[f"thr_{best_val_thr:.4f}"],
        },
        "prior_shift_bayes_threshold": {
            "threshold": prior_thr,
            "note": "Accounts for label prior shift from YouTube train (19.74% pos) to balanced UIT-VSMEC (50.0% pos)",
            **sweep[f"thr_{prior_thr:.4f}"],
        },
        "threshold_sweep": sweep,
    }


def main():
    print("=" * 75)
    print("ROUND 6 v2 FINAL SCIENTIFIC EVALUATION & STATISTICAL TESTING")
    print("=" * 75)

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    vsmec_df = pd.read_csv(VSMEC_FILE)

    val_texts = val_df["comment_text"].fillna("").astype(str).values
    val_labels = val_df["label"].astype(int).values
    test_texts = test_df["comment_text"].fillna("").astype(str).values
    test_labels = test_df["label"].astype(int).values
    vsmec_texts = (
        vsmec_df["comment_text"].fillna("").astype(str).values
        if "comment_text" in vsmec_df.columns
        else vsmec_df["text"].fillna("").astype(str).values
    )
    vsmec_labels = vsmec_df["label"].astype(int).values

    sadness_mask = vsmec_df["original_label"].isin(["Sadness", "Enjoyment"]).to_numpy()
    vsmec_sad_labels = vsmec_labels[sadness_mask]

    print(
        f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,} | "
        f"VSMEC Full: {len(vsmec_df):,} | VSMEC Sadness-Only: {len(vsmec_sad_labels):,}"
    )

    all_results: dict = {
        "timestamp": datetime.now().isoformat(),
        "protocol_version": "round6_v2_scientific_eval",
        "model_directory": str(PHOBERT_DIR),
        "dataset_info": {
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "vsmec_size": int(len(vsmec_df)),
            "vsmec_sadness_only_size": int(len(vsmec_sad_labels)),
            "train_label_dist": {
                str(int(k)): int(v) for k, v in train_df["label"].value_counts().sort_index().items()
            },
            "val_label_dist": {
                str(int(k)): int(v) for k, v in val_df["label"].value_counts().sort_index().items()
            },
            "test_label_dist": {
                str(int(k)): int(v) for k, v in test_df["label"].value_counts().sort_index().items()
            },
            "vsmec_label_dist": {
                str(int(k)): int(v) for k, v in vsmec_df["label"].value_counts().sort_index().items()
            },
        },
        "in_domain": {},
        "cross_domain": {},
        "cross_domain_sadness_only": {},
        "bootstrap_95_ci_in_domain": {},
        "bootstrap_95_ci_cross_domain": {},
        "statistical_significance_in_domain": [],
        "statistical_significance_cross_domain": [],
        "threshold_calibration_analysis": {},
    }

    in_domain_preds_store: dict[str, np.ndarray] = {}
    cross_domain_preds_store: dict[str, np.ndarray] = {}

    # ── 0. Reference Baselines (All-Negative & Weak-Labeler on n=1,371) ──────
    neg_test_preds = np.zeros_like(test_labels)
    neg_vsmec_preds = np.zeros_like(vsmec_labels)
    all_results["in_domain"]["all_negative_baseline"] = compute_metrics(
        test_labels, neg_test_preds
    )
    all_results["cross_domain"]["all_negative_baseline"] = compute_metrics(
        vsmec_labels, neg_vsmec_preds
    )
    all_results["cross_domain_sadness_only"]["all_negative_baseline"] = compute_metrics(
        vsmec_sad_labels, neg_vsmec_preds[sadness_mask]
    )
    in_domain_preds_store["all_negative_baseline"] = neg_test_preds
    cross_domain_preds_store["all_negative_baseline"] = neg_vsmec_preds

    # Weak-Labeler evaluated on all 1,371 test samples (abstentions counted as Normal, score >= 5)
    wl_test_res = [label_comment_detailed(t) for t in test_texts]
    wl_test_scores = np.array([r.depression_score for r in wl_test_res], dtype=float)
    wl_test_preds = (wl_test_scores >= 5).astype(int)

    wl_vsmec_res = [label_comment_detailed(t) for t in vsmec_texts]
    wl_vsmec_scores = np.array([r.depression_score for r in wl_vsmec_res], dtype=float)
    wl_vsmec_preds = (wl_vsmec_scores >= 5).astype(int)

    all_results["in_domain"]["weak_labeler_lexicon"] = compute_metrics(
        test_labels, wl_test_preds, wl_test_scores
    )
    all_results["cross_domain"]["weak_labeler_lexicon"] = compute_metrics(
        vsmec_labels, wl_vsmec_preds, wl_vsmec_scores
    )
    all_results["cross_domain_sadness_only"]["weak_labeler_lexicon"] = compute_metrics(
        vsmec_sad_labels, wl_vsmec_preds[sadness_mask], wl_vsmec_scores[sadness_mask]
    )
    all_results["bootstrap_95_ci_in_domain"]["weak_labeler_lexicon"] = bootstrap_ci(
        test_labels, wl_test_preds
    )
    in_domain_preds_store["weak_labeler_lexicon"] = wl_test_preds
    cross_domain_preds_store["weak_labeler_lexicon"] = wl_vsmec_preds

    # ── 1. PhoBERT (3 seeds) ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("1. PHOBERT (3 seeds: 42, 123, 2024)")
    print("=" * 75)

    phobert_preds_test, phobert_probs_test = {}, {}
    phobert_probs_val = {}
    phobert_preds_vsmec, phobert_probs_vsmec = {}, {}
    pho_in_list, pho_cr_list, pho_sad_list = [], [], []

    for seed in SEEDS:
        model_dir = PHOBERT_DIR / f"phobert_seed_{seed}" / "best_model"
        if not model_dir.exists():
            print(f"  Seed {seed}: model not found at {model_dir}, skipping")
            continue

        print(f"  Evaluating PhoBERT seed {seed}...")
        _, probs_val = predict_phobert(model_dir, val_texts)
        preds_test, probs_test = predict_phobert(model_dir, test_texts)
        preds_vsmec, probs_vsmec = predict_phobert(model_dir, vsmec_texts)

        metrics_test = compute_metrics(test_labels, preds_test, probs_test)
        metrics_vsmec = compute_metrics(vsmec_labels, preds_vsmec, probs_vsmec)
        metrics_sad = compute_metrics(
            vsmec_sad_labels, preds_vsmec[sadness_mask], probs_vsmec[sadness_mask]
        )

        all_results["in_domain"][f"phobert_seed{seed}"] = metrics_test
        all_results["cross_domain"][f"phobert_seed{seed}"] = metrics_vsmec
        all_results["cross_domain_sadness_only"][f"phobert_seed{seed}"] = metrics_sad

        phobert_probs_val[seed] = probs_val
        phobert_preds_test[seed] = preds_test
        phobert_probs_test[seed] = probs_test
        phobert_preds_vsmec[seed] = preds_vsmec
        phobert_probs_vsmec[seed] = probs_vsmec

        pho_in_list.append(metrics_test)
        pho_cr_list.append(metrics_vsmec)
        pho_sad_list.append(metrics_sad)

        print(
            f"    In-domain:    Acc={metrics_test['accuracy']:.4f}, F1-M={metrics_test['f1_macro']:.4f}, ROC-AUC={metrics_test['roc_auc']:.4f}"
        )
        print(
            f"    Cross-domain: Acc={metrics_vsmec['accuracy']:.4f}, F1-M={metrics_vsmec['f1_macro']:.4f}, ROC-AUC={metrics_vsmec['roc_auc']:.4f}"
        )

    if phobert_preds_test:
        seeds = sorted(phobert_preds_test.keys())
        maj_test = (
            np.mean(np.stack([phobert_preds_test[s] for s in seeds], axis=0), axis=0) >= 0.5
        ).astype(int)
        soft_prob_test = np.mean(
            np.stack([phobert_probs_test[s] for s in seeds], axis=0), axis=0
        )
        soft_test = (soft_prob_test >= 0.5).astype(int)
        soft_prob_val = np.mean(
            np.stack([phobert_probs_val[s] for s in seeds], axis=0), axis=0
        )

        maj_vsmec = (
            np.mean(np.stack([phobert_preds_vsmec[s] for s in seeds], axis=0), axis=0) >= 0.5
        ).astype(int)
        soft_prob_vsmec = np.mean(
            np.stack([phobert_probs_vsmec[s] for s in seeds], axis=0), axis=0
        )
        soft_vsmec = (soft_prob_vsmec >= 0.5).astype(int)

        all_results["in_domain"]["phobert_mean_std"] = summarize_seed_metrics(pho_in_list)
        all_results["in_domain"]["phobert_majority_vote"] = compute_metrics(
            test_labels, maj_test, soft_prob_test
        )
        all_results["in_domain"]["phobert_soft_vote"] = compute_metrics(
            test_labels, soft_test, soft_prob_test
        )
        all_results["in_domain"]["phobert_avg"] = dict(
            all_results["in_domain"]["phobert_soft_vote"],
            aggregation_rule="soft_vote_prob_avg_over_3_seeds",
            majority_vote_f1_macro=all_results["in_domain"]["phobert_majority_vote"]["f1_macro"],
            mean_f1_macro_across_seeds=all_results["in_domain"]["phobert_mean_std"]["f1_macro_mean"],
            std_f1_macro_across_seeds=all_results["in_domain"]["phobert_mean_std"]["f1_macro_std"],
        )

        all_results["cross_domain"]["phobert_mean_std"] = summarize_seed_metrics(pho_cr_list)
        all_results["cross_domain"]["phobert_majority_vote"] = compute_metrics(
            vsmec_labels, maj_vsmec, soft_prob_vsmec
        )
        all_results["cross_domain"]["phobert_soft_vote"] = compute_metrics(
            vsmec_labels, soft_vsmec, soft_prob_vsmec
        )
        all_results["cross_domain"]["phobert_avg"] = dict(
            all_results["cross_domain"]["phobert_soft_vote"],
            aggregation_rule="soft_vote_prob_avg_over_3_seeds",
            majority_vote_f1_macro=all_results["cross_domain"]["phobert_majority_vote"]["f1_macro"],
            mean_f1_macro_across_seeds=all_results["cross_domain"]["phobert_mean_std"]["f1_macro_mean"],
            std_f1_macro_across_seeds=all_results["cross_domain"]["phobert_mean_std"]["f1_macro_std"],
        )

        all_results["cross_domain_sadness_only"]["phobert_mean_std"] = summarize_seed_metrics(pho_sad_list)
        all_results["cross_domain_sadness_only"]["phobert_majority_vote"] = compute_metrics(
            vsmec_sad_labels, maj_vsmec[sadness_mask], soft_prob_vsmec[sadness_mask]
        )
        all_results["cross_domain_sadness_only"]["phobert_soft_vote"] = compute_metrics(
            vsmec_sad_labels, soft_vsmec[sadness_mask], soft_prob_vsmec[sadness_mask]
        )

        all_results["bootstrap_95_ci_in_domain"]["phobert_soft_vote"] = bootstrap_ci(
            test_labels, soft_test
        )
        all_results["bootstrap_95_ci_in_domain"]["phobert_majority_vote"] = bootstrap_ci(
            test_labels, maj_test
        )
        all_results["bootstrap_95_ci_cross_domain"]["phobert_soft_vote"] = bootstrap_ci(
            vsmec_labels, soft_vsmec
        )

        in_domain_preds_store["phobert_soft_vote"] = soft_test
        in_domain_preds_store["phobert_majority_vote"] = maj_test
        cross_domain_preds_store["phobert_soft_vote"] = soft_vsmec
        cross_domain_preds_store["phobert_majority_vote"] = maj_vsmec

        all_results["threshold_calibration_analysis"]["phobert_soft_vote"] = (
            threshold_calibration_analysis(
                val_labels,
                soft_prob_val,
                test_labels,
                soft_prob_test,
                vsmec_labels,
                soft_prob_vsmec,
                vsmec_sad_labels,
                soft_prob_vsmec[sadness_mask],
            )
        )

    # ── 2A & 2B. BiLSTM (Unweighted & Class-Weighted across 3 seeds) ────────
    for prefix in ["bilstm", "bilstm_weighted"]:
        print("\n" + "=" * 75)
        print(f"2. {prefix.upper()} (3 seeds: 42, 123, 2024)")
        print("=" * 75)

        b_preds_test, b_probs_test = {}, {}
        b_preds_vsmec, b_probs_vsmec = {}, {}
        b_in_list, b_cr_list, b_sad_list = [], [], []

        for seed in SEEDS:
            model_path = PHOBERT_DIR / f"{prefix}_seed_{seed}" / "best_model.pt"
            if not model_path.exists():
                print(f"  {prefix} seed {seed} not found at {model_path}, skipping")
                continue

            preds_test, probs_test = predict_bilstm(model_path, test_texts)
            preds_vsmec, probs_vsmec = predict_bilstm(model_path, vsmec_texts)

            m_test = compute_metrics(test_labels, preds_test, probs_test)
            m_vsmec = compute_metrics(vsmec_labels, preds_vsmec, probs_vsmec)
            m_sad = compute_metrics(
                vsmec_sad_labels, preds_vsmec[sadness_mask], probs_vsmec[sadness_mask]
            )

            all_results["in_domain"][f"{prefix}_seed{seed}"] = m_test
            all_results["cross_domain"][f"{prefix}_seed{seed}"] = m_vsmec
            all_results["cross_domain_sadness_only"][f"{prefix}_seed{seed}"] = m_sad

            b_preds_test[seed] = preds_test
            b_probs_test[seed] = probs_test
            b_preds_vsmec[seed] = preds_vsmec
            b_probs_vsmec[seed] = probs_vsmec

            b_in_list.append(m_test)
            b_cr_list.append(m_vsmec)
            b_sad_list.append(m_sad)

            print(
                f"  Seed {seed}: In-Domain F1-M={m_test['f1_macro']:.4f} | Cross-Domain F1-M={m_vsmec['f1_macro']:.4f}"
            )

        if b_preds_test:
            seeds = sorted(b_preds_test.keys())
            maj_test = (
                np.mean(np.stack([b_preds_test[s] for s in seeds], axis=0), axis=0) >= 0.5
            ).astype(int)
            soft_prob_test = np.mean(
                np.stack([b_probs_test[s] for s in seeds], axis=0), axis=0
            )
            soft_test = (soft_prob_test >= 0.5).astype(int)

            maj_vsmec = (
                np.mean(np.stack([b_preds_vsmec[s] for s in seeds], axis=0), axis=0) >= 0.5
            ).astype(int)
            soft_prob_vsmec = np.mean(
                np.stack([b_probs_vsmec[s] for s in seeds], axis=0), axis=0
            )
            soft_vsmec = (soft_prob_vsmec >= 0.5).astype(int)

            all_results["in_domain"][f"{prefix}_mean_std"] = summarize_seed_metrics(b_in_list)
            all_results["in_domain"][f"{prefix}_majority_vote"] = compute_metrics(
                test_labels, maj_test, soft_prob_test
            )
            all_results["in_domain"][f"{prefix}_soft_vote"] = compute_metrics(
                test_labels, soft_test, soft_prob_test
            )
            all_results["in_domain"][f"{prefix}_avg"] = dict(
                all_results["in_domain"][f"{prefix}_majority_vote"],
                aggregation_rule="majority_vote_over_3_seeds",
                mean_f1_macro_across_seeds=all_results["in_domain"][f"{prefix}_mean_std"]["f1_macro_mean"],
                std_f1_macro_across_seeds=all_results["in_domain"][f"{prefix}_mean_std"]["f1_macro_std"],
            )

            all_results["cross_domain"][f"{prefix}_mean_std"] = summarize_seed_metrics(b_cr_list)
            all_results["cross_domain"][f"{prefix}_majority_vote"] = compute_metrics(
                vsmec_labels, maj_vsmec, soft_prob_vsmec
            )
            all_results["cross_domain"][f"{prefix}_soft_vote"] = compute_metrics(
                vsmec_labels, soft_vsmec, soft_prob_vsmec
            )
            all_results["cross_domain"][f"{prefix}_avg"] = dict(
                all_results["cross_domain"][f"{prefix}_majority_vote"],
                aggregation_rule="majority_vote_over_3_seeds",
                mean_f1_macro_across_seeds=all_results["cross_domain"][f"{prefix}_mean_std"]["f1_macro_mean"],
                std_f1_macro_across_seeds=all_results["cross_domain"][f"{prefix}_mean_std"]["f1_macro_std"],
            )

            all_results["cross_domain_sadness_only"][f"{prefix}_mean_std"] = summarize_seed_metrics(b_sad_list)
            all_results["cross_domain_sadness_only"][f"{prefix}_majority_vote"] = compute_metrics(
                vsmec_sad_labels, maj_vsmec[sadness_mask], soft_prob_vsmec[sadness_mask]
            )

            all_results["bootstrap_95_ci_in_domain"][f"{prefix}_majority_vote"] = bootstrap_ci(
                test_labels, maj_test
            )
            all_results["bootstrap_95_ci_cross_domain"][f"{prefix}_majority_vote"] = bootstrap_ci(
                vsmec_labels, maj_vsmec
            )
            in_domain_preds_store[f"{prefix}_majority_vote"] = maj_test
            cross_domain_preds_store[f"{prefix}_majority_vote"] = maj_vsmec

    # ── 3. TF-IDF + Logistic Regression ──────────────────────────────────────
    print("\n" + "=" * 75)
    print("3. TF-IDF + LOGISTIC REGRESSION")
    print("=" * 75)
    tfidf_logreg_path = (
        PHOBERT_DIR / "tfidf_logreg_round6_v2.joblib"
        if (PHOBERT_DIR / "tfidf_logreg_round6_v2.joblib").exists()
        else PHOBERT_DIR / "tfidf_logreg_round6.joblib"
    )
    if tfidf_logreg_path.exists():
        preds_test, probs_test = predict_tfidf(tfidf_logreg_path, test_texts)
        preds_vsmec, probs_vsmec = predict_tfidf(tfidf_logreg_path, vsmec_texts)

        all_results["in_domain"]["tfidf_logreg"] = compute_metrics(
            test_labels, preds_test, probs_test
        )
        all_results["cross_domain"]["tfidf_logreg"] = compute_metrics(
            vsmec_labels, preds_vsmec, probs_vsmec
        )
        all_results["cross_domain_sadness_only"]["tfidf_logreg"] = compute_metrics(
            vsmec_sad_labels, preds_vsmec[sadness_mask], probs_vsmec[sadness_mask]
        )
        all_results["bootstrap_95_ci_in_domain"]["tfidf_logreg"] = bootstrap_ci(
            test_labels, preds_test
        )
        all_results["bootstrap_95_ci_cross_domain"]["tfidf_logreg"] = bootstrap_ci(
            vsmec_labels, preds_vsmec
        )
        in_domain_preds_store["tfidf_logreg"] = preds_test
        cross_domain_preds_store["tfidf_logreg"] = preds_vsmec

    # ── 4. TF-IDF + LinearSVC ────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("4. TF-IDF + LINEARSVC")
    print("=" * 75)
    tfidf_svc_path = (
        PHOBERT_DIR / "tfidf_linearsvc_round6_v2.joblib"
        if (PHOBERT_DIR / "tfidf_linearsvc_round6_v2.joblib").exists()
        else PHOBERT_DIR / "tfidf_linearsvc_round6.joblib"
    )
    if tfidf_svc_path.exists():
        preds_test, probs_test = predict_tfidf(tfidf_svc_path, test_texts)
        preds_vsmec, probs_vsmec = predict_tfidf(tfidf_svc_path, vsmec_texts)

        all_results["in_domain"]["tfidf_svc"] = compute_metrics(
            test_labels, preds_test, probs_test
        )
        all_results["cross_domain"]["tfidf_svc"] = compute_metrics(
            vsmec_labels, preds_vsmec, probs_vsmec
        )
        all_results["cross_domain_sadness_only"]["tfidf_svc"] = compute_metrics(
            vsmec_sad_labels, preds_vsmec[sadness_mask], probs_vsmec[sadness_mask]
        )
        all_results["bootstrap_95_ci_in_domain"]["tfidf_svc"] = bootstrap_ci(
            test_labels, preds_test
        )
        all_results["bootstrap_95_ci_cross_domain"]["tfidf_svc"] = bootstrap_ci(
            vsmec_labels, preds_vsmec
        )
        in_domain_preds_store["tfidf_svc"] = preds_test
        cross_domain_preds_store["tfidf_svc"] = preds_vsmec

    # ── 5. Merge Inductive BERTopic & PhoBERT+BERTopic Results if present ───
    bt_eval_file = PHOBERT_DIR / "bertopic_evaluation_results.json"
    if bt_eval_file.exists():
        bt_data = json.loads(bt_eval_file.read_text(encoding="utf-8"))
        if "bertopic_only" in bt_data:
            all_results["in_domain"]["bertopic_only"] = bt_data["bertopic_only"]["in_domain"]
            all_results["cross_domain"]["bertopic_only"] = bt_data["bertopic_only"]["cross_domain"]
            if "cross_domain_sadness_only" in bt_data["bertopic_only"]:
                all_results["cross_domain_sadness_only"]["bertopic_only"] = bt_data["bertopic_only"][
                    "cross_domain_sadness_only"
                ]
        if bt_data.get("phobert_bertopic_combined"):
            hyb = bt_data["phobert_bertopic_combined"]
            all_results["in_domain"]["phobert_bertopic"] = hyb["in_domain"]
            all_results["in_domain"]["phobert_bertopic_majority_vote"] = hyb["in_domain_majority_vote"]
            all_results["in_domain"]["phobert_bertopic_mean_std"] = hyb["in_domain_mean_std"]
            all_results["cross_domain"]["phobert_bertopic"] = hyb["cross_domain"]
            all_results["cross_domain"]["phobert_bertopic_majority_vote"] = hyb["cross_domain_majority_vote"]
            all_results["cross_domain"]["phobert_bertopic_mean_std"] = hyb["cross_domain_mean_std"]
            if "cross_domain_sadness_only" in hyb:
                all_results["cross_domain_sadness_only"]["phobert_bertopic"] = hyb[
                    "cross_domain_sadness_only"
                ]

        in_pred_csv = RESULTS_DIR / "phobert_bertopic_predictions_in_domain.csv"
        cr_pred_csv = RESULTS_DIR / "phobert_bertopic_predictions_cross_domain.csv"
        if in_pred_csv.exists():
            df_p_in = pd.read_csv(in_pred_csv)
            col_in = (
                "pred_label_soft_vote"
                if "pred_label_soft_vote" in df_p_in.columns
                else "pred_label"
            )
            preds_hyb_in = df_p_in[col_in].to_numpy(dtype=int)
            in_domain_preds_store["phobert_bertopic"] = preds_hyb_in
            all_results["bootstrap_95_ci_in_domain"]["phobert_bertopic"] = bootstrap_ci(
                test_labels, preds_hyb_in
            )
        if cr_pred_csv.exists():
            df_p_cr = pd.read_csv(cr_pred_csv)
            col_cr = (
                "pred_label_soft_vote"
                if "pred_label_soft_vote" in df_p_cr.columns
                else "pred_label"
            )
            preds_hyb_cr = df_p_cr[col_cr].to_numpy(dtype=int)
            cross_domain_preds_store["phobert_bertopic"] = preds_hyb_cr
            all_results["bootstrap_95_ci_cross_domain"]["phobert_bertopic"] = bootstrap_ci(
                vsmec_labels, preds_hyb_cr
            )

    # ── 6. Pairwise Statistical Significance Tests (McNemar + Paired Bootstrap)
    print("\n" + "=" * 75)
    print("5. PAIRWISE STATISTICAL SIGNIFICANCE TESTS (B=2,000 + McNemar)")
    print("=" * 75)

    ref_model = (
        "phobert_soft_vote"
        if "phobert_soft_vote" in in_domain_preds_store
        else next(iter(in_domain_preds_store.keys()), None)
    )
    if ref_model:
        for other_model, other_preds in in_domain_preds_store.items():
            if other_model == ref_model:
                continue
            sig_in = paired_significance_test(
                test_labels,
                in_domain_preds_store[ref_model],
                other_preds,
                ref_model,
                other_model,
            )
            all_results["statistical_significance_in_domain"].append(sig_in)
            print(
                f"  In-Domain [{ref_model} vs {other_model}]: "
                f"ΔF1={sig_in['delta_f1_macro_a_minus_b']:+.4f} "
                f"95% CI [{sig_in['paired_bootstrap']['ci95_lower']:+.4f}, {sig_in['paired_bootstrap']['ci95_upper']:+.4f}], "
                f"boot_p={sig_in['paired_bootstrap']['p_value_two_sided']:.4f}, "
                f"McNemar_p={sig_in['mcnemar_test']['exact_binomial_p_value']:.4f}"
            )

    if "tfidf_svc" in cross_domain_preds_store:
        for other_model, other_preds in cross_domain_preds_store.items():
            if other_model == "tfidf_svc":
                continue
            sig_cr = paired_significance_test(
                vsmec_labels,
                cross_domain_preds_store["tfidf_svc"],
                other_preds,
                "tfidf_svc",
                other_model,
            )
            all_results["statistical_significance_cross_domain"].append(sig_cr)

    # ── Save Results ────────────────────────────────────────────────────────
    results_file = OUTPUT_DIR / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    canonical_file = PHOBERT_DIR / "evaluation_results.json"
    with open(canonical_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {results_file}")
    print(f"Canonical copy:   {canonical_file}")
    return all_results


if __name__ == "__main__":
    main()
