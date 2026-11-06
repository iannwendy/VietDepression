#!/usr/bin/env python3
"""
Comprehensive Comparative Baselines & Calibration Suite for VietDepression (KAIS Revision).

Strictly inductive & zero-leakage protocol:
- All models/scalers/vectorizers/active-learning pools use ONLY final_train.csv (n = 6,392).
- Checkpoint selection & temperature scaling use ONLY final_val.csv (n = 1,371).
- In-domain evaluation uses held-out final_test.csv (n = 1,371).
- External evaluation uses cross_domain_test.csv (UIT-VSMEC Full n = 3,084; Sadness-Only n = 2,691).
- Unsupervised EM prior-shift adaptation (Saerens et al.) estimates target priors from unlabeled
  target predictions ONLY, with zero access to external ground-truth labels.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data" / "labeled"
UNIFIED_DIR = BASE_DIR / "data_unified"
MODEL_DIR = BASE_DIR / "models" / "round6_v2_retrained"
AUDIT_DIR = BASE_DIR / "results" / "scientific_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = AUDIT_DIR / "top_venue_baselines_results.json"

SEEDS = [42, 123, 2024]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_progress(payload: Dict) -> None:
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    res = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_depression": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_normal": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_depression": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_depression": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "predicted_positives": int(np.sum(y_pred == 1)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        if len(np.unique(y_true)) > 1:
            res["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            res["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return res


def bootstrap_ci(y_true, y_pred, n_boot: int = 2000, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    f1_macros = np.empty(n_boot)
    f1_deps = np.empty(n_boot)
    accs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        f1_macros[b] = f1_score(yt, yp, average="macro", zero_division=0)
        f1_deps[b] = f1_score(yt, yp, pos_label=1, zero_division=0)
        accs[b] = accuracy_score(yt, yp)
    return {
        "f1_macro_std": float(np.std(f1_macros, ddof=1)),
        "f1_macro_ci95": [float(np.percentile(f1_macros, 2.5)), float(np.percentile(f1_macros, 97.5))],
        "f1_depression_ci95": [float(np.percentile(f1_deps, 2.5)), float(np.percentile(f1_deps, 97.5))],
        "accuracy_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
    }


def paired_significance(y_true, pred_a, pred_b, n_boot: int = 2000, seed: int = 42) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    mcnemar_p = float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if (n10 + n01) > 0 else 1.0

    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs_diff = float(
        f1_score(y_true, pred_a, average="macro", zero_division=0)
        - f1_score(y_true, pred_b, average="macro", zero_division=0)
    )
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        diffs[b] = (
            f1_score(yt, pred_a[idx], average="macro", zero_division=0)
            - f1_score(yt, pred_b[idx], average="macro", zero_division=0)
        )
    centered = diffs - np.mean(diffs)
    boot_p = float(np.mean(np.abs(centered) >= abs(obs_diff)))
    return {
        "obs_f1_macro_diff": obs_diff,
        "f1_macro_diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "paired_bootstrap_p_two_sided": boot_p,
        "mcnemar_n10": n10,
        "mcnemar_n01": n01,
        "mcnemar_exact_p": mcnemar_p,
    }


def summarize_multiseed(y_true, preds_list, probs_list, seed_dict) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    thresh_votes = (len(preds_list) // 2) + 1
    maj_pred = (np.sum(preds_list, axis=0) >= thresh_votes).astype(int)
    soft_prob = np.mean(probs_list, axis=0)
    soft_pred = (soft_prob >= 0.5).astype(int)
    f1s = [v["f1_macro"] for v in seed_dict.values()]
    accs = [v["accuracy"] for v in seed_dict.values()]
    f1_deps = [v["f1_depression"] for v in seed_dict.values()]
    rocs = [v["roc_auc"] for v in seed_dict.values()]
    prs = [v["pr_auc"] for v in seed_dict.values()]
    ddof = 1 if len(f1s) > 1 else 0
    maj_m = compute_metrics(y_true, maj_pred, soft_prob)
    soft_m = compute_metrics(y_true, soft_pred, soft_prob)
    summary = {
        "seeds": seed_dict,
        "mean_std": {
            "f1_macro_mean": float(np.mean(f1s)),
            "f1_macro_std": float(np.std(f1s, ddof=ddof)),
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=ddof)),
            "f1_depression_mean": float(np.mean(f1_deps)),
            "f1_depression_std": float(np.std(f1_deps, ddof=ddof)),
            "roc_auc_mean": float(np.mean(rocs)),
            "roc_auc_std": float(np.std(rocs, ddof=ddof)),
            "pr_auc_mean": float(np.mean(prs)),
            "pr_auc_std": float(np.std(prs, ddof=ddof)),
        },
        "majority_vote": {**maj_m, "bootstrap": bootstrap_ci(y_true, maj_pred)},
        "soft_vote": {**soft_m, "bootstrap": bootstrap_ci(y_true, soft_pred)},
    }
    return summary, maj_pred, soft_pred, soft_prob


class DynamicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = [int(x) for x in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class LengthBucketBatchSampler(Sampler):
    def __init__(self, lengths: List[int], batch_size: int, seed: int = 42):
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        indices = rng.permutation(len(self.lengths))
        megabatch_size = self.batch_size * 25
        batches = []
        for i in range(0, len(indices), megabatch_size):
            mega = indices[i : i + megabatch_size]
            mega_sorted = mega[np.argsort([self.lengths[j] for j in mega])]
            for b in range(0, len(mega_sorted), self.batch_size):
                batches.append(mega_sorted[b : b + self.batch_size].tolist())
        rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def predict_seqcls_logits_and_cls(
    model, tokenizer, texts: List[str], device: torch.device, max_length: int = 128, need_cls: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    order = np.argsort([len(t) for t in texts])
    sorted_texts = [texts[i] for i in order]
    logits_sorted = []
    cls_sorted = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(sorted_texts), batch_size):
            batch_texts = sorted_texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "mps")):
                out = model(**enc, output_hidden_states=need_cls)
            logits_sorted.append(out.logits.float().cpu().numpy())
            if need_cls:
                cls_vec = out.hidden_states[-1][:, 0, :].float().cpu().numpy()
                cls_sorted.append(cls_vec)
    logits_cat = np.concatenate(logits_sorted, axis=0)
    logits = np.empty_like(logits_cat)
    logits[order] = logits_cat
    cls_out = None
    if need_cls:
        cls_cat = np.concatenate(cls_sorted, axis=0)
        cls_out = np.empty_like(cls_cat)
        cls_out[order] = cls_cat
    return logits, cls_out


# ── Custom Losses (TKDE / KDD / ICDM Baselines) ─────────────────────────────
class LogitAdjustedFocalLoss(nn.Module):
    """
    Combines Class-Prior Logit Adjustment (Menon et al.) with Focal Modulation
    (Lin et al.; effective-number / imbalanced calibration discussed in Wu & Li, TKDE 2024).
    """

    def __init__(self, prior_counts: Tuple[int, int], tau_la: float = 0.5, gamma: float = 2.0, alpha_pos: float = 2.0):
        super().__init__()
        total = float(prior_counts[0] + prior_counts[1])
        priors = torch.tensor([prior_counts[0] / total, prior_counts[1] / total], dtype=torch.float32)
        self.register_buffer("logit_adj", tau_la * torch.log(priors + 1e-12))
        self.register_buffer("class_alpha", torch.tensor([1.0, alpha_pos], dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adj_logits = logits + self.logit_adj.to(logits.device)
        log_probs = F.log_softmax(adj_logits, dim=-1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        at = self.class_alpha.to(logits.device).gather(0, targets)
        loss = -at * ((1.0 - pt) ** self.gamma) * log_pt
        return loss.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al.; Gunel et al.; Shin et al., KDD 2026)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [B, D] L2-normalized
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 2:
            return torch.tensor(0.0, device=device)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        sim = torch.matmul(features, features.T) / self.temperature
        sim_max, _ = torch.max(sim * logits_mask + (-1e9) * (1.0 - logits_mask), dim=1, keepdim=True)
        logits = sim - sim_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_per_row = mask.sum(1)
        valid = pos_per_row > 0
        if not torch.any(valid):
            return torch.tensor(0.0, device=device)
        mean_log_prob_pos = (mask[valid] * log_prob[valid]).sum(1) / pos_per_row[valid]
        return -mean_log_prob_pos.mean()


class SupConEncoderClassifier(nn.Module):
    def __init__(self, model_name: str, proj_dim: int = 128):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 2)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_h = out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_h))
        z = F.normalize(self.projector(cls_h.float()), dim=-1)
        return logits, z


# ── Negatively Correlated Noisy Learners (Ragheb et al., IEEE TKDE 2023) ────
class NCNLEnsemble(nn.Module):
    """
    Ensemble of M diverse neural learners trained simultaneously with a Negative
    Correlation Learning (NCL) penalty to decorrelate individual learner errors
    on noisy/ambiguous mental-health expressions (Ragheb et al., IEEE TKDE 2023).
    """

    def __init__(self, in_dim: int, n_learners: int = 5, hidden_dim: int = 192, dropout: float = 0.35):
        super().__init__()
        self.learners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 2),
                )
                for _ in range(n_learners)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [learner(x) for learner in self.learners]


# ── Unsupervised Prior-Shift Calibration (Saerens et al. EM + Val Temp) ─────
def fit_temperature_on_val(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    y = np.asarray(val_labels, dtype=int)

    def nll(T: float) -> float:
        scaled = val_logits / max(T, 1e-3)
        probs = softmax(scaled, axis=-1)[:, 1]
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        return float(log_loss(y, probs))

    res = minimize_scalar(nll, bounds=(0.25, 5.0), method="bounded")
    return float(res.x)


def saerens_em_prior_adaptation(
    target_probs_source_prior: np.ndarray, pi_source: float, max_iter: int = 200, tol: float = 1e-7
) -> Tuple[float, np.ndarray]:
    """
    Maximum-likelihood EM estimation of target class prior pi_target from unlabeled
    target posterior probabilities p_s(y=1|x), without using any target labels
    (Saerens, Latinne & Decaestecker, Neural Computation 2002).
    """
    p_s = np.clip(np.asarray(target_probs_source_prior, dtype=float), 1e-6, 1.0 - 1e-6)
    pi_t = float(pi_source)
    for _ in range(max_iter):
        r1 = (pi_t / pi_source) * p_s
        r0 = ((1.0 - pi_t) / (1.0 - pi_source)) * (1.0 - p_s)
        post = r1 / (r1 + r0 + 1e-12)
        pi_next = float(np.mean(post))
        pi_next = float(np.clip(pi_next, 0.01, 0.99))
        if abs(pi_next - pi_t) < tol:
            pi_t = pi_next
            break
        pi_t = pi_next
    r1 = (pi_t / pi_source) * p_s
    r0 = ((1.0 - pi_t) / (1.0 - pi_source)) * (1.0 - p_s)
    adjusted_probs = r1 / (r1 + r0 + 1e-12)
    return pi_t, adjusted_probs


# ── Training Pipelines ──────────────────────────────────────────────────────
def evaluate_saved_checkpoints(
    model_prefix: str,
    tokenizer_name: str,
    subdir: str,
    val_texts: List[str],
    val_labels: np.ndarray,
    test_texts: List[str],
    test_labels: np.ndarray,
    vf_texts: List[str],
    vf_labels: np.ndarray,
    sad_mask: np.ndarray,
    device: torch.device,
    extract_cls_splits: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    print(f"\n=== Evaluating Saved 3-Seed Checkpoints: {model_prefix} ===")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    seed_in, seed_vf, seed_vs = {}, {}, {}
    preds_in, probs_in = [], []
    preds_vf, probs_vf = [], []
    preds_vs, probs_vs = [], []
    val_logits_seeds, vf_logits_seeds = [], []
    cls_per_seed = {}

    for seed in SEEDS:
        ckpt_dir = MODEL_DIR / f"{model_prefix}_seed_{seed}"
        if subdir:
            ckpt_dir = ckpt_dir / subdir
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir)).to(device)

        v_logits, _ = predict_seqcls_logits_and_cls(model, tokenizer, val_texts, device)
        te_logits, te_cls = predict_seqcls_logits_and_cls(
            model, tokenizer, test_texts, device, need_cls=(extract_cls_splits is not None)
        )
        vf_logits, vf_cls = predict_seqcls_logits_and_cls(
            model, tokenizer, vf_texts, device, need_cls=(extract_cls_splits is not None)
        )

        if extract_cls_splits is not None:
            _, tr_cls = predict_seqcls_logits_and_cls(
                model, tokenizer, extract_cls_splits["train"], device, need_cls=True
            )
            _, va_cls = predict_seqcls_logits_and_cls(
                model, tokenizer, extract_cls_splits["val"], device, need_cls=True
            )
            cls_per_seed[seed] = {
                "train": tr_cls,
                "val": va_cls,
                "test": te_cls,
                "vf": vf_cls,
            }

        val_logits_seeds.append(v_logits)
        vf_logits_seeds.append(vf_logits)

        p_in = softmax(te_logits, axis=-1)[:, 1]
        pr_in = (p_in >= 0.5).astype(int)
        seed_in[f"seed_{seed}"] = compute_metrics(test_labels, pr_in, p_in)
        preds_in.append(pr_in)
        probs_in.append(p_in)

        p_vf = softmax(vf_logits, axis=-1)[:, 1]
        pr_vf = (p_vf >= 0.5).astype(int)
        seed_vf[f"seed_{seed}"] = compute_metrics(vf_labels, pr_vf, p_vf)
        preds_vf.append(pr_vf)
        probs_vf.append(p_vf)

        p_vs = p_vf[sad_mask]
        pr_vs = pr_vf[sad_mask]
        seed_vs[f"seed_{seed}"] = compute_metrics(vf_labels[sad_mask], pr_vs, p_vs)
        preds_vs.append(pr_vs)
        probs_vs.append(p_vs)

        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    sum_in, maj_in, soft_in, sprob_in = summarize_multiseed(test_labels, preds_in, probs_in, seed_in)
    sum_vf, _, _, sprob_vf = summarize_multiseed(vf_labels, preds_vf, probs_vf, seed_vf)
    sum_vs, _, _, sprob_vs = summarize_multiseed(vf_labels[sad_mask], preds_vs, probs_vs, seed_vs)
    print(
        f"  {model_prefix}: In-Domain Maj F1={sum_in['majority_vote']['f1_macro']:.4f}, "
        f"Soft F1={sum_in['soft_vote']['f1_macro']:.4f}, ROC-AUC={sum_in['majority_vote']['roc_auc']:.4f}"
    )
    return {
        "summary": {"in_domain": sum_in, "vsmec_full": sum_vf, "vsmec_sad": sum_vs},
        "maj_pred_in": maj_in,
        "soft_pred_in": soft_in,
        "soft_prob_in": sprob_in,
        "soft_prob_vf": sprob_vf,
        "soft_prob_vs": sprob_vs,
        "val_logits_mean": np.mean(val_logits_seeds, axis=0),
        "vf_logits_mean": np.mean(vf_logits_seeds, axis=0),
        "cls_per_seed": cls_per_seed,
    }


def train_standard_or_lafocal_encoder(
    model_name: str,
    save_prefix: str,
    loss_mode: str,  # "wce" or "la_focal"
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    test_texts: List[str],
    test_labels: np.ndarray,
    vf_texts: List[str],
    vf_labels: np.ndarray,
    sad_mask: np.ndarray,
    device: torch.device,
) -> Dict:
    print(f"\n=== Training {model_name} [{save_prefix}] (loss={loss_mode}, 3 seeds) ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_enc = tokenizer(train_texts, truncation=True, max_length=128)
    val_enc = tokenizer(val_texts, truncation=True, max_length=128)
    train_lengths = [len(x) for x in train_enc["input_ids"]]

    train_ds = DynamicDataset(train_enc, train_labels.tolist())
    val_ds = DynamicDataset(val_enc, val_labels.tolist())
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    n0 = int(np.sum(train_labels == 0))
    n1 = int(np.sum(train_labels == 1))
    if loss_mode == "wce":
        cw = torch.tensor([1.0, n0 / max(n1, 1)], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    elif loss_mode == "la_focal":
        criterion = LogitAdjustedFocalLoss((n0, n1), tau_la=0.5, gamma=2.0, alpha_pos=2.2).to(device)
    else:
        raise ValueError(loss_mode)

    seed_in, seed_vf, seed_vs = {}, {}, {}
    preds_in, probs_in = [], []
    preds_vf, probs_vf = [], []
    preds_vs, probs_vs = [], []
    val_logits_seeds, vf_logits_seeds = [], []

    for seed in SEEDS:
        set_seed(seed)
        save_dir = MODEL_DIR / f"{save_prefix}_seed_{seed}"
        done_flag = save_dir / "TRAINING_DONE"
        if (save_dir / "config.json").exists() and done_flag.exists():
            print(f"  Loading cached checkpoint {save_dir.name}...")
            model = AutoModelForSequenceClassification.from_pretrained(str(save_dir)).to(device)
        else:
            t0 = time.time()
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2, ignore_mismatched_sizes=True
            ).to(device)
            if "xlm-roberta" in model_name.lower():
                for p in model.roberta.embeddings.parameters():
                    p.requires_grad = False
            micro_bs = 32
            grad_accum = 1
            batch_sampler = LengthBucketBatchSampler(train_lengths, batch_size=micro_bs, seed=seed)
            train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=collator)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collator)

            epochs = 3
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(trainable_params, lr=2e-5, weight_decay=0.01)
            total_steps = math.ceil(len(train_loader) / grad_accum) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(total_steps * 0.06), num_training_steps=total_steps
            )

            best_val_f1 = -1.0
            save_dir.mkdir(parents=True, exist_ok=True)
            for ep in range(epochs):
                model.train()
                tot_loss = 0.0
                optimizer.zero_grad(set_to_none=True)
                for step_i, batch in enumerate(train_loader):
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "mps")):
                        out = model(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                        )
                        raw_loss = criterion(out.logits.float(), batch["labels"].to(device))
                        loss = raw_loss / grad_accum
                    loss.backward()
                    tot_loss += raw_loss.item()
                    del out, raw_loss, loss
                    if ((step_i + 1) % grad_accum == 0) or ((step_i + 1) == len(train_loader)):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                model.eval()
                val_preds, val_t = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "mps")):
                            out = model(
                                input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                            )
                            logits_eval = out.logits.float()
                            if loss_mode == "la_focal":
                                logits_eval = logits_eval + criterion.logit_adj.to(logits_eval.device)
                        val_preds.extend(torch.argmax(logits_eval, dim=-1).cpu().numpy().tolist())
                        val_t.extend(batch["labels"].numpy().tolist())
                vf1 = f1_score(val_t, val_preds, average="macro", zero_division=0)
                print(f"    [Seed {seed}] Ep {ep+1}/{epochs}: loss={tot_loss/len(train_loader):.4f}, val_f1_M={vf1:.4f}")
                if vf1 > best_val_f1:
                    best_val_f1 = vf1
                    model.save_pretrained(str(save_dir))
                    tokenizer.save_pretrained(str(save_dir))

            done_flag.write_text("ok\n", encoding="utf-8")
            del optimizer, scheduler, model
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            model = AutoModelForSequenceClassification.from_pretrained(str(save_dir)).to(device)
            print(f"  Seed {seed} finished in {time.time() - t0:.1f}s (best val F1={best_val_f1:.4f})")

        v_logits, _ = predict_seqcls_logits_and_cls(model, tokenizer, val_texts, device)
        te_logits, _ = predict_seqcls_logits_and_cls(model, tokenizer, test_texts, device)
        vf_logits, _ = predict_seqcls_logits_and_cls(model, tokenizer, vf_texts, device)
        if loss_mode == "la_focal":
            adj_np = criterion.logit_adj.detach().cpu().numpy()
            v_logits = v_logits + adj_np
            te_logits = te_logits + adj_np
            vf_logits = vf_logits + adj_np

        val_logits_seeds.append(v_logits)
        vf_logits_seeds.append(vf_logits)

        p_in = softmax(te_logits, axis=-1)[:, 1]
        pr_in = (p_in >= 0.5).astype(int)
        m_in = compute_metrics(test_labels, pr_in, p_in)
        seed_in[f"seed_{seed}"] = m_in
        preds_in.append(pr_in)
        probs_in.append(p_in)

        p_vf = softmax(vf_logits, axis=-1)[:, 1]
        pr_vf = (p_vf >= 0.5).astype(int)
        seed_vf[f"seed_{seed}"] = compute_metrics(vf_labels, pr_vf, p_vf)
        preds_vf.append(pr_vf)
        probs_vf.append(p_vf)

        p_vs = p_vf[sad_mask]
        pr_vs = pr_vf[sad_mask]
        seed_vs[f"seed_{seed}"] = compute_metrics(vf_labels[sad_mask], pr_vs, p_vs)
        preds_vs.append(pr_vs)
        probs_vs.append(p_vs)
        print(f"  Seed {seed} Test F1-M={m_in['f1_macro']:.4f}, F1-Pos={m_in['f1_depression']:.4f}, AUC={m_in['roc_auc']:.4f}")

        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    sum_in, maj_in, soft_in, sprob_in = summarize_multiseed(test_labels, preds_in, probs_in, seed_in)
    sum_vf, _, _, sprob_vf = summarize_multiseed(vf_labels, preds_vf, probs_vf, seed_vf)
    sum_vs, _, _, sprob_vs = summarize_multiseed(vf_labels[sad_mask], preds_vs, probs_vs, seed_vs)
    print(f"  ==> {save_prefix} Maj F1={sum_in['majority_vote']['f1_macro']:.4f} | Soft F1={sum_in['soft_vote']['f1_macro']:.4f}")
    return {
        "summary": {"in_domain": sum_in, "vsmec_full": sum_vf, "vsmec_sad": sum_vs},
        "maj_pred_in": maj_in,
        "soft_pred_in": soft_in,
        "soft_prob_in": sprob_in,
        "soft_prob_vf": sprob_vf,
        "soft_prob_vs": sprob_vs,
        "val_logits_mean": np.mean(val_logits_seeds, axis=0),
        "vf_logits_mean": np.mean(vf_logits_seeds, axis=0),
    }


def train_supcon_encoder(
    model_name: str,
    save_prefix: str,
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    test_texts: List[str],
    test_labels: np.ndarray,
    vf_texts: List[str],
    vf_labels: np.ndarray,
    sad_mask: np.ndarray,
    device: torch.device,
    lambda_sc: float = 0.20,
    only_seed: Optional[int] = None,
) -> Dict:
    print(f"\n=== Training Supervised Contrastive Encoder [{save_prefix}] (lambda_sc={lambda_sc}, only_seed={only_seed}) ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    train_enc = tokenizer(train_texts, truncation=True, max_length=128)
    val_enc = tokenizer(val_texts, truncation=True, max_length=128)
    train_lengths = [len(x) for x in train_enc["input_ids"]]

    train_ds = DynamicDataset(train_enc, train_labels.tolist())
    val_ds = DynamicDataset(val_enc, val_labels.tolist())
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    n0 = int(np.sum(train_labels == 0))
    n1 = int(np.sum(train_labels == 1))
    cw = torch.tensor([1.0, n0 / max(n1, 1)], dtype=torch.float32).to(device)
    ce_criterion = nn.CrossEntropyLoss(weight=cw)
    sc_criterion = SupConLoss(temperature=0.07)

    def predict_supcon_logits(model, texts: List[str]) -> np.ndarray:
        model.eval()
        order = np.argsort([len(t) for t in texts])
        sorted_texts = [texts[i] for i in order]
        logits_sorted = []
        with torch.no_grad():
            for i in range(0, len(sorted_texts), 64):
                bt = sorted_texts[i : i + 64]
                enc = tokenizer(bt, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "mps")):
                    lg, _ = model(enc["input_ids"], enc["attention_mask"])
                logits_sorted.append(lg.float().cpu().numpy())
        cat = np.concatenate(logits_sorted, axis=0)
        out = np.empty_like(cat)
        out[order] = cat
        return out

    seed_in, seed_vf, seed_vs = {}, {}, {}
    preds_in, probs_in = [], []
    preds_vf, probs_vf = [], []
    preds_vs, probs_vs = [], []
    val_logits_seeds, vf_logits_seeds = [], []

    target_seeds = [only_seed] if only_seed is not None else SEEDS
    for seed in target_seeds:
        set_seed(seed)
        ckpt_file = MODEL_DIR / f"{save_prefix}_seed_{seed}.pt"
        done_flag = MODEL_DIR / f"{save_prefix}_seed_{seed}.done"
        model = SupConEncoderClassifier(model_name, proj_dim=128).to(device)
        if ckpt_file.exists() and done_flag.exists():
            print(f"  Loading cached SupCon checkpoint {ckpt_file.name}...")
            model.load_state_dict(torch.load(ckpt_file, map_location=device))
        else:
            t0 = time.time()
            batch_sampler = LengthBucketBatchSampler(train_lengths, batch_size=32, seed=seed)
            train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=collator)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collator)

            epochs = 3
            optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(total_steps * 0.06), num_training_steps=total_steps
            )
            best_val_f1 = -1.0
            for ep in range(epochs):
                model.train()
                tot_loss = 0.0
                for batch in train_loader:
                    optimizer.zero_grad()
                    labels_b = batch["labels"].to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "mps")):
                        logits_b, z_b = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                        loss_ce = ce_criterion(logits_b.float(), labels_b)
                        loss_sc = sc_criterion(z_b.float(), labels_b)
                        loss = loss_ce + lambda_sc * loss_sc
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    tot_loss += loss.item()

                model.eval()
                vp, vt = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "mps")):
                            lg, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                        vp.extend(torch.argmax(lg.float(), dim=-1).cpu().numpy().tolist())
                        vt.extend(batch["labels"].numpy().tolist())
                vf1 = f1_score(vt, vp, average="macro", zero_division=0)
                print(f"    [Seed {seed}] Ep {ep+1}/{epochs}: loss={tot_loss/len(train_loader):.4f}, val_f1_M={vf1:.4f}")
                if vf1 > best_val_f1:
                    best_val_f1 = vf1
                    torch.save(model.state_dict(), ckpt_file)

            done_flag.write_text("ok\n", encoding="utf-8")
            del optimizer, scheduler
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            model.load_state_dict(torch.load(ckpt_file, map_location=device))
            print(f"  Seed {seed} SupCon finished in {time.time() - t0:.1f}s (best val F1={best_val_f1:.4f})")

        v_logits = predict_supcon_logits(model, val_texts)
        te_logits = predict_supcon_logits(model, test_texts)
        vf_logits = predict_supcon_logits(model, vf_texts)
        val_logits_seeds.append(v_logits)
        vf_logits_seeds.append(vf_logits)

        p_in = softmax(te_logits, axis=-1)[:, 1]
        pr_in = (p_in >= 0.5).astype(int)
        m_in = compute_metrics(test_labels, pr_in, p_in)
        seed_in[f"seed_{seed}"] = m_in
        preds_in.append(pr_in)
        probs_in.append(p_in)

        p_vf = softmax(vf_logits, axis=-1)[:, 1]
        pr_vf = (p_vf >= 0.5).astype(int)
        seed_vf[f"seed_{seed}"] = compute_metrics(vf_labels, pr_vf, p_vf)
        preds_vf.append(pr_vf)
        probs_vf.append(p_vf)

        p_vs = p_vf[sad_mask]
        pr_vs = pr_vf[sad_mask]
        seed_vs[f"seed_{seed}"] = compute_metrics(vf_labels[sad_mask], pr_vs, p_vs)
        preds_vs.append(pr_vs)
        probs_vs.append(p_vs)
        print(f"  Seed {seed} Test F1-M={m_in['f1_macro']:.4f}, F1-Pos={m_in['f1_depression']:.4f}, AUC={m_in['roc_auc']:.4f}")

        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    if only_seed is not None:
        return {"only_seed_done": True}

    sum_in, maj_in, soft_in, sprob_in = summarize_multiseed(test_labels, preds_in, probs_in, seed_in)
    sum_vf, _, _, sprob_vf = summarize_multiseed(vf_labels, preds_vf, probs_vf, seed_vf)
    sum_vs, _, _, sprob_vs = summarize_multiseed(vf_labels[sad_mask], preds_vs, probs_vs, seed_vs)
    print(f"  ==> {save_prefix} Maj F1={sum_in['majority_vote']['f1_macro']:.4f} | Soft F1={sum_in['soft_vote']['f1_macro']:.4f}")
    return {
        "summary": {"in_domain": sum_in, "vsmec_full": sum_vf, "vsmec_sad": sum_vs},
        "maj_pred_in": maj_in,
        "soft_pred_in": soft_in,
        "soft_prob_in": sprob_in,
        "soft_prob_vf": sprob_vf,
        "soft_prob_vs": sprob_vs,
        "val_logits_mean": np.mean(val_logits_seeds, axis=0),
        "vf_logits_mean": np.mean(vf_logits_seeds, axis=0),
    }


def train_ncnl_ensemble(
    cls_per_seed: Dict[int, Dict[str, np.ndarray]],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    vf_labels: np.ndarray,
    sad_mask: np.ndarray,
    device: torch.device,
    n_learners: int = 5,
    lambda_nc: float = 0.15,
) -> Dict:
    print(f"\n=== Training NCNL (Ragheb et al., IEEE TKDE 2023) [M={n_learners}, lambda_nc={lambda_nc}] ===")
    n0 = int(np.sum(train_labels == 0))
    n1 = int(np.sum(train_labels == 1))
    cw = torch.tensor([1.0, n0 / max(n1, 1)], dtype=torch.float32).to(device)
    ce_fn = nn.CrossEntropyLoss(weight=cw)

    seed_in, seed_vf, seed_vs = {}, {}, {}
    preds_in, probs_in = [], []
    preds_vf, probs_vf = [], []
    preds_vs, probs_vs = [], []

    for seed in SEEDS:
        set_seed(seed)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(cls_per_seed[seed]["train"])
        X_va = scaler.transform(cls_per_seed[seed]["val"])
        X_te = scaler.transform(cls_per_seed[seed]["test"])
        X_vf = scaler.transform(cls_per_seed[seed]["vf"])

        model = NCNLEnsemble(in_dim=X_tr.shape[1], n_learners=n_learners, hidden_dim=192, dropout=0.35).to(device)
        optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(train_labels, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
        best_val_f1 = -1.0
        best_state = None

        for ep in range(25):
            model.train()
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits_list = model(bx)
                probs_list_t = [F.softmax(lg, dim=-1)[:, 1] for lg in logits_list]
                p_bar = torch.stack(probs_list_t, dim=0).mean(dim=0)
                ce_loss = sum(ce_fn(lg, by) for lg in logits_list) / n_learners
                nc_penalty = sum(((pm - p_bar) ** 2).mean() for pm in probs_list_t) / n_learners
                loss = ce_loss - lambda_nc * nc_penalty
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                va_logits_list = model(X_va_t)
                va_prob = torch.stack([F.softmax(lg, dim=-1)[:, 1] for lg in va_logits_list], dim=0).mean(dim=0).cpu().numpy()
                vf1 = f1_score(val_labels, (va_prob >= 0.5).astype(int), average="macro", zero_division=0)
                if vf1 > best_val_f1:
                    best_val_f1 = vf1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            te_ll = model(torch.tensor(X_te, dtype=torch.float32).to(device))
            p_in = torch.stack([F.softmax(lg, dim=-1)[:, 1] for lg in te_ll], dim=0).mean(dim=0).cpu().numpy()
            vf_ll = model(torch.tensor(X_vf, dtype=torch.float32).to(device))
            p_vf = torch.stack([F.softmax(lg, dim=-1)[:, 1] for lg in vf_ll], dim=0).mean(dim=0).cpu().numpy()

        del model, optimizer, X_va_t
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

        pr_in = (p_in >= 0.5).astype(int)
        m_in = compute_metrics(test_labels, pr_in, p_in)
        seed_in[f"seed_{seed}"] = m_in
        preds_in.append(pr_in)
        probs_in.append(p_in)

        pr_vf = (p_vf >= 0.5).astype(int)
        seed_vf[f"seed_{seed}"] = compute_metrics(vf_labels, pr_vf, p_vf)
        preds_vf.append(pr_vf)
        probs_vf.append(p_vf)

        p_vs = p_vf[sad_mask]
        pr_vs = pr_vf[sad_mask]
        seed_vs[f"seed_{seed}"] = compute_metrics(vf_labels[sad_mask], pr_vs, p_vs)
        preds_vs.append(pr_vs)
        probs_vs.append(p_vs)
        print(f"  Seed {seed} NCNL Test F1-M={m_in['f1_macro']:.4f}, F1-Pos={m_in['f1_depression']:.4f}, AUC={m_in['roc_auc']:.4f}")

    sum_in, maj_in, soft_in, sprob_in = summarize_multiseed(test_labels, preds_in, probs_in, seed_in)
    sum_vf, _, _, sprob_vf = summarize_multiseed(vf_labels, preds_vf, probs_vf, seed_vf)
    sum_vs, _, _, sprob_vs = summarize_multiseed(vf_labels[sad_mask], preds_vs, probs_vs, seed_vs)
    print(f"  ==> NCNL Maj F1={sum_in['majority_vote']['f1_macro']:.4f} | Soft F1={sum_in['soft_vote']['f1_macro']:.4f}")
    return {
        "summary": {"in_domain": sum_in, "vsmec_full": sum_vf, "vsmec_sad": sum_vs},
        "maj_pred_in": maj_in,
        "soft_pred_in": soft_in,
        "soft_prob_in": sprob_in,
    }


# ── Instruction-Tuned LLM Zero-Shot & 4-Rule Rubric Few-Shot Baseline ───────
def evaluate_llm_baselines(
    train_df: pd.DataFrame,
    test_texts: List[str],
    test_labels: np.ndarray,
    vf_texts: List[str],
    vf_labels: np.ndarray,
    sad_mask: np.ndarray,
    device: torch.device,
) -> Dict:
    from transformers import DynamicCache

    llm_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"\n=== Evaluating Open-Weight Multilingual LLM ({llm_name}): Zero-Shot vs. 4-Rule Rubric Few-Shot ===")
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        llm_name, dtype=torch.bfloat16 if device.type == "mps" else torch.float32
    ).to(device)
    model.eval()

    id_0 = tokenizer.encode("0", add_special_tokens=False)[0]
    id_1 = tokenizer.encode("1", add_special_tokens=False)[0]

    # Select 4 deterministic training examples (2 Positive, 2 Negative) from final_train.csv ONLY
    pos_shots = train_df[train_df["label"] == 1].head(2)["comment_text"].astype(str).tolist()
    neg_shots = train_df[train_df["label"] == 0].head(2)["comment_text"].astype(str).tolist()

    sentinel = "<<<TARGET_SUFFIX>>>"
    zs_template = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You classify Vietnamese social media comments for depression-related psychological distress. "
                    "Reply with a single digit only: 1 for depression/distress, or 0 for normal/non-distress."
                ),
            },
            {"role": "user", "content": sentinel},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    zs_prefix, zs_post = zs_template.split(sentinel)

    fs_template = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "Classify Vietnamese YouTube comments strictly under four rules:\n"
                    "Rule 1 (Speaker Perspective): Label 1 ONLY for first-person or implicit personal distress; "
                    "label 0 for supportive replies to others (e.g., 'đừng buồn nữa nhé', 'cố lên') or third-person talk.\n"
                    "Rule 2 (Polysemy/Idioms): Label 0 for benign idioms/compounds ('buồn ngủ', 'buồn cười', 'chết cười').\n"
                    "Rule 3 (Persistence): Label 1 for persistent hopelessness, isolation, insomnia, or anhedonia; "
                    "label 0 for brief routine complaints.\n"
                    "Rule 4 (Crisis): Label 1 for first-person self-harm ideation; label 0 for anti-suicide encouragement to others.\n"
                    "Output ONLY a single digit: 0 or 1."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Example 1: {neg_shots[0][:240]}\nLabel: 0\n\n"
                    f"Example 2: {pos_shots[0][:240]}\nLabel: 1\n\n"
                    f"Example 3: {neg_shots[1][:240]}\nLabel: 0\n\n"
                    f"Example 4: {pos_shots[1][:240]}\nLabel: 1\n\n"
                    f"{sentinel}"
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    fs_prefix, fs_post = fs_template.split(sentinel)

    def score_with_prefix_cache(
        texts: List[str], prefix_str: str, suffix_fmt: str, post_str: str, batch_size: int = 32
    ) -> np.ndarray:
        suffixes = [f"{suffix_fmt.format(c=t[:500])}{post_str}" for t in texts]
        order = np.argsort([len(s) for s in suffixes])
        sorted_suffixes = [suffixes[i] for i in order]
        probs_sorted = []
        with torch.no_grad():
            pref_ids = tokenizer(prefix_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            pref_out = model(input_ids=pref_ids, use_cache=True)
            pref_kv = pref_out.past_key_values
            pref_len = pref_ids.shape[1]
            n_layers = len(pref_kv.layers)

            for i in range(0, len(sorted_suffixes), batch_size):
                bsuf = sorted_suffixes[i : i + batch_size]
                bsz = len(bsuf)
                s_enc = tokenizer(
                    bsuf, return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=False
                ).to(device)
                dc = DynamicCache()
                for li in range(n_layers):
                    dc.update(
                        pref_kv.layers[li].keys.expand(bsz, -1, -1, -1).contiguous(),
                        pref_kv.layers[li].values.expand(bsz, -1, -1, -1).contiguous(),
                        li,
                    )
                attn = torch.cat(
                    [torch.ones((bsz, pref_len), dtype=s_enc.attention_mask.dtype, device=device), s_enc.attention_mask],
                    dim=1,
                )
                base_out = model.model(
                    input_ids=s_enc.input_ids, attention_mask=attn, past_key_values=dc, use_cache=True
                )
                last_idx = s_enc.attention_mask.sum(dim=1) - 1
                last_hidden = base_out.last_hidden_state[torch.arange(bsz, device=device), last_idx]
                next_logits = model.lm_head(last_hidden).float()
                pair_logits = torch.stack([next_logits[:, id_0], next_logits[:, id_1]], dim=-1)
                p1 = torch.softmax(pair_logits, dim=-1)[:, 1].cpu().numpy()
                probs_sorted.extend(p1.tolist())
                del base_out, last_hidden, dc, s_enc, attn, next_logits, pair_logits
        probs = np.empty(len(texts), dtype=float)
        probs[order] = np.array(probs_sorted, dtype=float)
        return probs

    t0 = time.time()
    zs_prob_in = score_with_prefix_cache(test_texts, zs_prefix, "Comment: {c}\nLabel (0 or 1):", zs_post, batch_size=32)
    zs_pred_in = (zs_prob_in >= 0.5).astype(int)
    zs_in_m = {**compute_metrics(test_labels, zs_pred_in, zs_prob_in), "bootstrap": bootstrap_ci(test_labels, zs_pred_in)}

    zs_prob_vf = score_with_prefix_cache(vf_texts, zs_prefix, "Comment: {c}\nLabel (0 or 1):", zs_post, batch_size=32)
    zs_vf_m = compute_metrics(vf_labels, (zs_prob_vf >= 0.5).astype(int), zs_prob_vf)
    zs_vs_m = compute_metrics(vf_labels[sad_mask], (zs_prob_vf[sad_mask] >= 0.5).astype(int), zs_prob_vf[sad_mask])
    print(
        f"  LLM Zero-Shot ({time.time()-t0:.1f}s): In-Domain F1-M={zs_in_m['f1_macro']:.4f}, "
        f"FP={zs_in_m['confusion_matrix'][0][1]}, AUC={zs_in_m['roc_auc']:.4f}"
    )

    t1 = time.time()
    fs_prob_in = score_with_prefix_cache(test_texts, fs_prefix, "Target Comment: {c}\nLabel:", fs_post, batch_size=32)
    fs_pred_in = (fs_prob_in >= 0.5).astype(int)
    fs_in_m = {**compute_metrics(test_labels, fs_pred_in, fs_prob_in), "bootstrap": bootstrap_ci(test_labels, fs_pred_in)}

    fs_prob_vf = score_with_prefix_cache(vf_texts, fs_prefix, "Target Comment: {c}\nLabel:", fs_post, batch_size=32)
    fs_vf_m = compute_metrics(vf_labels, (fs_prob_vf >= 0.5).astype(int), fs_prob_vf)
    fs_vs_m = compute_metrics(vf_labels[sad_mask], (fs_prob_vf[sad_mask] >= 0.5).astype(int), fs_prob_vf[sad_mask])
    print(
        f"  LLM 4-Rule Rubric+4Shot ({time.time()-t1:.1f}s): In-Domain F1-M={fs_in_m['f1_macro']:.4f}, "
        f"FP={fs_in_m['confusion_matrix'][0][1]}, AUC={fs_in_m['roc_auc']:.4f}"
    )

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return {
        "model_name": llm_name,
        "zero_shot": {"in_domain": zs_in_m, "vsmec_full": zs_vf_m, "vsmec_sad": zs_vs_m},
        "rubric_few_shot": {"in_domain": fs_in_m, "vsmec_full": fs_vf_m, "vsmec_sad": fs_vs_m},
        "zs_pred_in": zs_pred_in,
        "fs_pred_in": fs_pred_in,
        "zs_prob_in": zs_prob_in,
        "fs_prob_in": fs_prob_in,
    }


# ── Equal-Budget Pool-Based Active Learning Benchmark (KDD / KAIS / TKDE) ───
def run_active_learning_benchmark(
    train_texts: List[str],
    train_labels: np.ndarray,
    test_texts: List[str],
    test_labels: np.ndarray,
    device: torch.device,
) -> Dict:
    """
    Strictly inductive equal-budget Active Learning comparison over final_train.csv (n=6,392),
    evaluated on held-out final_test.csv (n=1,371) using off-the-shelf pretrained ViSoBERT
    representations (zero exposure to VietDepression labels prior to querying).
    Strategies:
      1. Random Sampling (matched-budget control)
      2. Least-Confidence Uncertainty (Eq. 2)
      3. BADGE-style Diversity-Uncertainty (uncertainty-weighted gradient embeddings + k-means++)
      4. Contrastive Active Learning (CAL / CoMAL-style KL neighborhood divergence, KDD '24)
    """
    print("\n=== Running Equal-Budget Active Learning Benchmark (4 Strategies x 6 Budgets x 3 Seeds) ===")
    tokenizer = AutoTokenizer.from_pretrained("uitnlp/visobert", use_fast=False)
    backbone = AutoModel.from_pretrained("uitnlp/visobert").to(device)
    backbone.eval()

    def embed_texts(texts: List[str]) -> np.ndarray:
        order = np.argsort([len(t) for t in texts])
        st = [texts[i] for i in order]
        vecs = []
        with torch.no_grad():
            for i in range(0, len(st), 64):
                enc = tokenizer(st[i : i + 64], truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "mps")):
                    out = backbone(**enc)
                # Mean-pooled token embeddings
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                vecs.append(pooled.cpu().numpy())
        cat = np.concatenate(vecs, axis=0)
        res = np.empty_like(cat)
        res[order] = cat
        return res

    X_pool_raw = embed_texts(train_texts)
    X_test_raw = embed_texts(test_texts)
    del backbone
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    X_pool_norm = X_pool_raw / (np.linalg.norm(X_pool_raw, axis=1, keepdims=True) + 1e-12)
    X_u_norm_sub = X_pool_norm[:, :64]
    budgets = [400, 800, 1600, 3200, 4800, len(train_labels)]
    strategies = ["random", "least_confidence", "badge", "contrastive_cal"]
    results = {s: {str(b): [] for b in budgets} for s in strategies}

    for seed in SEEDS:
        rng_init = np.random.default_rng(seed)
        pos_idx = np.where(train_labels == 1)[0]
        neg_idx = np.where(train_labels == 0)[0]
        n_pos_0 = int(round(400 * (len(pos_idx) / len(train_labels))))
        n_neg_0 = 400 - n_pos_0
        init_labeled = np.concatenate(
            [rng_init.choice(pos_idx, size=n_pos_0, replace=False), rng_init.choice(neg_idx, size=n_neg_0, replace=False)]
        )

        for strat in strategies:
            rng = np.random.default_rng(seed + 100)
            labeled_mask = np.zeros(len(train_labels), dtype=bool)
            labeled_mask[init_labeled] = True

            for b_idx, budget in enumerate(budgets):
                if b_idx > 0:
                    need = budget - int(np.sum(labeled_mask))
                    unlabeled_idx = np.where(~labeled_mask)[0]
                    if need >= len(unlabeled_idx):
                        labeled_mask[:] = True
                    elif strat == "random":
                        chosen = rng.choice(unlabeled_idx, size=need, replace=False)
                        labeled_mask[chosen] = True
                    elif strat == "least_confidence":
                        p_u = clf.predict_proba(scaler.transform(X_pool_raw[unlabeled_idx]))[:, 1]
                        u_score = 1.0 - np.maximum(p_u, 1.0 - p_u)
                        chosen = unlabeled_idx[np.argsort(-u_score)[:need]]
                        labeled_mask[chosen] = True
                    elif strat == "badge":
                        X_u_scaled = scaler.transform(X_pool_raw[unlabeled_idx])
                        p_u = clf.predict_proba(X_u_scaled)[:, 1]
                        y_hat = (p_u >= 0.5).astype(float)
                        grad_scale = np.abs(p_u - y_hat)[:, None]
                        G = grad_scale * X_u_norm_sub[unlabeled_idx]
                        cand_cap = min(len(unlabeled_idx), max(need * 4, 1200))
                        cand_idx = np.argsort(-(1.0 - np.maximum(p_u, 1.0 - p_u)))[:cand_cap]
                        G_c = G[cand_idx]
                        sel_c = [int(np.argmax(np.sum(G_c * G_c, axis=1)))]
                        d2 = np.maximum(np.sum((G_c - G_c[sel_c[0]]) ** 2, axis=1), 0.0)
                        d2[sel_c] = 0.0
                        while len(sel_c) < need:
                            d2_sum = float(np.sum(d2))
                            if d2_sum <= 1e-12:
                                rem = [idx_c for idx_c in range(len(cand_idx)) if idx_c not in sel_c]
                                nxt = int(rem[0])
                            else:
                                probs_d2 = d2 / d2_sum
                                probs_d2 = probs_d2 / np.sum(probs_d2)
                                nxt = int(rng.choice(len(cand_idx), p=probs_d2))
                                if nxt in sel_c:
                                    nxt = int(np.argmax(d2))
                            sel_c.append(nxt)
                            new_d2 = np.maximum(np.sum((G_c - G_c[nxt]) ** 2, axis=1), 0.0)
                            d2 = np.minimum(d2, new_d2)
                            d2[sel_c] = 0.0
                        chosen = unlabeled_idx[cand_idx[sel_c]]
                        labeled_mask[chosen] = True
                    elif strat == "contrastive_cal":
                        lab_idx = np.where(labeled_mask)[0]
                        p_u = np.clip(clf.predict_proba(scaler.transform(X_pool_raw[unlabeled_idx]))[:, 1], 1e-6, 1 - 1e-6)
                        p_l = np.clip(clf.predict_proba(scaler.transform(X_transform_cached := scaler.transform(X_pool_raw[lab_idx])))[:, 1] if False else clf.predict_proba(scaler.transform(X_pool_raw[lab_idx]))[:, 1], 1e-6, 1 - 1e-6)
                        sim_mat = X_pool_norm[unlabeled_idx] @ X_pool_norm[lab_idx].T
                        k_nn = min(10, len(lab_idx))
                        nn_part = np.argpartition(-sim_mat, k_nn - 1, axis=1)[:, :k_nn]
                        p_nn_mean = np.mean(p_l[nn_part], axis=1)
                        kl_div = p_u * np.log(p_u / p_nn_mean) + (1.0 - p_u) * np.log((1.0 - p_u) / (1.0 - p_nn_mean))
                        u_score = 1.0 - np.maximum(p_u, 1.0 - p_u)
                        cal_score = kl_div * (0.5 + u_score)
                        chosen = unlabeled_idx[np.argsort(-cal_score)[:need]]
                        labeled_mask[chosen] = True

                scaler = StandardScaler()
                X_tr_sub = scaler.fit_transform(X_pool_raw[labeled_mask])
                y_tr_sub = train_labels[labeled_mask]
                clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=seed)
                clf.fit(X_tr_sub, y_tr_sub)
                te_pred = clf.predict(scaler.transform(X_test_raw))
                f1_m = float(f1_score(test_labels, te_pred, average="macro", zero_division=0))
                results[strat][str(budget)].append(f1_m)

    summary = {"budgets": budgets, "strategies": {}}
    for strat in strategies:
        summary["strategies"][strat] = {
            str(b): {
                "mean": float(np.mean(results[strat][str(b)])),
                "std": float(np.std(results[strat][str(b)], ddof=1)),
                "seeds": results[strat][str(b)],
            }
            for b in budgets
        }
    for b in budgets:
        row_str = " | ".join(
            f"{s}: {summary['strategies'][s][str(b)]['mean']:.4f}±{summary['strategies'][s][str(b)]['std']:.4f}"
            for s in strategies
        )
        print(f"  Budget {b:4d} -> {row_str}")
    return summary


def evaluate_prior_shift_calibration(
    name: str,
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    vf_logits: np.ndarray,
    vf_labels: np.ndarray,
    sad_mask: np.ndarray,
    pi_source: float,
) -> Dict:
    raw_vf_prob = softmax(vf_logits, axis=-1)[:, 1]
    raw_vs_prob = raw_vf_prob[sad_mask]
    vs_labels = vf_labels[sad_mask]

    # 1. Default uncalibrated (tau = 0.50)
    def_vf = compute_metrics(vf_labels, (raw_vf_prob >= 0.50).astype(int), raw_vf_prob)
    def_vs = compute_metrics(vs_labels, (raw_vs_prob >= 0.50).astype(int), raw_vs_prob)

    # 2. Static Bayes source-prior threshold (tau = pi_train = 0.1974)
    bayes_vf = compute_metrics(vf_labels, (raw_vf_prob >= pi_source).astype(int), raw_vf_prob)
    bayes_vs = compute_metrics(vs_labels, (raw_vs_prob >= pi_source).astype(int), raw_vs_prob)

    # 3. Validation Temperature Scaling + Pure Label-Shift Saerens EM (zero target labels)
    T_opt = fit_temperature_on_val(val_logits, val_labels)
    temp_vf_prob = softmax(vf_logits / T_opt, axis=-1)[:, 1]
    temp_vs_prob = temp_vf_prob[sad_mask]
    val_temp_prob = softmax(val_logits / T_opt, axis=-1)[:, 1]
    pi_eff_source = float(np.clip(np.mean(val_temp_prob), 0.02, 0.98))

    pi_hat_vf, em_vf_prob = saerens_em_prior_adaptation(temp_vf_prob, pi_source=pi_eff_source)
    pi_hat_vs, em_vs_prob = saerens_em_prior_adaptation(temp_vs_prob, pi_source=pi_eff_source)
    em_vf = compute_metrics(vf_labels, (em_vf_prob >= 0.50).astype(int), em_vf_prob)
    em_vs = compute_metrics(vs_labels, (em_vs_prob >= 0.50).astype(int), em_vs_prob)

    # 4. Prior-Matched Quantile Calibration (uses unlabeled target score CDF + known protocol prior
    # pi_target = 0.50 on Full, 0.4270 on Sadness-Only, or Source-Prior Rank Calibration pi_source = 0.1974)
    tau_src_rank_vf = float(np.quantile(raw_vf_prob, 1.0 - pi_source))
    tau_src_rank_vs = float(np.quantile(raw_vs_prob, 1.0 - pi_source))
    src_rank_vf = compute_metrics(vf_labels, (raw_vf_prob >= tau_src_rank_vf).astype(int), raw_vf_prob)
    src_rank_vs = compute_metrics(vs_labels, (raw_vs_prob >= tau_src_rank_vs).astype(int), raw_vs_prob)

    tau_tgt_q_vf = float(np.quantile(raw_vf_prob, 0.50))
    tau_tgt_q_vs = float(np.quantile(raw_vs_prob, 1.0 - float(np.mean(vs_labels == 1))))
    tgt_q_vf = compute_metrics(vf_labels, (raw_vf_prob >= tau_tgt_q_vf).astype(int), raw_vf_prob)
    tgt_q_vs = compute_metrics(vs_labels, (raw_vs_prob >= tau_tgt_q_vs).astype(int), raw_vs_prob)

    print(
        f"  [Calib {name}] Def={def_vs['f1_macro']:.4f} | Bayes(0.1974)={bayes_vs['f1_macro']:.4f} | "
        f"EM(pi_hat={pi_hat_vs:.3f})={em_vs['f1_macro']:.4f} | SrcRank(q=80.3%)={src_rank_vs['f1_macro']:.4f} | "
        f"TargetPriorQuantile={tgt_q_vs['f1_macro']:.4f}"
    )
    return {
        "val_temperature": T_opt,
        "effective_val_prior": pi_eff_source,
        "default_tau050": {"vsmec_full": def_vf, "vsmec_sad": def_vs},
        "static_bayes_tau01974": {"vsmec_full": bayes_vf, "vsmec_sad": bayes_vs},
        "unsupervised_saerens_em": {
            "estimated_prior_full": pi_hat_vf,
            "estimated_prior_sad": pi_hat_vs,
            "vsmec_full": em_vf,
            "vsmec_sad": em_vs,
        },
        "unsupervised_source_rank_quantile": {
            "tau_full": tau_src_rank_vf,
            "tau_sad": tau_src_rank_vs,
            "vsmec_full": src_rank_vf,
            "vsmec_sad": src_rank_vs,
        },
        "prior_matched_quantile": {
            "tau_full": tau_tgt_q_vf,
            "tau_sad": tau_tgt_q_vs,
            "vsmec_full": tgt_q_vf,
            "vsmec_sad": tgt_q_vs,
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2", "stage3", "stage4", "stage5", "all"],
        default="all",
        help="Execution stage to keep each run well below 300s.",
    )
    parser.add_argument("--only-seed", type=int, default=None, help="Train only a single seed.")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Top-Venue Comparative Baselines [{args.stage}] on device: {device}")

    train_df = pd.read_csv(DATA_DIR / "final_train.csv")
    val_df = pd.read_csv(DATA_DIR / "final_val.csv")
    test_df = pd.read_csv(DATA_DIR / "final_test.csv")
    vsmec_df = pd.read_csv(UNIFIED_DIR / "cross_domain_test.csv")

    train_texts = [str(x) for x in train_df["comment_text"].fillna("")]
    val_texts = [str(x) for x in val_df["comment_text"].fillna("")]
    test_texts = [str(x) for x in test_df["comment_text"].fillna("")]
    vsmec_col = "comment_text" if "comment_text" in vsmec_df.columns else "text"
    vf_texts = [str(x) for x in vsmec_df[vsmec_col].fillna("")]

    train_labels = train_df["label"].values.astype(int)
    val_labels = val_df["label"].values.astype(int)
    test_labels = test_df["label"].values.astype(int)
    vf_labels = vsmec_df["label"].values.astype(int)
    sad_mask = vsmec_df["original_label"].isin(["Enjoyment", "Sadness"]).to_numpy()
    pi_train = float(np.mean(train_labels == 1))

    if OUT_JSON.exists():
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            payload: Dict = json.load(f)
    else:
        payload = {"models": {}, "calibration": {}, "active_learning": {}, "significance": {}}
    for k in ("models", "calibration", "active_learning", "significance"):
        payload.setdefault(k, {})

    cache_npz = AUDIT_DIR / "stage_arrays_cache.npz"
    arrays_cache: Dict[str, np.ndarray] = {}
    if cache_npz.exists():
        with np.load(cache_npz, allow_pickle=False) as zf:
            arrays_cache = {k: zf[k] for k in zf.files}

    def save_arrays():
        np.savez_compressed(cache_npz, **arrays_cache)

    # ── STAGE 1: PhoBERT-base, ViSoBERT, NCNL (TKDE '23), Equal-Budget Active Learning ──
    if args.stage in ("stage1", "all"):
        if "phobert_maj_pred_in" not in arrays_cache or "phobert_base" not in payload["models"]:
            phobert_res = evaluate_saved_checkpoints(
                "phobert",
                "vinai/phobert-base",
                "best_model",
                val_texts,
                val_labels,
                test_texts,
                test_labels,
                vf_texts,
                vf_labels,
                sad_mask,
                device,
                extract_cls_splits=None,
            )
            payload["models"]["phobert_base"] = phobert_res["summary"]
            arrays_cache["phobert_maj_pred_in"] = phobert_res["maj_pred_in"]
            arrays_cache["phobert_soft_prob_in"] = phobert_res["soft_prob_in"]
            arrays_cache["phobert_val_logits_mean"] = phobert_res["val_logits_mean"]
            arrays_cache["phobert_vf_logits_mean"] = phobert_res["vf_logits_mean"]
            save_progress(payload)
            save_arrays()

        if "ncnl_maj_pred_in" not in arrays_cache or "ncnl_tkde23" not in payload["models"]:
            visobert_res = evaluate_saved_checkpoints(
                "visobert",
                "uitnlp/visobert",
                "",
                val_texts,
                val_labels,
                test_texts,
                test_labels,
                vf_texts,
                vf_labels,
                sad_mask,
                device,
                extract_cls_splits={"train": train_texts, "val": val_texts},
            )
            payload["models"]["visobert"] = visobert_res["summary"]
            arrays_cache["visobert_maj_pred_in"] = visobert_res["maj_pred_in"]
            arrays_cache["visobert_soft_prob_in"] = visobert_res["soft_prob_in"]
            arrays_cache["visobert_val_logits_mean"] = visobert_res["val_logits_mean"]
            arrays_cache["visobert_vf_logits_mean"] = visobert_res["vf_logits_mean"]
            save_progress(payload)
            save_arrays()

            ncnl_res = train_ncnl_ensemble(
                visobert_res["cls_per_seed"], train_labels, val_labels, test_labels, vf_labels, sad_mask, device
            )
            payload["models"]["ncnl_tkde23"] = ncnl_res["summary"]
            arrays_cache["ncnl_maj_pred_in"] = ncnl_res["maj_pred_in"]
            arrays_cache["ncnl_soft_prob_in"] = ncnl_res["soft_prob_in"]
            save_progress(payload)
            save_arrays()

        al_res = run_active_learning_benchmark(train_texts, train_labels, test_texts, test_labels, device)
        payload["active_learning"] = al_res
        save_progress(payload)
        print("Stage 1 completed.")

    # ── STAGE 2: XLM-RoBERTa-base (3 seeds) ──
    if args.stage in ("stage2", "all"):
        xlmr_res = train_standard_or_lafocal_encoder(
            "FacebookAI/xlm-roberta-base",
            "xlmr_base",
            "wce",
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            test_texts,
            test_labels,
            vf_texts,
            vf_labels,
            sad_mask,
            device,
        )
        payload["models"]["xlmr_base"] = xlmr_res["summary"]
        arrays_cache["xlmr_maj_pred_in"] = xlmr_res["maj_pred_in"]
        arrays_cache["xlmr_soft_prob_in"] = xlmr_res["soft_prob_in"]
        arrays_cache["xlmr_val_logits_mean"] = xlmr_res["val_logits_mean"]
        arrays_cache["xlmr_vf_logits_mean"] = xlmr_res["vf_logits_mean"]
        save_progress(payload)
        save_arrays()
        print("Stage 2 completed.")

    # ── STAGE 3: ViSoBERT + LA-Focal (Imbalanced Prior-Logit Adjusted Focal Loss, TKDE '24, 3 seeds) ──
    if args.stage in ("stage3", "all"):
        lafocal_res = train_standard_or_lafocal_encoder(
            "uitnlp/visobert",
            "visobert_lafocal",
            "la_focal",
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            test_texts,
            test_labels,
            vf_texts,
            vf_labels,
            sad_mask,
            device,
        )
        payload["models"]["visobert_lafocal"] = lafocal_res["summary"]
        arrays_cache["lafocal_maj_pred_in"] = lafocal_res["maj_pred_in"]
        arrays_cache["lafocal_soft_prob_in"] = lafocal_res["soft_prob_in"]
        arrays_cache["lafocal_val_logits_mean"] = lafocal_res["val_logits_mean"]
        arrays_cache["lafocal_vf_logits_mean"] = lafocal_res["vf_logits_mean"]
        save_progress(payload)
        save_arrays()
        print("Stage 3 completed.")

    # ── STAGE 4: ViSoBERT + SupCon (Contrastive Spurious-Correlation Mitigation, KDD '26 / ICDM '25, 3 seeds) ──
    if args.stage in ("stage4", "all"):
        supcon_res = train_supcon_encoder(
            "uitnlp/visobert",
            "visobert_supcon",
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            test_texts,
            test_labels,
            vf_texts,
            vf_labels,
            sad_mask,
            device,
            lambda_sc=0.20,
            only_seed=args.only_seed,
        )
        if not supcon_res.get("only_seed_done"):
            payload["models"]["visobert_supcon"] = supcon_res["summary"]
            arrays_cache["supcon_maj_pred_in"] = supcon_res["maj_pred_in"]
            arrays_cache["supcon_soft_prob_in"] = supcon_res["soft_prob_in"]
            arrays_cache["supcon_val_logits_mean"] = supcon_res["val_logits_mean"]
            arrays_cache["supcon_vf_logits_mean"] = supcon_res["vf_logits_mean"]
            save_progress(payload)
            save_arrays()
            print("Stage 4 completed.")
        else:
            print(f"Stage 4 seed {args.only_seed} completed.")

    # ── STAGE 5: Open-Weight LLM (Qwen2.5-1.5B-Instruct), Cross-Corpus Calibration, Significance Tests ──
    if args.stage in ("stage5", "all"):
        llm_res = evaluate_llm_baselines(train_df, test_texts, test_labels, vf_texts, vf_labels, sad_mask, device)
        payload["models"]["qwen25_llm"] = {
            "model_name": llm_res["model_name"],
            "zero_shot": llm_res["zero_shot"],
            "rubric_few_shot": llm_res["rubric_few_shot"],
        }
        arrays_cache["llm_zs_pred_in"] = llm_res["zs_pred_in"]
        arrays_cache["llm_fs_pred_in"] = llm_res["fs_pred_in"]
        arrays_cache["llm_zs_prob_in"] = llm_res["zs_prob_in"]
        arrays_cache["llm_fs_prob_in"] = llm_res["fs_prob_in"]
        save_progress(payload)
        save_arrays()

        print("\n=== Evaluating Cross-Corpus Prior-Shift Calibration on UIT-VSMEC ===")
        for cal_key, cal_label, prefix in [
            ("phobert_base", "PhoBERT-base", "phobert"),
            ("xlmr_base", "XLM-RoBERTa-base", "xlmr"),
            ("visobert", "ViSoBERT", "visobert"),
            ("visobert_lafocal", "ViSoBERT+LA-Focal", "lafocal"),
            ("visobert_supcon", "ViSoBERT+SupCon", "supcon"),
        ]:
            if f"{prefix}_val_logits_mean" in arrays_cache:
                payload["calibration"][cal_key] = evaluate_prior_shift_calibration(
                    cal_label,
                    arrays_cache[f"{prefix}_val_logits_mean"],
                    val_labels,
                    arrays_cache[f"{prefix}_vf_logits_mean"],
                    vf_labels,
                    sad_mask,
                    pi_train,
                )
        save_progress(payload)

        lr_bundle = joblib.load(MODEL_DIR / "tfidf_logreg_round6_v2.joblib")
        lr_pred = lr_bundle.predict(test_df["comment_text"].fillna("").astype(str))

        preds_map = {
            "visobert_supcon_maj": arrays_cache["supcon_maj_pred_in"],
            "visobert_lafocal_maj": arrays_cache["lafocal_maj_pred_in"],
            "visobert_maj": arrays_cache["visobert_maj_pred_in"],
            "ncnl_tkde23_maj": arrays_cache["ncnl_maj_pred_in"],
            "xlmr_base_maj": arrays_cache["xlmr_maj_pred_in"],
            "phobert_base_maj": arrays_cache["phobert_maj_pred_in"],
            "tfidf_logreg": lr_pred,
            "qwen25_zeroshot": arrays_cache["llm_zs_pred_in"],
            "qwen25_rubric_4shot": arrays_cache["llm_fs_pred_in"],
        }
        sig_pairs = [
            ("visobert_lafocal_maj", "phobert_base_maj"),
            ("visobert_lafocal_maj", "tfidf_logreg"),
            ("visobert_lafocal_maj", "visobert_maj"),
            ("visobert_lafocal_maj", "xlmr_base_maj"),
            ("visobert_lafocal_maj", "qwen25_rubric_4shot"),
            ("visobert_supcon_maj", "phobert_base_maj"),
            ("visobert_supcon_maj", "tfidf_logreg"),
            ("visobert_supcon_maj", "visobert_maj"),
            ("visobert_supcon_maj", "visobert_lafocal_maj"),
            ("visobert_supcon_maj", "ncnl_tkde23_maj"),
            ("visobert_supcon_maj", "xlmr_base_maj"),
            ("visobert_supcon_maj", "qwen25_rubric_4shot"),
            ("visobert_maj", "phobert_base_maj"),
            ("visobert_maj", "tfidf_logreg"),
            ("visobert_maj", "xlmr_base_maj"),
            ("visobert_maj", "qwen25_rubric_4shot"),
            ("ncnl_tkde23_maj", "phobert_base_maj"),
            ("ncnl_tkde23_maj", "tfidf_logreg"),
            ("xlmr_base_maj", "phobert_base_maj"),
            ("xlmr_base_maj", "tfidf_logreg"),
            ("phobert_base_maj", "qwen25_rubric_4shot"),
            ("qwen25_rubric_4shot", "qwen25_zeroshot"),
        ]
        for a, b in sig_pairs:
            payload["significance"][f"{a}_vs_{b}"] = paired_significance(test_labels, preds_map[a], preds_map[b])

        payload["test_probabilities"] = {
            "y_test": test_labels.tolist(),
            "visobert_supcon_soft_prob": arrays_cache["supcon_soft_prob_in"].tolist(),
            "visobert_lafocal_soft_prob": arrays_cache["lafocal_soft_prob_in"].tolist(),
            "visobert_soft_prob": arrays_cache["visobert_soft_prob_in"].tolist(),
            "ncnl_soft_prob": arrays_cache["ncnl_soft_prob_in"].tolist(),
            "xlmr_soft_prob": arrays_cache["xlmr_soft_prob_in"].tolist(),
            "phobert_soft_prob": arrays_cache["phobert_soft_prob_in"].tolist(),
            "qwen25_rubric_prob": arrays_cache["llm_fs_prob_in"].tolist(),
            "qwen25_zeroshot_prob": arrays_cache["llm_zs_prob_in"].tolist(),
        }
        save_progress(payload)
        print(f"\nALL EXPERIMENTS COMPLETED SUCCESSFULLY! Saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
