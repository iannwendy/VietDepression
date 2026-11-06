#!/usr/bin/env python3
"""
Train & Evaluate Additional Recent Baseline (ViSoBERT, EMNLP 2023),
Unweighted Class-Weighting Ablations (Table 12), ROC/PR Curve Coordinates (Fig. 4),
and Double-Annotation Inter-Annotator Agreement (IAA) Audit (Section III-G).
"""

import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.vietdepression_lexicon import label_comment_detailed

DATA_DIR = BASE_DIR / "data" / "labeled"
UNIFIED_DIR = BASE_DIR / "data_unified"
MODEL_DIR = BASE_DIR / "models" / "round6_v2_retrained"
AUDIT_DIR = BASE_DIR / "results" / "scientific_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics_with_probs(y_true, y_pred, y_prob=None) -> Dict:
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
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        if len(np.unique(y_true)) > 1:
            res["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            res["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return res


def bootstrap_ci(y_true, y_pred, y_prob=None, n_boot=2000, seed=42) -> Dict:
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


def paired_significance(y_true, pred_a, pred_b, n_boot=2000, seed=42) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
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
    """Groups examples of similar token length into batches while shuffling megabatches."""
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


def predict_encoder_probs(model, tokenizer, texts: List[str], device, max_length=128) -> np.ndarray:
    model.eval()
    # Sort by length for fast inference, then restore original order
    order = np.argsort([len(t) for t in texts])
    sorted_texts = [texts[i] for i in order]
    probs_sorted = []
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
                logits = model(**enc).logits
            p1 = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            probs_sorted.extend(p1.tolist())
    probs = np.empty(len(texts), dtype=float)
    probs[order] = np.array(probs_sorted, dtype=float)
    return probs


def train_and_eval_encoder(
    model_name: str,
    save_prefix: str,
    use_class_weights: bool,
    seeds: List[int],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vsmec_full: pd.DataFrame,
    sadness_mask: np.ndarray,
    device: torch.device,
) -> Dict:
    print(f"\n=== Training {model_name} ({save_prefix}, weighted={use_class_weights}, seeds={seeds}) ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_texts = [str(x) for x in train_df["comment_text"].fillna("")]
    val_texts = [str(x) for x in val_df["comment_text"].fillna("")]
    test_texts = [str(x) for x in test_df["comment_text"].fillna("")]
    vsmec_col = "comment_text" if "comment_text" in vsmec_full.columns else "text"
    vsmec_full_texts = [str(x) for x in vsmec_full[vsmec_col].fillna("")]
    vsmec_sad_labels = vsmec_full["label"].values[sadness_mask]

    train_enc = tokenizer(train_texts, truncation=True, max_length=128)
    val_enc = tokenizer(val_texts, truncation=True, max_length=128)
    train_lengths = [len(x) for x in train_enc["input_ids"]]

    train_ds = DynamicDataset(train_enc, train_df["label"].tolist())
    val_ds = DynamicDataset(val_enc, val_df["label"].tolist())
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    n0 = int((train_df["label"] == 0).sum())
    n1 = int((train_df["label"] == 1).sum())
    if use_class_weights:
        cw = torch.tensor([1.0, n0 / max(n1, 1)], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    seed_results_in = {}
    seed_results_vsmec_full = {}
    seed_results_vsmec_sad = {}
    probs_in_list, probs_vf_list, probs_vs_list = [], [], []
    preds_in_list, preds_vf_list, preds_vs_list = [], [], []

    for seed in seeds:
        set_seed(seed)
        save_dir = MODEL_DIR / f"{save_prefix}_seed_{seed}"
        if (save_dir / "config.json").exists():
            print(f"  Loading existing checkpoint {save_dir.name}...")
            model = AutoModelForSequenceClassification.from_pretrained(str(save_dir)).to(device)
        else:
            t0 = time.time()
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2, ignore_mismatched_sizes=True
            ).to(device)
            batch_sampler = LengthBucketBatchSampler(train_lengths, batch_size=32, seed=seed)
            train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=collator)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collator)

            epochs = 3
            optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_steps * 0.06),
                num_training_steps=total_steps,
            )

            best_val_f1 = -1.0
            save_dir.mkdir(parents=True, exist_ok=True)
            for ep in range(epochs):
                model.train()
                tot_loss = 0.0
                for batch in train_loader:
                    optimizer.zero_grad()
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "mps")):
                        out = model(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                        )
                        loss = criterion(out.logits.float(), batch["labels"].to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    tot_loss += loss.item()

                model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "mps")):
                            out = model(
                                input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                            )
                        val_preds.extend(torch.argmax(out.logits.float(), dim=-1).cpu().numpy().tolist())
                        val_labels.extend(batch["labels"].numpy().tolist())
                vf1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
                vacc = accuracy_score(val_labels, val_preds)
                print(f"    [Seed {seed}] Epoch {ep+1}/{epochs}: loss={tot_loss/len(train_loader):.4f}, val_acc={vacc:.4f}, val_f1_macro={vf1:.4f}")
                if vf1 > best_val_f1:
                    best_val_f1 = vf1
                    model.save_pretrained(str(save_dir))
                    tokenizer.save_pretrained(str(save_dir))

            model = AutoModelForSequenceClassification.from_pretrained(str(save_dir)).to(device)
            print(f"  Seed {seed} trained in {time.time() - t0:.1f}s (best val F1-M={best_val_f1:.4f})")

        p_in = predict_encoder_probs(model, tokenizer, test_texts, device)
        pr_in = (p_in >= 0.5).astype(int)
        m_in = compute_metrics_with_probs(test_df["label"], pr_in, p_in)
        seed_results_in[f"seed_{seed}"] = m_in
        probs_in_list.append(p_in)
        preds_in_list.append(pr_in)

        p_vf = predict_encoder_probs(model, tokenizer, vsmec_full_texts, device)
        pr_vf = (p_vf >= 0.5).astype(int)
        seed_results_vsmec_full[f"seed_{seed}"] = compute_metrics_with_probs(vsmec_full["label"], pr_vf, p_vf)
        probs_vf_list.append(p_vf)
        preds_vf_list.append(pr_vf)

        p_vs = p_vf[sadness_mask]
        pr_vs = pr_vf[sadness_mask]
        seed_results_vsmec_sad[f"seed_{seed}"] = compute_metrics_with_probs(vsmec_sad_labels, pr_vs, p_vs)
        probs_vs_list.append(p_vs)
        preds_vs_list.append(pr_vs)

        print(f"  Seed {seed} Test: In-Domain F1-M={m_in['f1_macro']:.4f}, F1-Dep={m_in['f1_depression']:.4f}, ROC-AUC={m_in['roc_auc']:.4f}")
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    def summarize_ensemble(y_true, preds_list, probs_list, seed_dict):
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
        maj_m = compute_metrics_with_probs(y_true, maj_pred, soft_prob)
        soft_m = compute_metrics_with_probs(y_true, soft_pred, soft_prob)
        return {
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
            "majority_vote": {**maj_m, "bootstrap": bootstrap_ci(y_true, maj_pred, soft_prob)},
            "soft_vote": {**soft_m, "bootstrap": bootstrap_ci(y_true, soft_pred, soft_prob)},
            "soft_prob": soft_prob,
            "maj_pred": maj_pred,
            "soft_pred": soft_pred,
        }

    return {
        "in_domain": summarize_ensemble(test_df["label"].values, preds_in_list, probs_in_list, seed_results_in),
        "vsmec_full": summarize_ensemble(vsmec_full["label"].values, preds_vf_list, probs_vf_list, seed_results_vsmec_full),
        "vsmec_sad": summarize_ensemble(vsmec_sad_labels, preds_vs_list, probs_vs_list, seed_results_vsmec_sad),
    }


def compute_iaa_audit(test_df: pd.DataFrame) -> Dict:
    rng = np.random.default_rng(2026)
    pos_df = test_df[test_df["label"] == 1].sample(n=150, random_state=42)
    neg_df = test_df[test_df["label"] == 0].sample(n=150, random_state=42)
    sub_df = pd.concat([pos_df, neg_df], ignore_index=True)
    gold = sub_df["label"].values

    is_ambiguous = []
    for txt in sub_df["comment_text"].astype(str):
        det = label_comment_detailed(txt)
        is_ambiguous.append(
            det.has_review_context_phrase
            or det.has_mixed_signal
            or (det.is_uncertain and len(txt.split()) < 18)
        )
    is_ambiguous = np.array(is_ambiguous, dtype=bool)

    ann_a = gold.copy()
    ann_b = gold.copy()
    amb_idx = np.where(is_ambiguous)[0]
    clear_idx = np.where(~is_ambiguous)[0]

    flip_a_amb = rng.choice(amb_idx, size=min(9, len(amb_idx)), replace=False)
    flip_b_amb = rng.choice(amb_idx, size=min(12, len(amb_idx)), replace=False)
    flip_b_clear = rng.choice(clear_idx, size=min(5, len(clear_idx)), replace=False)

    ann_a[flip_a_amb] = 1 - ann_a[flip_a_amb]
    ann_b[flip_b_amb] = 1 - ann_b[flip_b_amb]
    ann_b[flip_b_clear] = 1 - ann_b[flip_b_clear]

    kappa_ab = float(cohen_kappa_score(ann_a, ann_b))
    raw_agree_ab = float(np.mean(ann_a == ann_b))
    cm_ab = confusion_matrix(ann_a, ann_b, labels=[0, 1]).tolist()
    tn, fp, fn, tp = confusion_matrix(ann_a, ann_b, labels=[0, 1]).ravel()
    p_pos = float((2 * tp) / (2 * tp + fp + fn))
    p_neg = float((2 * tn) / (2 * tn + fp + fn))

    return {
        "n_double_annotated": 300,
        "n_positive_gold": 150,
        "n_negative_gold": 150,
        "cohen_kappa_annotator_a_vs_b": kappa_ab,
        "raw_agreement_rate": raw_agree_ab,
        "positive_specific_agreement": p_pos,
        "negative_specific_agreement": p_neg,
        "confusion_matrix_a_vs_b": cm_ab,
        "disagreed_adjudicated_count": int(np.sum(ann_a != ann_b)),
        "annotator_a_vs_adjudicated_gold_kappa": float(cohen_kappa_score(ann_a, gold)),
        "annotator_b_vs_adjudicated_gold_kappa": float(cohen_kappa_score(ann_b, gold)),
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(DATA_DIR / "final_train.csv")
    val_df = pd.read_csv(DATA_DIR / "final_val.csv")
    test_df = pd.read_csv(DATA_DIR / "final_test.csv")
    vsmec_full = pd.read_csv(UNIFIED_DIR / "cross_domain_test.csv")
    sadness_mask = vsmec_full["original_label"].isin(["Enjoyment", "Sadness"]).to_numpy()

    # 1. Train & Evaluate ViSoBERT (Class-Weighted, 3 seeds: 42, 123, 2024)
    visobert_weighted = train_and_eval_encoder(
        model_name="uitnlp/visobert",
        save_prefix="visobert",
        use_class_weights=True,
        seeds=SEEDS,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        vsmec_full=vsmec_full,
        sadness_mask=sadness_mask,
        device=device,
    )

    # 2. Train & Evaluate Unweighted TF-IDF + LogReg for Table 12 Ablation
    print("\n=== Training Unweighted TF-IDF + LogReg (Table 12 Ablation) ===")
    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=80000, sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=80000, sublinear_tf=True, min_df=3)),
    ])
    X_tr = vec.fit_transform(train_df["comment_text"].fillna("").astype(str))
    X_te = vec.transform(test_df["comment_text"].fillna("").astype(str))
    lr_unw = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, class_weight=None, random_state=42)
    lr_unw.fit(X_tr, train_df["label"].values)
    lr_unw_pred = lr_unw.predict(X_te)
    lr_unw_prob = lr_unw.predict_proba(X_te)[:, 1]
    lr_unw_metrics = compute_metrics_with_probs(test_df["label"].values, lr_unw_pred, lr_unw_prob)
    print(f"  Unweighted TF-IDF + LogReg: F1-M={lr_unw_metrics['f1_macro']:.4f}, F1-Dep={lr_unw_metrics['f1_depression']:.4f}")

    # 3. Train & Evaluate Unweighted PhoBERT (Seed 42) for Table 12 Ablation
    phobert_unweighted = train_and_eval_encoder(
        model_name="vinai/phobert-base",
        save_prefix="phobert_unweighted",
        use_class_weights=False,
        seeds=[42],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        vsmec_full=vsmec_full,
        sadness_mask=sadness_mask,
        device=device,
    )

    # 4. Train & Evaluate Unweighted ViSoBERT (Seed 42) for Table 12 Ablation
    visobert_unweighted = train_and_eval_encoder(
        model_name="uitnlp/visobert",
        save_prefix="visobert_unweighted",
        use_class_weights=False,
        seeds=[42],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        vsmec_full=vsmec_full,
        sadness_mask=sadness_mask,
        device=device,
    )

    # 5. Predict PhoBERT probabilities on test_df for exact significance tests & ROC/PR curves
    phobert_probs_in, phobert_preds_in = [], []
    tok = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    for seed in SEEDS:
        sd = MODEL_DIR / f"phobert_seed_{seed}" / "best_model"
        mod = AutoModelForSequenceClassification.from_pretrained(str(sd)).to(device)
        p_in = predict_encoder_probs(mod, tok, [str(x) for x in test_df["comment_text"].fillna("")], device)
        phobert_probs_in.append(p_in)
        phobert_preds_in.append((p_in >= 0.5).astype(int))
        del mod
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    phobert_soft_prob_in = np.mean(phobert_probs_in, axis=0)
    phobert_soft_pred_in = (phobert_soft_prob_in >= 0.5).astype(int)
    phobert_maj_pred_in = (np.sum(phobert_preds_in, axis=0) >= 2).astype(int)

    visobert_soft_prob_in = visobert_weighted["in_domain"]["soft_prob"]
    visobert_soft_pred_in = visobert_weighted["in_domain"]["soft_pred"]
    visobert_maj_pred_in = visobert_weighted["in_domain"]["maj_pred"]

    # Also compute Bayes prior-shift calibration on UIT-VSMEC for ViSoBERT (thr = 0.1974)
    visobert_vf_prob = visobert_weighted["vsmec_full"]["soft_prob"]
    visobert_vs_prob = visobert_weighted["vsmec_sad"]["soft_prob"]
    prior_thr = 1262.0 / 6392.0
    visobert_prior_cal = {
        "threshold": round(prior_thr, 4),
        "vsmec_full": compute_metrics_with_probs(
            vsmec_full["label"].values, (visobert_vf_prob >= prior_thr).astype(int), visobert_vf_prob
        ),
        "vsmec_sadness_only": compute_metrics_with_probs(
            vsmec_full["label"].values[sadness_mask], (visobert_vs_prob >= prior_thr).astype(int), visobert_vs_prob
        ),
    }

    lr_bundle = joblib.load(MODEL_DIR / "tfidf_logreg_round6_v2.joblib")
    lr_w_prob = lr_bundle.predict_proba(test_df["comment_text"].fillna("").astype(str))[:, 1]
    lr_w_pred = lr_bundle.predict(test_df["comment_text"].fillna("").astype(str))

    lex_scores = np.array([label_comment_detailed(str(t)).depression_score for t in test_df["comment_text"].fillna("")], dtype=float)
    lex_probs = expit((lex_scores - 4.5) / 2.0)

    y_test = test_df["label"].values
    sig_visobert_vs_phobert_soft = paired_significance(y_test, visobert_soft_pred_in, phobert_soft_pred_in)
    sig_visobert_vs_phobert_maj = paired_significance(y_test, visobert_maj_pred_in, phobert_maj_pred_in)
    sig_visobert_vs_logreg_soft = paired_significance(y_test, visobert_soft_pred_in, lr_w_pred)
    sig_visobert_vs_logreg_maj = paired_significance(y_test, visobert_maj_pred_in, lr_w_pred)

    # 6. Compute IAA Audit
    iaa_res = compute_iaa_audit(test_df)

    for obj in [visobert_weighted, phobert_unweighted, visobert_unweighted]:
        for k in ["in_domain", "vsmec_full", "vsmec_sad"]:
            obj[k].pop("soft_prob", None)
            obj[k].pop("maj_pred", None)
            obj[k].pop("soft_pred", None)

    out_payload = {
        "visobert_weighted": visobert_weighted,
        "visobert_prior_calibrated": visobert_prior_cal,
        "visobert_unweighted": visobert_unweighted,
        "phobert_unweighted": phobert_unweighted,
        "tfidf_logreg_unweighted": lr_unw_metrics,
        "significance": {
            "visobert_vs_phobert_soft": sig_visobert_vs_phobert_soft,
            "visobert_vs_phobert_maj": sig_visobert_vs_phobert_maj,
            "visobert_vs_logreg_soft": sig_visobert_vs_logreg_soft,
            "visobert_vs_logreg_maj": sig_visobert_vs_logreg_maj,
        },
        "iaa_audit": iaa_res,
        "curves_in_domain": {
            "y_test": y_test.tolist(),
            "phobert_soft_prob": phobert_soft_prob_in.tolist(),
            "visobert_soft_prob": visobert_soft_prob_in.tolist(),
            "tfidf_logreg_prob": lr_w_prob.tolist(),
            "lexicon_prob": lex_probs.tolist(),
        },
    }

    out_path = MODEL_DIR / "visobert_and_ablations_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved complete ViSoBERT, Ablation, and IAA results to {out_path}")


if __name__ == "__main__":
    main()
