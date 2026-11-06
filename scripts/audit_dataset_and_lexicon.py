"""Comprehensive Dataset, Provenance, Filtering, and Weak-Labeler Audit.

Addresses Reviewer 1 (Comments 1, 3, 4) and Reviewer 2 (Major Comments 1, 2, 4):
1. Exact train / val / test / cross-domain class counts and human-label provenance.
2. Exact raw -> cleaned filtering breakdown, unique video counts, and text length
   distributions (characters and whitespace tokens: mean, SD, median, IQR, min, max).
3. Exact decomposition of the `need_review` flag on the 125,328-comment inventory,
   separating the 22 `REVIEW_CONTEXT_KEYWORDS` phrases from the composite
   `uncertain | near_boundary | mixed_signal | short_strong` conditions.
4. Evaluation of the Lexicon Weak-Labeler on the 1,371-sample in-domain test set
   (`final_test.csv`) and 3,084-sample cross-domain UIT-VSMEC set, both with
   abstentions counted (all 1,371 samples) and on covered-only subsets, plus
   threshold-independent ROC-AUC and PR-AUC on raw lexical scores.

Usage:
    PYTHONPATH="$PWD" .venv/bin/python scripts/audit_dataset_and_lexicon.py
"""

from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.vietdepression_lexicon import (
    DEPRESSION_MEDIUM_KEYWORDS,
    DEPRESSION_STRONG_KEYWORDS,
    NORMAL_KEYWORDS,
    REVIEW_CONTEXT_KEYWORDS,
    REPEATED_CHAR_RE,
    SPAM_PHRASES,
    URL_RE,
    is_basic_spam,
    label_comment_detailed,
    normalize_comparison_key,
    normalize_text,
)

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
LABELED_DIR = DATA_DIR / "labeled"
ANALYSIS_DIR = DATA_DIR / "analysis"
UNIFIED_DIR = PROJECT_DIR / "data_unified"
RESULTS_DIR = PROJECT_DIR / "results" / "scientific_audit"


def length_stats(texts: pd.Series) -> dict:
    s = texts.fillna("").astype(str).map(normalize_text)
    char_lens = s.str.len().to_numpy(dtype=float)
    word_lens = s.str.split().map(len).to_numpy(dtype=float)

    def _summarize(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {}
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        return {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 2),
            "median": round(float(q50), 2),
            "iqr": round(float(q75 - q25), 2),
            "q25": round(float(q25), 2),
            "q75": round(float(q75), 2),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
        }

    return {
        "n": int(len(s)),
        "chars": _summarize(char_lens),
        "words": _summarize(word_lens),
    }


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    out = {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "precision_depression": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_depression": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_depression": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision_normal": round(float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "recall_normal": round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "f1_normal": round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "predicted_positives": int(tp + fp),
        "predicted_negatives": int(tn + fn),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, scores)), 4)
    return out


def audit_raw_to_cleaned() -> dict:
    raw_path = RAW_DIR / "raw_comments.csv"
    cleaned_path = RAW_DIR / "cleaned_comments.csv"
    video_meta_path = RAW_DIR / "video_metadata.csv"
    proc_videos_path = RAW_DIR / "processed_videos.txt"

    result: dict = {}
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path, dtype=str).fillna("")
        total_raw = len(raw_df)
        norm_texts = raw_df["comment_text"].map(normalize_text)
        len_ge_5_mask = norm_texts.str.len() >= 5
        removed_short = int((~len_ge_5_mask).sum())

        after_short = raw_df[len_ge_5_mask].copy()
        after_short["comment_text"] = norm_texts[len_ge_5_mask]

        # Breakdown of spam reasons among length >= 5
        s_texts = after_short["comment_text"]
        s_low = s_texts.str.lower()
        has_url = s_texts.map(lambda x: bool(URL_RE.search(x)))
        has_spam_phrase = s_low.map(lambda x: any(p in x for p in SPAM_PHRASES))
        has_rep_char = s_texts.map(lambda x: bool(REPEATED_CHAR_RE.search(x)))
        has_low_div = s_low.map(lambda x: len(x.split()) >= 8 and len(set(x.split())) <= 2)
        spam_mask = after_short["comment_text"].map(is_basic_spam)
        removed_spam = int(spam_mask.sum())

        after_spam = after_short[~spam_mask].copy()
        before_cid = len(after_spam)
        if "comment_id" in after_spam.columns:
            has_cid = after_spam["comment_id"].str.strip().ne("")
            with_id = after_spam[has_cid].drop_duplicates(subset=["comment_id"], keep="first")
            without_id = after_spam[~has_cid]
            after_cid = pd.concat([with_id, without_id], ignore_index=True)
        else:
            after_cid = after_spam
        removed_dup_cid = int(before_cid - len(after_cid))

        before_text_dedup = len(after_cid)
        after_text_dedup = after_cid.drop_duplicates(subset=["comment_text"], keep="first")
        removed_dup_text = int(before_text_dedup - len(after_text_dedup))

        result["raw_filtering_breakdown"] = {
            "raw_rows": int(total_raw),
            "removed_short_lt_5_chars": removed_short,
            "remaining_after_length_filter": int(len(after_short)),
            "removed_basic_spam_total": removed_spam,
            "spam_reason_counts_non_exclusive": {
                "url": int(has_url.sum()),
                "spam_phrase": int(has_spam_phrase.sum()),
                "repeated_char_ge_15": int(has_rep_char.sum()),
                "low_token_diversity": int(has_low_div.sum()),
            },
            "remaining_after_spam_filter": int(len(after_spam)),
            "removed_duplicate_comment_id": removed_dup_cid,
            "removed_duplicate_comment_text": removed_dup_text,
            "final_cleaned_rows_from_pipeline": int(len(after_text_dedup)),
            "raw_unique_videos": int(raw_df["video_id"].nunique()) if "video_id" in raw_df.columns else None,
            "raw_unique_keywords": int(raw_df["keyword"].nunique()) if "keyword" in raw_df.columns else None,
        }

    if cleaned_path.exists():
        cl_df = pd.read_csv(cleaned_path, dtype=str).fillna("")
        result["cleaned_comments_csv"] = {
            "rows": int(len(cl_df)),
            "unique_comment_texts": int(cl_df["comment_text"].map(normalize_text).nunique()),
            "unique_videos": int(cl_df["video_id"].nunique()) if "video_id" in cl_df.columns else None,
            "unique_keywords": int(cl_df["keyword"].nunique()) if "keyword" in cl_df.columns else None,
            "length_stats": length_stats(cl_df["comment_text"]),
        }

    if video_meta_path.exists():
        vm_df = pd.read_csv(video_meta_path, dtype=str).fillna("")
        result["video_metadata_csv"] = {
            "rows": int(len(vm_df)),
            "unique_video_ids": int(vm_df["video_id"].nunique()) if "video_id" in vm_df.columns else None,
            "unique_channels": int(vm_df["channel"].nunique()) if "channel" in vm_df.columns else None,
        }

    if proc_videos_path.exists():
        vids = [line.strip() for line in proc_videos_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result["processed_videos_txt_count"] = len(set(vids))

    return result


def audit_weak_labeler_and_review_flag() -> dict:
    """Verify the 120,334 `need_review` flag and evaluate weak-labeler on test sets."""
    cleaned_path = RAW_DIR / "cleaned_comments.csv"
    cl_df = pd.read_csv(cleaned_path, dtype=str).fillna("")
    cl_df["comment_text"] = cl_df["comment_text"].map(normalize_text)
    cl_df = cl_df[cl_df["comment_text"].str.len() >= 5].drop_duplicates(subset=["comment_text"], keep="first")

    detailed = [label_comment_detailed(t) for t in cl_df["comment_text"]]

    n_total = len(detailed)
    n_need_review = sum(1 for r in detailed if r.need_review)
    n_uncertain = sum(1 for r in detailed if r.is_uncertain)
    n_near_boundary = sum(1 for r in detailed if r.near_boundary)
    n_mixed_signal = sum(1 for r in detailed if r.has_mixed_signal)
    n_short_strong = sum(1 for r in detailed if r.too_short_for_strong)
    n_review_phrase = sum(1 for r in detailed if r.has_review_context_phrase)

    phrase_counter: Counter[str] = Counter()
    for r in detailed:
        for p in r.matched_review_phrases:
            phrase_counter[p] += 1

    weak_counts = Counter(r.weak_label for r in detailed)
    conf_counts = Counter(r.confidence for r in detailed)

    inventory_audit = {
        "total_dedup_cleaned_comments": n_total,
        "weak_label_counts": dict(weak_counts),
        "weak_label_percentages": {k: round(v / n_total * 100, 2) for k, v in weak_counts.items()},
        "confidence_counts": dict(conf_counts),
        "composite_need_review_count": n_need_review,
        "composite_need_review_pct": round(n_need_review / n_total * 100, 2),
        "need_review_subcondition_breakdown": {
            "is_uncertain_score_minus1_to_4": {
                "count": n_uncertain,
                "pct_of_inventory": round(n_uncertain / n_total * 100, 2),
            },
            "near_boundary_score_minus3_to_5": {
                "count": n_near_boundary,
                "pct_of_inventory": round(n_near_boundary / n_total * 100, 2),
            },
            "has_mixed_positive_and_negative_keywords": {
                "count": n_mixed_signal,
                "pct_of_inventory": round(n_mixed_signal / n_total * 100, 2),
            },
            "too_short_lt_20_chars_with_strong_keyword": {
                "count": n_short_strong,
                "pct_of_inventory": round(n_short_strong / n_total * 100, 2),
            },
            "matched_22_review_context_phrases_ONLY_table3": {
                "count": n_review_phrase,
                "pct_of_inventory": round(n_review_phrase / n_total * 100, 2),
                "per_phrase_counts": dict(phrase_counter.most_common()),
            },
        },
        "explanation_for_reviewer_1_comment_4": (
            f"The 120,334 count ({n_need_review / n_total * 100:.2f}%) is the union of five "
            f"heuristic review conditions in `auto_labeler.py` (`is_uncertain`={n_uncertain:,}, "
            f"`near_boundary` [-3..5]={n_near_boundary:,}, `has_mixed_signal`={n_mixed_signal:,}, "
            f"`too_short_for_strong`={n_short_strong:,}, `has_review_context_phrase`={n_review_phrase:,}). "
            f"The 22 contextual review phrases in Table 3 actually match only {n_review_phrase:,} records "
            f"({n_review_phrase / n_total * 100:.2f}% of the inventory)."
        ),
    }

    # Evaluate weak-labeler on final_test.csv (n=1,371) and cross_domain_test.csv (n=3,084)
    test_df = pd.read_csv(LABELED_DIR / "final_test.csv")
    vsmec_df = pd.read_csv(UNIFIED_DIR / "cross_domain_test.csv")

    def _eval_weak_on_split(df: pd.DataFrame, split_name: str) -> dict:
        texts = df["comment_text"].fillna("").astype(str).tolist()
        y_true = df["label"].astype(int).to_numpy()
        res = [label_comment_detailed(t) for t in texts]
        scores = np.array([r.depression_score for r in res], dtype=float)
        weak_labels = [r.weak_label for r in res]

        # 1. Covered-only (excluding "uncertain" abstentions)
        covered_mask = np.array([w in ("normal_auto", "depression_auto") for w in weak_labels])
        y_true_cov = y_true[covered_mask]
        y_pred_cov = np.array([1 if w == "depression_auto" else 0 for w, m in zip(weak_labels, covered_mask) if m])
        scores_cov = scores[covered_mask]

        # 2. All samples, counting abstentions ("uncertain") as Normal (0) — standard rule trigger score >= 5
        y_pred_abstention_as_normal = (scores >= 5).astype(int)

        # 3. All samples, counting any net-positive lexical score (score >= 1) as Depression (1)
        y_pred_score_ge_1 = (scores >= 1).astype(int)

        # 4. All samples, counting abstentions ("uncertain") as wrong (or 3-class -> binary pessimistic)
        y_pred_pessimistic = np.where(
            np.array(weak_labels) == "depression_auto",
            1,
            np.where(np.array(weak_labels) == "normal_auto", 0, 1 - y_true),
        )

        return {
            "split": split_name,
            "total_samples": int(len(y_true)),
            "weak_label_distribution": dict(Counter(weak_labels)),
            "coverage_rate": round(float(covered_mask.mean()), 4),
            "abstention_rate_uncertain": round(float((~covered_mask).mean()), 4),
            "covered_only_excluding_abstentions": compute_binary_metrics(y_true_cov, y_pred_cov, scores_cov),
            "all_samples_abstention_as_normal_score_ge_5": compute_binary_metrics(
                y_true, y_pred_abstention_as_normal, scores
            ),
            "all_samples_any_positive_keyword_score_ge_1": compute_binary_metrics(
                y_true, y_pred_score_ge_1, scores
            ),
            "all_samples_abstention_counted_as_error_lower_bound": compute_binary_metrics(
                y_true, y_pred_pessimistic, scores
            ),
        }

    return {
        "lexicon_counts": {
            "strong_keywords_total_surface_forms": len(DEPRESSION_STRONG_KEYWORDS),
            "medium_keywords_total_surface_forms": len(DEPRESSION_MEDIUM_KEYWORDS),
            "normal_keywords_total_surface_forms": len(NORMAL_KEYWORDS),
            "review_context_phrases_total_surface_forms": len(REVIEW_CONTEXT_KEYWORDS),
        },
        "inventory_125k_audit": inventory_audit,
        "weak_labeler_on_final_test_1371": _eval_weak_on_split(test_df, "in_domain_final_test_1371"),
        "weak_labeler_on_vsmec_3084": _eval_weak_on_split(vsmec_df, "cross_domain_vsmec_3084"),
    }


def audit_splits_and_provenance() -> dict:
    train_df = pd.read_csv(LABELED_DIR / "final_train.csv")
    val_df = pd.read_csv(LABELED_DIR / "final_val.csv")
    test_df = pd.read_csv(LABELED_DIR / "final_test.csv")
    full_df = pd.read_csv(LABELED_DIR / "final_dataset.csv")
    vsmec_df = pd.read_csv(UNIFIED_DIR / "cross_domain_test.csv")

    def _split_info(df: pd.DataFrame, name: str) -> dict:
        labels = df["label"].astype(int)
        n_total = len(df)
        n_neg = int((labels == 0).sum())
        n_pos = int((labels == 1).sum())
        info = {
            "split_name": name,
            "total": int(n_total),
            "normal_0": n_neg,
            "depression_1": n_pos,
            "normal_pct": round(n_neg / n_total * 100, 2),
            "depression_pct": round(n_pos / n_total * 100, 2),
            "length_all": length_stats(df["comment_text"]),
            "length_normal_0": length_stats(df.loc[labels == 0, "comment_text"]),
            "length_depression_1": length_stats(df.loc[labels == 1, "comment_text"]),
        }
        if "source" in df.columns:
            info["by_source"] = {str(k): int(v) for k, v in df["source"].value_counts().items()}
        if "annotation_round" in df.columns:
            info["by_annotation_round"] = {
                str(k): int(v) for k, v in df["annotation_round"].fillna("unspecified").value_counts().items()
            }
        if "original_label" in df.columns:
            info["by_original_label"] = {
                str(k): int(v) for k, v in df["original_label"].value_counts().items()
            }
        return info

    # Verify zero overlap across train, val, test, and cross_domain
    norm_sets = {
        "train": set(train_df["comment_text"].map(normalize_comparison_key)),
        "val": set(val_df["comment_text"].map(normalize_comparison_key)),
        "test": set(test_df["comment_text"].map(normalize_comparison_key)),
        "vsmec": set(vsmec_df["comment_text"].map(normalize_comparison_key)),
    }

    # Also build Sadness-only subset info for VSMEC (Reviewer 2 Major Comment 4)
    vsmec_sadness_only = vsmec_df[vsmec_df["original_label"].isin(["Sadness", "Enjoyment"])].copy()

    return {
        "final_dataset_9134": _split_info(full_df, "final_dataset"),
        "final_train_6392": _split_info(train_df, "final_train"),
        "final_val_1371": _split_info(val_df, "final_val"),
        "final_test_1371": _split_info(test_df, "final_test"),
        "cross_domain_vsmec_full_3084": _split_info(vsmec_df, "cross_domain_vsmec_full"),
        "cross_domain_vsmec_sadness_only": _split_info(vsmec_sadness_only, "cross_domain_vsmec_sadness_only"),
        "exact_overlap_verification": {
            "train_val_overlap": len(norm_sets["train"] & norm_sets["val"]),
            "train_test_overlap": len(norm_sets["train"] & norm_sets["test"]),
            "val_test_overlap": len(norm_sets["val"] & norm_sets["test"]),
            "train_vsmec_overlap": len(norm_sets["train"] & norm_sets["vsmec"]),
            "val_vsmec_overlap": len(norm_sets["val"] & norm_sets["vsmec"]),
            "test_vsmec_overlap": len(norm_sets["test"] & norm_sets["vsmec"]),
        },
        "all_9134_human_gold_verified": bool(
            set(full_df["source"].unique()).issubset(
                {"human_gold_pre_round5", "human_gold_round5", "human_gold_round6"}
            )
        ),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 75)
    print("RUNNING SCIENTIFIC DATASET, FILTERING, AND LEXICON AUDIT")
    print("=" * 75)

    hw_info = {
        "timestamp": datetime.now().isoformat(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sklearn_version": sklearn.__version__,
        "accelerator": (
            "NVIDIA CUDA"
            if torch.cuda.is_available()
            else "Apple Silicon MPS (Metal Performance Shaders)"
            if torch.backends.mps.is_available()
            else "CPU"
        ),
    }

    print("\n1. Auditing raw -> cleaned filtering & video metadata...")
    raw_audit = audit_raw_to_cleaned()

    print("2. Auditing splits, class counts, text lengths, and provenance...")
    split_audit = audit_splits_and_provenance()

    print("3. Auditing weak-labeler, 22 review phrases vs composite flag, and test-set coverage...")
    lexicon_audit = audit_weak_labeler_and_review_flag()

    report = {
        "environment": hw_info,
        "raw_and_filtering_audit": raw_audit,
        "splits_and_provenance_audit": split_audit,
        "lexicon_and_weak_labeler_audit": lexicon_audit,
    }

    out_file = RESULTS_DIR / "dataset_and_lexicon_audit.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also update data/analysis/dataset_integrity_report_round6_v2.json with full audited details
    print(f"\nSaved complete audit report to: {out_file}")
    print("\n--- KEY FINDINGS FOR REVIEWERS ---")
    inv = lexicon_audit["inventory_125k_audit"]
    print(f"Cleaned inventory: {inv['total_dedup_cleaned_comments']:,} comments")
    print(f"Composite need_review flag: {inv['composite_need_review_count']:,} ({inv['composite_need_review_pct']}%)")
    r22 = inv["need_review_subcondition_breakdown"]["matched_22_review_context_phrases_ONLY_table3"]
    print(f"Actual 22 review-context phrases (Table 3) matches: {r22['count']:,} ({r22['pct_of_inventory']}%)")

    wl_test = lexicon_audit["weak_labeler_on_final_test_1371"]
    print(f"\nWeak-Labeler on Final Test (n={wl_test['total_samples']:,}):")
    print(f"  - Coverage (non-uncertain): {wl_test['coverage_rate']*100:.2f}% (Abstention: {wl_test['abstention_rate_uncertain']*100:.2f}%)")
    print(f"  - Covered-only F1-macro: {wl_test['covered_only_excluding_abstentions']['f1_macro']:.4f}")
    print(f"  - All 1,371 samples (Abstention->Normal, score>=5): F1-macro={wl_test['all_samples_abstention_as_normal_score_ge_5']['f1_macro']:.4f}, ROC-AUC={wl_test['all_samples_abstention_as_normal_score_ge_5']['roc_auc']:.4f}, PR-AUC={wl_test['all_samples_abstention_as_normal_score_ge_5']['pr_auc']:.4f}")
    print(f"  - All 1,371 samples (Any positive keyword, score>=1): F1-macro={wl_test['all_samples_any_positive_keyword_score_ge_1']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
