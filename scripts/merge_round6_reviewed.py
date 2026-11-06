"""Merge Round 6 human-reviewed labels into the training pipeline.

Round 6: 6,813 PhoBERT-positive candidates manually labeled
- normal: 4,458
- depression: 1,244
- uncertain: 684 (excluded)
- exclude: 427 (excluded)

Valid samples: 4,458 + 1,244 = 5,702

Usage:
    PYTHONPATH="$PWD" .venv/bin/python scripts/merge_round6_reviewed.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_PRED_DIR = DATA_DIR / "model_predictions"
LABELED_DIR = DATA_DIR / "labeled"
ROUND6_DIR = DATA_DIR / "round6"
ANALYSIS_DIR = DATA_DIR / "analysis"
UNIFIED_DIR = PROJECT_DIR / "data_unified"

# Input files
RECOVERED_LABELS = PROJECT_DIR / "project-17-labels-recovered.csv"
AUTO_LABELED_FILE = DATA_DIR / "raw" / "auto_labeled_comments.csv"
CROSS_DOMAIN_FILE = UNIFIED_DIR / "cross_domain_test.csv"

# Output files
ROUND6_CANONICAL_FILE = ROUND6_DIR / "round6_reviewed_clean.csv"
INTEGRITY_REPORT_FILE = ANALYSIS_DIR / "dataset_integrity_report_round6.json"

OUTPUT_COLUMNS = [
    "comment_text", "label", "weak_label", "confidence",
    "depression_score", "matched_keywords", "source", "weight", "annotation_round",
]


def normalize_text(value: object) -> str:
    """Return a Unicode-normalized, whitespace-collapsed comparison key."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def label_counts(df: pd.DataFrame) -> dict:
    counts = df["label"].astype(int).value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def standardize_rows(df: pd.DataFrame, *, source: str, weight: int, annotation_round: str) -> pd.DataFrame:
    result = pd.DataFrame()
    result["comment_text"] = df["comment_text"].astype(str).str.strip()
    result["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    result["weak_label"] = df.get("weak_label", "")
    result["confidence"] = df.get("confidence", "")
    result["depression_score"] = df.get("depression_score", 0)
    result["matched_keywords"] = df.get("matched_keywords", "")
    result["source"] = source
    result["weight"] = weight
    result["annotation_round"] = annotation_round
    return result[OUTPUT_COLUMNS]


def main() -> None:
    print("=" * 70)
    print("ROUND 6 MERGE: Building Final Dataset with Round 6 Labels")
    print("=" * 70)

    # Create directories
    for directory in [ROUND6_DIR, ANALYSIS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and process Round 6 labels ──────────────────────────────────
    print("\n1. Loading recovered labels...")
    recovered = pd.read_csv(RECOVERED_LABELS, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    print(f"   Total: {len(recovered)} rows")
    print(f"   Distribution: {dict(recovered['final_label'].value_counts())}")

    # ── 2. Filter valid labels ───────────────────────────────────────────────
    print("\n2. Filtering valid labels...")
    valid_df = recovered[recovered['final_label'].isin(['normal', 'depression'])].copy()
    valid_df['label'] = valid_df['final_label'].map({'normal': 0, 'depression': 1})
    print(f"   Valid: {len(valid_df)} (excluded {len(recovered) - len(valid_df)})")

    # ── 3. Create Round 6 canonical file ────────────────────────────────────
    print("\n3. Creating Round 6 canonical file...")
    round6_canonical = valid_df[['comment_text', 'label']].copy()
    round6_canonical['source'] = 'human_gold_round6'
    round6_canonical['weight'] = 3
    round6_canonical['annotation_round'] = 'round6'
    round6_canonical.to_csv(ROUND6_CANONICAL_FILE, index=False, encoding='utf-8-sig')
    print(f"   Saved: {len(round6_canonical)} rows")

    # ── 4. Load validation and test sets (FIXED - don't modify) ─────────────
    print("\n4. Loading validation and test sets (fixed)...")
    val_gold = pd.read_csv(LABELED_DIR / "val_gold.csv", dtype=str, encoding='utf-8-sig').fillna("")
    test_gold = pd.read_csv(LABELED_DIR / "test_gold.csv", dtype=str, encoding='utf-8-sig').fillna("")
    val_standardized = standardize_rows(val_gold, source="human_gold_fixed_validation", weight=3, annotation_round="round3_or_earlier")
    test_standardized = standardize_rows(test_gold, source="human_gold_fixed_test", weight=3, annotation_round="round3_or_earlier")
    print(f"   Val: {len(val_standardized)} | Test: {len(test_standardized)}")

    # Create holdout set (val + test) for overlap removal
    holdout_norm = set(val_standardized["comment_text"].map(normalize_text)) | set(test_standardized["comment_text"].map(normalize_text))

    # ── 5. Load train_gold (pre-R5, R5) ────────────────────────────────────
    print("\n5. Loading train_gold...")
    train_gold = pd.read_csv(LABELED_DIR / "train_gold.csv", dtype=str, encoding='utf-8-sig').fillna("")
    print(f"   Total: {len(train_gold)} | Sources: {dict(train_gold['source'].value_counts())}")

    # ── 6. Standardize all gold data ────────────────────────────────────────
    print("\n6. Standardizing gold data...")
    pre_r5 = train_gold[train_gold['source'] == 'human_gold_pre_round5'].copy()
    r5 = train_gold[train_gold['source'] == 'human_gold_round5'].copy()
    r6 = round6_canonical.copy()

    pre_r5_std = standardize_rows(pre_r5, source="human_gold_pre_round5", weight=3, annotation_round="round4_or_earlier")
    r5_std = standardize_rows(r5, source="human_gold_round5", weight=3, annotation_round="round5")
    r6_std = standardize_rows(r6, source="human_gold_round6", weight=3, annotation_round="round6")

    print(f"   Pre-R5: {len(pre_r5_std)} | R5: {len(r5_std)} | R6: {len(r6_std)}")

    # ── 7. Load and process weak labels ──────────────────────────────────────
    print("\n7. Loading weak labels...")
    auto = pd.read_csv(AUTO_LABELED_FILE, dtype=str).fillna("")
    auto = auto[auto["confidence"].eq("high") & auto["weak_label"].isin(["normal_auto", "depression_auto"])].copy()
    auto["label"] = auto["weak_label"].map({"normal_auto": 0, "depression_auto": 1})
    weak = standardize_rows(auto, source="weak_high_conf", weight=2, annotation_round="weak_label")
    print(f"   Total weak high-conf: {len(weak)}")

    # ── 8. Load VSMEC for overlap check ──────────────────────────────────────
    print("\n8. Loading VSMEC cross-domain test...")
    vsmec = pd.read_csv(CROSS_DOMAIN_FILE)
    vsmec_norm = set(vsmec["comment_text"].map(normalize_text))
    print(f"   VSMEC: {len(vsmec)} rows")

    # ── 9. Remove overlap from each component ─────────────────────────────────
    print("\n9. Removing overlap from components...")

    # Helper to remove overlaps
    def remove_overlaps(df, exclude_norms):
        df['_norm'] = df['comment_text'].map(normalize_text)
        before = len(df)
        df = df[~df['_norm'].isin(exclude_norms)]
        df = df.drop(columns=['_norm'])
        return df, before - len(df)

    # Remove holdout overlap
    pre_r5_clean, pre_r5_removed = remove_overlaps(pre_r5_std, holdout_norm)
    r5_clean, r5_removed = remove_overlaps(r5_std, holdout_norm)
    r6_clean, r6_removed = remove_overlaps(r6_std, holdout_norm)
    weak_clean, weak_removed = remove_overlaps(weak, holdout_norm)

    print(f"   Removed for holdout overlap:")
    print(f"      Pre-R5: {pre_r5_removed} | R5: {r5_removed} | R6: {r6_removed} | Weak: {weak_removed}")

    # Remove VSMEC overlap
    pre_r5_final, pre_r5_vsmec = remove_overlaps(pre_r5_clean, vsmec_norm)
    r5_final, r5_vsmec = remove_overlaps(r5_clean, vsmec_norm)
    r6_final, r6_vsmec = remove_overlaps(r6_clean, vsmec_norm)
    weak_final, weak_vsmec = remove_overlaps(weak_clean, vsmec_norm)

    print(f"   Removed for VSMEC overlap:")
    print(f"      Pre-R5: {pre_r5_vsmec} | R5: {r5_vsmec} | R6: {r6_vsmec} | Weak: {weak_vsmec}")

    print(f"\n   Final counts:")
    print(f"      Pre-R5: {len(pre_r5_final)} | R5: {len(r5_final)} | R6: {len(r6_final)} | Weak: {len(weak_final)}")

    # ── 10. Build final training set ──────────────────────────────────────────
    print("\n10. Building final training set...")

    # Combine all gold first (human labels have priority)
    all_gold = pd.concat([pre_r5_final, r5_final, r6_final], ignore_index=True)

    # Remove duplicates within gold (keep first)
    all_gold['_norm'] = all_gold['comment_text'].map(normalize_text)
    all_gold = all_gold.drop_duplicates(subset=['_norm'], keep='first')
    all_gold = all_gold.drop(columns=['_norm'])

    # Get texts already in gold
    gold_norms = set(all_gold['comment_text'].map(normalize_text))

    # Add weak labels (only those not in gold)
    weak_to_add = weak_final[~weak_final['comment_text'].map(normalize_text).isin(gold_norms)]

    # Combine
    final_train = pd.concat([all_gold, weak_to_add], ignore_index=True)

    # Final deduplication
    final_train['_norm'] = final_train['comment_text'].map(normalize_text)
    final_train = final_train.drop_duplicates(subset=['_norm'], keep='first')
    final_train = final_train.drop(columns=['_norm'])

    final_val = val_standardized
    final_test = test_standardized
    final_dataset = pd.concat([final_train, final_val, final_test], ignore_index=True)

    print(f"\n   Final train: {len(final_train)} rows")
    print(f"   Final val: {len(final_val)} rows")
    print(f"   Final test: {len(final_test)} rows")
    print(f"   Final dataset: {len(final_dataset)} rows")

    # ── 11. Save final datasets ───────────────────────────────────────────────
    print("\n11. Saving final datasets...")
    final_train.to_csv(LABELED_DIR / "final_train.csv", index=False, encoding='utf-8-sig')
    final_val.to_csv(LABELED_DIR / "final_val.csv", index=False, encoding='utf-8-sig')
    final_test.to_csv(LABELED_DIR / "final_test.csv", index=False, encoding='utf-8-sig')
    final_dataset.to_csv(LABELED_DIR / "final_dataset.csv", index=False, encoding='utf-8-sig')
    print(f"   Saved!")

    # ── 12. Generate integrity report ─────────────────────────────────────────
    print("\n12. Generating integrity report...")

    splits = {
        "train": final_train,
        "val": final_val,
        "test": final_test,
        "cross_domain": vsmec,
    }
    normalized = {name: set(df["comment_text"].map(normalize_text)) for name, df in splits.items()}

    integrity = {
        "null_or_blank_text": {name: int(df["comment_text"].isna().sum()) for name, df in splits.items()},
        "duplicate_text_within_split": {name: int(df["comment_text"].map(normalize_text).duplicated().sum()) for name, df in splits.items()},
        "overlap": {
            "train_val": int(len(normalized["train"] & normalized["val"])),
            "train_test": int(len(normalized["train"] & normalized["test"])),
            "val_test": int(len(normalized["val"] & normalized["test"])),
            "train_cross_domain": int(len(normalized["train"] & normalized["cross_domain"])),
        },
        "invalid_labels": {name: int((~df["label"].astype(int).isin([0, 1])).sum()) for name, df in splits.items()},
    }

    all_zero = all(
        value == 0
        for group in integrity.values()
        for value in group.values()
    )

    report = {
        "protocol_version": "round6_v1",
        "timestamp": datetime.now().isoformat(),
        "round6_review": {
            "total_reviewed": int(len(recovered)),
            "valid_labels": int(len(valid_df)),
            "excluded": int(len(recovered) - len(valid_df)),
            "distribution": {
                "normal": int((valid_df['label'] == 0).sum()),
                "depression": int((valid_df['label'] == 1).sum())
            },
        },
        "datasets": {
            "final_train": {
                "rows": int(len(final_train)),
                "labels": label_counts(final_train),
                "sources": {k: int(v) for k, v in final_train["source"].value_counts().items()}
            },
            "final_val": {"rows": int(len(final_val)), "labels": label_counts(final_val)},
            "final_test": {"rows": int(len(final_test)), "labels": label_counts(final_test)},
            "cross_domain": {"rows": int(len(vsmec)), "labels": label_counts(vsmec)},
        },
        "integrity": integrity,
        "integrity_pass": all_zero,
    }

    INTEGRITY_REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if all_zero:
        print("\n✅ All integrity checks PASSED!")
    else:
        print("\n⚠️  WARNING: Some integrity checks FAILED!")
        print(f"   Issues: {integrity}")

    print("\n" + "=" * 70)
    print("ROUND 6 MERGE COMPLETE")
    print("=" * 70)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
