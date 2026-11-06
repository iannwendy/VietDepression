"""Leakage-free augmentation pipeline.

Correct approach:
- Keep original val and test sets UNCHANGED
- Only generate/apply augmentation to TRAINING data
- Add augmented samples ONLY to the training set
- Check for near-duplicates between train and test

Usage:
  .venv/bin/python scripts/merge_augmented_leakage_free.py
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
LABELED_DIR = DATA_DIR / "labeled"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Unicode-normalized, whitespace-collapsed comparison key."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def check_near_duplicates(df1: pd.DataFrame, df2: pd.DataFrame,
                          text_col: str = "comment_text", threshold: float = 0.85) -> list:
    """Check for near-duplicates between two dataframes."""
    duplicates = []
    df1_texts = df1[text_col].tolist()
    df2_texts = df2[text_col].tolist()

    for i, text1 in enumerate(df1_texts):
        for j, text2 in enumerate(df2_texts):
            sim = jaccard_similarity(text1, text2)
            if sim >= threshold:
                duplicates.append({
                    "df1_idx": i,
                    "df2_idx": j,
                    "similarity": sim,
                    "text1": text1[:100],
                    "text2": text2[:100]
                })
    return duplicates


def main():
    # ── Files ─────────────────────────────────────────────────────────
    ORIGINAL_TRAIN = LABELED_DIR / "final_train.csv"
    ORIGINAL_VAL = LABELED_DIR / "final_val.csv"
    ORIGINAL_TEST = LABELED_DIR / "final_test.csv"
    AUGMENTED_DEP = LABELED_DIR / "final_train_augmented_depression.csv"

    OUTPUT_DIR = DATA_DIR / "augmented_v2"  # New version to avoid confusion

    # ── Load original data ─────────────────────────────────────────────
    logger.info("Loading original train/val/test sets...")
    train_df = pd.read_csv(ORIGINAL_TRAIN, dtype=str).fillna("")
    val_df = pd.read_csv(ORIGINAL_VAL, dtype=str).fillna("")
    test_df = pd.read_csv(ORIGINAL_TEST, dtype=str).fillna("")

    logger.info(f"Original train: {len(train_df)} rows | Labels: {dict(train_df['label'].value_counts())}")
    logger.info(f"Original val: {len(val_df)} rows | Labels: {dict(val_df['label'].value_counts())}")
    logger.info(f"Original test: {len(test_df)} rows | Labels: {dict(test_df['label'].value_counts())}")

    # ── Load augmented data ────────────────────────────────────────────
    logger.info("Loading augmented depression data...")
    aug_df = pd.read_csv(AUGMENTED_DEP, dtype=str).fillna("")
    # Filter only augmented samples (not originals)
    aug_df = aug_df[aug_df["augmented"] == "True"].copy()
    logger.info(f"Augmented samples: {len(aug_df)} rows")

    # ── CRITICAL: Check near-duplicates BEFORE adding augmentation ──────
    logger.info("\n=== NEAR-DUPLICATE CHECK ===")
    logger.info("Checking for near-duplicates (Jaccard >= 0.85) between splits...")

    # Check train-val overlap
    train_val_dups = check_near_duplicates(train_df, val_df, threshold=0.85)
    logger.info(f"Train-Val near-duplicates: {len(train_val_dups)}")

    # Check train-test overlap
    train_test_dups = check_near_duplicates(train_df, test_df, threshold=0.85)
    logger.info(f"Train-Test near-duplicates: {len(train_test_dups)}")

    # Check val-test overlap
    val_test_dups = check_near_duplicates(val_df, test_df, threshold=0.85)
    logger.info(f"Val-Test near-duplicates: {len(val_test_dups)}")

    if train_val_dups or train_test_dups or val_test_dups:
        logger.warning("⚠️  Near-duplicates found! This indicates potential data leakage.")
        # Save duplicate report
        dup_report = {
            "train_val": train_val_dups,
            "train_test": train_test_dups,
            "val_test": val_test_dups
        }
    else:
        logger.info("✅ No near-duplicates found. Splits are clean.")

    # ── Deduplicate augmented data ─────────────────────────────────────
    train_texts_norm = set(train_df["comment_text"].map(normalize_text))

    # Normalize augmented texts
    aug_df["_norm"] = aug_df["comment_text"].map(normalize_text)

    # Remove augmented samples that already exist in train
    new_aug = aug_df[~aug_df["_norm"].isin(train_texts_norm)].copy()
    logger.info(f"\nNew augmented samples (after dedup from train): {len(new_aug)} rows")

    # Also check overlap with val and test
    val_texts_norm = set(val_df["comment_text"].map(normalize_text))
    test_texts_norm = set(test_df["comment_text"].map(normalize_text))

    # Remove augmented samples that overlap with val or test
    new_aug = new_aug[~new_aug["_norm"].isin(val_texts_norm | test_texts_norm)].copy()
    logger.info(f"New augmented samples (after dedup from val/test): {len(new_aug)} rows")

    # ── Prepare augmented training data ─────────────────────────────────
    new_aug_clean = new_aug[["comment_text", "label"]].copy()
    new_aug_clean["weight"] = 1.0
    new_aug_clean["source"] = "augmented"

    # Ensure train has required columns
    for col in ["weight", "source", "weak_label", "confidence", "depression_score", "matched_keywords"]:
        if col not in train_df.columns:
            train_df[col] = ""
    if "source" not in train_df.columns:
        train_df["source"] = "original"

    # ── NO RESPLIT: Keep val and test unchanged ─────────────────────────
    # Only add augmented samples to training set

    augmented_train = pd.concat([train_df, new_aug_clean], ignore_index=True)

    # Final column order
    columns = ["comment_text", "label", "weak_label", "confidence", "depression_score",
               "matched_keywords", "source", "weight"]
    augmented_train = augmented_train[[c for c in columns if c in augmented_train.columns]]

    logger.info(f"\n=== AUGMENTED TRAINING SET ===")
    logger.info(f"Original train: {len(train_df)} rows")
    logger.info(f"New augmented: {len(new_aug_clean)} rows")
    logger.info(f"Total augmented train: {len(augmented_train)} rows")
    logger.info(f"Label distribution:\n{augmented_train['label'].value_counts()}")
    logger.info(f"Source distribution:\n{augmented_train['source'].value_counts()}")

    # ── Save ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    augmented_train.to_csv(OUTPUT_DIR / "final_train_aug.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(OUTPUT_DIR / "final_val_aug.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_DIR / "final_test_aug.csv", index=False, encoding="utf-8-sig")

    # Also save combined dataset for reference
    combined = pd.concat([augmented_train, val_df, test_df], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "final_dataset_aug.csv", index=False, encoding="utf-8-sig")

    logger.info(f"\n{'='*60}")
    logger.info(f"LEAKAGE-FREE AUGMENTATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Train (augmented): {len(augmented_train):,} rows")
    logger.info(f"Val (unchanged):   {len(val_df):,} rows")
    logger.info(f"Test (unchanged):  {len(test_df):,} rows")
    logger.info(f"\nOutput directory: {OUTPUT_DIR}")
    logger.info(f"\n⚠️  IMPORTANT: Val and test sets are UNCHANGED.")
    logger.info(f"   Only training set was augmented.")
    logger.info(f"   Evaluation should be run on original test set.")

    return {
        "train_size": len(augmented_train),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "augmented_added": len(new_aug_clean),
        "near_duplicates": {
            "train_val": len(train_val_dups),
            "train_test": len(train_test_dups),
            "val_test": len(val_test_dups)
        }
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
