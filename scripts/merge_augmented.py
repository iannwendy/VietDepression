"""Merge augmented depression samples with existing training data.

Usage:
  .venv/bin/python scripts/merge_augmented.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    # ── Files ─────────────────────────────────────────────────────────
    ORIGINAL_TRAIN = DATA_DIR / "final_train.csv"
    AUGMENTED_DEP = DATA_DIR / "final_train_augmented_depression.csv"
    GOLD_FILE = DATA_DIR / "gold_review.csv"
    CROSS_DOMAIN = DATA_DIR.parent / "data_unified" / "cross_domain_test.csv"

    OUTPUT_DIR = DATA_DIR / "augmented_version"

    # ── Load data ────────────────────────────────────────────────────
    logger.info("Loading original train data...")
    train_df = pd.read_csv(ORIGINAL_TRAIN, dtype=str).fillna("")
    logger.info(f"Original train: {len(train_df)} rows")
    logger.info(f"Original distribution:\n{train_df['label'].value_counts()}")

    logger.info("Loading augmented depression data...")
    aug_df = pd.read_csv(AUGMENTED_DEP, dtype=str).fillna("")
    # Filter only augmented samples (not originals)
    aug_df = aug_df[aug_df["augmented"] == "True"].copy()
    logger.info(f"Augmented samples: {len(aug_df)} rows")

    # ── Load gold for reference ──────────────────────────────────────
    gold_df = pd.read_csv(GOLD_FILE, dtype=str).fillna("")
    gold_texts = set(gold_df["comment_text"].str.strip().str.lower())
    logger.info(f"Gold samples: {len(gold_df)} rows")

    # ── Deduplicate ─────────────────────────────────────────────────
    # Normalize texts for comparison
    train_df["text_normalized"] = train_df["comment_text"].str.strip().str.lower()
    aug_df["text_normalized"] = aug_df["comment_text"].str.strip().str.lower()

    # Remove augmented samples that already exist in train
    existing_texts = set(train_df["text_normalized"])
    new_aug = aug_df[~aug_df["text_normalized"].isin(existing_texts)].copy()
    logger.info(f"New augmented samples (after dedup): {len(new_aug)} rows")

    # ── Prepare merged dataset ────────────────────────────────────────
    # Original samples keep their original weight/source
    train_df = train_df.drop(columns=["text_normalized"])

    # Augmented samples get weight=1 and source="augmented"
    new_aug_clean = new_aug[["comment_text", "label"]].copy()
    new_aug_clean["weight"] = 1.0
    new_aug_clean["source"] = "augmented"

    # Combine
    if "weight" not in train_df.columns:
        train_df["weight"] = 1.0
    if "source" not in train_df.columns:
        train_df["source"] = "original"

    combined_df = pd.concat([train_df, new_aug_clean], ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} rows")
    logger.info(f"Label distribution:\n{combined_df['label'].value_counts()}")
    logger.info(f"Source distribution:\n{combined_df['source'].value_counts()}")

    # ── Stratified split ─────────────────────────────────────────────
    labels = combined_df["label"].astype(int)
    texts = combined_df["comment_text"]

    # 70/15/15 split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )

    train_final = combined_df[combined_df["comment_text"].isin(train_texts)].copy()
    val_final = combined_df[combined_df["comment_text"].isin(val_texts)].copy()
    test_final = combined_df[combined_df["comment_text"].isin(test_texts)].copy()

    # ── Save ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_final.to_csv(OUTPUT_DIR / "final_train_aug.csv", index=False, encoding="utf-8-sig")
    val_final.to_csv(OUTPUT_DIR / "final_val_aug.csv", index=False, encoding="utf-8-sig")
    test_final.to_csv(OUTPUT_DIR / "final_test_aug.csv", index=False, encoding="utf-8-sig")
    combined_df.to_csv(OUTPUT_DIR / "final_dataset_aug.csv", index=False, encoding="utf-8-sig")

    logger.info(f"\nSaved augmented dataset to: {OUTPUT_DIR}")
    logger.info(f"  train: {len(train_final)} rows")
    logger.info(f"  val:   {len(val_final)} rows")
    logger.info(f"  test:  {len(test_final)} rows")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"AUGMENTED DATASET CREATED")
    print(f"{'='*60}")
    print(f"Original train:    {len(train_df):,} rows")
    print(f"New augmented:     {len(new_aug):,} rows")
    print(f"Combined:          {len(combined_df):,} rows")
    print(f"\nNew splits:")
    print(f"  train: {len(train_final):,} rows")
    print(f"  val:   {len(val_final):,} rows")
    print(f"  test:  {len(test_final):,} rows")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\n⚠️  Next steps:")
    print(f"  1. Copy augmented versions over original:")
    print(f"     cp {OUTPUT_DIR}/final_train_aug.csv data/final_train.csv")
    print(f"     cp {OUTPUT_DIR}/final_val_aug.csv data/final_val.csv")
    print(f"     cp {OUTPUT_DIR}/final_test_aug.csv data/final_test.csv")
    print(f"  2. Rerun model training")
    print(f"  3. Compare results")


if __name__ == "__main__":
    main()
