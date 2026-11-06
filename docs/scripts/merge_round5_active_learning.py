"""Merge Round 5 Active Learning review results into gold + final_dataset.

Usage:
  .venv/bin/python docs/merge_round5_active_learning.py

Prerequisites:
  1. Run retrain: scripts/retrain_phobert_for_round5.py
  2. Run selection: scripts/prepare_round5_active_learning.py
  3. Import into Label Studio and review
  4. Export CSV → docs/export_round5_active_learning.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DOCS_DIR = PROJECT_DIR / "docs"

# ── Inputs ───────────────────────────────────────────────────────────
EXPORT_FILE = DOCS_DIR / "export_round5_active_learning.csv"
KEY_FILE = DOCS_DIR / "label_studio_round5_active_learning_key.csv"
GOLD_FILE = DATA_DIR / "gold_review.csv"
FINAL_DATASET_FILE = DATA_DIR / "final_dataset.csv"

# ── Outputs ──────────────────────────────────────────────────────────
OUTPUT_GOLD_FILE = DATA_DIR / "gold_review.csv"  # Overwrite
OUTPUT_FINAL_FILE = DATA_DIR / "final_dataset.csv"  # Overwrite
REPORT_FILE = DOCS_DIR / "round5_merge_report.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    # ── 1. Validate inputs ──────────────────────────────────────────────
    if not EXPORT_FILE.exists():
        logger.error("Missing export file: %s", EXPORT_FILE)
        logger.error("Please export from Label Studio and save as: %s", EXPORT_FILE)
        sys.exit(1)

    if not KEY_FILE.exists():
        logger.error("Missing key file: %s", KEY_FILE)
        sys.exit(1)

    logger.info("Loading export file: %s", EXPORT_FILE)
    export_df = pd.read_csv(EXPORT_FILE, dtype=str).fillna("")
    logger.info("Export rows: %d", len(export_df))

    logger.info("Loading key file: %s", KEY_FILE)
    key_df = pd.read_csv(KEY_FILE, dtype=str).fillna("")
    logger.info("Key rows: %d", len(key_df))

    # ── 2. Merge labels from export into key ────────────────────────────
    label_col = None
    possible_labels = ["depression", "normal", "final_label", "label", "annotations"]
    for col in possible_labels:
        if col in export_df.columns:
            label_col = col
            break

    if label_col is None:
        logger.error("Could not find label column. Available columns: %s", export_df.columns.tolist())
        sys.exit(1)

    logger.info("Label column: %s", label_col)

    merged = key_df.merge(
        export_df[["row_id", label_col]],
        on="row_id",
        how="left",
        suffixes=("", "_export")
    )
    merged["final_label"] = merged[label_col]

    # ── 3. Filter valid labels ─────────────────────────────────────────
    valid_labels = ["depression", "normal"]
    valid_df = merged[merged["final_label"].isin(valid_labels)].copy()
    logger.info("Valid labels: %d/%d", len(valid_df), len(merged))

    uncertain_df = merged[~merged["final_label"].isin(valid_labels)].copy()
    logger.info("Excluded (uncertain/exclude): %d", len(uncertain_df))

    # ── 4. Load existing gold and merge ────────────────────────────────
    logger.info("Loading existing gold: %s", GOLD_FILE)
    gold_df = pd.read_csv(GOLD_FILE, dtype=str).fillna("")
    logger.info("Existing gold rows: %d", len(gold_df))

    valid_df["text_normalized"] = valid_df["comment_text"].str.strip()
    gold_df["text_normalized"] = gold_df["comment_text"].str.strip()

    existing_texts = set(gold_df["text_normalized"])
    new_df = valid_df[~valid_df["text_normalized"].isin(existing_texts)].copy()
    logger.info("New entries to add: %d", len(new_df))

    new_gold = new_df[["comment_text", "final_label"]].copy()
    new_gold.columns = ["comment_text", "label"]
    new_gold["label"] = new_gold["label"].map({"depression": 1, "normal": 0})

    combined_gold = pd.concat([gold_df[["comment_text", "label"]], new_gold], ignore_index=True)
    combined_gold = combined_gold.drop_duplicates(subset=["comment_text"], keep="first")

    logger.info("Combined gold rows: %d", len(combined_gold))
    logger.info("Gold distribution: %s", combined_gold["label"].value_counts().to_dict())

    # ── 5. Rebuild final_dataset ─────────────────────────────────────────
    logger.info("Loading existing final_dataset: %s", FINAL_DATASET_FILE)
    final_df = pd.read_csv(FINAL_DATASET_FILE, dtype=str).fillna("")

    new_final = new_df[["comment_text", "final_label"]].copy()
    new_final.columns = ["comment_text", "label"]
    new_final["label"] = new_final["label"].map({"depression": 1, "normal": 0})
    new_final["source"] = "human_gold"
    new_final["weight"] = 3

    combined_final = pd.concat([final_df, new_final], ignore_index=True)
    combined_final = combined_final.drop_duplicates(subset=["comment_text"], keep="first")

    logger.info("Combined final_dataset rows: %d", len(combined_final))
    logger.info("Final distribution: %s", combined_final["source"].value_counts().to_dict())

    # ── 6. Re-stratified split ─────────────────────────────────────────
    labels = combined_final["label"].astype(int)
    texts = combined_final["comment_text"]

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )

    train_df = combined_final[combined_final["comment_text"].isin(train_texts)].copy()
    val_df = combined_final[combined_final["comment_text"].isin(val_texts)].copy()
    test_df = combined_final[combined_final["comment_text"].isin(test_texts)].copy()

    logger.info("New splits:")
    logger.info("  Train: %d rows", len(train_df))
    logger.info("  Val:   %d rows", len(val_df))
    logger.info("  Test:  %d rows", len(test_df))

    # ── 7. Save outputs ────────────────────────────────────────────────
    combined_gold.to_csv(OUTPUT_GOLD_FILE, index=False, encoding="utf-8-sig")
    logger.info("Saved gold: %s", OUTPUT_GOLD_FILE)

    combined_final.to_csv(OUTPUT_FINAL_FILE, index=False, encoding="utf-8-sig")
    logger.info("Saved final_dataset: %s", OUTPUT_FINAL_FILE)

    train_df.to_csv(DATA_DIR / "final_train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(DATA_DIR / "final_val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(DATA_DIR / "final_test.csv", index=False, encoding="utf-8-sig")
    logger.info("Saved train/val/test splits")

    # ── 8. Generate report ─────────────────────────────────────────────
    report = {
        "round": 5,
        "timestamp": pd.Timestamp.now().isoformat(),
        "review_stats": {
            "total_exported": int(len(export_df)),
            "valid_labels": int(len(valid_df)),
            "excluded": int(len(uncertain_df)),
            "new_entries_added": int(len(new_df)),
        },
        "gold_stats": {
            "before": int(len(gold_df)),
            "after": int(len(combined_gold)),
            "depression": int(combined_gold["label"].sum()),
            "normal": int(len(combined_gold) - combined_gold["label"].sum()),
        },
        "final_dataset_stats": {
            "before": int(len(final_df)),
            "after": int(len(combined_final)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "source_distribution": combined_final["source"].value_counts().to_dict(),
        "output_files": {
            "gold": str(OUTPUT_GOLD_FILE),
            "final_dataset": str(OUTPUT_FINAL_FILE),
            "final_train": str(DATA_DIR / "final_train.csv"),
            "final_val": str(DATA_DIR / "final_val.csv"),
            "final_test": str(DATA_DIR / "final_test.csv"),
        },
        "next_steps": [
            "1. Rerun model training on new final_train.csv",
            "2. Evaluate on final_test.csv and cross_domain_test.csv",
            "3. Compare results with previous round",
            "4. Update paper with new numbers",
        ],
    }

    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Report saved: %s", REPORT_FILE)

    print(f"\n{'='*60}")
    print(f"ROUND 5 MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"New gold: {len(combined_gold)} rows ({combined_gold['label'].sum()} depression / {len(combined_gold) - combined_gold['label'].sum()} normal)")
    print(f"New final_dataset: {len(combined_final)} rows")
    print(f"New splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"\nNext: Rerun models and evaluate!")


if __name__ == "__main__":
    main()
