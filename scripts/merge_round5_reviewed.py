"""Repair Round 5 labels and rebuild leakage-safe supervised splits.

The original Round 5 merge wrote new annotations to a ``text`` column while
the project schema used ``comment_text``. Re-running that merge duplicated the
same 1,360 annotations four times and then collapsed them to one null row in
``final_dataset.csv``. This script repairs that state deterministically.

Protocol
--------
* Recover the clean pre-Round-5 gold set from non-null ``comment_text`` rows.
* Recover one unique Round-5 annotation per normalized ``text`` value.
* Build validation and test exclusively from pre-Round-5 human gold.
* Put all Round-5 annotations and high-confidence weak labels in training only.
* Remove exact VSMEC overlap before constructing supervised data.
* Emit a machine-readable integrity report and a canonical Round-5 export.

Usage:
    PYTHONPATH="$PWD" .venv/bin/python scripts/merge_round5_reviewed.py
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
LABELED_DIR = DATA_DIR / "labeled"
ROUND5_DIR = DATA_DIR / "round5"
ANALYSIS_DIR = DATA_DIR / "analysis"
ARCHIVE_DIR = DATA_DIR / "_archive"
UNIFIED_DIR = PROJECT_DIR / "data_unified"

TRAIN_GOLD_FILE = LABELED_DIR / "train_gold.csv"
VAL_GOLD_FILE = LABELED_DIR / "val_gold.csv"
TEST_GOLD_FILE = LABELED_DIR / "test_gold.csv"
AUTO_LABELED_FILE = RAW_DIR / "auto_labeled_comments.csv"
CROSS_DOMAIN_FILE = UNIFIED_DIR / "cross_domain_test.csv"
ROUND5_CANONICAL_FILE = ROUND5_DIR / "round5_reviewed_clean.csv"
INTEGRITY_REPORT_FILE = ANALYSIS_DIR / "dataset_integrity_report.json"

FINAL_DATASET_FILE = LABELED_DIR / "final_dataset.csv"
FINAL_TRAIN_FILE = LABELED_DIR / "final_train.csv"
FINAL_VAL_FILE = LABELED_DIR / "final_val.csv"
FINAL_TEST_FILE = LABELED_DIR / "final_test.csv"

RANDOM_SEED = 42

OUTPUT_COLUMNS = [
    "comment_text",
    "label",
    "weak_label",
    "confidence",
    "depression_score",
    "matched_keywords",
    "source",
    "weight",
    "annotation_round",
]


def normalize_text(value: object) -> str:
    """Return a Unicode-normalized, whitespace-collapsed comparison key."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].astype(int).value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def standardize_rows(
    df: pd.DataFrame,
    *,
    source: str,
    weight: int,
    annotation_round: str,
) -> pd.DataFrame:
    result = pd.DataFrame()
    result["comment_text"] = df["comment_text"].astype(str).str.strip()
    result["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    result["weak_label"] = df.get("weak_label", "")
    result["confidence"] = df.get("confidence", "")
    result["depression_score"] = df.get("depression_score", 0)
    result["matched_keywords"] = df.get("matched_keywords", "")
    result["source"] = source
    result["weight"] = weight
    result["annotation_round"] = annotation_round
    return result[OUTPUT_COLUMNS]


def recover_gold_sets(raw_gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover clean pre-R5 gold and unique R5 annotations from the bad merge."""
    if "comment_text" not in raw_gold.columns:
        raise ValueError("train_gold.csv has no comment_text column")

    base_mask = raw_gold["comment_text"].notna() & raw_gold["comment_text"].astype(str).str.strip().ne("")
    if "source" in raw_gold.columns:
        base_mask &= ~raw_gold["source"].isin(["round5_active_learning", "human_gold_round5"])
    base = raw_gold.loc[base_mask].copy()
    base["_norm"] = base["comment_text"].map(normalize_text)

    if "text" in raw_gold.columns:
        r5_mask = raw_gold.get("source", "").eq("round5_active_learning")
        r5_mask &= raw_gold["text"].notna() & raw_gold["text"].astype(str).str.strip().ne("")
        r5 = raw_gold.loc[r5_mask, ["text", "label"]].copy()
        r5 = r5.rename(columns={"text": "comment_text"})
    else:
        r5_mask = raw_gold.get("source", "").isin(["round5_active_learning", "human_gold_round5"])
        r5 = raw_gold.loc[r5_mask, ["comment_text", "label"]].copy()

    if ROUND5_CANONICAL_FILE.exists():
        canonical = pd.read_csv(ROUND5_CANONICAL_FILE)
        canonical = canonical[["comment_text", "label"]].copy()
        if not canonical.empty:
            r5 = canonical

    r5["_norm"] = r5["comment_text"].map(normalize_text)

    conflict_count = int(r5.groupby("_norm")["label"].nunique().gt(1).sum())
    if conflict_count:
        raise ValueError(f"Round 5 contains {conflict_count} conflicting text labels")

    base = base.drop_duplicates("_norm", keep="first")
    r5 = r5.drop_duplicates("_norm", keep="first")
    r5 = r5[~r5["_norm"].isin(set(base["_norm"]))].copy()

    # Validate row counts with configurable thresholds (not hardcoded)
    # Pre-R5 gold: expect at least 1500 rows (flexible)
    MIN_PRE_R5_GOLD = 1500
    if len(base) < MIN_PRE_R5_GOLD:
        raise ValueError(f"Pre-R5 gold set too small: {len(base):,} rows (expected at least {MIN_PRE_R5_GOLD:,})")

    # Round 5 gold: expect at least 1000 rows (flexible)
    MIN_ROUND5_GOLD = 1000
    if len(r5) < MIN_ROUND5_GOLD:
        raise ValueError(f"Round 5 gold set too small: {len(r5):,} rows (expected at least {MIN_ROUND5_GOLD:,})")

    # Log actual counts (informational, not assertion)
    print(f"  Recovered: {len(base):,} pre-R5 gold, {len(r5):,} Round 5 gold")

    return base.drop(columns="_norm"), r5.drop(columns="_norm")


def integrity_metrics(
    final_train: pd.DataFrame,
    final_val: pd.DataFrame,
    final_test: pd.DataFrame,
    cross_domain: pd.DataFrame,
) -> dict[str, object]:
    splits = {
        "train": final_train,
        "val": final_val,
        "test": final_test,
        "cross_domain": cross_domain,
    }
    normalized = {
        name: set(df["comment_text"].map(normalize_text))
        for name, df in splits.items()
    }
    return {
        "null_or_blank_text": {
            name: int(df["comment_text"].isna().sum() + df["comment_text"].fillna("").astype(str).str.strip().eq("").sum())
            for name, df in splits.items()
        },
        "duplicate_text_within_split": {
            name: int(df["comment_text"].map(normalize_text).duplicated().sum())
            for name, df in splits.items()
        },
        "overlap": {
            "train_val": len(normalized["train"] & normalized["val"]),
            "train_test": len(normalized["train"] & normalized["test"]),
            "val_test": len(normalized["val"] & normalized["test"]),
            "train_cross_domain": len(normalized["train"] & normalized["cross_domain"]),
            "val_cross_domain": len(normalized["val"] & normalized["cross_domain"]),
            "test_cross_domain": len(normalized["test"] & normalized["cross_domain"]),
        },
        "invalid_labels": {
            name: int((~df["label"].astype(int).isin([0, 1])).sum())
            for name, df in splits.items()
        },
    }


def main() -> None:
    for directory in [ROUND5_DIR, ANALYSIS_DIR, ARCHIVE_DIR, LABELED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    if not TRAIN_GOLD_FILE.exists():
        raise FileNotFoundError(TRAIN_GOLD_FILE)

    archive_file = ARCHIVE_DIR / "train_gold_corrupted_round5_20260724.csv"
    if not archive_file.exists():
        shutil.copy2(TRAIN_GOLD_FILE, archive_file)

    raw_gold = pd.read_csv(TRAIN_GOLD_FILE)
    base_gold_raw, round5_raw = recover_gold_sets(raw_gold)

    base_gold = standardize_rows(
        base_gold_raw,
        source="human_gold_pre_round5",
        weight=3,
        annotation_round="round4_or_earlier",
    )
    round5_gold = standardize_rows(
        round5_raw,
        source="human_gold_round5",
        weight=3,
        annotation_round="round5",
    )

    round5_export = round5_gold[["comment_text", "label", "source", "weight", "annotation_round"]].copy()
    round5_export["final_label"] = round5_export["label"].map({0: "normal", 1: "depression"})
    round5_export.to_csv(ROUND5_CANONICAL_FILE, index=False, encoding="utf-8-sig")

    val_gold_raw = pd.read_csv(VAL_GOLD_FILE)
    test_gold_raw = pd.read_csv(TEST_GOLD_FILE)
    base_val_raw = standardize_rows(
        val_gold_raw,
        source="human_gold_fixed_validation",
        weight=3,
        annotation_round="round3_or_earlier",
    )
    base_test_raw = standardize_rows(
        test_gold_raw,
        source="human_gold_fixed_test",
        weight=3,
        annotation_round="round3_or_earlier",
    )

    holdout_norm = set(base_val_raw["comment_text"].map(normalize_text)) | set(
        base_test_raw["comment_text"].map(normalize_text)
    )
    base_before_holdout_filter = len(base_gold)
    round5_before_holdout_filter = len(round5_gold)
    base_gold = base_gold[~base_gold["comment_text"].map(normalize_text).isin(holdout_norm)].copy()
    round5_gold = round5_gold[~round5_gold["comment_text"].map(normalize_text).isin(holdout_norm)].copy()
    train_gold_holdout_overlap_removed = (
        base_before_holdout_filter - len(base_gold)
        + round5_before_holdout_filter - len(round5_gold)
    )

    combined_gold = pd.concat([base_gold, round5_gold], ignore_index=True)
    combined_gold["_norm"] = combined_gold["comment_text"].map(normalize_text)
    combined_gold = combined_gold.drop_duplicates("_norm", keep="first").drop(columns="_norm")

    auto = pd.read_csv(AUTO_LABELED_FILE, dtype=str).fillna("")
    auto = auto[auto["confidence"].eq("high") & auto["weak_label"].isin(["normal_auto", "depression_auto"])].copy()
    auto["label"] = auto["weak_label"].map({"normal_auto": 0, "depression_auto": 1})
    weak = standardize_rows(
        auto,
        source="weak_high_conf",
        weight=2,
        annotation_round="weak_label",
    )

    cross = pd.read_csv(CROSS_DOMAIN_FILE)
    cross_norm = set(cross["comment_text"].map(normalize_text))
    gold_norm = set(combined_gold["comment_text"].map(normalize_text)) | holdout_norm
    weak["_norm"] = weak["comment_text"].map(normalize_text)
    weak_before = len(weak)
    weak = weak[~weak["_norm"].isin(gold_norm | cross_norm)].copy()
    removed_weak_overlap = weak_before - len(weak)
    weak = weak.drop_duplicates("_norm", keep="first").drop(columns="_norm")

    final_train = pd.concat([base_gold, round5_gold, weak], ignore_index=True)
    final_val = base_val_raw.copy()
    final_test = base_test_raw.copy()
    final_dataset = pd.concat([final_train, final_val, final_test], ignore_index=True)

    combined_gold.to_csv(TRAIN_GOLD_FILE, index=False, encoding="utf-8-sig")
    final_dataset.to_csv(FINAL_DATASET_FILE, index=False, encoding="utf-8-sig")
    final_train.to_csv(FINAL_TRAIN_FILE, index=False, encoding="utf-8-sig")
    final_val.to_csv(FINAL_VAL_FILE, index=False, encoding="utf-8-sig")
    final_test.to_csv(FINAL_TEST_FILE, index=False, encoding="utf-8-sig")

    integrity = integrity_metrics(final_train, final_val, final_test, cross)
    all_zero = all(
        value == 0
        for group in integrity.values()
        for value in group.values()
    )
    report = {
        "protocol_version": "round5_repair_v2",
        "random_seed": RANDOM_SEED,
        "split_policy": {
            "pre_round5_training_gold": "existing unique train_gold rows",
            "validation_test": "fixed historical val_gold/test_gold; never resplit",
            "round5_human_gold": "train only",
            "weak_high_confidence": "train only",
            "validation_test_labels": "human only",
            "cross_domain": "strict holdout; exact overlap removed before training",
        },
        "recovery": {
            "pre_round5_training_gold_rows": len(base_gold),
            "fixed_validation_gold_rows": len(base_val_raw),
            "fixed_test_gold_rows": len(base_test_raw),
            "round5_unique_rows": len(round5_gold),
            "round5_label_distribution": label_counts(round5_gold),
            "combined_gold_rows": len(combined_gold),
            "all_human_gold_rows_including_fixed_holdouts": len(combined_gold) + len(base_val_raw) + len(base_test_raw),
            "training_gold_rows_removed_for_fixed_holdout_overlap": train_gold_holdout_overlap_removed,
            "corrupted_snapshot": str(archive_file.relative_to(PROJECT_DIR)),
            "canonical_round5_file": str(ROUND5_CANONICAL_FILE.relative_to(PROJECT_DIR)),
        },
        "datasets": {
            "final_dataset": {"rows": len(final_dataset), "labels": label_counts(final_dataset)},
            "final_train": {
                "rows": len(final_train),
                "labels": label_counts(final_train),
                "sources": {str(k): int(v) for k, v in final_train["source"].value_counts().items()},
            },
            "final_val": {"rows": len(final_val), "labels": label_counts(final_val)},
            "final_test": {"rows": len(final_test), "labels": label_counts(final_test)},
            "cross_domain": {"rows": len(cross), "labels": label_counts(cross)},
        },
        "weak_rows_removed_for_gold_or_cross_overlap": removed_weak_overlap,
        "integrity": integrity,
        "integrity_pass": all_zero,
    }
    INTEGRITY_REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not all_zero:
        raise RuntimeError(f"Dataset integrity checks failed; see {INTEGRITY_REPORT_FILE}")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
