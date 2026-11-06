"""Round 5 Active Learning - Select hardest samples from existing predictions.

This script (NO training needed):
1. Uses existing predictions from Round 4
2. Filters out already reviewed samples
3. Selects top 2000 samples with probability CLOSEST to 0.5 (most uncertain)

Usage:
    .venv/bin/python scripts/select_hardest_samples.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DOCS_DIR = PROJECT_DIR / "docs"

# ── Inputs ──────────────────────────────────────────────────────────────
PREDICTIONS_FILE = DATA_DIR / "phobert_remaining_predictions.csv"
GOLD_FILE = DATA_DIR / "gold_review.csv"

# ── Outputs ─────────────────────────────────────────────────────────────
IMPORT_FILE = DOCS_DIR / "label_studio_round5_active_learning_import.csv"
KEY_FILE = DOCS_DIR / "label_studio_round5_active_learning_key.csv"
REPORT_FILE = DOCS_DIR / "round5_selection_report.json"

# ── Settings ────────────────────────────────────────────────────────────
# Lấy top N samples KHÓ NHẤT (probability gần 0.5 nhất)
TARGET_SAMPLES = 2000

BUCKET_COLUMNS = [
    "review_bucket",
    "comment_text",
    "source_weak_label",
    "source_confidence",
    "source_depression_score",
    "phobert_label",
    "probability",
    "prob_normal",
    "prob_depression",
    "need_review",
    "suggested_label",
    "final_label",
    "reviewer_note",
]


def _text_hash(text: str) -> str:
    return hashlib.sha1(str(text).strip().encode("utf-8")).hexdigest()[:10]


def main() -> None:
    # ── 1. Load predictions ──────────────────────────────────────────────
    print(f"Loading predictions from: {PREDICTIONS_FILE}")
    preds = pd.read_csv(PREDICTIONS_FILE, dtype=str).fillna("")
    print(f"Total remaining samples: {len(preds):,}")

    # ── 2. Load already-reviewed texts ─────────────────────────────────
    gold = pd.read_csv(GOLD_FILE, dtype=str).fillna("")
    gold_texts = set(gold["comment_text"].str.strip().tolist())
    print(f"Already reviewed (gold): {len(gold_texts):,}")

    # ── 3. Filter out already-reviewed ─────────────────────────────────
    preds = preds[~preds["comment_text"].str.strip().isin(gold_texts)].copy()
    print(f"Remaining after filtering: {len(preds):,}")

    # ── 4. Parse numeric columns ───────────────────────────────────────
    preds["probability"] = pd.to_numeric(preds["probability"], errors="coerce").fillna(0.5)

    # Calculate distance from 0.5 (most uncertain = closest to 0.5)
    preds["distance_from_05"] = abs(preds["probability"] - 0.5)

    # ── 5. Lọc samples KHÓ ─────────────────────────────────────────────
    # Chỉ lấy samples có prob < 0.75 (model không chắc chắn)
    uncertain_mask = preds["probability"] < 0.75
    uncertain = preds[uncertain_mask].copy()
    uncertain["review_bucket"] = "model_uncertain"

    print(f"\n=== Sample Selection ===")
    print(f"Uncertain samples (P < 0.75): {len(uncertain):,}")

    # ── 6. Sort by distance from 0.5 (hardest first) ──────────────────
    uncertain = uncertain.sort_values("distance_from_05", ascending=True)

    # Take top N hardest
    selected = uncertain.head(TARGET_SAMPLES).copy()

    print(f"Selected (top {TARGET_SAMPLES} hardest): {len(selected):,}")
    print(f"\nProbability range of selected:")
    print(f"  Min: {selected['probability'].min():.4f}")
    print(f"  Max: {selected['probability'].max():.4f}")
    print(f"  Median: {selected['probability'].median():.4f}")
    print(f"  Mean distance from 0.5: {selected['distance_from_05'].mean():.4f}")

    # ── 7. Add metadata columns ────────────────────────────────────────
    selected["need_review"] = True
    selected["suggested_label"] = selected["phobert_label"]
    selected["final_label"] = ""
    selected["reviewer_note"] = ""

    # Store distance before reindex
    selected_distance = selected["distance_from_05"].mean()
    selected = selected.reindex(columns=BUCKET_COLUMNS)

    # ── 8. Create row_id and blind import file ─────────────────────────
    selected = selected.reset_index(drop=True)
    selected["row_id"] = [f"r5-{i:04d}" for i in range(len(selected))]

    # BLIND file: only row_id + text
    blind = selected[["row_id", "comment_text"]].copy()
    blind = blind.rename(columns={"comment_text": "text"})
    blind.to_csv(IMPORT_FILE, index=False, encoding="utf-8-sig")

    # KEY file: full columns
    selected.to_csv(KEY_FILE, index=False, encoding="utf-8-sig")

    # ── 9. Stats ───────────────────────────────────────────────────────
    phobert_label_counts = selected["phobert_label"].value_counts().to_dict()
    source_label_counts = selected["source_weak_label"].value_counts().to_dict()

    depression_count = (selected["phobert_label"] == "depression").sum()
    normal_count = (selected["phobert_label"] == "normal").sum()

    # ── 10. Save report ────────────────────────────────────────────────
    report = {
        "round": 5,
        "total_selected": int(len(selected)),
        "input_predictions_file": str(PREDICTIONS_FILE),
        "already_reviewed_gold": int(len(gold_texts)),
        "remaining_pool_size": int(len(preds)),
        "selection_strategy": "probability_closest_to_0.5",
        "uncertain_threshold": 0.75,
        "phobert_label_distribution": phobert_label_counts,
        "phobert_depression_count": int(depression_count),
        "phobert_normal_count": int(normal_count),
        "source_weak_label_distribution": source_label_counts,
        "probability_range": {
            "min": round(float(selected["probability"].min()), 4),
            "max": round(float(selected["probability"].max()), 4),
            "median": round(float(selected["probability"].median()), 4),
            "mean": round(float(selected["probability"].mean()), 4),
        },
        "mean_distance_from_05": round(float(selected_distance), 4),
        "output_import_file": str(IMPORT_FILE),
        "output_key_file": str(KEY_FILE),
        "instructions": (
            "Round 5: Only HARDEST samples (model uncertain). "
            "1. Import label_studio_round5_active_learning_import.csv into Label Studio. "
            "2. Review samples (model predictions are in KEY file for reference only). "
            "3. Export as export_round5_active_learning.csv. "
            "4. Run merge script to update gold + final_dataset."
        ),
    }

    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Print Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ACTIVE LEARNING ROUND 5 — READY")
    print(f"{'='*60}")
    print(f"Strategy: TOP {TARGET_SAMPLES} HARDEST SAMPLES")
    print(f"Selection: Probability closest to 0.5 (model uncertain)")
    print(f"\nSelected: {len(selected):,} samples")
    print(f"  - Depression (PhoBERT): {depression_count:,}")
    print(f"  - Normal (PhoBERT): {normal_count:,}")
    print(f"\nProbability Stats:")
    print(f"  Range: [{selected['probability'].min():.4f}, {selected['probability'].max():.4f}]")
    print(f"  Median: {selected['probability'].median():.4f}")
    print(f"  Mean distance from 0.5: {selected_distance:.4f}")
    print(f"\nBucket breakdown:")
    print(f"  model_uncertain: {len(selected):,}")
    print(f"\nImport file (BLIND): {IMPORT_FILE}")
    print(f"Key file (KEEP SECRET): {KEY_FILE}")
    print(f"Report: {REPORT_FILE}")

    # Show some examples
    print(f"\n{'='*60}")
    print(f"TOP 10 HARDEST SAMPLES:")
    print(f"{'='*60}")
    for i, (_, row) in enumerate(selected.head(10).iterrows()):
        text = row['comment_text'][:70] if len(row['comment_text']) > 70 else row['comment_text']
        print(f"{i+1}. P={row['probability']:.4f} | {row['phobert_label']:10s} | {text}...")

    print(f"\n⚠️  Next steps:")
    print(f"   1. Import CSV into Label Studio")
    print(f"   2. Review samples (prioritize depression candidate)")
    print(f"   3. Export as export_round5_active_learning.csv")
    print(f"   4. Run docs/merge_round5_active_learning.py to update gold + final_dataset")


if __name__ == "__main__":
    main()
