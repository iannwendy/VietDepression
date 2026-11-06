"""Round 5 Active Learning - Prepare samples for human review.

This script selects the HARDEST samples (probability closest to 0.5)
from the new PhoBERT predictions after Round 4 retraining.

Strategy: Focus ONLY on uncertain samples (model không chắc chắn)
- Take samples with probability GẦN 0.5 NHẤT
- No random sampling

Usage:
    .venv/bin/python scripts/prepare_round5_active_learning.py
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
MODEL_DIR = PROJECT_DIR / "models"

# ── Inputs ──────────────────────────────────────────────────────────────
PREDICTIONS_FILE = MODEL_DIR / "round5_predictions" / "round5_phobert_predictions.csv"
GOLD_FILE = DATA_DIR / "gold_review.csv"

# ── Outputs ─────────────────────────────────────────────────────────────
IMPORT_FILE = DOCS_DIR / "label_studio_round5_active_learning_import.csv"
KEY_FILE = DOCS_DIR / "label_studio_round5_active_learning_key.csv"
REPORT_FILE = DOCS_DIR / "round5_selection_report.json"

# ── Settings ────────────────────────────────────────────────────────────
# Lấy top 2000 samples KHÓ NHẤT (probability gần 0.5 nhất)
TARGET_SAMPLES = 2000
PROBABILITY_CUTOFF = 0.70  # Chỉ lấy samples có prob < 0.70 (model không chắc chắn)

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
    # ── 1. Load predictions từ model mới ──────────────────────────────
    if not PREDICTIONS_FILE.exists():
        print(f"❌ Predictions file not found: {PREDICTIONS_FILE}")
        print("   Run retrain_phobert_for_round5.py first!")
        sys.exit(1)

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
    preds["phobert_prob"] = pd.to_numeric(preds["phobert_prob"], errors="coerce").fillna(0.5)
    preds["prob_normal"] = preds["phobert_prob"].apply(lambda x: 1 - x)
    preds["prob_depression"] = preds["phobert_prob"]

    # ── 5. Lọc samples KHÓ (model không chắc chắn) ───────────────────
    # Chỉ lấy samples có probability < 0.70 (model không chắc chắn)
    uncertain = preds[preds["probability"] < PROBABILITY_CUTOFF].copy()
    uncertain["distance_from_05"] = abs(uncertain["probability"] - 0.5)
    uncertain["review_bucket"] = "model_uncertain"

    print(f"\n=== Sample Selection ===")
    print(f"Uncertain samples (P < {PROBABILITY_CUTOFF}): {len(uncertain):,}")

    # ── 6. Sort by distance from 0.5 (hardest first) ──────────────────
    uncertain = uncertain.sort_values("distance_from_05", ascending=True)

    # Take top N hardest
    selected = uncertain.head(TARGET_SAMPLES).copy()

    print(f"Selected (top {TARGET_SAMPLES} hardest): {len(selected):,}")
    print(f"\nProbability range of selected:")
    print(f"  Min: {selected['probability'].min():.4f}")
    print(f"  Max: {selected['probability'].max():.4f}")
    print(f"  Median: {selected['probability'].median():.4f}")

    # ── 7. Add metadata columns ────────────────────────────────────────
    selected["need_review"] = True
    selected["suggested_label"] = selected["phobert_label"]
    selected["final_label"] = ""
    selected["reviewer_note"] = ""
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
        "probability_cutoff": PROBABILITY_CUTOFF,
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
    print(f"\nBucket breakdown:")
    print(f"  model_uncertain: {len(selected):,}")
    print(f"\nImport file (BLIND): {IMPORT_FILE}")
    print(f"Key file (KEEP SECRET): {KEY_FILE}")
    print(f"Report: {REPORT_FILE}")
    print(f"\n⚠️  Next steps:")
    print(f"   1. Import CSV into Label Studio")
    print(f"   2. Review samples (prioritize depression candidate)")
    print(f"   3. Export as export_round5_active_learning.csv")
    print(f"   4. Run merge script to update gold + final_dataset")


if __name__ == "__main__":
    main()
