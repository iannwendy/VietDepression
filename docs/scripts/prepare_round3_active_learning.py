"""Active Learning Round 3 — Select samples for human review and create Label Studio import files.

Selection strategy — prioritize samples with highest information value:
  1. Weak-label vs PhoBERT disagreement (most valuable — model contradicts keyword)
  2. Boundary probability (0.45–0.55 — model is maximally uncertain)
  3. Low confidence (0.55–0.70 — model is somewhat uncertain)
  4. Source uncertain (weak-labeler said uncertain, PhoBERT took a side)

Outputs (in docs/):
  - label_studio_round3_active_learning_import.csv  (BLIND — only row_id + text)
  - label_studio_round3_active_learning_key.csv     (full columns for merge later)

Usage:
  .venv/bin/python docs/prepare_round3_active_learning.py
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

# ── Inputs ───────────────────────────────────────────────────────────
PREDICTIONS_FILE = DATA_DIR / "phobert_remaining_predictions_v2.csv"
GOLD_FILE = DATA_DIR / "gold_review.csv"
ACTIVE_LEARNING_FILE = DATA_DIR / "phobert_active_learning_samples.csv"

# ── Outputs ──────────────────────────────────────────────────────────
IMPORT_FILE = DOCS_DIR / "label_studio_round3_active_learning_import.csv"
KEY_FILE = DOCS_DIR / "label_studio_round3_active_learning_key.csv"
REPORT_FILE = DOCS_DIR / "round3_selection_report.json"

# ── Settings ─────────────────────────────────────────────────────────
TOTAL_SAMPLES = 1500  # Total to select
DISAGREEMENT_TARGET = 500
BOUNDARY_TARGET = 400
LOW_CONFIDENCE_TARGET = 350
SOURCED_UNCERTAIN_TARGET = 250
RANDOM_SEED = 42

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
    # ── 1. Load data ──────────────────────────────────────────────────
    preds = pd.read_csv(PREDICTIONS_FILE, dtype=str).fillna("")
    print(f"Loaded predictions: {len(preds):,} rows")

    # Load already-reviewed text keys
    gold = pd.read_csv(GOLD_FILE, dtype=str).fillna("")
    gold_texts = set(gold["comment_text"].str.strip().tolist())

    al = pd.read_csv(ACTIVE_LEARNING_FILE, dtype=str).fillna("")
    al_texts = set(al["comment_text"].str.strip().tolist())

    already_reviewed = gold_texts | al_texts
    print(f"Already reviewed: {len(already_reviewed):,} unique texts "
          f"(gold={len(gold_texts):,}, active_learning={len(al_texts):,})")

    # ── 2. Filter out already-reviewed ─────────────────────────────────
    preds = preds[~preds["comment_text"].str.strip().isin(already_reviewed)].copy()
    print(f"Remaining after filtering reviewed: {len(preds):,}")

    # ── 3. Parse numeric columns ───────────────────────────────────────
    preds["probability"] = pd.to_numeric(preds["probability"], errors="coerce").fillna(0.0)
    preds["prob_normal"] = pd.to_numeric(preds["prob_normal"], errors="coerce").fillna(0.0)
    preds["prob_depression"] = pd.to_numeric(preds["prob_depression"], errors="coerce").fillna(0.0)
    preds["source_depression_score"] = pd.to_numeric(
        preds["source_depression_score"], errors="coerce"
    ).fillna(0).astype(int)

    # ── 4. Categorize into buckets ─────────────────────────────────────
    source_label = preds["source_weak_label"].fillna("")
    weak_normal = source_label.eq("normal_auto")
    weak_depression = source_label.eq("depression_auto")
    weak_uncertain = source_label.eq("uncertain")

    # 4a. Disagreement: weak-label says one thing, PhoBERT says another
    disagreement_mask = (
        (weak_normal & (preds["phobert_label"] == "depression"))
        | (weak_depression & (preds["phobert_label"] == "normal"))
    )
    disagreement = preds[disagreement_mask].copy()
    disagreement["review_bucket"] = "weak_model_disagreement"

    # 4b. Boundary: probability exactly where model is most confused
    boundary_mask = (preds["probability"] >= 0.45) & (preds["probability"] <= 0.55)
    boundary = preds[boundary_mask & ~disagreement_mask].copy()
    boundary["review_bucket"] = "boundary_probability"

    # 4c. Low confidence: somewhat uncertain but not boundary
    low_conf_mask = (preds["probability"] > 0.55) & (preds["probability"] < 0.70)
    low_conf = preds[low_conf_mask & ~disagreement_mask & ~boundary_mask].copy()
    low_conf["review_bucket"] = "low_confidence"

    # 4d. Source uncertain + PhoBERT confident — does PhoBERT guess right?
    uncertain_mask = weak_uncertain & ~disagreement_mask & ~boundary_mask & ~low_conf_mask
    uncertain = preds[uncertain_mask].copy()
    uncertain["review_bucket"] = "source_uncertain"

    print(f"\nBucket sizes before sampling:")
    print(f"  Disagreement:        {len(disagreement):,}")
    print(f"  Boundary probability: {len(boundary):,}")
    print(f"  Low confidence:       {len(low_conf):,}")
    print(f"  Source uncertain:     {len(uncertain):,}")

    # ── 5. Sample from each bucket ─────────────────────────────────────
    used_texts: set[str] = set()
    samples: list[pd.DataFrame] = []
    bucket_stats: dict[str, dict] = {}

    buckets = [
        ("weak_model_disagreement", disagreement, DISAGREEMENT_TARGET),
        ("boundary_probability", boundary, BOUNDARY_TARGET),
        ("low_confidence", low_conf, LOW_CONFIDENCE_TARGET),
        ("source_uncertain", uncertain, SOURCED_UNCERTAIN_TARGET),
    ]

    for idx, (name, bucket_df, target) in enumerate(buckets):
        available = bucket_df[~bucket_df["comment_text"].isin(used_texts)]
        take = min(target, len(available))

        if take > 0:
            # Sort by probability ascending (most uncertain first)
            available = available.sort_values("probability", ascending=True)
            sampled = available.head(take).copy()
            used_texts.update(sampled["comment_text"].tolist())
            samples.append(sampled)

        bucket_stats[name] = {
            "available": int(len(bucket_df)),
            "taken": take,
            "target": target,
        }

    combined = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()

    # ── 6. Fill up to target with additional samples from any bucket ──
    if len(combined) < TOTAL_SAMPLES:
        remaining_pool = preds[
            ~preds["comment_text"].isin(used_texts)
            & ~preds["comment_text"].isin(already_reviewed)
        ].copy()
        remaining_pool = remaining_pool.sort_values("probability", ascending=True)
        extra_needed = TOTAL_SAMPLES - len(combined)
        extra = remaining_pool.head(extra_needed).copy()
        if len(extra) > 0:
            extra["review_bucket"] = "extra_low_probability"
            combined = pd.concat([combined, extra], ignore_index=True)
            bucket_stats["extra_low_probability"] = {
                "available": int(len(remaining_pool)),
                "taken": int(len(extra)),
                "target": extra_needed,
            }

    # ── 7. Add metadata columns ────────────────────────────────────────
    combined["need_review"] = True
    combined["suggested_label"] = combined["phobert_label"]
    combined["final_label"] = ""
    combined["reviewer_note"] = ""
    combined = combined.reindex(columns=BUCKET_COLUMNS)

    # ── 8. Create row_id and blind import file ─────────────────────────
    combined = combined.reset_index(drop=True)
    combined["row_id"] = [f"r3-{i:04d}" for i in range(len(combined))]
    combined["text_hash"] = combined["comment_text"].map(_text_hash)

    # BLIND file: only row_id + text (hide ALL machine suggestions)
    blind = combined[["row_id", "comment_text"]].copy()
    blind = blind.rename(columns={"comment_text": "text"})
    blind = blind[blind["text"].str.strip().ne("")]
    blind.to_csv(IMPORT_FILE, index=False, encoding="utf-8-sig")

    # KEY file: full columns for merge after review
    combined.to_csv(KEY_FILE, index=False, encoding="utf-8-sig")

    # ── 9. Stats ───────────────────────────────────────────────────────
    phobert_label_counts = (
        combined["phobert_label"].value_counts().to_dict() if not combined.empty else {}
    )
    source_label_counts = (
        combined["source_weak_label"].value_counts().to_dict() if not combined.empty else {}
    )

    report = {
        "round": 3,
        "total_selected": int(len(combined)),
        "input_predictions_file": str(PREDICTIONS_FILE),
        "already_reviewed_gold": int(len(gold_texts)),
        "already_reviewed_active_learning": int(len(al_texts)),
        "remaining_pool_size": int(len(preds)),
        "bucket_stats": bucket_stats,
        "phobert_label_distribution": phobert_label_counts,
        "source_weak_label_distribution": source_label_counts,
        "probability_range": {
            "min": round(float(combined["probability"].min()), 4) if not combined.empty else None,
            "max": round(float(combined["probability"].max()), 4) if not combined.empty else None,
            "median": round(float(combined["probability"].median()), 4) if not combined.empty else None,
        },
        "output_import_file": str(IMPORT_FILE),
        "output_key_file": str(KEY_FILE),
        "instructions": (
            "1. Import label_studio_round3_active_learning_import.csv into Label Studio "
            "(Settings → Labeling Interface → Code — use the same XML config as before). "
            "2. After review, export CSV and rename to export_round3_active_learning.csv. "
            "3. Run the merge script to combine final_label back via row_id in the key file. "
            "IMPORTANT: This is a BLIND review file — annotators see ONLY row_id + text. "
            "DO NOT share the key file with annotators."
        ),
    }

    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"ACTIVE LEARNING ROUND 3 — READY")
    print(f"{'='*60}")
    print(f"Selected: {len(combined):,} samples")
    print(f"PhoBERT label distribution: {phobert_label_counts}")
    print(f"Source weak-label distribution: {source_label_counts}")
    print(f"Probability range: [{report['probability_range']['min']}, "
          f"{report['probability_range']['max']}], "
          f"median={report['probability_range']['median']}")
    print(f"\nBucket breakdown:")
    for name, stats in bucket_stats.items():
        print(f"  {name:30s} → {stats['taken']:>4d}/{stats['available']:>6,d} taken")
    print(f"\nImport file (BLIND — for reviewer): {IMPORT_FILE}")
    print(f"Key file    (KEEP SECRET):         {KEY_FILE}")
    print(f"Report:                              {REPORT_FILE}")


if __name__ == "__main__":
    main()
