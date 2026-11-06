"""Phase 1 — Merge human labels, quality check, rebuild gold.

1. Merge export_step5_review.csv → key_step5 → data/review_samples.csv
2. Merge export_step8_active_learning.csv → key_step8 → data/phobert_active_learning_samples.csv
3. Merge export_round3_active_learning.csv → key_round3 → data/round3_review.csv (new source)
4. Quality check: human vs machine disagreement (step5 + step8 + round3)
5. Rebuild gold_review.csv
6. Eval weak-label + baseline on new gold
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ── path setup ────────────────────────────────────────────────────────
DATA_DIR = PROJECT_DIR / "data"
DOCS_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ── helpers ───────────────────────────────────────────────────────────

def _text_hash(text: str) -> str:
    return hashlib.sha1(str(text).strip().encode("utf-8")).hexdigest()[:10]


def _row_id_to_idx(row_id: str) -> int:
    """s5-0000 → 0, s8-0042 → 42"""
    return int(row_id.split("-")[1])


# ── step 1: merge ─────────────────────────────────────────────────────

def merge_step5() -> dict:
    """Merge step5 export → update review_samples.csv with human labels."""
    src = DATA_DIR / "review_samples.csv"
    export = DOCS_DIR / "export_step5_review.csv"
    key = DOCS_DIR / "label_studio_step5_review_key.csv"


    # Backup
    backup = DATA_DIR / "review_samples.backup_before_phase1.csv"
    if not backup.exists():
        src_df = pd.read_csv(src, dtype=str).fillna("")
        src_df.to_csv(backup, index=False, encoding="utf-8-sig")
        logging.info("Backed up review_samples.csv → %s", backup.name)

    src_df = pd.read_csv(src, dtype=str).fillna("")
    exp_df = pd.read_csv(export, dtype=str).fillna("")
    key_df = pd.read_csv(key, dtype=str).fillna("")

    logging.info("Step5: source=%d rows, export=%d rows, key=%d rows",
                 len(src_df), len(exp_df), len(key_df))

    # Build lookup: row_id → human final_label + reviewer_note
    human = exp_df.set_index("row_id")[["final_label", "reviewer_note"]]

    # Map export row_id to source index
    idx_map = {f"s5-{i:04d}": i for i in range(len(src_df))}

    updated = 0
    for row_id, row in human.iterrows():
        if row_id in idx_map:
            si = idx_map[row_id]
            src_df.at[si, "final_label"] = row["final_label"]
            src_df.at[si, "reviewer_note"] = row.get("reviewer_note", "")
            updated += 1

    src_df.to_csv(src, index=False, encoding="utf-8-sig")
    logging.info("Step5: merged %d/%d rows → %s", updated, len(human), src.name)

    # Also save an annotated key copy for traceability
    key_df["human_final_label"] = key_df["row_id"].map(
        lambda rid: human.at[rid, "final_label"] if rid in human.index else ""
    )
    key_df["original_final_label"] = key_df["final_label"]
    key_df["final_label"] = key_df["human_final_label"]
    key_df.to_csv(DOCS_DIR / "label_studio_step5_review_key_MERGED.csv", index=False, encoding="utf-8-sig")

    return {"file": str(src), "rows": len(src_df), "updated": updated}


def merge_step8() -> dict:
    """Merge step8 export → update phobert_active_learning_samples.csv with human labels."""
    src = DATA_DIR / "phobert_active_learning_samples.csv"
    export = DOCS_DIR / "export_step8_active_learning.csv"
    key = DOCS_DIR / "label_studio_step8_active_learning_key.csv"

    # Backup
    backup = DATA_DIR / "phobert_active_learning_samples.backup_before_phase1.csv"
    if not backup.exists():
        src_df = pd.read_csv(src, dtype=str).fillna("")
        src_df.to_csv(backup, index=False, encoding="utf-8-sig")
        logging.info("Backed up phobert_active_learning_samples.csv → %s", backup.name)

    src_df = pd.read_csv(src, dtype=str).fillna("")
    exp_df = pd.read_csv(export, dtype=str).fillna("")
    key_df = pd.read_csv(key, dtype=str).fillna("")

    logging.info("Step8: source=%d rows, export=%d rows, key=%d rows",
                 len(src_df), len(exp_df), len(key_df))

    human = exp_df.set_index("row_id")[["final_label", "reviewer_note"]]
    idx_map = {f"s8-{i:04d}": i for i in range(len(src_df))}

    updated = 0
    for row_id, row in human.iterrows():
        if row_id in idx_map:
            si = idx_map[row_id]
            src_df.at[si, "final_label"] = row["final_label"]
            src_df.at[si, "reviewer_note"] = row.get("reviewer_note", "")
            updated += 1

    src_df.to_csv(src, index=False, encoding="utf-8-sig")
    logging.info("Step8: merged %d/%d rows → %s", updated, len(human), src.name)

    key_df["human_final_label"] = key_df["row_id"].map(
        lambda rid: human.at[rid, "final_label"] if rid in human.index else ""
    )
    key_df["original_final_label"] = key_df["final_label"]
    key_df["final_label"] = key_df["human_final_label"]
    key_df.to_csv(DOCS_DIR / "label_studio_step8_active_learning_key_MERGED.csv", index=False, encoding="utf-8-sig")

    return {"file": str(src), "rows": len(src_df), "updated": updated}


def merge_round3() -> dict:
    """Merge round 3 export → create data/round3_review.csv.

    Khác với step 5/8: round 3 CHƯA có source file nào trong data/. Source được
    tạo mới từ key file (đã có sẵn từ prepare_round3_active_learning.py) + export
    Label Studio. Join trên row_id (`r3-XXXX` format), lấy final_label +
    reviewer_note từ người review. Output file giữ nguyên schema của key để
    rebuild_gold() có thể concat cùng step5/step8.
    """
    export = DOCS_DIR / "export_round3_active_learning.csv"
    key = DOCS_DIR / "label_studio_round3_active_learning_key.csv"
    out = DATA_DIR / "round3_review.csv"

    if not export.exists():
        raise FileNotFoundError(
            f"Missing round 3 export: {export}. Run Label Studio review first."
        )
    if not key.exists():
        raise FileNotFoundError(
            f"Missing round 3 key: {key}. Run prepare_round3_active_learning.py first."
        )

    # Backup key trước khi merge (pattern giống step 5/8 đã có key_MERGED riêng).
    key_backup = DOCS_DIR / "label_studio_round3_active_learning_key.backup_pre_merge.csv"
    if not key_backup.exists():
        key_df_raw = pd.read_csv(key, dtype=str).fillna("")
        key_df_raw.to_csv(key_backup, index=False, encoding="utf-8-sig")
        logging.info("Backed up round3 key → %s", key_backup.name)

    exp_df = pd.read_csv(export, dtype=str).fillna("")
    key_df = pd.read_csv(key, dtype=str).fillna("")

    logging.info("Round3: export=%d rows, key=%d rows", len(exp_df), len(key_df))

    # Lookup row_id → human label.
    human = exp_df.set_index("row_id")[["final_label", "reviewer_note"]]

    # Annotate key với human_final_label + original_final_label, đồng thời
    # ghi đè final_label bằng human — pattern y hệt step 5/8.
    key_df["human_final_label"] = key_df["row_id"].map(
        lambda rid: human.at[rid, "final_label"] if rid in human.index else ""
    )
    key_df["original_final_label"] = key_df["final_label"]
    key_df["final_label"] = key_df["human_final_label"]
    key_merged = DOCS_DIR / "label_studio_round3_active_learning_key_MERGED.csv"
    key_df.to_csv(key_merged, index=False, encoding="utf-8-sig")
    logging.info("Wrote merged key → %s", key_merged.name)

    # Ghi source file mới với schema giống key (giữ nguyên text_hash, row_id...).
    out.parent.mkdir(parents=True, exist_ok=True)
    key_df.to_csv(out, index=False, encoding="utf-8-sig")

    valid_labels = {"normal", "depression"}
    valid_mask = key_df["final_label"].str.strip().str.lower().isin(valid_labels)
    updated = int(valid_mask.sum())

    label_counts = key_df.loc[valid_mask, "final_label"].str.lower().value_counts().to_dict()
    excluded = int(len(key_df) - updated)

    print("\n" + "=" * 60)
    print(f"ROUND 3 MERGE → {out.name}")
    print("=" * 60)
    print(f"  Total rows:        {len(key_df)}")
    print(f"  Valid (gold):      {updated}")
    print(f"  Excluded:          {excluded} (uncertain + exclude + blank)")
    print(f"  Label distribution: {label_counts}")

    return {
        "file": str(out),
        "rows": len(key_df),
        "updated": updated,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "excluded": excluded,
    }


# ── step 2: quality check ─────────────────────────────────────────────

def quality_check() -> dict:
    """Measure human vs machine disagreement on BOTH review sets."""
    results = {}

    # ── Step 5 ──
    s5 = pd.read_csv(DATA_DIR / "review_samples.csv", dtype=str).fillna("")
    s5["hl"] = s5["final_label"].str.strip().str.lower()
    s5["ml"] = s5["suggested_label"].str.strip().str.lower()

    # Only rows with valid labels
    valid_labels = {"depression", "normal"}
    s5_eval = s5[s5["hl"].isin(valid_labels) & s5["ml"].isin(valid_labels)].copy()

    agree = (s5_eval["hl"] == s5_eval["ml"]).sum()
    disagree = (s5_eval["hl"] != s5_eval["ml"]).sum()
    total = len(s5_eval)

    # Counts
    hl_counts = s5["hl"].value_counts().to_dict()
    ml_counts = s5["ml"].value_counts().to_dict()

    results["step5"] = {
        "total_rows": len(s5),
        "evaluable_rows": total,
        "agreement_count": int(agree),
        "disagreement_count": int(disagree),
        "agreement_rate": round(agree / total * 100, 2) if total else 0,
        "disagreement_rate": round(disagree / total * 100, 2) if total else 0,
        "human_label_counts": hl_counts,
        "machine_label_counts": ml_counts,
    }

    print("\n" + "=" * 60)
    print("QUALITY CHECK — Step 5 (review_samples, n=%d)" % total)
    print("=" * 60)
    print(f"  Human label distribution:  {hl_counts}")
    print(f"  Machine label distribution: {ml_counts}")
    print(f"  Agreement:   {agree}/{total} ({results['step5']['agreement_rate']}%)")
    print(f"  Disagreement: {disagree}/{total} ({results['step5']['disagreement_rate']}%)")

    if total >= 2:
        kappa = cohen_kappa_score(
            s5_eval["hl"].map({"normal": 0, "depression": 1}),
            s5_eval["ml"].map({"normal": 0, "depression": 1}),
        )
        results["step5"]["cohens_kappa"] = round(kappa, 4)
        print(f"  Cohen's kappa: {results['step5']['cohens_kappa']}")

    # Show some disagreements
    dis_df = s5_eval[s5_eval["hl"] != s5_eval["ml"]]
    if len(dis_df) > 0:
        print(f"\n  Sample disagreements (first 5):")
        for _, row in dis_df.head(5).iterrows():
            txt = row["comment_text"][:80].replace("\n", " ")
            print(f"    human={row['hl']:12s} machine={row['ml']:12s}  text: {txt}...")

    # ── Step 8 ──
    s8 = pd.read_csv(DATA_DIR / "phobert_active_learning_samples.csv", dtype=str).fillna("")
    s8["hl"] = s8["final_label"].str.strip().str.lower()
    s8["ml"] = s8["suggested_label"].str.strip().str.lower()

    s8_eval = s8[s8["hl"].isin(valid_labels) & s8["ml"].isin(valid_labels)].copy()

    agree8 = (s8_eval["hl"] == s8_eval["ml"]).sum()
    disagree8 = (s8_eval["hl"] != s8_eval["ml"]).sum()
    total8 = len(s8_eval)

    hl8 = s8["hl"].value_counts().to_dict()
    ml8 = s8["ml"].value_counts().to_dict()

    # Also compare vs phobert_label
    s8["pl"] = s8["phobert_label"].str.strip().str.lower()
    s8_eval2 = s8[s8["hl"].isin(valid_labels) & s8["pl"].isin(valid_labels)].copy()
    agree_phobert = (s8_eval2["hl"] == s8_eval2["pl"]).sum()
    disagree_phobert = (s8_eval2["hl"] != s8_eval2["pl"]).sum()
    total_phobert = len(s8_eval2)

    results["step8"] = {
        "total_rows": len(s8),
        "evaluable_rows": total8,
        "agreement_count": int(agree8),
        "disagreement_count": int(disagree8),
        "agreement_rate": round(agree8 / total8 * 100, 2) if total8 else 0,
        "disagreement_rate": round(disagree8 / total8 * 100, 2) if total8 else 0,
        "human_label_counts": hl8,
        "machine_label_counts": ml8,
        "phobert_vs_human": {
            "agreement": int(agree_phobert),
            "disagreement": int(disagree_phobert),
            "agreement_rate": round(agree_phobert / total_phobert * 100, 2) if total_phobert else 0,
        },
    }

    print("\n" + "=" * 60)
    print("QUALITY CHECK — Step 8 (active_learning, n=%d)" % total8)
    print("=" * 60)
    print(f"  Human label distribution:  {hl8}")
    print(f"  Machine label distribution: {ml8}")
    print(f"  Agreement:   {agree8}/{total8} ({results['step8']['agreement_rate']}%)")
    print(f"  Disagreement: {disagree8}/{total8} ({results['step8']['disagreement_rate']}%)")
    print(f"  PhoBERT vs human agreement: {agree_phobert}/{total_phobert} ({results['step8']['phobert_vs_human']['agreement_rate']}%)")

    if total8 >= 2:
        kappa8 = cohen_kappa_score(
            s8_eval["hl"].map({"normal": 0, "depression": 1}),
            s8_eval["ml"].map({"normal": 0, "depression": 1}),
        )
        results["step8"]["cohens_kappa"] = round(kappa8, 4)
        print(f"  Cohen's kappa: {results['step8']['cohens_kappa']}")

    dis8_df = s8_eval[s8_eval["hl"] != s8_eval["ml"]]
    if len(dis8_df) > 0:
        print(f"\n  Sample disagreements (first 5):")
        for _, row in dis8_df.head(5).iterrows():
            txt = row["comment_text"][:80].replace("\n", " ")
            print(f"    human={row['hl']:12s} machine={row['ml']:12s}  text: {txt}...")

    # ── Round 3 (active learning, n=1500) ──
    r3_path = DATA_DIR / "round3_review.csv"
    r3_summary: dict = {"available": False}
    if r3_path.exists():
        r3_df = pd.read_csv(r3_path, dtype=str).fillna("")
        r3_df["hl"] = r3_df["final_label"].str.strip().str.lower()
        r3_df["ml"] = r3_df["suggested_label"].str.strip().str.lower()

        r3_eval = r3_df[r3_df["hl"].isin(valid_labels) & r3_df["ml"].isin(valid_labels)].copy()
        agree_r3 = int((r3_eval["hl"] == r3_eval["ml"]).sum())
        disagree_r3 = int((r3_eval["hl"] != r3_eval["ml"]).sum())
        total_r3 = len(r3_eval)

        # Chi tiết flip: người review ngược với máy.
        flipped_dep_to_norm = int(
            ((r3_eval["ml"] == "depression") & (r3_eval["hl"] == "normal")).sum()
        )
        flipped_norm_to_dep = int(
            ((r3_eval["ml"] == "normal") & (r3_eval["hl"] == "depression")).sum()
        )

        hl_r3 = r3_df["hl"].value_counts().to_dict()
        ml_r3 = r3_df["ml"].value_counts().to_dict()

        r3_summary = {
            "available": True,
            "total_rows": int(len(r3_df)),
            "evaluable_rows": int(total_r3),
            "agreement_count": agree_r3,
            "disagreement_count": disagree_r3,
            "agreement_rate": round(agree_r3 / total_r3 * 100, 2) if total_r3 else 0,
            "disagreement_rate": round(disagree_r3 / total_r3 * 100, 2) if total_r3 else 0,
            "flipped_depression_to_normal": flipped_dep_to_norm,
            "flipped_normal_to_depression": flipped_norm_to_dep,
            "human_label_counts": hl_r3,
            "machine_label_counts": ml_r3,
        }

        print("\n" + "=" * 60)
        print(f"QUALITY CHECK — Round 3 (active_learning, n={total_r3})")
        print("=" * 60)
        print(f"  Human label distribution:    {hl_r3}")
        print(f"  Machine label distribution:  {ml_r3}")
        print(f"  Agreement:    {agree_r3}/{total_r3} ({r3_summary['agreement_rate']}%)")
        print(f"  Disagreement: {disagree_r3}/{total_r3} ({r3_summary['disagreement_rate']}%)")
        print(f"  Flipped depression→normal: {flipped_dep_to_norm}")
        print(f"  Flipped normal→depression: {flipped_norm_to_dep}")

        if total_r3 >= 2:
            kappa_r3 = cohen_kappa_score(
                r3_eval["hl"].map({"normal": 0, "depression": 1}),
                r3_eval["ml"].map({"normal": 0, "depression": 1}),
            )
            r3_summary["cohens_kappa"] = round(float(kappa_r3), 4)
            print(f"  Cohen's kappa: {r3_summary['cohens_kappa']}")

        dis_r3_df = r3_eval[r3_eval["hl"] != r3_eval["ml"]]
        if len(dis_r3_df) > 0:
            print(f"\n  Sample disagreements (first 5):")
            for _, row in dis_r3_df.head(5).iterrows():
                txt = row["comment_text"][:80].replace("\n", " ")
                print(f"    human={row['hl']:12s} machine={row['ml']:12s}  text: {txt}...")
    else:
        print("\n[round3 quality check] SKIPPED — data/round3_review.csv not found.")

    results["round3"] = r3_summary

    # ── Combined ──
    r3_hl = r3_summary.get("human_label_counts", {}) if r3_summary.get("available") else {}
    results["combined"] = {
        "total_human_labeled": len(s5) + len(s8) + (r3_summary.get("total_rows", 0) if r3_summary.get("available") else 0),
        "depression_total": hl_counts.get("depression", 0) + hl8.get("depression", 0) + r3_hl.get("depression", 0),
        "normal_total": hl_counts.get("normal", 0) + hl8.get("normal", 0) + r3_hl.get("normal", 0),
        "uncertain_total": hl_counts.get("uncertain", 0) + hl8.get("uncertain", 0) + r3_hl.get("uncertain", 0),
        "exclude_total": hl_counts.get("exclude", 0) + hl8.get("exclude", 0) + r3_hl.get("exclude", 0),
    }

    print("\n" + "=" * 60)
    print("COMBINED SUMMARY")
    print("=" * 60)
    c = results["combined"]
    print(f"  Total labeled:        {c['total_human_labeled']}")
    print(f"  Depression (gold):    {c['depression_total']}")
    print(f"  Normal (gold):        {c['normal_total']}")
    print(f"  Uncertain (excluded): {c['uncertain_total']}")
    print(f"  Exclude (excluded):   {c['exclude_total']}")
    print(f"  Usable gold:          {c['depression_total'] + c['normal_total']}")

    return results


# ── step 3: rebuild gold ──────────────────────────────────────────────

def rebuild_gold() -> dict:
    """Build new gold_review.csv from BOTH review_samples + active_learning."""
    from yt_depression_crawler.processing.cleaner import normalize_text

    gold_cols = [
        "comment_text", "label", "final_label", "weak_label",
        "confidence", "depression_score", "matched_keywords",
        "review_bucket", "reviewer_note",
        "source",  # step5 or step8
    ]

    # Load step5
    s5 = pd.read_csv(DATA_DIR / "review_samples.csv", dtype=str).fillna("")
    s5["source"] = "step5"

    # Load step8
    s8 = pd.read_csv(DATA_DIR / "phobert_active_learning_samples.csv", dtype=str).fillna("")
    s8["source"] = "step8"

    # Load round 3 (optional — gracefully skip nếu chưa merge).
    r3_path = DATA_DIR / "round3_review.csv"
    if r3_path.exists():
        r3 = pd.read_csv(r3_path, dtype=str).fillna("")
        r3["source"] = "round3"
    else:
        r3 = pd.DataFrame()

    # Combine
    df = pd.concat([s5, s8, r3], ignore_index=True)

    # Filter to valid labels only
    df["fl"] = df["final_label"].str.strip().str.lower()
    valid = {"normal": 0, "depression": 1}
    gold = df[df["fl"].isin(valid)].copy()

    gold["comment_text"] = gold["comment_text"].map(normalize_text)
    gold = gold.drop_duplicates(subset=["comment_text"], keep="first")
    gold["label"] = gold["fl"].map(valid).astype(int)

    # depression_score might not exist in step8, handle gracefully
    if "depression_score" in gold.columns:
        gold["depression_score"] = pd.to_numeric(gold["depression_score"], errors="coerce").fillna(0).astype(int)

    # Select columns that exist
    available_cols = [c for c in gold_cols if c in gold.columns]
    gold = gold.reindex(columns=available_cols)

    gold_file = DATA_DIR / "gold_review.csv"
    gold.to_csv(gold_file, index=False, encoding="utf-8-sig")

    label_counts = gold["final_label"].value_counts().to_dict()

    report = {
        "total_rows_before_filter": int(len(df)),
        "gold_rows": int(len(gold)),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "sources": {
            "step5": int((gold["source"] == "step5").sum()) if "source" in gold.columns else 0,
            "step8": int((gold["source"] == "step8").sum()) if "source" in gold.columns else 0,
            "round3": int((gold["source"] == "round3").sum()) if "source" in gold.columns else 0,
        },
        "excluded": int(len(df) - len(gold)),
    }

    print("\n" + "=" * 60)
    print("REBUILD GOLD")
    print("=" * 60)
    print(f"  Combined rows:        {len(df)}")
    print(f"  Gold rows (valid):    {len(gold)}")
    print(f"  Excluded:             {len(df) - len(gold)}")
    print(f"  Label distribution:   {label_counts}")
    print(f"  Output:               {gold_file}")

    return report


# ── step 4: evaluate ──────────────────────────────────────────────────

def run_evaluations():
    """Re-run weak-label eval + baseline eval on new gold."""
    print("\n" + "=" * 60)
    print("EVALUATING WEAK LABELS ON NEW GOLD")
    print("=" * 60)

    from yt_depression_crawler.labeling.review_evaluator import evaluate_weak_labels_on_gold

    weak_report = evaluate_weak_labels_on_gold()
    print(json.dumps(weak_report, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("EVALUATING BASELINE ON NEW GOLD")
    print("=" * 60)

    from yt_depression_crawler.labeling.gold_baseline_eval import evaluate_baseline_on_gold

    baseline_report = evaluate_baseline_on_gold()
    print(json.dumps(baseline_report, ensure_ascii=False, indent=2))

    # Save full report
    report = {
        "phase1_timestamp": pd.Timestamp.now().isoformat(),
        "weak_label_eval": weak_report,
        "baseline_gold_eval": baseline_report,
    }
    report_file = DOCS_DIR / "phase1_eval_report.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report saved to {report_file}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 1: Merge human labels → Quality Check → Rebuild Gold")
    print("=" * 60)

    # Step 1
    print("\n>>> Step 1: Merging human labels into source files...")
    r5 = merge_step5()
    r8 = merge_step8()
    r3 = merge_round3()
    print(f"  Step5:  {r5['updated']}/{r5['rows']} rows updated")
    print(f"  Step8:  {r8['updated']}/{r8['rows']} rows updated")
    print(f"  Round3: {r3['updated']}/{r3['rows']} rows updated ({r3['excluded']} excluded)")

    # Step 2
    print("\n>>> Step 2: Quality check...")
    qc = quality_check()

    # Step 3
    print("\n>>> Step 3: Rebuilding gold...")
    gold = rebuild_gold()

    # Step 4
    print("\n>>> Step 4: Re-evaluating...")
    run_evaluations()

    # Final summary
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"  review_samples.csv updated:   {r5['updated']} rows")
    print(f"  active_learning_samples.csv:  {r8['updated']} rows")
    print(f"  Gold set size:                {gold['gold_rows']} rows")
    print(f"  Step5 disagreement rate:      {qc['step5']['disagreement_rate']}%")
    print(f"  Step8 disagreement rate:      {qc['step8']['disagreement_rate']}%")


if __name__ == "__main__":
    main()
