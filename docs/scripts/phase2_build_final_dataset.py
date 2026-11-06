"""Phase 2 — Train PhoBERT v2 on gold → Build final_dataset.csv."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
DOCS_DIR = Path(__file__).resolve().parent
UNIFIED_DIR = PROJECT_DIR / "data_unified"

PHOBERT_V2_DIR = MODEL_DIR / "phobert_second"
GOLD_REVIEW = DATA_DIR / "gold_review.csv"
AUTO_LABELED = DATA_DIR / "auto_labeled_comments.csv"
REMAINING_PREDS_V2 = DATA_DIR / "phobert_remaining_predictions_v2.csv"
CONFIDENT_PREDS_V2 = DATA_DIR / "phobert_confident_predictions_v2.csv"
FINAL_DATASET = DATA_DIR / "final_dataset.csv"
FINAL_TRAIN = DATA_DIR / "final_train.csv"
FINAL_VAL = DATA_DIR / "final_val.csv"
FINAL_TEST = DATA_DIR / "final_test.csv"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def create_gold_splits():
    from yt_depression_crawler.processing.cleaner import normalize_text

    gold = pd.read_csv(GOLD_REVIEW, dtype=str).fillna("")
    gold["label"] = gold["final_label"].map({"normal": 0, "depression": 1}).astype(int)
    gold["comment_text"] = gold["comment_text"].map(normalize_text)
    gold = gold.drop_duplicates(subset=["comment_text"], keep="first")

    split_cols = ["comment_text", "label", "weak_label", "confidence", "depression_score", "matched_keywords"]
    available = [c for c in split_cols if c in gold.columns]
    gold = gold.reindex(columns=available)

    temp_ratio = VAL_RATIO + TEST_RATIO
    train_df, temp_df = train_test_split(gold, test_size=temp_ratio, random_state=SEED, stratify=gold["label"])
    relative_test_ratio = TEST_RATIO / temp_ratio
    val_df, test_df = train_test_split(temp_df, test_size=relative_test_ratio, random_state=SEED, stratify=temp_df["label"])

    for path, split_df in [
        (DATA_DIR / "train_gold.csv", train_df),
        (DATA_DIR / "val_gold.csv", val_df),
        (DATA_DIR / "test_gold.csv", test_df),
    ]:
        split_df.to_csv(path, index=False, encoding="utf-8-sig")

    report = {
        "total_rows": int(len(gold)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_labels": train_df["label"].value_counts().to_dict(),
    }

    print("\n>>> Gold Split Report")
    print(f"  Total: {report['total_rows']}")
    print(f"  Train: {report['train_rows']} ({report['train_labels']})")
    print(f"  Val:   {report['val_rows']}")
    print(f"  Test:  {report['test_rows']}")
    return report


def train_phobert_v2():
    from yt_depression_crawler.modeling.phobert.phobert_train import train_phobert_first

    PHOBERT_V2_DIR.mkdir(parents=True, exist_ok=True)

    report = train_phobert_first(
        train_file=DATA_DIR / "train_gold.csv",
        val_file=DATA_DIR / "val_gold.csv",
        test_file=DATA_DIR / "test_gold.csv",
        output_dir=PHOBERT_V2_DIR,
        metrics_file=PHOBERT_V2_DIR / "phobert_metrics.json",
    )

    print(f"\n>>> PhoBERT v2 Training Report")
    print(f"  Best epoch: {report['best_epoch']} (val F1: {report['best_val_f1_macro']})")
    print(f"  Test F1 macro: {report['test']['f1_macro']}")
    print(f"  Test F1 depression: {report['test']['f1_depression']}")
    if report.get("gold"):
        print(f"  Gold F1 macro: {report['gold']['f1_macro']}")
        print(f"  Gold F1 depression: {report['gold']['f1_depression']}")
    return report


def predict_with_v2():
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from tqdm import tqdm

    from yt_depression_crawler.core.config import (
        PHOBERT_MAX_LENGTH, PHOBERT_PREDICT_BATCH_SIZE, PHOBERT_PREDICT_CHUNK_SIZE,
        PHOBERT_PREDICTION_COLUMNS,
    )
    from yt_depression_crawler.modeling.phobert.phobert_utils import (
        ID_TO_LABEL, PhoBertDataset, get_device, prepare_many_texts,
    )

    if not (PHOBERT_V2_DIR / "config.json").exists():
        raise FileNotFoundError(f"Missing PhoBERT v2 model in {PHOBERT_V2_DIR}")

    gold = pd.read_csv(GOLD_REVIEW, dtype=str).fillna("")
    gold_texts = set(gold["comment_text"].str.strip().tolist())

    df = pd.read_csv(AUTO_LABELED, dtype=str).fillna("")
    df = df[df["comment_text"].str.strip().ne("")].copy()
    df = df.drop_duplicates(subset=["comment_text"], keep="first")
    remaining = df[~df["comment_text"].isin(gold_texts)].copy()
    logging.info("Predicting on %d comments (excluded %d gold)", len(remaining), len(gold_texts))

    device = get_device()
    logging.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(str(PHOBERT_V2_DIR), use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(PHOBERT_V2_DIR))
    model.to(device)
    model.eval()

    output_file = REMAINING_PREDS_V2
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = True

    for start in tqdm(range(0, len(remaining), PHOBERT_PREDICT_CHUNK_SIZE), desc="PhoBERT v2 predict"):
        chunk_df = remaining.iloc[start : start + PHOBERT_PREDICT_CHUNK_SIZE].copy()
        texts = prepare_many_texts(chunk_df["comment_text"].tolist())
        dataset = PhoBertDataset(texts, None, tokenizer, PHOBERT_MAX_LENGTH)
        data_loader = DataLoader(dataset, batch_size=PHOBERT_PREDICT_BATCH_SIZE)

        probs_normal, probs_depression, labels, probabilities = [], [], [], []
        with torch.no_grad():
            for batch in data_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1).detach().cpu()
                pred_ids = torch.argmax(probs, dim=-1).tolist()
                for idx, pred_id in enumerate(pred_ids):
                    np_val = float(probs[idx, 0].item())
                    dp_val = float(probs[idx, 1].item())
                    prob = dp_val if pred_id == 1 else np_val
                    labels.append(ID_TO_LABEL[pred_id])
                    probabilities.append(round(prob, 4))
                    probs_normal.append(round(np_val, 4))
                    probs_depression.append(round(dp_val, 4))

        out_df = pd.DataFrame({
            "comment_text": chunk_df["comment_text"].tolist(),
            "source_weak_label": chunk_df.get("weak_label", "").tolist(),
            "source_confidence": chunk_df.get("confidence", "").tolist(),
            "source_depression_score": chunk_df.get("depression_score", "").tolist(),
            "phobert_label": labels,
            "probability": probabilities,
            "prob_normal": probs_normal,
            "prob_depression": probs_depression,
        }).reindex(columns=PHOBERT_PREDICTION_COLUMNS)

        out_df.to_csv(output_file, mode="a", header=write_header, index=False, encoding="utf-8-sig")
        write_header = False

    result = pd.read_csv(output_file, dtype=str).fillna("")
    result["probability"] = pd.to_numeric(result["probability"], errors="coerce")
    conf_mask = result["probability"] >= 0.9
    confident = result[conf_mask].copy()
    confident["label"] = confident["phobert_label"]
    confident.to_csv(CONFIDENT_PREDS_V2, index=False, encoding="utf-8-sig")

    report = {
        "predicted_total": int(len(result)),
        "confident_total": int(len(confident)),
        "confident_labels": confident["phobert_label"].value_counts().to_dict() if not confident.empty else {},
    }
    print(f"\n>>> PhoBERT v2 Predictions: {report['predicted_total']} total, {report['confident_total']} confident (prob>=0.9)")
    print(f"  Confident labels: {report['confident_labels']}")
    return report


def build_final_dataset():
    from yt_depression_crawler.processing.cleaner import normalize_text

    print("\n>>> Building final dataset...")

    # Source A: Human gold
    gold = pd.read_csv(GOLD_REVIEW, dtype=str).fillna("")
    gold["label"] = gold["final_label"].map({"normal": 0, "depression": 1}).astype(int)
    gold["source"] = "human_gold"
    gold["weight"] = 3
    print(f"  A (human gold): {len(gold)}")

    # Source B: Weak high-confidence
    weak = pd.read_csv(AUTO_LABELED, dtype=str).fillna("")
    weak_hi = weak[weak["confidence"] == "high"].copy()
    weak_hi["label"] = weak_hi["weak_label"].map({"normal_auto": 0, "depression_auto": 1}).astype(int)
    weak_hi = weak_hi[weak_hi["weak_label"].isin(["normal_auto", "depression_auto"])]
    weak_hi["source"] = "weak_high_conf"
    weak_hi["weight"] = 2
    print(f"  B (weak high-conf): {len(weak_hi)}")

    # Source C: PhoBERT v2 confident
    if CONFIDENT_PREDS_V2.exists():
        pc = pd.read_csv(CONFIDENT_PREDS_V2, dtype=str).fillna("")
        pc["probability"] = pd.to_numeric(pc["probability"], errors="coerce")
        pc = pc[pc["probability"] >= 0.9].copy()
        pc["label"] = pc["phobert_label"].map({"normal": 0, "depression": 1}).astype(int)
        pc["source"] = "phobert_v2_confident"
        pc["weight"] = 1
        max_pc = 20000
        if len(pc) > max_pc:
            pc = pc.sample(n=max_pc, random_state=SEED)
        print(f"  C (PhoBERT v2 confident): {len(pc)}")
    else:
        pc = pd.DataFrame()
        print("  C (PhoBERT v2 confident): SKIPPED")

    # Source D: External negatives
    ext = pd.read_csv(UNIFIED_DIR / "augmentation_negatives.csv", dtype=str).fillna("")
    ext["label"] = pd.to_numeric(ext["label"], errors="coerce").fillna(0).astype(int)
    ext = ext[ext["label"] == 0].copy()
    ext["source"] = "external_negative"
    ext["weight"] = 2
    n_ext = min(10000, len(ext))
    ext = ext.sample(n=n_ext, random_state=SEED)
    print(f"  D (external negatives): {len(ext)}")

    # Combine
    combine_cols = ["comment_text", "label", "source", "weight"]
    parts = []
    for part in [gold, weak_hi, pc, ext]:
        if part.empty:
            continue
        avail = [c for c in combine_cols if c in part.columns]
        parts.append(part.reindex(columns=avail))

    combined = pd.concat(parts, ignore_index=True)
    print(f"\n  Combined before dedup: {len(combined)}")

    combined["comment_text"] = combined["comment_text"].map(normalize_text)
    combined = combined.drop_duplicates(subset=["comment_text"], keep="first")
    print(f"  After dedup: {len(combined)}")

    label_counts = combined["label"].value_counts().to_dict()
    n_dep = label_counts.get(1, 0)
    n_norm = label_counts.get(0, 0)
    print(f"  Labels: {label_counts}")

    # Balance: downsample normal to 2x depression
    if n_dep > 0 and n_norm > n_dep * 2:
        target_norm = n_dep * 2
        norm_idx = combined[combined["label"] == 0]
        # Keep human_gold and weak_high_conf normals first (higher weight)
        norm_priority = norm_idx[norm_idx["source"] != "external_negative"]
        norm_ext = norm_idx[norm_idx["source"] == "external_negative"]

        # Take priority normals first (up to target)
        n_priority_take = min(len(norm_priority), target_norm)
        norm_priority = norm_priority.sample(n=n_priority_take, random_state=SEED)

        # Fill remaining with external negatives
        n_ext_take = max(0, target_norm - n_priority_take)
        if len(norm_ext) > n_ext_take:
            norm_ext = norm_ext.sample(n=n_ext_take, random_state=SEED)

        dep_idx = combined[combined["label"] == 1]
        combined = pd.concat([dep_idx, norm_priority, norm_ext], ignore_index=True)
        print(f"  Balanced: {len(combined)} rows ({n_dep} dep + {len(norm_priority) + len(norm_ext)} norm)")

    combined = combined.reindex(columns=combine_cols)
    combined.to_csv(FINAL_DATASET, index=False, encoding="utf-8-sig")
    print(f"  Final dataset: {FINAL_DATASET} ({len(combined)} rows)")

    # Split
    if combined["label"].nunique() >= 2:
        temp_r = VAL_RATIO + TEST_RATIO
        tr, tp = train_test_split(combined, test_size=temp_r, random_state=SEED, stratify=combined["label"])
        rt_ratio = TEST_RATIO / temp_r
        va, te = train_test_split(tp, test_size=rt_ratio, random_state=SEED, stratify=tp["label"])

        for p, d in [(FINAL_TRAIN, tr), (FINAL_VAL, va), (FINAL_TEST, te)]:
            d.to_csv(p, index=False, encoding="utf-8-sig")

        print(f"\n  Final splits:")
        print(f"    Train: {len(tr)} ({tr['label'].value_counts().to_dict()})")
        print(f"    Val:   {len(va)} ({va['label'].value_counts().to_dict()})")
        print(f"    Test:  {len(te)} ({te['label'].value_counts().to_dict()})")

    return {"final_rows": int(len(combined)), "depression": int(n_dep), "normal": int(len(combined) - n_dep),
            "sources": combined["source"].value_counts().to_dict(), "output": str(FINAL_DATASET)}


def main():
    print("=" * 60)
    print("PHASE 2: PhoBERT v2 -> final_dataset.csv")
    print("=" * 60)

    print("\n>>> STEP 1: Creating gold splits")
    split_report = create_gold_splits()

    print("\n>>> STEP 2: Training PhoBERT v2 on gold")
    train_report = train_phobert_v2()

    print("\n>>> STEP 3: Predicting with PhoBERT v2")
    pred_report = predict_with_v2()

    print("\n>>> STEP 4: Building final_dataset.csv")
    final_report = build_final_dataset()

    full_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "phase": 2,
        "gold_splits": split_report,
        "phobert_v2_train": {
            "best_epoch": train_report["best_epoch"],
            "best_val_f1": train_report["best_val_f1_macro"],
            "test_f1_macro": train_report["test"]["f1_macro"],
            "test_f1_depression": train_report["test"]["f1_depression"],
        },
        "phobert_v2_predictions": pred_report,
        "final_dataset": final_report,
    }
    (DOCS_DIR / "phase2_report.json").write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report: docs/phase2_report.json")
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"  PhoBERT v2 test F1: {train_report['test']['f1_macro']}")
    print(f"  Final dataset: {final_report['final_rows']} rows -> {FINAL_DATASET}")


if __name__ == "__main__":
    main()
