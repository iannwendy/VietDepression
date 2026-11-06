"""Chấm điểm cross-domain: baseline + PhoBERT-lần-1 trên tập human-labeled external.

Mục tiêu: đo NĂNG LỰC TỔNG QUÁT HÓA thật của model train trên dữ liệu YouTube
weak-label, khi đem ra một domain khác (VSMEC) có nhãn do NGƯỜI gán độc lập.

Tập test: external_human_eval.csv (3,084 dòng, cân bằng)
  label=1 <- VSMEC Sadness/Fear (distress)
  label=0 <- VSMEC Enjoyment

So sánh với điểm "gold" nội bộ (gần như 1.0) để lộ khoảng cách tổng quát hóa.
Dùng chính metric/tokenizer/word-segmentation của dự án để số liệu nhất quán.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from yt_depression_crawler.core.config import (
    BASELINE_MODEL_FILE,
    PHOBERT_MAX_LENGTH,
    PHOBERT_EVAL_BATCH_SIZE,
    PHOBERT_OUTPUT_DIR,
)
from yt_depression_crawler.modeling.phobert.phobert_utils import (
    PhoBertDataset,
    compute_metrics,
    get_device,
    prepare_many_texts,
)

BASE_DIR = Path(__file__).resolve().parent
HUMAN_EVAL_FILE = BASE_DIR / "external_human_eval.csv"
REPORT_FILE = BASE_DIR / "external_cross_domain_report.json"


def score_baseline(texts: list[str], y_true: list[int]) -> dict:
    model = joblib.load(BASELINE_MODEL_FILE)
    y_pred = [int(p) for p in model.predict(texts)]
    return compute_metrics(y_true, y_pred)


def score_phobert(texts: list[str], y_true: list[int]) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_OUTPUT_DIR, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(PHOBERT_OUTPUT_DIR)
    model.to(device)
    model.train(False)  # inference mode (tương đương .eval())

    seg_texts = prepare_many_texts(texts)
    dataset = PhoBertDataset(seg_texts, None, tokenizer, PHOBERT_MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=PHOBERT_EVAL_BATCH_SIZE)

    preds: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
    return compute_metrics(y_true, preds)


def main() -> None:
    df = pd.read_csv(HUMAN_EVAL_FILE, dtype={"comment_text": str}).fillna("")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
    texts = df["comment_text"].astype(str).tolist()
    y_true = df["label"].astype(int).tolist()
    print(f"Test set: {len(df):,} dong, can bang {df['label'].value_counts().to_dict()}")

    print("\n[1/2] Baseline TF-IDF + LogReg ...")
    baseline = score_baseline(texts, y_true)
    print(f"  acc={baseline['accuracy']} f1_macro={baseline['f1_macro']} f1_dep={baseline['f1_depression']}")

    print("\n[2/2] PhoBERT lan 1 ...")
    phobert = score_phobert(texts, y_true)
    print(f"  acc={phobert['accuracy']} f1_macro={phobert['f1_macro']} f1_dep={phobert['f1_depression']}")

    report = {
        "test_set": {
            "file": str(HUMAN_EVAL_FILE),
            "rows": int(len(df)),
            "label_counts": df["label"].value_counts().sort_index().to_dict(),
            "domain": "VSMEC (cross-domain vs YouTube train)",
            "label_provenance": "human emotion annotation (Sadness/Fear=1, Enjoyment=0)",
        },
        "baseline_tfidf_logreg": baseline,
        "phobert_first": phobert,
        "internal_gold_for_reference": {
            "note": "Diem gold noi bo gan nhu 1.0 do gold set chap nhan lai goi y he thong.",
            "baseline_gold_accuracy": 1.0,
            "phobert_gold_f1_macro": 0.954,
        },
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== CROSS-DOMAIN vs INTERNAL GOLD ==========")
    print(f"baseline       cross-domain f1_macro={baseline['f1_macro']}   internal gold ~1.00")
    print(f"phobert_first  cross-domain f1_macro={phobert['f1_macro']}   internal gold 0.954")
    print(f"\nReport: {REPORT_FILE.name}")


if __name__ == "__main__":
    main()
