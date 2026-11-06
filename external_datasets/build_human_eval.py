"""Build tập eval/augmentation cross-domain TỪ NHÃN NGƯỜI GÁN (không dùng keyword).

Lý do: weak-labeler keyword của dự án có recall ~0.91% trên distress người-gán
(chỉ bắt 14/1542 câu VSMEC Sadness/Fear). Vì vậy dùng keyword để gán nhãn external
data là sai hướng. Thứ external data CÓ mà YouTube weak-label KHÔNG có là nhãn cảm
xúc do annotator thật gán -> khai thác chính nó.

Sản phẩm:
1. external_human_eval.csv  -- tập eval nhị phân SẠCH, chỉ từ VSMEC (1 annotation scheme):
     label=1  <- Sadness + Fear   (distress: tín hiệu trầm cảm gần nhất do người gán)
     label=0  <- Enjoyment        (positive rõ ràng)
   Dùng để ĐÁNH GIÁ KHÁCH QUAN cross-domain model train trên dữ liệu YouTube.

2. external_augmentation_negatives.csv -- pool hard-negative lớn: text positive-affect
   từ mọi nguồn, làm dữ liệu bổ sung lớp normal khi train (giúp model bớt coi mọi
   câu tiêu cực là trầm cảm).

Nguyên tắc: distress != trầm cảm lâm sàng. Đây là PROXY cảm xúc do người gán, dùng
cho nghiên cứu/đánh giá, không phải chẩn đoán. Schema khớp pipeline: comment_text,label.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_FILE = BASE_DIR / "external_processed.csv"
HUMAN_EVAL_FILE = BASE_DIR / "external_human_eval.csv"
AUG_NEGATIVES_FILE = BASE_DIR / "external_augmentation_negatives.csv"
REPORT_FILE = BASE_DIR / "external_human_eval_report.json"

EVAL_COLUMNS = [
    "comment_text",
    "label",
    "label_name",
    "affect_signal",
    "original_label",
    "source_dataset",
    "label_provenance",
]

RANDOM_SEED = 42


def main() -> None:
    df = pd.read_csv(PROCESSED_FILE, dtype=str).fillna("")
    df["text"] = df["text"].astype(str)

    # ---- 1. tập eval sạch chỉ từ VSMEC (emotion_7class, 1 annotation scheme) ----
    vsmec = df[df["label_scheme"] == "emotion_7class"].copy()
    positive = vsmec[vsmec["original_label"].isin(["Sadness", "Fear"])].copy()
    negative = vsmec[vsmec["original_label"] == "Enjoyment"].copy()

    # cân bằng lớp theo lớp nhỏ hơn
    n = min(len(positive), len(negative))
    positive = positive.sample(n=n, random_state=RANDOM_SEED)
    negative = negative.sample(n=n, random_state=RANDOM_SEED)

    positive["label"] = 1
    positive["label_name"] = "distress"
    negative["label"] = 0
    negative["label_name"] = "normal"

    eval_df = pd.concat([positive, negative], ignore_index=True)
    eval_df = eval_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    eval_df["comment_text"] = eval_df["text"]
    eval_df["label_provenance"] = "VSMEC human emotion annotation (Sadness/Fear=1, Enjoyment=0)"
    eval_df = eval_df.reindex(columns=EVAL_COLUMNS)
    eval_df.to_csv(HUMAN_EVAL_FILE, index=False, encoding="utf-8-sig")

    # ---- 2. pool hard-negative: positive-affect người-gán, từ mọi nguồn ----
    aug_neg = df[df["affect_signal"] == "positive"].copy()
    aug_neg = aug_neg.drop_duplicates(subset=["text"], keep="first")
    aug_neg["comment_text"] = aug_neg["text"]
    aug_neg["label"] = 0
    aug_neg["label_name"] = "normal"
    aug_neg["label_provenance"] = "human positive-affect label (sentiment/emotion) -> normal"
    aug_neg = aug_neg.reindex(columns=EVAL_COLUMNS)
    aug_neg.to_csv(AUG_NEGATIVES_FILE, index=False, encoding="utf-8-sig")

    report = {
        "human_eval": {
            "total": int(len(eval_df)),
            "label_counts": eval_df["label"].value_counts().sort_index().to_dict(),
            "source": "VSMEC only (single annotation scheme)",
            "positive_from": "Sadness + Fear",
            "negative_from": "Enjoyment",
            "balanced_per_class": int(n),
            "file": str(HUMAN_EVAL_FILE),
        },
        "augmentation_negatives": {
            "total": int(len(aug_neg)),
            "by_source": aug_neg["source_dataset"].value_counts().to_dict(),
            "file": str(AUG_NEGATIVES_FILE),
        },
        "caveat": (
            "distress (Sadness/Fear) là proxy cảm xúc do người gán, KHÔNG phải chẩn đoán "
            "trầm cảm lâm sàng. Dùng cho đánh giá cross-domain và augmentation, không kết luận y tế."
        ),
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== HUMAN-LABELED ARTIFACTS ==========")
    print(f"Eval set (VSMEC, balanced): {len(eval_df):,} dòng ({n:,}/lớp) -> {HUMAN_EVAL_FILE.name}")
    print(f"Augmentation negatives:     {len(aug_neg):,} dòng -> {AUG_NEGATIVES_FILE.name}")
    print(f"Report: {REPORT_FILE.name}")


if __name__ == "__main__":
    main()
