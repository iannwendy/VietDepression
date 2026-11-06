"""Xử lý external dataset cho đúng đề tài phát hiện trầm cảm.

Mục tiêu: biến 192K dòng cảm xúc/sentiment thô thành dữ liệu dùng được, NHẤT QUÁN
với pipeline YouTube, KHÔNG tự bịa nhãn "depression".

Quy trình:
1. Làm sạch text bằng CHÍNH cleaner.py của dự án (cùng chuẩn với YouTube).
2. Chạy weak-labeler keyword của dự án (auto_labeler.label_comment) lên external text
   -> mỗi dòng có depression_score / weak_label / confidence, giống hệt comment YouTube.
3. Suy ra affect_signal ĐỘC LẬP từ nhãn gốc do người gán (Sadness/Fear -> distress, v.v.).
4. Đối chiếu 2 tín hiệu để (a) validate weak-labeler, (b) lọc tập high-precision.

Nguyên tắc khoa học: sentiment tiêu cực != trầm cảm. Review 1 sao "đồ ăn dở" là
negative nhưng không phải distress cá nhân. Nên tín hiệu trầm cảm chính vẫn là
keyword weak-labeler; nhãn cảm xúc gốc chỉ để validate (mạnh nhất ở VSMEC).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.vietdepression_lexicon import is_basic_spam, label_comment, normalize_text

BASE_DIR = Path(__file__).resolve().parent
COMBINED_FILE = BASE_DIR / "external_vietnamese_combined.csv"
PROCESSED_FILE = BASE_DIR / "external_processed.csv"
DEPRESSION_CANDIDATES_FILE = BASE_DIR / "external_depression_candidates.csv"
HARD_NEGATIVES_FILE = BASE_DIR / "external_hard_negatives.csv"
REPORT_FILE = BASE_DIR / "external_processed_report.json"

# affect_signal: tín hiệu cảm xúc do NGƯỜI gán nhãn gốc, độc lập với keyword scoring.
#   distress  = buồn/sợ/tuyệt vọng -> sát trầm cảm nhất
#   negative  = tiêu cực chung (chê bai, không hài lòng)
#   toxic     = độc hại/công kích
#   neutral / positive
DISTRESS = "distress"
NEGATIVE = "negative"
TOXIC = "toxic"
NEUTRAL = "neutral"
POSITIVE = "positive"
UNKNOWN = "unknown"


def affect_from_label(label_scheme: str, original_label: str) -> str:
    raw = str(original_label).strip()
    low = raw.lower()

    if label_scheme == "emotion_7class":
        if raw in {"Sadness", "Fear"}:
            return DISTRESS
        if raw in {"Disgust", "Anger"}:
            return NEGATIVE
        if raw == "Enjoyment":
            return POSITIVE
        return NEUTRAL  # Surprise, Other

    if label_scheme in {"sentiment_binary", "sentiment_3class"}:
        if low == "negative":
            return NEGATIVE
        if low == "positive":
            return POSITIVE
        return NEUTRAL

    if label_scheme == "sentiment_3class_vi":
        if low in {"negative", "tiêu cực"}:
            return NEGATIVE
        if low in {"positive", "tích cực"}:
            return POSITIVE
        return NEUTRAL

    if label_scheme == "sentiment_5star":
        star = _to_int(raw)
        if star in {1, 2}:
            return NEGATIVE
        if star in {4, 5}:
            return POSITIVE
        if star == 3:
            return NEUTRAL
        return UNKNOWN

    if label_scheme in {"toxic_binary", "toxic_binary_translated"}:
        if low in {"toxic", "1"}:
            return TOXIC
        if low in {"clean", "0"}:
            return NEUTRAL
        return UNKNOWN

    return UNKNOWN


def _to_int(value: object) -> object:
    try:
        return int(value)
    except (ValueError, TypeError):
        return value


def main() -> None:
    df = pd.read_csv(COMBINED_FILE, dtype=str).fillna("")
    print(f"Đọc {len(df):,} dòng từ {COMBINED_FILE.name}")

    # 1. làm sạch bằng cleaner của dự án
    df["text"] = df["text"].map(normalize_text)
    df = df[df["text"].str.len() >= 5]
    df = df[~df["text"].map(is_basic_spam)]
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"Sau làm sạch: {len(df):,} dòng")

    # 2. weak-label keyword của dự án (cùng lexicon với YouTube)
    results = [label_comment(text) for text in df["text"]]
    df["depression_score"] = [r.depression_score for r in results]
    df["weak_label"] = [r.weak_label for r in results]
    df["confidence"] = [r.confidence for r in results]
    df["matched_keywords"] = [r.matched_keywords for r in results]
    df["need_review"] = [r.need_review for r in results]

    # 3. affect_signal độc lập từ nhãn gốc
    df["affect_signal"] = [
        affect_from_label(scheme, label)
        for scheme, label in zip(df["label_scheme"], df["original_label"])
    ]

    df.to_csv(PROCESSED_FILE, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {PROCESSED_FILE.name}")

    # 4a. tập depression-positive high-precision:
    #     keyword nói depression_auto. Trong đó, đồng thuận với distress (VSMEC Sadness/Fear)
    #     là chắc chắn nhất -> đánh dấu agreement.
    dep = df[df["weak_label"] == "depression_auto"].copy()
    dep["signal_agreement"] = dep["affect_signal"].isin([DISTRESS, NEGATIVE, TOXIC])
    dep = dep.sort_values(
        ["signal_agreement", "depression_score"], ascending=[False, False]
    )
    dep.to_csv(DEPRESSION_CANDIDATES_FILE, index=False, encoding="utf-8-sig")

    # 4b. hard-negatives: tích cực rõ ràng nhưng keyword không bắt được trầm cảm.
    #     Dạy model phân biệt "tiêu cực/bình thường" với "trầm cảm".
    hard_neg = df[
        (df["affect_signal"] == POSITIVE) & (df["weak_label"] != "depression_auto")
    ].copy()
    hard_neg.to_csv(HARD_NEGATIVES_FILE, index=False, encoding="utf-8-sig")

    # 5. cross-tab + đồng thuận
    crosstab = (
        pd.crosstab(df["affect_signal"], df["weak_label"]).to_dict(orient="index")
    )
    report = {
        "input_rows": int(len(df)),
        "weak_label_counts": df["weak_label"].value_counts().to_dict(),
        "affect_signal_counts": df["affect_signal"].value_counts().to_dict(),
        "crosstab_affect_x_weaklabel": crosstab,
        "depression_candidates": {
            "total": int(len(dep)),
            "agree_with_affect": int(dep["signal_agreement"].sum()),
            "by_affect": dep["affect_signal"].value_counts().to_dict(),
            "by_source": dep["source_dataset"].value_counts().to_dict(),
            "file": str(DEPRESSION_CANDIDATES_FILE),
        },
        "hard_negatives": {
            "total": int(len(hard_neg)),
            "by_source": hard_neg["source_dataset"].value_counts().to_dict(),
            "file": str(HARD_NEGATIVES_FILE),
        },
        "distress_keyword_recall": _distress_recall(df),
        "processed_file": str(PROCESSED_FILE),
    }
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n========== TỔNG KẾT ==========")
    print(f"weak_label: {report['weak_label_counts']}")
    print(f"affect_signal: {report['affect_signal_counts']}")
    print(
        f"Depression candidates: {len(dep):,} "
        f"(đồng thuận affect: {int(dep['signal_agreement'].sum()):,})"
    )
    print(f"Hard negatives: {len(hard_neg):,}")
    print(f"Report: {REPORT_FILE.name}")


def _distress_recall(df: pd.DataFrame) -> dict:
    """Trong các dòng người gán là distress (VSMEC Sadness/Fear), keyword bắt được bao nhiêu %?"""
    distress = df[df["affect_signal"] == DISTRESS]
    if distress.empty:
        return {"distress_rows": 0}
    caught = int((distress["weak_label"] == "depression_auto").sum())
    return {
        "distress_rows": int(len(distress)),
        "keyword_flagged_depression": caught,
        "recall": round(caught / len(distress), 4),
    }


if __name__ == "__main__":
    main()
