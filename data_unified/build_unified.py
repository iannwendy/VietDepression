"""Build corpus hợp nhất theo Hướng 2: 3 tier rõ ràng + manifest.

Quyết định đã chốt với nhóm:
- Bài toán đích: NHỊ PHÂN depression/normal.
- neutral + toxic external: CHỈ vào corpus text/topic, KHÔNG vào tập train depression.
- Mục tiêu gộp 190K: nền text/topic + test cross-domain (không cố phình tập train).

Chống leakage (quan trọng): VSMEC đã dùng làm test cross-domain ở bước đánh giá A.
Vì vậy:
- cross_domain_test.csv  = VSMEC holdout, GIỮ NGUYÊN, không bao giờ vào train.
- augmentation_negatives = positive-affect TỪ NGUỒN != VSMEC, đã lọc trùng với holdout.
- corpus_text_all        = toàn bộ text (YouTube + external) cho clean/topic/MLM,
                           có cột is_holdout_text để topic/MLM tùy chọn loại text test.

Mọi text đi qua CHÍNH cleaner.py của dự án để đồng chuẩn với pipeline YouTube.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.vietdepression_lexicon import is_basic_spam, normalize_text
EXTERNAL_DIR = PROJECT_DIR / "external_datasets"
OUT_DIR = Path(__file__).resolve().parent

YOUTUBE_CLEANED = PROJECT_DIR / "data" / "cleaned_comments.csv"
EXTERNAL_PROCESSED = EXTERNAL_DIR / "external_processed.csv"
HOLDOUT_EVAL = EXTERNAL_DIR / "external_human_eval.csv"

CORPUS_FILE = OUT_DIR / "corpus_text_all.csv"
TEST_FILE = OUT_DIR / "cross_domain_test.csv"
AUG_NEG_FILE = OUT_DIR / "augmentation_negatives.csv"
MANIFEST_FILE = OUT_DIR / "manifest.json"

VSMEC_SOURCE = "duwuonline/UIT-VSMEC"


def _dedupe_key(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def _clean_series(texts: pd.Series) -> pd.Series:
    return texts.map(normalize_text)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- nguồn YouTube ----------
    yt = pd.read_csv(YOUTUBE_CLEANED, dtype=str).fillna("")
    yt_text = _clean_series(yt["comment_text"])
    yt_df = pd.DataFrame(
        {
            "text": yt_text,
            "source": "youtube",
            "source_dataset": "youtube_crawl",
            "affect_signal": "",
        }
    )

    # ---------- nguồn external (đã có affect_signal) ----------
    ext = pd.read_csv(EXTERNAL_PROCESSED, dtype=str).fillna("")
    ext_text = _clean_series(ext["text"])
    ext_df = pd.DataFrame(
        {
            "text": ext_text,
            "source": "external",
            "source_dataset": ext["source_dataset"],
            "affect_signal": ext["affect_signal"],
        }
    )

    # ---------- holdout text keys (để đánh dấu + chống leakage) ----------
    holdout = pd.read_csv(HOLDOUT_EVAL, dtype=str).fillna("")
    holdout_keys = {_dedupe_key(t) for t in holdout["comment_text"]}

    # ===== 1. corpus_text_all: toàn bộ text cho clean/topic/MLM =====
    corpus = pd.concat([yt_df, ext_df], ignore_index=True)
    corpus["text"] = corpus["text"].map(normalize_text)
    corpus = corpus[corpus["text"].str.len() >= 5]
    corpus = corpus[~corpus["text"].map(is_basic_spam)]
    corpus["_key"] = corpus["text"].map(_dedupe_key)
    corpus = corpus.drop_duplicates(subset=["_key"], keep="first")
    corpus["is_holdout_text"] = corpus["_key"].isin(holdout_keys)
    corpus = corpus.drop(columns=["_key"])
    corpus.to_csv(CORPUS_FILE, index=False, encoding="utf-8-sig")

    # ===== 2. cross_domain_test: VSMEC holdout, giữ nguyên =====
    holdout.to_csv(TEST_FILE, index=False, encoding="utf-8-sig")

    # ===== 3. augmentation_negatives: positive-affect, nguồn != VSMEC, không trùng holdout =====
    aug = ext.copy()
    aug["text"] = _clean_series(aug["text"])
    aug = aug[aug["affect_signal"] == "positive"]
    aug = aug[aug["source_dataset"] != VSMEC_SOURCE]
    aug = aug[aug["text"].str.len() >= 5]
    aug = aug[~aug["text"].map(is_basic_spam)]
    aug["_key"] = aug["text"].map(_dedupe_key)
    aug = aug[~aug["_key"].isin(holdout_keys)]
    aug = aug.drop_duplicates(subset=["_key"], keep="first")
    aug_out = pd.DataFrame(
        {
            "comment_text": aug["text"],
            "label": 0,
            "label_name": "normal",
            "affect_signal": aug["affect_signal"],
            "original_label": aug["original_label"],
            "source_dataset": aug["source_dataset"],
            "label_provenance": "human positive-affect (sentiment/emotion) -> normal; source != VSMEC",
        }
    )
    aug_out.to_csv(AUG_NEG_FILE, index=False, encoding="utf-8-sig")

    # ===== manifest: bước nào ăn file nào =====
    manifest = {
        "approach": "Huong 2 - tach theo muc dich buoc, mot corpus chung",
        "target_task": "binary depression/normal",
        "files": {
            "corpus_text_all.csv": {
                "rows": int(len(corpus)),
                "columns": list(corpus.columns),
                "used_by": [
                    "Lam sach chung",
                    "BERTopic (topic modeling) -> models/bertopic/",
                    "Domain-adapt PhoBERT (MLM) - co the loc is_holdout_text=True",
                ],
                "note": "Toan bo text YouTube + external. KHONG dung lam nhan supervised.",
                "by_source": corpus["source"].value_counts().to_dict(),
                "holdout_text_rows": int(corpus["is_holdout_text"].sum()),
            },
            "cross_domain_test.csv": {
                "rows": int(len(holdout)),
                "used_by": ["Test set CO DINH cho moi model (baseline/SVM/BiLSTM/PhoBERT/PhoBERT+BERTopic)"],
                "note": "VSMEC Sadness/Fear=1, Enjoyment=0. KHONG BAO GIO vao train.",
            },
            "augmentation_negatives.csv": {
                "rows": int(len(aug_out)),
                "used_by": ["Bo sung lop normal (hard-negative) khi train depression"],
                "note": "Positive-affect nguoi-gan, nguon != VSMEC, da loc trung voi holdout.",
                "by_source": aug_out["source_dataset"].value_counts().to_dict(),
            },
        },
        "excluded_from_train": {
            "neutral_toxic_negative_external": "Chi nam trong corpus_text_all cho topic/text, KHONG vao tap train depression",
            "reason": "Recall keyword tren distress = 0.91%; negative sentiment != tram cam",
        },
        "supervised_train_sources": {
            "positive_label_1": "YouTube depression_auto (weak) + review thu cong (gold/active-learning sau khi review)",
            "negative_label_0": "YouTube normal_auto (weak) + augmentation_negatives (external positive-affect)",
            "note": "VSMEC KHONG vao train de tranh leakage voi cross_domain_test",
        },
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== DATA_UNIFIED (Huong 2) ==========")
    print(f"corpus_text_all.csv      : {len(corpus):,} dong "
          f"(youtube {corpus['source'].value_counts().get('youtube', 0):,} / "
          f"external {corpus['source'].value_counts().get('external', 0):,}; "
          f"holdout_text {int(corpus['is_holdout_text'].sum()):,})")
    print(f"cross_domain_test.csv    : {len(holdout):,} dong (VSMEC holdout)")
    print(f"augmentation_negatives   : {len(aug_out):,} dong (positive-affect, != VSMEC)")
    print(f"manifest.json            : da ghi")


if __name__ == "__main__":
    main()
