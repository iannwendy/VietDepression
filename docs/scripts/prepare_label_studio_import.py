"""Sinh file import BLIND cho Label Studio + file key để merge nhãn về sau.

Vì sao BLIND: gold review hiện tại bị "ảo" (baseline acc 1.0) do người review
chấp nhận lại gợi ý của hệ thống (96.3% final_label trùng weak_label). Để nhãn
người có giá trị thật, file import CHỈ chứa `row_id` + `text`, ẩn toàn bộ cột gợi ý
(suggested_label, weak_label, phobert_label, depression_score, probability, bucket...).

Sinh ra cho mỗi bước 2 file:
  - *_import.csv : đưa cho người review (Label Studio). Chỉ row_id + text.
  - *_key.csv    : GIỮ LẠI, KHÔNG đưa reviewer. Map row_id -> đầy đủ cột gốc,
                   dùng để gộp final_label trở lại sau khi export từ Label Studio.

row_id = source_tag + chỉ số dòng gốc, ổn định để map 2 chiều.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = Path(__file__).resolve().parent

STEP5_SRC = PROJECT_DIR / "data" / "review_samples.csv"
STEP8_SRC = PROJECT_DIR / "data" / "phobert_active_learning_samples.csv"


def _text_hash(text: str) -> str:
    return hashlib.sha1(str(text).strip().encode("utf-8")).hexdigest()[:10]


def build(src: Path, tag: str, import_out: Path, key_out: Path) -> dict:
    df = pd.read_csv(src, dtype=str).fillna("")
    df = df.reset_index(drop=True)
    df["row_id"] = [f"{tag}-{i:04d}" for i in range(len(df))]
    df["text_hash"] = df["comment_text"].map(_text_hash)

    # ---- file BLIND cho reviewer: chỉ row_id + text ----
    blind = df[["row_id", "comment_text"]].copy()
    blind = blind.rename(columns={"comment_text": "text"})
    blind = blind[blind["text"].str.strip().ne("")]
    blind.to_csv(import_out, index=False, encoding="utf-8-sig")

    # ---- file KEY (giữ lại): row_id + toàn bộ cột gốc để merge sau ----
    df.to_csv(key_out, index=False, encoding="utf-8-sig")

    return {"rows": int(len(blind)), "import": import_out.name, "key": key_out.name}


def main() -> None:
    r5 = build(
        STEP5_SRC,
        "s5",
        DOCS_DIR / "label_studio_step5_review_import.csv",
        DOCS_DIR / "label_studio_step5_review_key.csv",
    )
    r8 = build(
        STEP8_SRC,
        "s8",
        DOCS_DIR / "label_studio_step8_active_learning_import.csv",
        DOCS_DIR / "label_studio_step8_active_learning_key.csv",
    )

    print("========== LABEL STUDIO IMPORT FILES ==========")
    print(f"Buoc 5 (review_samples)      : {r5['rows']} dong -> {r5['import']} (key: {r5['key']})")
    print(f"Buoc 8 (active_learning)     : {r8['rows']} dong -> {r8['import']} (key: {r8['key']})")
    print(f"Tong can review              : {r5['rows'] + r8['rows']} dong")


if __name__ == "__main__":
    main()
