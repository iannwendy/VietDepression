"""Tải các dataset tiếng Việt liên quan depression/emotion/sentiment từ HuggingFace.

Mục tiêu: gom ~200K dòng dữ liệu tiếng Việt KHÁC nguồn YouTube, đặt ở folder riêng
biệt (external_datasets/) để không lẫn với pipeline crawl YouTube.

Nguyên tắc:
- GIỮ NGUYÊN nhãn gốc (original_label) của từng dataset, KHÔNG tự gán "depression".
- Gắn relevance_tier (1=emotion sát nhất, 2=sentiment, 3=toxic/tin tức/dịch máy).
- Ghi rõ provenance: source_dataset, label_scheme, native_vi (người Việt viết hay dịch máy).

Không có bộ "depression tiếng Việt" công khai cỡ lớn; đây là các tín hiệu liên quan
gần nhất (cảm xúc tiêu cực, sentiment tiêu cực) để mở rộng dữ liệu nghiên cứu.
"""

from __future__ import annotations

import io
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_per_dataset"
COMBINED_FILE = BASE_DIR / "external_vietnamese_combined.csv"
REPORT_FILE = BASE_DIR / "external_datasets_report.json"

PARQUET_API = "https://huggingface.co/api/datasets/{hf_id}/parquet"

COMMON_COLUMNS = [
    "text",
    "original_label",
    "label_scheme",
    "relevance_tier",
    "native_vi",
    "source_dataset",
    "source_split",
]


@dataclass
class DatasetSpec:
    hf_id: str
    text_col: str
    label_col: str | None
    label_scheme: str
    relevance_tier: int
    native_vi: bool
    note: str
    cap: int | None = None  # giới hạn số dòng (lấy đầu) nếu cần
    config: str = "default"
    label_map: dict | None = field(default=None)  # map nhãn số -> tên người đọc


SPECS: list[DatasetSpec] = [
    DatasetSpec(
        hf_id="duwuonline/UIT-VSMEC",
        text_col="Sentence",
        label_col="Emotion",
        label_scheme="emotion_7class",
        relevance_tier=1,
        native_vi=True,
        note="UIT-VSMEC: cảm xúc 7 lớp; Sadness/Fear/Disgust sát đề tài trầm cảm.",
    ),
    DatasetSpec(
        hf_id="thainq107/ntc-scv",
        text_col="sentence",
        label_col="label",
        label_scheme="sentiment_binary",
        relevance_tier=2,
        native_vi=True,
        note="NTC-SCV: review Foody người Việt viết; 0=negative,1=positive.",
        label_map={0: "negative", 1: "positive"},
    ),
    DatasetSpec(
        hf_id="tridm/UIT-VSFC",
        text_col="Sentence",
        label_col="Sentiment",
        label_scheme="sentiment_3class",
        relevance_tier=2,
        native_vi=True,
        note="UIT-VSFC: phản hồi sinh viên; 0=negative,1=neutral,2=positive.",
        label_map={0: "negative", 1: "neutral", 2: "positive"},
    ),
    DatasetSpec(
        hf_id="anotherpolarbear/vietnamese-sentiment-analysis",
        text_col="comment",
        label_col="label",
        label_scheme="sentiment_5star",
        relevance_tier=2,
        native_vi=True,
        note="Review sản phẩm tiếng Việt; nhãn 1-5 sao.",
    ),
    DatasetSpec(
        hf_id="sepidmnorozy/Vietnamese_sentiment",
        text_col="text",
        label_col="label",
        label_scheme="sentiment_binary",
        relevance_tier=2,
        native_vi=True,
        note="Sentiment nhị phân tiếng Việt; 0=negative,1=positive.",
        label_map={0: "negative", 1: "positive"},
    ),
    DatasetSpec(
        hf_id="minhtoan/vietnamese-comment-sentiment",
        text_col="Content",
        label_col="Sentiment",
        label_scheme="sentiment_3class_vi",
        relevance_tier=3,
        native_vi=True,
        note="Sentiment văn bản tin tức/tài chính tiếng Việt (Tích cực/Trung lập/Tiêu cực).",
    ),
    DatasetSpec(
        hf_id="zerostratos/vietnamese_toxic_core",
        text_col="text",
        label_col="label",
        label_scheme="toxic_binary",
        relevance_tier=3,
        native_vi=True,
        note="Bình luận độc hại tiếng Việt; 0=clean,1=toxic.",
        label_map={0: "clean", 1: "toxic"},
    ),
    DatasetSpec(
        hf_id="thanh29nt/vietnamese-toxic-comment",
        text_col="translated_comment_text",
        label_col="toxic",
        label_scheme="toxic_binary_translated",
        relevance_tier=3,
        native_vi=False,
        note="Jigsaw toxic DỊCH MÁY sang tiếng Việt (đã tách từ); 0=clean,1=toxic. Chất lượng thấp.",
        cap=52000,
    ),
]


def fetch_parquet_urls(hf_id: str, config: str) -> dict[str, list[str]]:
    url = PARQUET_API.format(hf_id=hf_id)
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if config in payload:
        return payload[config]
    # fallback: lấy config đầu tiên
    first_key = next(iter(payload))
    return payload[first_key]


def download_parquet(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=300) as resp:
        data = resp.read()
    return pd.read_parquet(io.BytesIO(data))


def load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    splits = fetch_parquet_urls(spec.hf_id, spec.config)
    frames: list[pd.DataFrame] = []
    for split_name, urls in splits.items():
        for url in urls:
            raw = download_parquet(url)
            if spec.text_col not in raw.columns:
                raise ValueError(f"{spec.hf_id}: thiếu cột text {spec.text_col!r}; có {list(raw.columns)}")
            text = raw[spec.text_col].astype(str)
            if spec.label_col and spec.label_col in raw.columns:
                original = raw[spec.label_col]
                if spec.label_map:
                    original = original.map(lambda v: spec.label_map.get(_to_int(v), v))
                original = original.astype(str)
            else:
                original = ""
            part = pd.DataFrame(
                {
                    "text": text,
                    "original_label": original,
                    "label_scheme": spec.label_scheme,
                    "relevance_tier": spec.relevance_tier,
                    "native_vi": spec.native_vi,
                    "source_dataset": spec.hf_id,
                    "source_split": split_name,
                }
            )
            frames.append(part)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COMMON_COLUMNS)
    df["text"] = df["text"].map(_normalize_ws)
    df = df[df["text"].str.len() >= 5]
    df = df.drop_duplicates(subset=["text"], keep="first")
    if spec.cap is not None and len(df) > spec.cap:
        df = df.head(spec.cap)
    return df.reindex(columns=COMMON_COLUMNS)


def _to_int(value: object) -> object:
    try:
        return int(value)
    except (ValueError, TypeError):
        return value


def _normalize_ws(text: str) -> str:
    return " ".join(str(text).strip().split())


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"datasets": [], "total_rows": 0, "by_tier": {}, "by_source_native": {}}
    all_frames: list[pd.DataFrame] = []

    for spec in SPECS:
        print(f"\n=== {spec.hf_id} (tier {spec.relevance_tier}) ===")
        try:
            df = load_dataset(spec)
        except Exception as exc:  # noqa: BLE001 - log và bỏ qua dataset lỗi
            print(f"  LỖI: {exc}")
            report["datasets"].append({"hf_id": spec.hf_id, "status": "failed", "error": str(exc)})
            continue

        out_path = RAW_DIR / (spec.hf_id.replace("/", "__") + ".csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        all_frames.append(df)
        print(f"  OK {len(df):,} dòng -> {out_path.name}")
        report["datasets"].append(
            {
                "hf_id": spec.hf_id,
                "status": "ok",
                "rows": int(len(df)),
                "label_scheme": spec.label_scheme,
                "relevance_tier": spec.relevance_tier,
                "native_vi": spec.native_vi,
                "note": spec.note,
                "label_distribution": df["original_label"].value_counts().head(15).to_dict(),
                "file": str(out_path),
            }
        )

    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=COMMON_COLUMNS)
    # chống trùng chéo giữa các dataset theo nội dung text
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text"], keep="first")
    combined.to_csv(COMBINED_FILE, index=False, encoding="utf-8-sig")

    report["total_rows"] = int(len(combined))
    report["cross_dataset_duplicates_removed"] = int(before - len(combined))
    report["by_tier"] = combined["relevance_tier"].value_counts().sort_index().to_dict()
    report["by_source_native"] = combined["native_vi"].value_counts().to_dict()
    report["by_source_dataset"] = combined["source_dataset"].value_counts().to_dict()
    report["combined_file"] = str(COMBINED_FILE)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== TỔNG KẾT ==========")
    print(f"Tổng dòng (sau khử trùng chéo): {len(combined):,}")
    print(f"Theo tier: {report['by_tier']}")
    print(f"Người Việt viết vs dịch máy: {report['by_source_native']}")
    print(f"Combined: {COMBINED_FILE}")
    print(f"Report:   {REPORT_FILE}")


if __name__ == "__main__":
    main()
