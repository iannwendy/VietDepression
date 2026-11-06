#!/usr/bin/env python3
"""
Build and verify Tier-1 public de-identified SHA-256 split manifests for the
VietDepression benchmark, and audit the Git index to guarantee zero leakage of
raw YouTube comment text or platform identifiers in the public repository.

Usage:
  python3 scripts/build_and_verify_dua_manifests.py --build
  python3 scripts/build_and_verify_dua_manifests.py --verify-local
  python3 scripts/build_and_verify_dua_manifests.py --audit-git
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.vietdepression_lexicon import label_comment_detailed

LABELED_DIR = BASE_DIR / "data" / "labeled"
MANIFEST_DIR = BASE_DIR / "data" / "manifests"
VSMEC_CSV = BASE_DIR / "data_unified" / "cross_domain_test.csv"

SPLIT_SPECS = [
    ("train", "final_train.csv", "final_train_manifest.csv", 6392, 1262, 5130),
    ("val", "final_val.csv", "final_val_manifest.csv", 1371, 271, 1100),
    ("test", "final_test.csv", "final_test_manifest.csv", 1371, 271, 1100),
]


def normalize_nfc(text: str) -> str:
    """Canonical Unicode NFC + whitespace normalization used for SHA-256 hashing."""
    s = unicodedata.normalize("NFC", str(text).strip())
    return re.sub(r"\s+", " ", s)


def sha256_nfc(text: str) -> str:
    return hashlib.sha256(normalize_nfc(text).encode("utf-8")).hexdigest()


def build_split_manifest(split_name: str, src_csv: Path) -> pd.DataFrame:
    if not src_csv.exists():
        raise FileNotFoundError(f"Controlled-access split file not found locally: {src_csv}")

    df = pd.read_csv(src_csv)
    rows: List[Dict[str, object]] = []
    prefix = {"train": "VD-TRN", "val": "VD-VAL", "test": "VD-TST"}[split_name]

    for idx, row in df.iterrows():
        raw_txt = str(row["comment_text"])
        norm_txt = normalize_nfc(raw_txt)
        lex = label_comment_detailed(norm_txt)
        rows.append(
            {
                "record_id": f"{prefix}-{idx + 1:04d}",
                "split": split_name,
                "text_sha256": sha256_nfc(norm_txt),
                "label": int(row["label"]),
                "source": str(row.get("source", "human_gold")),
                "annotation_round": str(row.get("annotation_round", "unspecified")),
                "weak_score": int(lex.depression_score),
                "weak_label": str(lex.weak_label),
                "has_review_context_phrase": bool(lex.has_review_context_phrase),
                "need_review_composite": bool(lex.need_review),
                "char_len": len(norm_txt),
                "word_len": len(norm_txt.split()),
            }
        )
    return pd.DataFrame(rows)


def build_vsmec_manifest(vsmec_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(vsmec_csv)
    rows: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        norm_txt = normalize_nfc(str(row["comment_text"]))
        orig_lbl = str(row["original_label"])
        rows.append(
            {
                "record_id": f"VSMEC-{idx + 1:04d}",
                "text_sha256": sha256_nfc(norm_txt),
                "proxy_label": int(row["label"]),
                "original_emotion": orig_lbl,
                "in_sadness_only_subset": orig_lbl in ("Sadness", "Enjoyment"),
                "source_dataset": str(row.get("source_dataset", "duwuonline/UIT-VSMEC")),
                "char_len": len(norm_txt),
                "word_len": len(norm_txt.split()),
            }
        )
    return pd.DataFrame(rows)


def cmd_build() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    split_frames: List[pd.DataFrame] = []
    summary_splits: Dict[str, Dict[str, object]] = {}

    for split_name, src_file, out_file, exp_total, exp_pos, exp_neg in SPLIT_SPECS:
        mdf = build_split_manifest(split_name, LABELED_DIR / src_file)
        if len(mdf) != exp_total:
            raise AssertionError(f"{split_name}: expected {exp_total} rows, got {len(mdf)}")
        pos_cnt = int((mdf["label"] == 1).sum())
        neg_cnt = int((mdf["label"] == 0).sum())
        if pos_cnt != exp_pos or neg_cnt != exp_neg:
            raise AssertionError(
                f"{split_name}: expected ({exp_pos} P, {exp_neg} N), got ({pos_cnt} P, {neg_cnt} N)"
            )
        out_path = MANIFEST_DIR / out_file
        mdf.to_csv(out_path, index=False, encoding="utf-8")
        split_frames.append(mdf)
        summary_splits[split_name] = {
            "manifest_file": f"data/manifests/{out_file}",
            "total": len(mdf),
            "positive_1": pos_cnt,
            "negative_0": neg_cnt,
            "positive_rate_pct": round(pos_cnt / len(mdf) * 100, 2),
            "unique_sha256": int(mdf["text_sha256"].nunique()),
        }
        print(f"[OK] Wrote {out_path.relative_to(BASE_DIR)} ({len(mdf)} rows: {pos_cnt} P / {neg_cnt} N)")

    full_manifest = pd.concat(split_frames, ignore_index=True)
    if full_manifest["text_sha256"].nunique() != len(full_manifest):
        raise AssertionError("Duplicate SHA-256 hash detected across benchmark splits!")

    full_out = MANIFEST_DIR / "final_dataset_manifest.csv"
    full_manifest.to_csv(full_out, index=False, encoding="utf-8")
    print(f"[OK] Wrote {full_out.relative_to(BASE_DIR)} ({len(full_manifest)} rows, 0% overlap)")

    vsmec_summary: Dict[str, object] = {}
    if VSMEC_CSV.exists():
        vdf = build_vsmec_manifest(VSMEC_CSV)
        v_out = MANIFEST_DIR / "uit_vsmec_proxy_manifest.csv"
        vdf.to_csv(v_out, index=False, encoding="utf-8")
        sad_df = vdf[vdf["in_sadness_only_subset"]]
        vsmec_summary = {
            "manifest_file": "data/manifests/uit_vsmec_proxy_manifest.csv",
            "full_proxy_total": int(len(vdf)),
            "full_proxy_positive": int((vdf["proxy_label"] == 1).sum()),
            "full_proxy_negative": int((vdf["proxy_label"] == 0).sum()),
            "sadness_only_total": int(len(sad_df)),
            "sadness_only_positive": int((sad_df["proxy_label"] == 1).sum()),
            "sadness_only_negative": int((sad_df["proxy_label"] == 0).sum()),
        }
        print(f"[OK] Wrote {v_out.relative_to(BASE_DIR)} ({len(vdf)} Full / {len(sad_df)} Sadness-Only)")

    summary_payload = {
        "benchmark": "VietDepression",
        "release_governance": "Two-Tier Protocol (Public De-Identified Manifests + Controlled-Access Text via DUA)",
        "hash_algorithm": "SHA-256 over Unicode NFC + single-space normalized UTF-8 comment text",
        "total_gold_records": int(len(full_manifest)),
        "total_positive_1": int((full_manifest["label"] == 1).sum()),
        "total_negative_0": int((full_manifest["label"] == 0).sum()),
        "unique_sha256_hashes": int(full_manifest["text_sha256"].nunique()),
        "cross_split_hash_overlap": 0,
        "contains_raw_comment_text": False,
        "contains_youtube_comment_or_video_ids": False,
        "splits": summary_splits,
        "external_vsmec_proxy": vsmec_summary,
    }
    sum_out = MANIFEST_DIR / "manifest_integrity_summary.json"
    sum_out.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {sum_out.relative_to(BASE_DIR)}")


def cmd_verify_local() -> None:
    seen_hashes = set()
    for split_name, src_file, man_file, exp_total, exp_pos, exp_neg in SPLIT_SPECS:
        src_path = LABELED_DIR / src_file
        man_path = MANIFEST_DIR / man_file
        if not src_path.exists():
            raise FileNotFoundError(
                f"Missing DUA-governed split file {src_path}. Obtain normalized text under DATA_USE_AGREEMENT.md."
            )
        if not man_path.exists():
            raise FileNotFoundError(f"Missing public manifest file {man_path}.")

        sdf = pd.read_csv(src_path)
        mdf = pd.read_csv(man_path)
        if len(sdf) != exp_total or len(mdf) != exp_total:
            raise AssertionError(f"{split_name}: length mismatch ({len(sdf)} vs {len(mdf)})")

        computed_hashes = [sha256_nfc(t) for t in sdf["comment_text"].astype(str)]
        if computed_hashes != mdf["text_sha256"].tolist():
            raise AssertionError(f"{split_name}: SHA-256 hash mismatch against public manifest!")
        if sdf["label"].astype(int).tolist() != mdf["label"].astype(int).tolist():
            raise AssertionError(f"{split_name}: gold label mismatch against public manifest!")

        overlap = seen_hashes.intersection(computed_hashes)
        if overlap:
            raise AssertionError(f"{split_name}: detected {len(overlap)} overlapping hashes across splits!")
        seen_hashes.update(computed_hashes)

        pos_cnt = int((sdf["label"] == 1).sum())
        neg_cnt = int((sdf["label"] == 0).sum())
        assert pos_cnt == exp_pos and neg_cnt == exp_neg
        print(f"[VERIFIED] {split_name}: {len(sdf)} rows ({pos_cnt} P / {neg_cnt} N) match SHA-256 manifest 100%")

    print(f"[VERIFIED] All {len(seen_hashes)} gold records match public manifests with 0% cross-split overlap.")


def cmd_audit_git() -> None:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=BASE_DIR).decode("utf-8").splitlines()
    violations: List[str] = []

    gold_snippets: set[str] = set()
    gold_path = LABELED_DIR / "final_dataset.csv"
    if gold_path.exists():
        gdf = pd.read_csv(gold_path)
        gold_snippets = {
            t.strip()[:35]
            for t in gdf["comment_text"].dropna().astype(str)
            if len(t.strip()) >= 35
        }

    video_ids: set[str] = set()
    vid_meta_path = BASE_DIR / "data" / "raw" / "video_metadata.csv"
    if vid_meta_path.exists():
        vdf = pd.read_csv(vid_meta_path)
        if "video_id" in vdf.columns:
            video_ids = {
                v.strip()
                for v in vdf["video_id"].dropna().astype(str)
                if len(v.strip()) >= 8
            }

    yt_cid_re = re.compile(r"\bUg[wxyz][A-Za-z0-9_-]{15,30}\b")
    forbidden_text_cols = {"comment_text", "text", "clean_text", "original_text", "video_id", "comment_id"}

    for rel in tracked:
        p = BASE_DIR / rel
        if not p.exists():
            continue

        if rel.endswith((".db", ".sqlite", ".sqlite3")):
            violations.append(f"Tracked database file prohibited under DUA governance: {rel}")
            continue

        if rel.endswith(".csv"):
            with p.open(encoding="utf-8-sig", errors="replace") as fh:
                header_cols = {c.strip().strip('"') for c in fh.readline().strip().split(",")}
            bad_cols = header_cols.intersection(forbidden_text_cols)
            if bad_cols:
                violations.append(f"Tracked CSV contains raw text/ID column(s) {sorted(bad_cols)}: {rel}")
            continue

        if rel.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf", ".joblib", ".pyc")):
            continue

        if p.stat().st_size > 10_000_000:
            continue

        txt = p.read_text(encoding="utf-8", errors="replace")
        if yt_cid_re.search(txt):
            violations.append(f"Tracked file contains YouTube comment_id pattern: {rel}")

        if video_ids:
            vid_hits = sum(1 for v in video_ids if v in txt)
            if vid_hits > 0:
                violations.append(f"Tracked file contains {vid_hits} raw YouTube video_id(s): {rel}")

        audit_script_files = {
            "scripts/verify_manuscript_100pct.py",
            "scripts/build_and_verify_dua_manifests.py",
        }
        if rel not in audit_script_files and "iannwendy/aitraining" in txt:
            violations.append(f"Tracked file references legacy repository name 'iannwendy/aitraining': {rel}")

        if rel not in audit_script_files and "raw.githubusercontent.com/iannwendy/VietDepression/main/data/labeled/" in txt:
            violations.append(f"Tracked file links directly to DUA-restricted raw CSVs: {rel}")

        if gold_snippets and not rel.startswith("tests/"):
            hits = sum(1 for s in gold_snippets if s in txt)
            if hits > 0:
                violations.append(f"Tracked file contains {hits} verbatim gold YouTube comment snippet(s): {rel}")

    if violations:
        print("[AUDIT FAILED] DUA / Privacy violations detected in git-tracked files:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)

    manifest_csvs = [f for f in tracked if f.startswith("data/manifests/") and f.endswith(".csv")]
    print(f"[AUDIT PASSED] Scanned {len(tracked)} git-tracked files:")
    print("  - 0 tracked CSVs contain raw comment_text, video_id, or comment_id")
    print("  - 0 tracked files leak verbatim gold YouTube comments or YouTube comment IDs")
    print("  - 0 tracked docs reference 'iannwendy/aitraining' or direct raw CSV download links")
    print(f"  - {len(manifest_csvs)} public de-identified SHA-256 manifest CSVs tracked in data/manifests/")


def main() -> None:
    parser = argparse.ArgumentParser(description="VietDepression Two-Tier DUA Manifest Builder & Privacy Auditor")
    parser.add_argument("--build", action="store_true", help="Generate de-identified manifests in data/manifests/")
    parser.add_argument("--verify-local", action="store_true", help="Verify local DUA CSVs against public SHA-256 manifests")
    parser.add_argument("--audit-git", action="store_true", help="Audit all git-tracked files for zero raw-text leakage")
    args = parser.parse_args()

    if not (args.build or args.verify_local or args.audit_git):
        parser.print_help()
        sys.exit(1)

    if args.build:
        cmd_build()
    if args.verify_local:
        cmd_verify_local()
    if args.audit_git:
        cmd_audit_git()


if __name__ == "__main__":
    main()
