"""Smoke tests for yt_depression_crawler.labeling.gold_builder.

CPU-only. Run:
    .venv/bin/python tests/test_gold_builder.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from yt_depression_crawler.labeling.gold_builder import (  # noqa: E402
    GOLD_COLUMNS,
    build_gold_review_set,
)


def _write_review_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


class TestBuildGoldReviewSet(unittest.TestCase):
    def test_filters_to_valid_final_labels_only(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "review.csv"
            out = Path(d) / "gold.csv"
            _write_review_csv(inp, [
                {"comment_text": "cảm ơn bạn", "final_label": "normal",
                 "weak_label": "normal_auto", "confidence": "high",
                 "depression_score": "-2", "matched_keywords": "cảm ơn",
                 "review_bucket": "normal_high", "reviewer_note": ""},
                {"comment_text": "tôi muốn chết", "final_label": "depression",
                 "weak_label": "depression_auto", "confidence": "high",
                 "depression_score": "5", "matched_keywords": "muốn chết",
                 "review_bucket": "depression_high", "reviewer_note": ""},
                # These should be excluded.
                {"comment_text": "hơi mệt", "final_label": "uncertain",
                 "weak_label": "uncertain", "confidence": "low",
                 "depression_score": "0", "matched_keywords": "",
                 "review_bucket": "uncertain", "reviewer_note": ""},
                {"comment_text": "spammer", "final_label": "exclude",
                 "weak_label": "normal_auto", "confidence": "medium",
                 "depression_score": "-2", "matched_keywords": "",
                 "review_bucket": "need_review", "reviewer_note": "spam"},
                {"comment_text": "chưa review", "final_label": "",
                 "weak_label": "uncertain", "confidence": "low",
                 "depression_score": "0", "matched_keywords": "",
                 "review_bucket": "boundary", "reviewer_note": ""},
            ])
            report = build_gold_review_set(inp, out)
            self.assertEqual(report["gold_rows"], 2)
            self.assertEqual(report["excluded_from_gold"], 3)
            df = pd.read_csv(out)
            self.assertEqual(set(df["final_label"]), {"normal", "depression"})

    def test_label_mapping_normal_depression(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "r.csv"
            out = Path(d) / "g.csv"
            _write_review_csv(inp, [
                {"comment_text": "video hay", "final_label": "normal",
                 "weak_label": "normal_auto", "confidence": "high",
                 "depression_score": "-2", "matched_keywords": "",
                 "review_bucket": "normal_high", "reviewer_note": ""},
                {"comment_text": "muốn chết", "final_label": "depression",
                 "weak_label": "depression_auto", "confidence": "high",
                 "depression_score": "5", "matched_keywords": "",
                 "review_bucket": "depression_high", "reviewer_note": ""},
            ])
            build_gold_review_set(inp, out)
            df = pd.read_csv(out)
            label_map = dict(zip(df["final_label"], df["label"]))
            self.assertEqual(label_map["normal"], 0)
            self.assertEqual(label_map["depression"], 1)

    def test_drops_duplicate_comment_text(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "r.csv"
            out = Path(d) / "g.csv"
            _write_review_csv(inp, [
                {"comment_text": "cảm ơn", "final_label": "normal",
                 "weak_label": "normal_auto", "confidence": "high",
                 "depression_score": "-2", "matched_keywords": "",
                 "review_bucket": "normal_high", "reviewer_note": "first"},
                {"comment_text": "cảm ơn", "final_label": "normal",
                 "weak_label": "normal_auto", "confidence": "high",
                 "depression_score": "-2", "matched_keywords": "",
                 "review_bucket": "normal_high", "reviewer_note": "dup"},
            ])
            report = build_gold_review_set(inp, out)
            self.assertEqual(report["gold_rows"], 1)
            df = pd.read_csv(out)
            self.assertEqual(len(df), 1)
            # First occurrence wins.
            self.assertEqual(df.iloc[0]["reviewer_note"], "first")

    def test_missing_input_file_writes_empty_gold(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "missing.csv"
            out = Path(d) / "g.csv"
            report = build_gold_review_set(inp, out)
            self.assertEqual(report["gold_rows"], 0)
            self.assertTrue(out.exists())
            df = pd.read_csv(out)
            self.assertEqual(list(df.columns), GOLD_COLUMNS)
            self.assertEqual(len(df), 0)

    def test_output_columns_in_declared_order(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "r.csv"
            out = Path(d) / "g.csv"
            _write_review_csv(inp, [
                {"comment_text": "video hay", "final_label": "normal",
                 "weak_label": "normal_auto", "confidence": "high",
                 "depression_score": "-2", "matched_keywords": "",
                 "review_bucket": "normal_high", "reviewer_note": ""},
            ])
            build_gold_review_set(inp, out)
            df = pd.read_csv(out)
            self.assertEqual(list(df.columns), GOLD_COLUMNS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
