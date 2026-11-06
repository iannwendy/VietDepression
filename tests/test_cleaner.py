"""Smoke tests for yt_depression_crawler.processing.cleaner.

CPU-only. Run:
    .venv/bin/python -m pytest tests/test_cleaner.py -v
or:  .venv/bin/python tests/test_cleaner.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vietdepression_lexicon import (  # noqa: E402
    RAW_COMMENT_COLUMNS,
    clean_comments,
    is_basic_spam,
    normalize_text,
)


class TestNormalizeText(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(normalize_text("  hello   world  "), "hello world")

    def test_preserves_vietnamese_diacritics(self):
        self.assertEqual(normalize_text("trầm cảm"), "trầm cảm")

    def test_preserves_emoji(self):
        # Emoji are not in the regex's class, so they pass through.
        # Internal double-spaces are collapsed to single (regex matches \s+).
        self.assertEqual(normalize_text("vui quá  😊  "), "vui quá 😊")

    def test_strips_only_outer_whitespace(self):
        # Inner tabs are collapsed to single space; control chars remain.
        self.assertEqual(normalize_text("\n\txin  chào\t"), "xin chào")


class TestIsBasicSpam(unittest.TestCase):
    def test_url_is_spam(self):
        self.assertTrue(is_basic_spam("xem tại https://example.com nhé"))

    def test_spam_phrase_dang_ky_kenh(self):
        self.assertTrue(is_basic_spam("đăng ký kênh để ủng hộ mình"))

    def test_spam_phrase_sub4sub(self):
        self.assertTrue(is_basic_spam("sub4sub nha mọi người"))

    def test_repeated_char_is_spam(self):
        self.assertTrue(is_basic_spam("aaaaaaaaaaaaaaaaaaaaaaaaa"))

    def test_normal_vietnamese_not_spam(self):
        self.assertFalse(is_basic_spam("cảm ơn bạn đã chia sẻ video rất hay"))

    def test_short_normal_text_not_spam_at_function_level(self):
        # is_basic_spam doesn't enforce length; clean_comments does.
        self.assertFalse(is_basic_spam("ok"))


class TestCleanComments(unittest.TestCase):
    def test_empty_input_writes_header_only(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "raw.csv"
            out = Path(d) / "cleaned.csv"
            inp.write_text("", encoding="utf-8")
            count = clean_comments(inp, out)
            self.assertEqual(count, 0)
            self.assertTrue(out.exists())
            df = pd.read_csv(out)
            self.assertEqual(list(df.columns), RAW_COMMENT_COLUMNS)

    def test_drops_duplicates_keeps_first(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "raw.csv"
            out = Path(d) / "cleaned.csv"
            rows = [
                {
                    "comment_id": "c1",
                    "video_id": "v1",
                    "video_title": "t",
                    "keyword": "k",
                    "comment_text": "cảm ơn bạn nhiều",
                    "like_count": "5",
                    "published_at": "2025-01-01",
                },
                {
                    "comment_id": "c2",
                    "video_id": "v1",
                    "video_title": "t",
                    "keyword": "k",
                    "comment_text": "cảm ơn bạn nhiều",  # duplicate text
                    "like_count": "3",
                    "published_at": "2025-01-02",
                },
                {
                    "comment_id": "c3",
                    "video_id": "v1",
                    "video_title": "t",
                    "keyword": "k",
                    "comment_text": "video rất hay và bổ ích",
                    "like_count": "2",
                    "published_at": "2025-01-03",
                },
            ]
            pd.DataFrame(rows).to_csv(inp, index=False, encoding="utf-8-sig")
            count = clean_comments(inp, out)
            df = pd.read_csv(out)
            self.assertEqual(count, 2)
            self.assertEqual(df.iloc[0]["comment_id"], "c1")
            self.assertIn("video rất hay và bổ ích", df["comment_text"].tolist())

    def test_drops_short_and_spam_rows(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "raw.csv"
            out = Path(d) / "cleaned.csv"
            rows = [
                {"comment_id": "a", "video_id": "v", "video_title": "t",
                 "keyword": "k", "comment_text": "hi",  # too short
                 "like_count": "0", "published_at": ""},
                {"comment_id": "b", "video_id": "v", "video_title": "t",
                 "keyword": "k", "comment_text": "aaaaaaaaaaaaaaaaaaaaaaaaa",  # spam
                 "like_count": "0", "published_at": ""},
                {"comment_id": "c", "video_id": "v", "video_title": "t",
                 "keyword": "k", "comment_text": "video rất hay và ý nghĩa",
                 "like_count": "0", "published_at": ""},
            ]
            pd.DataFrame(rows).to_csv(inp, index=False, encoding="utf-8-sig")
            count = clean_comments(inp, out)
            self.assertEqual(count, 1)
            df = pd.read_csv(out)
            self.assertEqual(df.iloc[0]["comment_id"], "c")


if __name__ == "__main__":
    unittest.main(verbosity=2)
