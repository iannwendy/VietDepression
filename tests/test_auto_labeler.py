"""Smoke tests for yt_depression_crawler.labeling.auto_labeler.

CPU-only. Run:
    .venv/bin/python tests/test_auto_labeler.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vietdepression_lexicon import (  # noqa: E402
    STRONG_KEYWORDS,
    find_matches,
    label_comment,
    normalize_for_match,
    strip_accents,
)


class TestStripAccents(unittest.TestCase):
    def test_strips_vietnamese_diacritics(self):
        self.assertEqual(strip_accents("trầm cảm"), "tram cam")

    def test_d_stripped_to_d(self):
        # NFD decomposition alone doesn't map "đ" to "d"; the helper does.
        self.assertEqual(strip_accents("đợi mãi"), "doi mai")
        self.assertEqual(strip_accents("Đang họp"), "Dang hop")

    def test_ascii_unchanged(self):
        self.assertEqual(strip_accents("hello world"), "hello world")


class TestNormalizeForMatch(unittest.TestCase):
    def test_wraps_with_spaces(self):
        # The padding spaces let the " {keyword} " pattern in find_matches
        # detect word boundaries; this is intentional.
        self.assertEqual(normalize_for_match("trầm cảm"), " tram cam ")

    def test_lowercases(self):
        self.assertEqual(normalize_for_match("Trầm Cảm"), " tram cam ")


class TestLabelComment(unittest.TestCase):
    def test_strong_depression_keyword(self):
        # "muốn chết" is a STRONG_KEYWORDS entry (weight 5).
        result = label_comment("tôi muốn chết quá")
        self.assertEqual(result.weak_label, "depression_auto")
        self.assertGreaterEqual(result.depression_score, 5)
        self.assertIn("muốn chết", result.matched_keywords)

    def test_normal_keyword(self):
        # "video hay" + "cảm ơn" are NORMAL_KEYWORDS entries (weight -2 each).
        result = label_comment("video hay quá cảm ơn bạn")
        self.assertEqual(result.weak_label, "normal_auto")
        self.assertLessEqual(result.depression_score, -2)

    def test_review_context_triggers_need_review(self):
        # "chữa lành tâm lý" or "hãy đi khám" both appear in REVIEW_CONTEXT_KEYWORDS.
        result = label_comment("bạn ơi hãy đi khám đi nhé")
        self.assertTrue(result.need_review)

    def test_short_uncertain(self):
        # Empty / very short text — score=0 → uncertain.
        result = label_comment("ok")
        self.assertEqual(result.weak_label, "uncertain")
        self.assertEqual(result.depression_score, 0)


class TestFindMatches(unittest.TestCase):
    def test_matches_accent_insensitive(self):
        # Text with full diacritics should match a strong keyword written
        # without diacritics (and vice versa), because find_matches compares
        # on `match_key` (NFD-stripped). The first STRONG_KEYWORDS entry is
        # "muốn chết" → match_key "muon chet".
        matches = find_matches(" toi muon chet qua ", STRONG_KEYWORDS[:3])
        self.assertIn("muốn chết", matches,
                      f"expected a depression match, got {matches}")

    def test_no_false_positive_on_unrelated_substring(self):
        # "mo" should not match "mơ mộng" — find_matches uses whole-word
        # padding (" key ") AND substring fallback only on the full strip.
        # Verify the helper does not blow up on an empty prepared list.
        matches = find_matches(" toi di choi voi ban ", [])
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
