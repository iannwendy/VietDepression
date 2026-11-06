"""Smoke tests for yt_depression_crawler.modeling.phobert.phobert_postprocess.

CPU-only — no model loading. Run:
    .venv/bin/python tests/test_phobert_postprocess.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from yt_depression_crawler.core.config import (  # noqa: E402
    PHOBERT_ACTIVE_LEARNING_MAX_SAMPLES,
    PHOBERT_ACTIVE_LEARNING_PROB_THRESHOLD,
    PHOBERT_PSEUDO_LABEL_PROB_THRESHOLD,
)
from yt_depression_crawler.modeling.phobert.phobert_postprocess import (  # noqa: E402
    ACTIVE_COLUMNS,
    CONFIDENT_COLUMNS,
    build_phobert_followup_files,
)


def _write_predictions(path: Path, rows: list[dict]) -> None:
    cols = [
        "comment_text", "source_weak_label", "source_confidence",
        "source_depression_score", "phobert_label", "probability",
        "prob_normal", "prob_depression",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False, encoding="utf-8-sig")


class TestBuildPhobertFollowupFiles(unittest.TestCase):
    def test_empty_predictions_writes_headers(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "pred.csv"
            out_conf = Path(d) / "conf.csv"
            out_act = Path(d) / "act.csv"
            inp.write_text("", encoding="utf-8")
            report = build_phobert_followup_files(inp, out_conf, out_act)
            self.assertEqual(report["confident_rows"], 0)
            self.assertEqual(report["active_rows"], 0)
            conf_df = pd.read_csv(out_conf)
            self.assertEqual(list(conf_df.columns), CONFIDENT_COLUMNS)
            act_df = pd.read_csv(out_act)
            self.assertEqual(list(act_df.columns), ACTIVE_COLUMNS)

    def test_confident_threshold_inclusive(self):
        # probability >= 0.9 → confident. Test exactly at threshold.
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "pred.csv"
            out_conf = Path(d) / "conf.csv"
            out_act = Path(d) / "act.csv"
            _write_predictions(inp, [
                {"comment_text": f"c{i}",
                 "source_weak_label": "uncertain", "source_confidence": "low",
                 "source_depression_score": "0",
                 "phobert_label": "depression" if i < 2 else "normal",
                 "probability": str(prob),
                 "prob_normal": str(1 - prob), "prob_depression": str(prob)}
                for i, prob in enumerate([0.91, 0.90, 0.89, 0.50])
            ])
            report = build_phobert_followup_files(inp, out_conf, out_act)
            self.assertEqual(report["confident_rows"], 2)  # 0.91 and 0.90
            self.assertEqual(report["confident_threshold"], PHOBERT_PSEUDO_LABEL_PROB_THRESHOLD)

    def test_active_learning_categorizes_four_buckets(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "pred.csv"
            out_conf = Path(d) / "conf.csv"
            out_act = Path(d) / "act.csv"
            _write_predictions(inp, [
                # weak_model_disagreement: weak=normal_auto, model=depression
                {"comment_text": "disagree 1", "source_weak_label": "normal_auto",
                 "source_confidence": "high", "source_depression_score": "-2",
                 "phobert_label": "depression", "probability": "0.85",
                 "prob_normal": "0.15", "prob_depression": "0.85"},
                # boundary_probability: 0.45 <= p <= 0.55
                {"comment_text": "boundary 1", "source_weak_label": "uncertain",
                 "source_confidence": "low", "source_depression_score": "0",
                 "phobert_label": "depression", "probability": "0.50",
                 "prob_normal": "0.50", "prob_depression": "0.50"},
                # low_confidence: p < 0.7
                {"comment_text": "low 1", "source_weak_label": "uncertain",
                 "source_confidence": "low", "source_depression_score": "0",
                 "phobert_label": "normal", "probability": "0.40",
                 "prob_normal": "0.60", "prob_depression": "0.40"},
                # source_uncertain: weak_label == uncertain
                {"comment_text": "src unc 1", "source_weak_label": "uncertain",
                 "source_confidence": "low", "source_depression_score": "0",
                 "phobert_label": "depression", "probability": "0.80",
                 "prob_normal": "0.20", "prob_depression": "0.80"},
                # High confidence (>= 0.9), NOT active
                {"comment_text": "high conf 1", "source_weak_label": "normal_auto",
                 "source_confidence": "high", "source_depression_score": "-2",
                 "phobert_label": "normal", "probability": "0.95",
                 "prob_normal": "0.95", "prob_depression": "0.05"},
            ])
            report = build_phobert_followup_files(inp, out_conf, out_act)
            # 4 candidates (all except "high conf 1") → capped to MAX_SAMPLES
            # but here it's well under cap.
            act_df = pd.read_csv(out_act)
            buckets = set(act_df["review_bucket"].dropna())
            self.assertEqual(buckets, {
                "weak_model_disagreement",
                "boundary_probability",
                "low_confidence",
                "source_uncertain",
            })
            self.assertEqual(len(act_df), 4)

    def test_active_learning_capped_at_max_samples(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "pred.csv"
            out_conf = Path(d) / "conf.csv"
            out_act = Path(d) / "act.csv"
            n = PHOBERT_ACTIVE_LEARNING_MAX_SAMPLES + 500
            rows = [
                {"comment_text": f"boundary {i}",
                 "source_weak_label": "uncertain", "source_confidence": "low",
                 "source_depression_score": "0",
                 "phobert_label": "depression", "probability": "0.50",
                 "prob_normal": "0.50", "prob_depression": "0.50"}
                for i in range(n)
            ]
            _write_predictions(inp, rows)
            report = build_phobert_followup_files(inp, out_conf, out_act)
            self.assertEqual(report["active_rows"], PHOBERT_ACTIVE_LEARNING_MAX_SAMPLES)

    def test_low_confidence_threshold_matches_constant(self):
        # low_confidence bucket uses PHOBERT_ACTIVE_LEARNING_PROB_THRESHOLD = 0.7.
        # A row with p=0.7 exactly should NOT enter (it's >= 0.7).
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "pred.csv"
            out_conf = Path(d) / "conf.csv"
            out_act = Path(d) / "act.csv"
            _write_predictions(inp, [
                {"comment_text": "exactly 0.7",
                 "source_weak_label": "uncertain", "source_confidence": "low",
                 "source_depression_score": "0",
                 "phobert_label": "normal", "probability": str(PHOBERT_ACTIVE_LEARNING_PROB_THRESHOLD),
                 "prob_normal": "0.50", "prob_depression": "0.50"},
            ])
            report = build_phobert_followup_files(inp, out_conf, out_act)
            # The single row qualifies only via "source_uncertain" (since
            # weak_label is uncertain), NOT via low_confidence (boundary).
            act_df = pd.read_csv(out_act)
            buckets = set(act_df["review_bucket"].dropna())
            self.assertNotIn("low_confidence", buckets)

    def test_confident_dedupes_by_comment_text(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "pred.csv"
            out_conf = Path(d) / "conf.csv"
            out_act = Path(d) / "act.csv"
            _write_predictions(inp, [
                {"comment_text": "same text", "source_weak_label": "uncertain",
                 "source_confidence": "low", "source_depression_score": "0",
                 "phobert_label": "depression", "probability": "0.95",
                 "prob_normal": "0.05", "prob_depression": "0.95"},
                {"comment_text": "same text", "source_weak_label": "uncertain",
                 "source_confidence": "low", "source_depression_score": "0",
                 "phobert_label": "normal", "probability": "0.91",
                 "prob_normal": "0.91", "prob_depression": "0.09"},
            ])
            report = build_phobert_followup_files(inp, out_conf, out_act)
            self.assertEqual(report["confident_rows"], 1)
            conf_df = pd.read_csv(out_conf)
            self.assertEqual(conf_df.iloc[0]["label"], "depression")  # first wins


if __name__ == "__main__":
    unittest.main(verbosity=2)
