"""Smoke tests for scripts/domain_adapted_eval_utils.py and the DAPT orchestrator.

CPU-only. Skips GPU/MPS-heavy helpers (run_finetune, run_eval).
Run:  .venv/bin/python -m pytest tests/test_dapt_eval_helpers.py -v
or:    .venv/bin/python tests/test_dapt_eval_helpers.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Make scripts/ importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.domain_adapted_eval_utils import (  # noqa: E402
    aggregate_results,
    compute_cross_domain_metrics,
    run_finetune,
)


class TestComputeCrossDomainMetrics(unittest.TestCase):
    """Tests the text-keyed alignment fix for compute_cross_domain_metrics."""

    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as d:
            pred = Path(d) / "p.csv"
            gold = Path(d) / "g.csv"
            pd.DataFrame({
                "comment_text": ["a", "b", "c", "d"],
                "predicted_label": [1, 0, 1, 1],
            }).to_csv(pred, index=False)
            pd.DataFrame({
                "comment_text": ["a", "b", "c", "d"],
                "label": [1, 0, 0, 1],
            }).to_csv(gold, index=False)
            metrics = compute_cross_domain_metrics(str(pred), str(gold))
            self.assertIn("f1_macro", metrics)
            self.assertIn("accuracy", metrics)
            self.assertAlmostEqual(metrics["accuracy"], 0.75, places=4)
            # Gold=[1,0,0,1] pred=[1,0,1,1] -> 3/4 correct
            self.assertAlmostEqual(metrics["f1_macro"], 0.7333, places=3)

    def test_reorder_is_safe(self):
        """Reordering predictions should NOT corrupt metrics (text-keyed join)."""
        with tempfile.TemporaryDirectory() as d:
            pred = Path(d) / "p.csv"
            gold = Path(d) / "g.csv"
            # Predictions in scrambled order; gold in alphabetical.
            pd.DataFrame({
                "comment_text": ["d", "a", "c", "b"],
                "predicted_label": [1, 1, 1, 0],
            }).to_csv(pred, index=False)
            pd.DataFrame({
                "comment_text": ["a", "b", "c", "d"],
                "label": [1, 0, 0, 1],
            }).to_csv(gold, index=False)
            metrics = compute_cross_domain_metrics(str(pred), str(gold))
            # a: pred=1 gold=1 ✓; b: pred=0 gold=0 ✓; c: pred=1 gold=0 ✗; d: pred=1 gold=1 ✓
            self.assertAlmostEqual(metrics["accuracy"], 0.75, places=4)

    def test_missing_predictions_raises(self):
        """If predictions are missing for some gold rows, raise loudly."""
        with tempfile.TemporaryDirectory() as d:
            pred = Path(d) / "p.csv"
            gold = Path(d) / "g.csv"
            pd.DataFrame({
                "comment_text": ["a", "b", "c"],
                "predicted_label": [1, 0, 1],
            }).to_csv(pred, index=False)
            pd.DataFrame({
                "comment_text": ["a", "b", "c", "d"],
                "label": [1, 0, 0, 1],
            }).to_csv(gold, index=False)
            with self.assertRaises(ValueError) as ctx:
                compute_cross_domain_metrics(str(pred), str(gold))
            self.assertIn("merged=", str(ctx.exception))

    def test_missing_comment_text_column_raises(self):
        with tempfile.TemporaryDirectory() as d:
            pred = Path(d) / "p.csv"
            gold = Path(d) / "g.csv"
            pd.DataFrame({"foo": [1, 0], "predicted_label": [1, 0]}).to_csv(pred, index=False)
            pd.DataFrame({"foo": [1, 0], "label": [1, 0]}).to_csv(gold, index=False)
            with self.assertRaises(ValueError):
                compute_cross_domain_metrics(str(pred), str(gold))


class TestAggregateResults(unittest.TestCase):
    """Tests aggregate_results: metrics.json + comparison_table.md."""

    def _make_run(self, model_tag, seed, test_set, f1, status="ok"):
        return {
            "model_tag": model_tag,
            "seed": seed,
            "test_set": test_set,
            "status": status,
            "accuracy": 0.7,
            "f1_macro": f1,
            "f1_depression": f1 - 0.05,
            "precision_macro": 0.75,
            "recall_macro": 0.7,
            "confusion_matrix": [[1, 1], [0, 2]],
        }

    def test_writes_metrics_json_and_table(self):
        with tempfile.TemporaryDirectory() as d:
            runs = [
                self._make_run("original", 42, "final_test", 0.80),
                self._make_run("original", 123, "final_test", 0.78),
                self._make_run("original", 2024, "final_test", 0.79),
                self._make_run("domain_adapted", 42, "final_test", 0.74),
                self._make_run("domain_adapted", 123, "final_test", 0.75),
                self._make_run("domain_adapted", 2024, "final_test", 0.76),
                self._make_run("original", 42, "vsmec", 0.76),
            ]
            aggregate_results(runs, d, "abc123", "2026-06-26_120000")

            metrics_path = Path(d) / "metrics.json"
            table_path = Path(d) / "comparison_table.md"
            self.assertTrue(metrics_path.exists())
            self.assertTrue(table_path.exists())

            payload = json.loads(metrics_path.read_text())
            self.assertEqual(payload["git_commit"], "abc123")
            self.assertEqual(payload["timestamp"], "2026-06-26_120000")
            self.assertEqual(len(payload["runs"]), 7)

            table = table_path.read_text()
            self.assertIn("| Model | Test set | n_seeds |", table)
            self.assertIn("| original | final_test | 3 |", table)
            self.assertIn("| domain_adapted | final_test | 3 |", table)
            self.assertIn("| original | vsmec | 1 |", table)
            # Failed runs footer
            self.assertIn("Failed runs: 0 / 7", table)

    def test_failed_runs_excluded_from_aggregation(self):
        with tempfile.TemporaryDirectory() as d:
            runs = [
                self._make_run("original", 42, "final_test", 0.80),
                self._make_run("original", 123, "final_test", 0.78, status="failed"),
                self._make_run("original", 2024, "final_test", 0.79),
            ]
            aggregate_results(runs, d, "x", "y")
            table = (Path(d) / "comparison_table.md").read_text()
            # Failed row drops seed=123; only 2 seeds aggregated.
            self.assertIn("| original | final_test | 2 |", table)
            self.assertIn("Failed runs: 1 / 3", table)

    def test_single_seed_shows_na_std(self):
        with tempfile.TemporaryDirectory() as d:
            runs = [self._make_run("original", 42, "final_test", 0.80)]
            aggregate_results(runs, d, "x", "y")
            table = (Path(d) / "comparison_table.md").read_text()
            # stddev undefined for n=1 -> should show "n/a" rather than 0.0
            self.assertIn("+/- n/a", table)


class TestRunFinetuneDataPath(unittest.TestCase):
    """Regression test for the bug fixed in commit f09fddc.

    Earlier versions of run_finetune() did not override the CSV paths, so
    the subprocess fell back to TRAIN_FILE/VAL_FILE/TEST_FILE in
    core/config.py — which still pointed to the pre-Phase-1 splits
    (data/train.csv, 2,632 rows). The original DAPT evaluation
    (results/domain_adapted_eval_2026-06-25_123440/) was inadvertently
    trained on those OLD splits.

    Updated for Round 6 v2: files now in data/labeled/ with sizes:
    - final_train.csv: 6,392 rows
    - final_val.csv: 1,371 rows
    - final_test.csv: 1,371 rows

    These tests guard against the regression by static-checking both the
    function body and the default argument values. They do NOT execute
    run_finetune (which requires GPU/MPS) and are safe to run on CPU.
    """

    def test_run_finetune_source_references_final_splits(self):
        import inspect

        src = inspect.getsource(run_finetune)
        # Subprocess script body must explicitly override each split.
        self.assertIn("data/labeled/final_train.csv", src,
                      "run_finetune() body must reference data/labeled/final_train.csv")
        self.assertIn("data/labeled/final_val.csv", src,
                      "run_finetune() body must reference data/labeled/final_val.csv")
        self.assertIn("data/labeled/final_test.csv", src,
                      "run_finetune() body must reference data/labeled/final_test.csv")
        # And must NOT silently fall back to the legacy train.csv path.
        self.assertNotIn("train_file=TRAIN_FILE", src,
                         "run_finetune() must not default to legacy TRAIN_FILE")

    def test_run_finetune_default_args_point_to_final_splits(self):
        import inspect

        sig = inspect.signature(run_finetune)
        self.assertEqual(
            sig.parameters["train_csv"].default, "data/labeled/final_train.csv",
            "Default train_csv must point at data/labeled/final_train.csv",
        )
        self.assertEqual(
            sig.parameters["val_csv"].default, "data/labeled/final_val.csv",
            "Default val_csv must point at data/labeled/final_val.csv",
        )
        self.assertEqual(
            sig.parameters["test_csv"].default, "data/labeled/final_test.csv",
            "Default test_csv must point at data/labeled/final_test.csv",
        )


class TestModelTag(unittest.TestCase):
    """Tests scripts/evaluate_domain_adapted_phobert.model_tag."""

    def test_original(self):
        from scripts.evaluate_domain_adapted_phobert import model_tag
        self.assertEqual(model_tag("vinai/phobert-base"), "original")
        self.assertEqual(model_tag("/path/to/phobert_base_local"), "original")

    def test_domain_adapted(self):
        from scripts.evaluate_domain_adapted_phobert import model_tag
        self.assertEqual(model_tag("models/phobert_domain_adapted"), "domain_adapted")
        self.assertEqual(model_tag("some/random/domain_adapted_v2"), "domain_adapted")


class TestSanityCheckDatasets(unittest.TestCase):
    """Tests scripts/evaluate_domain_adapted_phobert.sanity_check_datasets.

    Uses the real on-disk datasets (read-only smoke check).
    """

    def test_passes_on_real_data(self):
        from scripts.evaluate_domain_adapted_phobert import sanity_check_datasets
        # Should not raise. If schemas drift, this is the early warning.
        sanity_check_datasets()


if __name__ == "__main__":
    unittest.main(verbosity=2)