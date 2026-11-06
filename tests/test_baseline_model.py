"""Smoke tests for yt_depression_crawler.modeling.baseline.baseline_model.

CPU-only, no model download. Trains tiny TF-IDF + (LogReg | LinearSVC) on a
synthetic 60-row fixture and asserts the pipeline + persisted metrics behave
as documented.

Run:
    .venv/bin/python tests/test_baseline_model.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from yt_depression_crawler.modeling.baseline.baseline_model import (  # noqa: E402
    train_baseline_model,
    train_linear_svc_model,
)


def _build_split_csvs(d: Path, n_per_class: int = 25) -> tuple[Path, Path, Path]:
    """Write train.csv / val.csv / test.csv in the schema baseline_model expects.

    The schema mirrors SPLIT_COLUMNS (comment_text, label, weak_label, ...).
    """
    rng_seed = 42
    normal_texts = [
        f"video hay qua cam on ban {i}" for i in range(n_per_class)
    ]
    depression_texts = [
        f"toi muon chet qua khong con ly do song {i}" for i in range(n_per_class)
    ]
    rows = []
    for t in normal_texts:
        rows.append({"comment_text": t, "label": 0,
                     "weak_label": "normal_auto", "confidence": "high",
                     "depression_score": -2, "matched_keywords": "cam on"})
    for t in depression_texts:
        rows.append({"comment_text": t, "label": 1,
                     "weak_label": "depression_auto", "confidence": "high",
                     "depression_score": 5, "matched_keywords": "muon chet"})
    df = pd.DataFrame(rows).sample(frac=1, random_state=rng_seed).reset_index(drop=True)
    # 70/15/15 stratified-ish split (use the first 70% as train, 15% val, 15% test).
    n = len(df)
    train_df = df.iloc[: int(0.7 * n)]
    val_df = df.iloc[int(0.7 * n): int(0.85 * n)]
    test_df = df.iloc[int(0.85 * n):]

    train_path = d / "train.csv"
    val_path = d / "val.csv"
    test_path = d / "test.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    return train_path, val_path, test_path


class TestTrainBaselineModel(unittest.TestCase):
    def test_smoke_train_returns_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            d_path = Path(d)
            train, val, test = _build_split_csvs(d_path)
            metrics_path = d_path / "logreg_metrics.json"
            model_path = d_path / "logreg.joblib"
            report_path = d_path / "report.json"
            metrics = train_baseline_model(
                train_file=train, val_file=val, test_file=test,
                model_file=model_path, metrics_file=metrics_path,
                report_file=report_path,
            )
            self.assertIn("test", metrics)
            self.assertIn("f1_macro", metrics["test"])
            # Sanity: with strong-keyword signal, F1 should be high.
            self.assertGreaterEqual(metrics["test"]["f1_macro"], 0.8)
            self.assertTrue(model_path.exists())
            self.assertTrue(metrics_path.exists())

    def test_persisted_model_predicts(self):
        with tempfile.TemporaryDirectory() as d:
            d_path = Path(d)
            train, val, test = _build_split_csvs(d_path)
            model_path = d_path / "logreg.joblib"
            train_baseline_model(
                train_file=train, val_file=val, test_file=test,
                model_file=model_path,
                metrics_file=d_path / "metrics.json",
                report_file=d_path / "report.json",
            )
            loaded = joblib.load(model_path)
            preds = loaded.predict(["cam on ban rat nhieu", "toi muon chet"])
            self.assertEqual(len(preds), 2)
            self.assertIn(int(preds[0]), (0, 1))
            self.assertIn(int(preds[1]), (0, 1))

    def test_handles_class_imbalance_without_error(self):
        # 40 normal / 10 depression — class_weight=balanced prevents the
        # trivial "predict all 0" solution. We don't assert accuracy, just
        # that training succeeds and metrics are well-formed.
        with tempfile.TemporaryDirectory() as d:
            d_path = Path(d)
            normal = [{"comment_text": f"hay qua cam on {i}", "label": 0,
                       "weak_label": "normal_auto", "confidence": "high",
                       "depression_score": -2, "matched_keywords": "hay qua"}
                      for i in range(40)]
            depression = [{"comment_text": f"toi muon chet {i}", "label": 1,
                           "weak_label": "depression_auto", "confidence": "high",
                           "depression_score": 5, "matched_keywords": "muon chet"}
                          for i in range(10)]
            df = pd.DataFrame(normal + depression).sample(frac=1, random_state=1).reset_index(drop=True)
            train = df.iloc[:35]
            val = df.iloc[35:45]
            test = df.iloc[45:]
            train_path = d_path / "train.csv"
            val_path = d_path / "val.csv"
            test_path = d_path / "test.csv"
            train.to_csv(train_path, index=False, encoding="utf-8-sig")
            val.to_csv(val_path, index=False, encoding="utf-8-sig")
            test.to_csv(test_path, index=False, encoding="utf-8-sig")
            metrics = train_baseline_model(
                train_file=train_path, val_file=val_path, test_file=test_path,
                model_file=d_path / "m.joblib",
                metrics_file=d_path / "met.json",
                report_file=d_path / "rep.json",
            )
            self.assertIn("accuracy", metrics["test"])
            # Confusion matrix must be 2x2 (labels 0, 1).
            cm = metrics["test"]["confusion_matrix"]
            self.assertEqual(len(cm), 2)
            self.assertEqual(len(cm[0]), 2)


class TestTrainLinearSvcModel(unittest.TestCase):
    def test_smoke_train_returns_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            d_path = Path(d)
            train, val, test = _build_split_csvs(d_path)
            metrics_path = d_path / "svc_metrics.json"
            model_path = d_path / "svc.joblib"
            metrics = train_linear_svc_model(
                train_file=train, val_file=val, test_file=test,
                model_file=model_path, metrics_file=metrics_path,
                report_file=d_path / "report.json",
            )
            self.assertIn("test", metrics)
            self.assertIn("f1_macro", metrics["test"])
            self.assertGreaterEqual(metrics["test"]["f1_macro"], 0.8)
            self.assertEqual(metrics["C"], 1.0)
            self.assertTrue(model_path.exists())
            # Metrics file contains both training and C parameter.
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["C"], 1.0)
            self.assertEqual(payload["label_mapping"], {"normal": 0, "depression": 1})

    def test_svc_persisted_model_predicts(self):
        with tempfile.TemporaryDirectory() as d:
            d_path = Path(d)
            train, val, test = _build_split_csvs(d_path)
            model_path = d_path / "svc.joblib"
            train_linear_svc_model(
                train_file=train, val_file=val, test_file=test,
                model_file=model_path,
                metrics_file=d_path / "metrics.json",
                report_file=d_path / "report.json",
            )
            loaded = joblib.load(model_path)
            preds = loaded.predict(["hay qua cam on ban", "toi muon chet that su"])
            self.assertEqual(len(preds), 2)
            # First example should be predicted 0, second 1.
            self.assertEqual(int(preds[0]), 0)
            self.assertEqual(int(preds[1]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
