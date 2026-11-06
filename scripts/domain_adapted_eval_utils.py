"""Helpers for domain-adapted PhoBERT evaluation."""
import os
import subprocess
import sys
from pathlib import Path


def run_finetune(
    model_path: str,
    seed: int,
    output_dir: str,
    train_csv: str = "data/labeled/final_train.csv",
    val_csv: str = "data/labeled/final_val.csv",
    test_csv: str = "data/labeled/final_test.csv",
) -> str:
    """Fine-tune a PhoBERT model on the FINAL dataset (post-Phase-1/2 rebuild).

    Returns path to the best checkpoint directory.

    Earlier versions of this function omitted the CSV path overrides, which
    caused the trainer to fall back to `TRAIN_FILE`/`VAL_FILE`/`TEST_FILE` in
    `core/config.py` — those point to the pre-round-3 `data/train.csv` (2,632
    rows), not the new `data/labeled/final_train.csv` (6,392 rows, Round 6 v2).
    The defaults here make the new behaviour explicit; callers can override for ablation.

    The trainer (`phobert_train.train_phobert_first`) saves the best
    checkpoint directly to its `output_dir` argument via
    `model.save_pretrained(output_dir)` / `tokenizer.save_pretrained(output_dir)`
    whenever validation macro F1 improves (early-stopping tracking). There is
    no extra `best_model/` or `checkpoint-<N>/` subdirectory, so the override
    must point at `output_dir` itself.
    """
    os.environ["PHOBERT_MODEL_NAME_OVERRIDE"] = model_path
    os.environ["PHOBERT_OUTPUT_DIR_OVERRIDE"] = output_dir
    os.environ["PHOBERT_RANDOM_SEED_OVERRIDE"] = str(seed)

    # Subprocess snippet — must import Path before using it in kwargs.
    script = (
        "from pathlib import Path; "
        "from yt_depression_crawler.modeling.phobert.phobert_train import "
        "train_phobert_first; "
        f"train_phobert_first("
        f"train_file=Path('{train_csv}'), "
        f"val_file=Path('{val_csv}'), "
        f"test_file=Path('{test_csv}'))"
    )
    # Tee subprocess output to a per-run log so training progress is recoverable
    # when the run succeeds (subprocess.run silently drops stdout otherwise).
    log_path = os.path.join(output_dir, "finetune_stdout.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        result = subprocess.run(
            [sys.executable, "-c", script],
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Fine-tune failed for {model_path} seed={seed}. "
            f"See log: {log_path}"
        )

    # Trainer saves best weights straight to `output_dir` (no `best_model` subdir).
    # Accept either `output_dir` itself or a nested `best_model/` for forward-compat
    # in case the trainer is later refactored to nest.
    candidates = [output_dir, os.path.join(output_dir, "best_model")]
    expected = next((c for c in candidates if os.path.isdir(c)), None)
    if expected is None:
        raise RuntimeError(
            f"Expected checkpoint at one of {candidates} but none exist"
        )
    return expected


def run_eval(checkpoint_dir: str, eval_csv: str, output_csv: str) -> None:
    """Run prediction on an eval CSV using a fine-tuned checkpoint.

    Writes predictions (comment_text, prob_normal, prob_depression,
    predicted_label) to output_csv.
    """
    import pandas as pd
    from yt_depression_crawler.modeling.phobert import phobert_predict

    df = pd.read_csv(eval_csv)
    if "comment_text" not in df.columns:
        raise ValueError(f"{eval_csv} must have a 'comment_text' column")

    texts = df["comment_text"].astype(str).tolist()
    preds = phobert_predict.predict_texts(
        texts=texts,
        model_dir=checkpoint_dir,
    )
    # preds is a list of dicts with keys: predicted_label, prob_normal, prob_depression
    out_df = pd.DataFrame({
        "comment_text": texts,
        "predicted_label": [p["predicted_label"] for p in preds],
        "prob_normal": [p["prob_normal"] for p in preds],
        "prob_depression": [p["prob_depression"] for p in preds],
    })
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)


import json


def compute_cross_domain_metrics(pred_csv: str, gold_csv: str) -> dict:
    """Compute classification metrics for predictions joined with gold labels.

    Both CSVs are joined by ``comment_text`` (text-keyed alignment, not
    positional — robust against reordering or row-count drift). Returns
    a dict matching the metrics schema in the design spec.
    """
    import pandas as pd
    from yt_depression_crawler.modeling.phobert.phobert_utils import (
        compute_metrics as _compute_metrics,
    )

    preds = pd.read_csv(pred_csv)
    gold = pd.read_csv(gold_csv)
    if "comment_text" not in preds.columns or "comment_text" not in gold.columns:
        raise ValueError(
            f"Both CSVs must have 'comment_text'; "
            f"preds cols={list(preds.columns)}, gold cols={list(gold.columns)}"
        )

    # Text-keyed inner join: drop predictions whose text isn't in gold (shouldn't
    # happen, but guards against silent drop) and gold rows that didn't get a
    # prediction (also shouldn't happen, but worth flagging).
    merged = preds.merge(gold[["comment_text", "label"]], on="comment_text", how="inner")
    if len(merged) != len(gold):
        raise ValueError(
            f"After text-keyed join: merged={len(merged)} gold={len(gold)}. "
            f"Predictions missing for some gold rows or vice versa."
        )

    y_true = merged["label"].astype(int).values
    y_pred = merged["predicted_label"].astype(int).values

    return _compute_metrics(y_true, y_pred)


def aggregate_results(runs: list, output_dir: str, git_commit: str, timestamp: str) -> None:
    """Aggregate per-run metrics into metrics.json and comparison_table.md."""
    import statistics
    from collections import defaultdict

    metrics_path = os.path.join(output_dir, "metrics.json")
    table_path = os.path.join(output_dir, "comparison_table.md")

    payload = {
        "git_commit": git_commit,
        "timestamp": timestamp,
        "runs": runs,
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Group by (model_tag, test_set)
    groups = defaultdict(list)
    for r in runs:
        if r.get("status") == "failed":
            continue
        key = (r["model_tag"], r["test_set"])
        groups[key].append(r)

    lines = [
        "# Domain-Adapted PhoBERT Evaluation Results",
        "",
        f"Git commit: `{git_commit}`  ",
        f"Timestamp: `{timestamp}`",
        "",
        "| Model | Test set | n_seeds | Accuracy (mean+/-std) | "
        "F1 macro (mean+/-std) | F1 depression (mean+/-std) | "
        "Precision macro (mean+/-std) | Recall macro (mean+/-std) |",
        "|-------|----------|---------|------------------------|"
        "------------------------|-----------------------------|"
        "------------------------------|----------------------------|",
    ]

    def fmt(vals):
        if not vals:
            return "n/a"
        mean = statistics.mean(vals)
        if len(vals) < 2:
            return f"{mean:.4f} +/- n/a (n=1)"
        std = statistics.stdev(vals)
        return f"{mean:.4f} +/- {std:.4f}"

    for (model_tag, test_set), group in sorted(groups.items()):
        n = len(group)
        lines.append(
            f"| {model_tag} | {test_set} | {n} | "
            f"{fmt([g['accuracy'] for g in group])} | "
            f"{fmt([g['f1_macro'] for g in group])} | "
            f"{fmt([g['f1_depression'] for g in group])} | "
            f"{fmt([g['precision_macro'] for g in group])} | "
            f"{fmt([g['recall_macro'] for g in group])} |"
        )

    failed_count = sum(1 for r in runs if r.get("status") == "failed")
    lines.append("")
    lines.append(f"_Failed runs: {failed_count} / {len(runs)}_")
    lines.append("")

    with open(table_path, "w") as f:
        f.write("\n".join(lines))