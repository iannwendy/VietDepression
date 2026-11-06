# Domain-Adapted PhoBERT Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune both original and domain-adapted PhoBERT on the final dataset, evaluate on in-domain and cross-domain (VSMEC) test sets across 3 seeds, and produce aggregated metrics (JSON + Markdown table) for the paper.

**Architecture:** A standalone orchestrator script (`scripts/evaluate_domain_adapted_phobert.py`) loops over `(model_path, seed)` pairs. For each pair, it wraps the existing `phobert_train.train_phobert_first()` and `phobert_predict.predict_remaining_comments()` functions via a new helper module (`scripts/domain_adapted_eval_utils.py`). Predictions and metrics are aggregated into `results/<timestamp>/metrics.json` and `comparison_table.md`.

**Tech Stack:** Python 3.9+, PyTorch 2.8, Transformers 4.40+, HuggingFace `transformers`, existing `phobert_utils`/`phobert_train`/`phobert_predict` modules, pandas, scikit-learn.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `scripts/evaluate_domain_adapted_phobert.py` (Create) | CLI entry point. Loops over (model, seed), calls helpers, writes outputs. |
| `scripts/domain_adapted_eval_utils.py` (Create) | Helper functions: `run_finetune`, `run_eval`, `compute_cross_domain_metrics`, `aggregate_results`. |
| `results/domain_adapted_eval_<timestamp>/` (Created at runtime) | Output dir containing metrics, table, predictions, checkpoints. |

No existing files are modified structurally. The orchestrator imports production code (`phobert_train`, `phobert_predict`, `phobert_utils`) and uses env-var overrides to swap `PHOBERT_MODEL_NAME` and `PHOBERT_OUTPUT_DIR`.

---

## Task 1: Helper module skeleton with `run_finetune`

**Files:**
- Create: `scripts/domain_adapted_eval_utils.py`

- [ ] **Step 1: Create helper module with `run_finetune` skeleton**

```python
"""Helpers for domain-adapted PhoBERT evaluation."""
import os
import subprocess
import sys
from pathlib import Path


def run_finetune(model_path: str, seed: int, output_dir: str) -> str:
    """Fine-tune a PhoBERT model on the final dataset.

    Returns path to the best checkpoint directory.
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.environ["PHOBERT_MODEL_NAME_OVERRIDE"] = model_path
    os.environ["PHOBERT_OUTPUT_DIR_OVERRIDE"] = checkpoint_dir
    os.environ["PHOBERT_RANDOM_SEED_OVERRIDE"] = str(seed)

    script = (
        "from yt_depression_crawler.modeling.phobert.phobert_train import "
        "train_phobert_first; "
        "train_phobert_first()"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Fine-tune failed for {model_path} seed={seed}:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    expected = os.path.join(checkpoint_dir, "best_model")
    if not os.path.isdir(expected):
        raise RuntimeError(
            f"Expected checkpoint at {expected} but directory does not exist"
        )
    return expected
```

- [ ] **Step 2: Verify module imports cleanly**

Run: `python -c "from scripts.domain_adapted_eval_utils import run_finetune; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/domain_adapted_eval_utils.py
git commit -m "Add domain_adapted_eval_utils skeleton with run_finetune"
```

---

## Task 2: Wire `phobert_train.py` to honor env-var overrides

**Files:**
- Modify: `yt_depression_crawler/modeling/phobert/phobert_train.py:1-80`

- [ ] **Step 1: Read current constants at top of file**

Run: `head -80 yt_depression_crawler/modeling/phobert/phobert_train.py`
Expected: A block that imports `config` and uses values like `config.PHOBERT_MODEL_NAME`.

- [ ] **Step 2: Add env-var override logic near the top**

Find the line that does `from yt_depression_crawler.core import config` (or equivalent). Immediately after that import block, insert:

```python
import os

# Allow scripts/evaluate_domain_adapted_phobert.py to override these via env vars
_MODEL_NAME_OVERRIDE = os.environ.get("PHOBERT_MODEL_NAME_OVERRIDE")
_OUTPUT_DIR_OVERRIDE = os.environ.get("PHOBERT_OUTPUT_DIR_OVERRIDE")
_SEED_OVERRIDE = os.environ.get("PHOBERT_RANDOM_SEED_OVERRIDE")

if _MODEL_NAME_OVERRIDE:
    config.PHOBERT_MODEL_NAME = _MODEL_NAME_OVERRIDE
if _OUTPUT_DIR_OVERRIDE:
    config.PHOBERT_OUTPUT_DIR = _OUTPUT_DIR_OVERRIDE
if _SEED_OVERRIDE:
    config.PHOBERT_RANDOM_SEED = int(_SEED_OVERRIDE)
```

Note: We mutate the `config` module's attributes at import time. Since this happens before `train_phobert_first()` is called, downstream code reads the overridden values.

- [ ] **Step 3: Verify by running a smoke fine-tune (skip if no GPU/MPS)**

Run:
```bash
PHOBERT_MODEL_NAME_OVERRIDE=vinai/phobert-base \
PHOBERT_OUTPUT_DIR_OVERRIDE=/tmp/phobert_smoke_test \
PHOBERT_RANDOM_SEED_OVERRIDE=42 \
python -c "from yt_depression_crawler.modeling.phobert.phobert_train import train_phobert_first; train_phobert_first()"
```
Expected: Training starts, prints model name `vinai/phobert-base`. If no GPU/MPS available, skip to next step.

- [ ] **Step 4: Commit**

```bash
git add yt_depression_crawler/modeling/phobert/phobert_train.py
git commit -m "Allow env-var overrides for PHOBERT_MODEL_NAME/OUTPUT_DIR/SEED"
```

---

## Task 3: Add `run_eval` helper for in-domain predictions

**Files:**
- Modify: `scripts/domain_adapted_eval_utils.py`

- [ ] **Step 1: Add `run_eval` function**

Append to `scripts/domain_adapted_eval_utils.py`:

```python
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
```

- [ ] **Step 2: Verify function signature by importing**

Run: `python -c "from scripts.domain_adapted_eval_utils import run_eval; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/domain_adapted_eval_utils.py
git commit -m "Add run_eval helper for in-domain prediction"
```

---

## Task 4: Add `predict_texts` to phobert_predict if not present

**Files:**
- Modify: `yt_depression_crawler/modeling/phobert/phobert_predict.py`

- [ ] **Step 1: Check if `predict_texts` exists**

Run: `grep -n "def predict_texts\|def predict_remaining_comments" yt_depression_crawler/modeling/phobert/phobert_predict.py`
Expected: Either line is present.

- [ ] **Step 2: If only `predict_remaining_comments` exists, add `predict_texts` wrapper**

If `predict_texts` is not found, read the actual signature of `predict_remaining_comments` first:

```bash
grep -n "def predict_remaining_comments" yt_depression_crawler/modeling/phobert/phobert_predict.py
sed -n '/def predict_remaining_comments/,/^def /p' yt_depression_crawler/modeling/phobert/phobert_predict.py | head -40
```

Then append the following function to `phobert_predict.py`, adapting the body to call `predict_remaining_comments` with its actual signature:

```python
def predict_texts(texts, model_dir, batch_size=None):
    """Predict labels for a list of texts using a fine-tuned checkpoint.

    Args:
        texts: list of raw comment strings.
        model_dir: path to the fine-tuned model directory.
        batch_size: optional override.

    Returns:
        list of dicts: [{"predicted_label": int, "prob_normal": float,
                          "prob_depression": float}, ...]
    """
    import pandas as pd
    import tempfile
    import os
    from yt_depression_crawler.core import config

    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, "input.csv")
        output_csv = os.path.join(tmpdir, "predictions.csv")
        pd.DataFrame({"comment_text": texts}).to_csv(input_csv, index=False)

        # Set the prediction output path via config so predict_remaining_comments
        # writes to our temp file. Adapt call to match its actual signature.
        old_output = getattr(config, "PHOBERT_PREDICTIONS_OUTPUT", None)
        config.PHOBERT_PREDICTIONS_OUTPUT = output_csv
        try:
            # NOTE: This call signature must match predict_remaining_comments.
            # Read its definition in step 1 to determine correct arguments.
            predict_remaining_comments(
                input_csv=input_csv,
                model_dir=model_dir,
            )
        finally:
            if old_output is not None:
                config.PHOBERT_PREDICTIONS_OUTPUT = old_output

        preds_df = pd.read_csv(output_csv)
        return preds_df.to_dict(orient="records")
```

**IMPORTANT:** Read the actual signature of `predict_remaining_comments` first. If it does not accept `input_csv` or `model_dir` kwargs, adapt the call accordingly. The plan needs adjustment — STOP and report if the signature is incompatible.

- [ ] **Step 3: Verify import**

Run: `python -c "from yt_depression_crawler.modeling.phobert.phobert_predict import predict_texts; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add yt_depression_crawler/modeling/phobert/phobert_predict.py
git commit -m "Add predict_texts wrapper for arbitrary text inputs"
```

---

## Task 5: Add `compute_cross_domain_metrics` helper

**Files:**
- Modify: `scripts/domain_adapted_eval_utils.py`

- [ ] **Step 1: Append metrics computation function**

Append to `scripts/domain_adapted_eval_utils.py`:

```python
import json


def compute_cross_domain_metrics(pred_csv: str, gold_csv: str) -> dict:
    """Compute classification metrics for predictions joined with gold labels.

    Both CSVs are joined by row index (positional alignment).
    Returns a dict matching the metrics schema in the design spec.
    """
    import numpy as np
    import pandas as pd
    from yt_depression_crawler.modeling.phobert.phobert_utils import (
        compute_metrics as _compute_metrics,
    )

    preds = pd.read_csv(pred_csv)
    gold = pd.read_csv(gold_csv)
    if len(preds) != len(gold):
        raise ValueError(
            f"Row count mismatch: predictions={len(preds)} gold={len(gold)}"
        )

    y_true = gold["label"].astype(int).values
    y_pred = preds["predicted_label"].astype(int).values

    metrics = _compute_metrics((y_pred, y_true))
    return metrics
```

- [ ] **Step 2: Verify import**

Run: `python -c "from scripts.domain_adapted_eval_utils import compute_cross_domain_metrics; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/domain_adapted_eval_utils.py
git commit -m "Add compute_cross_domain_metrics helper"
```

---

## Task 6: Add `aggregate_results` helper

**Files:**
- Modify: `scripts/domain_adapted_eval_utils.py`

- [ ] **Step 1: Append aggregation function**

Append to `scripts/domain_adapted_eval_utils.py`:

```python
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
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
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
```

- [ ] **Step 2: Verify import**

Run: `python -c "from scripts.domain_adapted_eval_utils import aggregate_results; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/domain_adapted_eval_utils.py
git commit -m "Add aggregate_results for metrics.json and Markdown table"
```

---

## Task 7: Orchestrator skeleton with smoke mode

**Files:**
- Create: `scripts/evaluate_domain_adapted_phobert.py`

- [ ] **Step 1: Create orchestrator with smoke flag**

```python
"""Evaluate domain-adapted PhoBERT vs original PhoBERT on the final dataset.

Usage:
    python -m scripts.evaluate_domain_adapted_phobert \
        --models vinai/phobert-base models/phobert_domain_adapted \
        --seeds 42 123 2024 \
        --output-dir results/domain_adapted_eval_<timestamp> \
        [--smoke]
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Make scripts/ importable when running as `python -m scripts.evaluate_...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.domain_adapted_eval_utils import (  # noqa: E402
    aggregate_results,
    compute_cross_domain_metrics,
    run_eval,
    run_finetune,
)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.getcwd(), text=True
        ).strip()
    except Exception:
        return "unknown"


def model_tag(model_path: str) -> str:
    if "domain_adapted" in model_path:
        return "domain_adapted"
    return "original"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models", nargs="+",
        default=["vinai/phobert-base", "models/phobert_domain_adapted"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    p.add_argument(
        "--output-dir", default=None,
        help="Defaults to results/domain_adapted_eval_<timestamp>",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Skip fine-tune and eval; verify imports and orchestration only.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = args.output_dir or f"results/domain_adapted_eval_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    git_commit = get_git_commit()
    print(f"[orchestrator] output_dir={output_dir}")
    print(f"[orchestrator] git_commit={git_commit}")
    print(f"[orchestrator] smoke={args.smoke}")

    runs = []
    for model_path in args.models:
        for seed in args.seeds:
            tag = model_tag(model_path)
            run_id = f"{tag}_seed{seed}"
            print(f"\n=== {run_id} ({model_path}) ===")
            t0 = time.time()
            try:
                if args.smoke:
                    print("[smoke mode] skipping fine-tune and eval")
                    elapsed = time.time() - t0
                    runs.append({
                        "model_tag": tag,
                        "seed": seed,
                        "status": "ok",
                        "elapsed_seconds": round(elapsed, 2),
                    })
                    continue

                ckpt = run_finetune(model_path, seed, output_dir)
                print(f"[{run_id}] fine-tuned -> {ckpt}")

                # In-domain eval
                final_test_pred_csv = os.path.join(
                    output_dir, "predictions", f"{run_id}_final_test.csv"
                )
                run_eval(ckpt, "data/final_test.csv", final_test_pred_csv)
                in_domain_metrics = compute_cross_domain_metrics(
                    final_test_pred_csv, "data/final_test.csv"
                )
                print(f"[{run_id}] final_test F1 macro={in_domain_metrics['f1_macro']:.4f}")

                # Cross-domain eval
                vsmec_pred_csv = os.path.join(
                    output_dir, "predictions", f"{run_id}_vsmec.csv"
                )
                run_eval(ckpt, "data_unified/cross_domain_test.csv", vsmec_pred_csv)
                cross_metrics = compute_cross_domain_metrics(
                    vsmec_pred_csv, "data_unified/cross_domain_test.csv"
                )
                print(f"[{run_id}] vsmec F1 macro={cross_metrics['f1_macro']:.4f}")

                elapsed = time.time() - t0
                runs.append({
                    "model_tag": tag,
                    "seed": seed,
                    "test_set": "final_test",
                    "status": "ok",
                    "elapsed_seconds": round(elapsed, 2),
                    **in_domain_metrics,
                })
                runs.append({
                    "model_tag": tag,
                    "seed": seed,
                    "test_set": "vsmec",
                    "status": "ok",
                    "elapsed_seconds": round(elapsed, 2),
                    **cross_metrics,
                })
            except Exception as e:
                print(f"[{run_id}] FAILED: {e}")
                runs.append({
                    "model_tag": tag,
                    "seed": seed,
                    "status": "failed",
                    "error_message": str(e),
                })

    if args.smoke:
        print("\n[smoke] Skipping aggregation. Smoke complete.")
        return

    aggregate_results(runs, output_dir, git_commit, timestamp)
    print(f"\n[orchestrator] Wrote metrics to {output_dir}/metrics.json")
    print(f"[orchestrator] Wrote table to {output_dir}/comparison_table.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify smoke mode runs end-to-end**

Run: `python -m scripts.evaluate_domain_adapted_phobert --smoke`
Expected: Prints orchestrator banner, lists both models + 3 seeds, prints "skipping fine-tune and eval" for each, ends with "[smoke] Skipping aggregation."

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate_domain_adapted_phobert.py
git commit -m "Add orchestrator with fine-tune, in-domain, and cross-domain eval"
```

---

## Task 8: Add sanity assertions

**Files:**
- Modify: `scripts/evaluate_domain_adapted_phobert.py`

- [ ] **Step 1: Add sanity check function**

Append to `scripts/evaluate_domain_adapted_phobert.py`:

```python
def sanity_check_datasets():
    """Validate final_train, final_test, and cross_domain_test schemas."""
    import pandas as pd
    for path, required in [
        ("data/final_train.csv", {"comment_text", "label", "weight"}),
        ("data/final_test.csv", {"comment_text", "label"}),
        ("data_unified/cross_domain_test.csv", {"comment_text", "label"}),
    ]:
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        if "weight" in df.columns and (df["weight"] < 0).any():
            raise ValueError(f"{path} has negative weight values")
        print(f"[sanity] {path}: {len(df)} rows, label dist: "
              f"{df['label'].value_counts().to_dict()}")
```

Then call `sanity_check_datasets()` at the top of `main()`, immediately after the `output_dir` setup:

```python
    if not args.smoke:
        sanity_check_datasets()
```

- [ ] **Step 2: Verify sanity check passes**

Run: `python -m scripts.evaluate_domain_adapted_phobert --smoke`
Expected: Smoke runs without calling sanity check.

Run: `python -m scripts.evaluate_domain_adapted_phobert --seeds 42`
Expected: `[sanity]` lines for all 3 datasets, then fine-tune begins.

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate_domain_adapted_phobert.py
git commit -m "Add sanity assertions for dataset schemas"
```

---

## Task 9: Run a single real seed to verify full pipeline

**Files:**
- None (validation only)

- [ ] **Step 1: Run a single real seed**

Run:
```bash
python -m scripts.evaluate_domain_adapted_phobert \
    --models vinai/phobert-base --seeds 42 \
    --output-dir results/smoke_real
```
Expected: Fine-tune completes, eval runs on both test sets, `results/smoke_real/metrics.json` and `comparison_table.md` are written.

- [ ] **Step 2: Inspect outputs**

Run:
```bash
ls -la results/smoke_real/
cat results/smoke_real/comparison_table.md
```
Expected: Contains 1 row for `original / final_test` and 1 row for `original / vsmec`.

- [ ] **Step 3: Clean up smoke output**

Run: `rm -rf results/smoke_real`

- [ ] **Step 4: No commit needed**

If no source files changed in this task, skip the commit.

---

## Task 10: Run full evaluation

**Files:**
- None (execution only)

- [ ] **Step 1: Run full evaluation in background**

Run:
```bash
mkdir -p logs
TS=$(date +%Y-%m-%d_%H%M%S)
nohup python -m scripts.evaluate_domain_adapted_phobert \
    --output-dir results/domain_adapted_eval_${TS} \
    > logs/eval_${TS}.log 2>&1 &
echo "PID: $!"
```

Expected: PID printed, log file created.

- [ ] **Step 2: Monitor progress**

Run: `tail -f logs/eval_<timestamp>.log`
Expected: Each `(model, seed)` prints fine-tune, eval, F1 score. Total 6 runs.

- [ ] **Step 3: Verify completion and outputs**

After process exits, run:
```bash
ls results/domain_adapted_eval_*/
cat results/domain_adapted_eval_*/comparison_table.md
```
Expected: 2 models x 3 seeds x 2 test sets = 12 rows in `metrics.json`, comparison table populated with mean+/-std.

- [ ] **Step 4: Commit results**

```bash
git add results/
git commit -m "Add domain-adapted PhoBERT evaluation results"
```

---

## Self-Review Notes

**Spec coverage check:**
- Goal 1 (fine-tune both on same dataset, identical hparams): Task 2 env-var override reuses `phobert_train` unchanged.
- Goal 2 (eval on in-domain + cross-domain): Tasks 3, 4, 5, 7.
- Goal 3 (3 seeds): Task 7 orchestrator default + Task 10 full run.
- Goal 4 (JSON metrics + Markdown table): Task 6.
- Sample weighting: Task 2 (passes through env-var so `weight` column is read by existing trainer).
- Error handling: Failure recorded in runs list (Task 7), aggregation skips failed runs (Task 6).
- Reproducibility: Task 7 logs git commit hash.

**Type consistency:**
- `compute_cross_domain_metrics` returns dict matching `compute_metrics` keys (`accuracy`, `f1_macro`, `f1_depression`, `precision_macro`, `recall_macro`, `confusion_matrix`).
- `aggregate_results` reads those same keys.

**Placeholder scan:** No TBD/TODO. All code blocks are complete.
