# Design: Evaluate Domain-Adapted PhoBERT vs. Original PhoBERT

**Date:** 2026-06-25
**Status:** Approved
**Owner:** User

## 1. Problem Statement

The domain-adaptive PhoBERT base model (`models/phobert_domain_adapted/`) was successfully pretrained via continued MLM on 119,649 YouTube comments (final eval perplexity: 18.01). However, this base model has **not yet been evaluated** on the downstream classification task. The paper needs a direct comparison between:

- **Original PhoBERT** (`vinai/phobert-base`) fine-tuned on the final dataset
- **Domain-adapted PhoBERT** (continued MLM on YouTube text) fine-tuned on the final dataset

This comparison answers the question: *Does domain adaptation help on informal Vietnamese social-media text?*

## 2. Goals

- Fine-tune both models on the **same** final dataset with **identical** hyperparameters.
- Evaluate on **two** test sets: in-domain (`final_test.csv`) and cross-domain (`cross_domain_test.csv`, VSMEC).
- Run **3 seeds** per model (42, 123, 2024) for statistical significance.
- Produce outputs that can be inserted directly into the academic paper: JSON metrics, Markdown comparison table, per-prediction CSVs.

## 3. Non-Goals

- Re-pretraining the MLM (already done).
- Modifying `phobert_train.py` or `phobert_predict.py` (production stable; we wrap them).
- Training on test sets beyond `final_test` and VSMEC.
- Writing the paper's Discussion section (only output the table).

## 4. Architecture

```
scripts/evaluate_domain_adapted_phobert.py   (entry point)
        |
        v
scripts/domain_adapted_eval_utils.py        (helpers)
        |
        +-> phobert_train.train_phobert_first()  (existing, reused)
        +-> phobert_predict.predict_remaining_comments()  (existing, reused)
        +-> phobert_utils.compute_metrics()     (existing, reused)
```

The orchestrator loops over `(model_path, seed)` and calls helpers. Helpers wrap existing functions with the model path swapped. No changes to production code.

## 5. Components

### 5.1 Entry point: `scripts/evaluate_domain_adapted_phobert.py`

CLI:

```
python -m scripts.evaluate_domain_adapted_phobert \
    --models vinai/phobert-base models/phobert_domain_adapted \
    --seeds 42 123 2024 \
    --output-dir results/domain_adapted_eval_2026-06-25 \
    [--smoke]
```

Behavior:

1. For each `(model_path, seed)`:
   1. Fine-tune on `data/final_train.csv` / `data/final_val.csv`
   2. Evaluate on `data/final_test.csv` (in-domain)
   3. Evaluate on `data_unified/cross_domain_test.csv` (cross-domain)
   4. Save predictions and per-run metrics
2. Aggregate all runs into `metrics.json` and `comparison_table.md`.

### 5.2 Helpers: `scripts/domain_adapted_eval_utils.py`

| Function | Responsibility |
|----------|----------------|
| `run_finetune(model_path, seed, output_dir)` | Wraps `phobert_train.train_phobert_first()` with `PHOBERT_MODEL_NAME` overridden via env var or argument pass-through. Saves checkpoint to `output_dir/checkpoint/`. |
| `run_eval(checkpoint_dir, eval_csv, output_csv)` | Wraps `phobert_predict.predict_remaining_comments()` on the eval CSV. |
| `compute_cross_domain_metrics(pred_csv, gold_csv)` | Joins predictions with gold labels by row index, computes metrics via `phobert_utils.compute_metrics()`. |
| `aggregate_results(runs)` | Groups by `(model_tag, test_set)`, computes mean and std of metrics, writes `metrics.json` and `comparison_table.md`. |

### 5.3 Sample weighting

Both models use the `weight` column from `final_train.csv` as sample weights in `CrossEntropyLoss`. Sources and weights:

| source | weight |
|--------|--------|
| `human_gold` | 3 |
| `weak_high_conf` | 2 |
| `external_negative` | 2 |
| `phobert_v2_confident` | 1 |

`phobert_train.py` already accepts `sample_weights`; we pass the `weight` column directly.

### 5.4 Outputs

```
results/domain_adapted_eval_<timestamp>/
|-- metrics.json                  # per-(model, seed, test_set)
|-- comparison_table.md           # aggregated mean +/- std
|-- predictions/
|   |-- <tag>_seed42_final_test.csv
|   |-- <tag>_seed42_vsmec.csv
|   +-- ...
+-- checkpoints/                  # best checkpoint per run
    |-- <tag>_seed42/
    +-- ...
```

`metrics.json` schema:

```json
{
  "git_commit": "abc1234",
  "timestamp": "2026-06-25T10:30:00",
  "runs": [
    {
      "model_tag": "original",
      "seed": 42,
      "test_set": "final_test",
      "accuracy": 0.95,
      "f1_macro": 0.94,
      "f1_depression": 0.90,
      "precision_macro": 0.94,
      "recall_macro": 0.93,
      "confusion_matrix": [[480, 31], [15, 241]]
    }
  ]
}
```

`comparison_table.md` format:

```
| Model | Test set | F1 macro (mean+/-std) | F1 depression (mean+/-std) | Accuracy (mean+/-std) |
|-------|----------|-----------------------|----------------------------|-----------------------|
| PhoBERT (original) | final_test | 0.94 +/- 0.01 | 0.90 +/- 0.02 | 0.95 +/- 0.01 |
| PhoBERT (domain-adapted) | final_test | ... | ... | ... |
| PhoBERT (original) | vsmec | ... | ... | ... |
| PhoBERT (domain-adapted) | vsmec | ... | ... | ... |
```

## 6. Data Flow

| Stage | Input | Output |
|-------|-------|--------|
| Fine-tune | `data/final_train.csv` (3576 rows), `data/final_val.csv` (766 rows), `model_path` | `checkpoints/<tag>_seed<N>/best_model/` |
| In-domain eval | `checkpoints/...`, `data/final_test.csv` (767 rows) | `predictions/<tag>_seed<N>_final_test.csv`, metrics row |
| Cross-domain eval | `checkpoints/...`, `data_unified/cross_domain_test.csv` (3084 rows) | `predictions/<tag>_seed<N>_vsmec.csv`, metrics row |
| Aggregate | All metrics rows | `metrics.json`, `comparison_table.md` |

## 7. Error Handling

| Scenario | Handling |
|----------|----------|
| OOM (CUDA/MPS) | Catch `RuntimeError`, halve batch size, retry once. If still OOM, mark run as failed and continue. |
| Empty predictions | Log warning, skip run, mark as failed. |
| Missing checkpoint | Fail fast with clear message (training must have succeeded). |
| Tokenizer/model load failure | Fail fast before training. |
| Cross-domain label column missing | Validate schema before eval. |
| Disk full | Catch `IOError`, clean partial outputs, exit non-zero. |

Failed runs are recorded in `metrics.json` under `runs[]` with `"status": "failed"` and `error_message`. Aggregation skips failed runs but reports the count.

## 8. Testing

No external test framework. Built-in checks:

| Check | Where |
|-------|-------|
| Smoke mode: 1 epoch on 100 samples, full pipeline | `--smoke` flag |
| Sanity assertions: dataset shape, label distribution, weight >= 0 | At start of orchestrator |
| Output completeness: 2 models x 3 seeds x 2 test sets = 12 rows (or fewer if failures) | At end of aggregation |
| Comparison sanity: assert each (model, test_set) has >=1 successful seed | Before writing final table |

## 9. Reproducibility

- Reuse `phobert_utils.set_seed(seed)` for Python, numpy, torch.
- Log git commit hash (`git rev-parse HEAD`) into `metrics.json`.
- Pin torch/transformers via existing `requirements.txt`.
- Save tokenizer alongside each checkpoint.

## 10. Open Questions

None.

## 11. Implementation Plan

After approval, invoke `superpowers:writing-plans` to create the implementation plan.
