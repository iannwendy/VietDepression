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


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.getcwd(), text=True
        ).strip()
    except Exception:
        return "unknown"


def model_tag(model_path):
    if "domain_adapted" in model_path:
        return "domain_adapted"
    return "original"


def sanity_check_datasets():
    """Validate final_train, final_val, final_test, and cross_domain_test schemas.

    All splits in `data/labeled/final_*.csv` are produced by Phase 2 of the pipeline
    (after the round-3 review merge). The orchestrator must train on these,
    not the legacy `data/train.csv` from the original weak-label split.
    """
    import pandas as pd
    for path, required in [
        ("data/labeled/final_train.csv", {"comment_text", "label", "weight"}),
        ("data/labeled/final_val.csv", {"comment_text", "label"}),
        ("data/labeled/final_test.csv", {"comment_text", "label"}),
        ("data_unified/cross_domain_test.csv", {"comment_text", "label"}),
    ]:
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        if "weight" in df.columns and (df["weight"] < 0).any():
            raise ValueError(f"{path} has negative weight values")
        weight_str = (
            f", weight dist: {df['weight'].value_counts().to_dict()}"
            if "weight" in df.columns else ""
        )
        print(f"[sanity] {path}: {len(df)} rows, "
              f"label dist: {df['label'].value_counts().to_dict()}{weight_str}")


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

    if not args.smoke:
        sanity_check_datasets()

    runs = []
    for model_path in args.models:
        for seed in args.seeds:
            tag = model_tag(model_path)
            run_id = f"{tag}_seed{seed}"
            per_run_dir = os.path.join(output_dir, f"{run_id}")
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

                ckpt = run_finetune(model_path, seed, per_run_dir)
                print(f"[{run_id}] fine-tuned -> {ckpt}")

                # In-domain eval
                final_test_pred_csv = os.path.join(
                    output_dir, "predictions", f"{run_id}_final_test.csv"
                )
                run_eval(ckpt, "data/labeled/final_test.csv", final_test_pred_csv)
                in_domain_metrics = compute_cross_domain_metrics(
                    final_test_pred_csv, "data/labeled/final_test.csv"
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
