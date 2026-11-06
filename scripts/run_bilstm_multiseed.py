"""A3: Run BiLSTM with 2 additional seeds for multi-seed evaluation.

The single-seed BiLSTM runs (random + phobert-frozen) already exist in
models/bilstm/{random,phobert}/ with seed=42. This script adds seeds 123
and 2024, then aggregates all 3 seeds (42, 123, 2024) into mean +/- std
for both variants, matching the statistical-rigor pattern used for the
DAPT counter-experiment.

Output:
- models/bilstm/random/bilstm_metrics_seed123.json
- models/bilstm/random/bilstm_metrics_seed2024.json
- models/bilstm/phobert/bilstm_metrics_seed123.json
- models/bilstm/phobert/bilstm_metrics_seed2024.json
- models/bilstm/multiseed_summary.json (aggregated mean +/- std)

Usage:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    python3 scripts/run_bilstm_multiseed.py [--seeds 123 2024]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from yt_depression_crawler.modeling.bilstm.bilstm_model import train_bilstm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("bilstm_multiseed")


VARIANTS = ["random", "phobert"]
TRAIN_FILE = PROJECT_DIR / "data" / "final_train.csv"
VAL_FILE = PROJECT_DIR / "data" / "final_val.csv"
TEST_FILE = PROJECT_DIR / "data" / "final_test.csv"
PHOBERT_LOCAL_DIR = str(PROJECT_DIR / "models" / "phobert_base_local")


def run_variant(variant: str, seeds: list[int], epochs: int = 8) -> dict:
    """Train BiLSTM-<variant> for each seed, return aggregated metrics."""
    base_outdir = PROJECT_DIR / "models" / "bilstm" / variant
    base_outdir.mkdir(parents=True, exist_ok=True)

    seed_metrics: dict[int, dict] = {}
    for seed in seeds:
        # seed=42 uses the canonical naming (bilstm_metrics.json); extra
        # seeds use bilstm_metrics_seed{seed}.json to avoid clobbering.
        suffix = "" if seed == 42 else f"_seed{seed}"
        outdir = base_outdir / f"seed{seed}" if seed != 42 else base_outdir
        outdir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("BiLSTM-%s | seed=%d | output=%s", variant, seed, outdir)
        logger.info("=" * 60)

        metrics = train_bilstm(
            variant=variant,
            train_file=TRAIN_FILE,
            val_file=VAL_FILE,
            test_file=TEST_FILE,
            output_dir=outdir,
            epochs=epochs,
            phobert_local_dir=PHOBERT_LOCAL_DIR,
            seed=seed,
        )
        seed_metrics[seed] = metrics
        logger.info("seed=%d test_f1=%.4f vsmec_f1=%.4f",
                    seed, metrics["test"]["f1_macro"],
                    metrics["cross_domain_vsmec"]["f1_macro"])

    return seed_metrics


def aggregate(variant: str, seed_metrics: dict[int, dict]) -> dict:
    """Aggregate per-seed metrics into mean +/- std."""
    seeds = sorted(seed_metrics.keys())

    def stats(key_path: str, split: str) -> dict:
        vals = [seed_metrics[s][split][key_path] for s in seeds]
        return {
            "seeds": seeds,
            "per_seed": vals,
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    return {
        "variant": variant,
        "n_seeds": len(seeds),
        "test": {
            "f1_macro": stats("f1_macro", "test"),
            "accuracy": stats("accuracy", "test"),
            "f1_depression": stats("f1_depression", "test"),
            "precision_macro": stats("precision_macro", "test"),
            "recall_macro": stats("recall_macro", "test"),
        },
        "cross_domain_vsmec": {
            "f1_macro": stats("f1_macro", "cross_domain_vsmec"),
            "accuracy": stats("accuracy", "cross_domain_vsmec"),
            "f1_depression": stats("f1_depression", "cross_domain_vsmec"),
            "precision_macro": stats("precision_macro", "cross_domain_vsmec"),
            "recall_macro": stats("recall_macro", "cross_domain_vsmec"),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="BiLSTM multi-seed sweep")
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 2024],
                   help="Additional seeds to run (seed=42 is already done)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--variants", choices=VARIANTS, nargs="+", default=VARIANTS,
                   help="Which variants to run")
    args = p.parse_args()

    # For aggregation, include the existing seed=42 run + new seeds.
    existing_seed = 42
    all_seeds = sorted(set([existing_seed] + args.seeds))

    summary = {
        "timestamp": __import__("pandas").Timestamp.now().isoformat(),
        "seeds": all_seeds,
        "epochs": args.epochs,
        "variants": {},
    }

    for variant in args.variants:
        # Load existing seed=42 metrics from canonical path
        existing_metrics_path = PROJECT_DIR / "models" / "bilstm" / variant / "bilstm_metrics.json"
        seed_metrics: dict[int, dict] = {}
        if existing_metrics_path.exists():
            with open(existing_metrics_path) as f:
                m = json.load(f)
            seed_metrics[existing_seed] = m
            logger.info("Loaded existing seed=%d metrics from %s",
                        existing_seed, existing_metrics_path)
        else:
            logger.warning("No existing seed=%d metrics at %s — running all seeds",
                           existing_seed, existing_metrics_path)

        # Run new seeds
        new_seed_metrics = run_variant(variant, args.seeds, epochs=args.epochs)
        seed_metrics.update(new_seed_metrics)

        agg = aggregate(variant, seed_metrics)
        summary["variants"][variant] = agg

        logger.info("=" * 60)
        logger.info("BiLSTM-%s multi-seed summary (n=%d seeds: %s)",
                    variant, len(seed_metrics), list(seed_metrics.keys()))
        logger.info("  In-domain    F1-macro: %.4f +/- %.4f",
                    agg["test"]["f1_macro"]["mean"], agg["test"]["f1_macro"]["std"])
        logger.info("  Cross-domain F1-macro: %.4f +/- %.4f",
                    agg["cross_domain_vsmec"]["f1_macro"]["mean"],
                    agg["cross_domain_vsmec"]["f1_macro"]["std"])
        logger.info("=" * 60)

    out_path = PROJECT_DIR / "models" / "bilstm" / "multiseed_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Aggregated summary: %s", out_path)

    # Print final summary table
    print()
    print("=" * 80)
    print("BiLSTM MULTI-SEED SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<25s} {'In-domain F1':>14s} {'Cross-domain F1':>17s} {'F1-dep (cross)':>15s}")
    print("-" * 80)
    for variant, agg in summary["variants"].items():
        in_f1 = f"{agg['test']['f1_macro']['mean']:.4f} +/- {agg['test']['f1_macro']['std']:.4f}"
        cross_f1 = f"{agg['cross_domain_vsmec']['f1_macro']['mean']:.4f} +/- {agg['cross_domain_vsmec']['f1_macro']['std']:.4f}"
        cross_dep = f"{agg['cross_domain_vsmec']['f1_depression']['mean']:.4f} +/- {agg['cross_domain_vsmec']['f1_depression']['std']:.4f}"
        print(f"BiLSTM-{variant:<18s} {in_f1:>14s} {cross_f1:>17s} {cross_dep:>15s}")
    print("=" * 80)


if __name__ == "__main__":
    main()