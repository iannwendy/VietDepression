"""CLI entry point for BiLSTM training.

Usage:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python3 scripts/run_bilstm.py --variant random --output-dir models/bilstm/random
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python3 scripts/run_bilstm.py --variant phobert --output-dir models/bilstm/phobert

The variant "random" uses learned embeddings (paper Section 4.2 spec);
"phobert" initializes the embedding layer from vinai/phobert-base and
freezes it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from yt_depression_crawler.modeling.bilstm.bilstm_model import train_bilstm


def main() -> None:
    p = argparse.ArgumentParser(description="Train BiLSTM on final_dataset")
    p.add_argument("--variant", choices=["random", "phobert"], default="random",
                   help="Embedding variant: random (paper spec) or phobert-frozen")
    p.add_argument("--train-file", default="data/final_train.csv")
    p.add_argument("--val-file", default="data/final_val.csv")
    p.add_argument("--test-file", default="data/final_test.csv")
    p.add_argument("--output-dir", default="models/bilstm/random")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-len", type=int, default=100)
    p.add_argument("--vocab-size", type=int, default=15_000)
    p.add_argument("--phobert-local-dir", default="models/phobert_base_local",
                   help="Local PhoBERT directory (skips HuggingFace download)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    metrics = train_bilstm(
        variant=args.variant,
        train_file=Path(args.train_file),
        val_file=Path(args.val_file),
        test_file=Path(args.test_file),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        vocab_size=args.vocab_size,
        phobert_local_dir=args.phobert_local_dir,
    )
    print(json.dumps({
        "variant": metrics["variant"],
        "train_rows": metrics["train_rows"],
        "best_epoch": metrics["best_epoch"],
        "test_f1_macro": metrics["test"]["f1_macro"],
        "test_accuracy": metrics["test"]["accuracy"],
        "test_f1_depression": metrics["test"]["f1_depression"],
        "vsmec_f1_macro": metrics["cross_domain_vsmec"]["f1_macro"],
        "vsmec_accuracy": metrics["cross_domain_vsmec"]["accuracy"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()