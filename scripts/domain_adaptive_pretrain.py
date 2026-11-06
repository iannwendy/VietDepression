"""Domain-adaptive pretraining — continue PhoBERT MLM on YouTube informal Vietnamese.

Rationale: PhoBERT was pretrained on formal Vietnamese text (news, Wikipedia).
YouTube comments have a fundamentally different register — no diacritics, teen
slang, abbreviations ("ko", "dc", "j", "vkl"), code-switching with English.
This script continues masked language modeling on 125K YouTube comments so the
model adapts its tokenizer statistics and internal representations to the
target domain before fine-tuning for depression classification.

Expected impact: reduce the generalization gap (0.53 F1) between in-domain and
cross-domain evaluation by making the model less reliant on formal-text cues
that are absent in social media.

Usage:
  .venv/bin/python scripts/domain_adaptive_pretrain.py                # full corpus
  .venv/bin/python scripts/domain_adaptive_pretrain.py --max-steps 5000  # quick test
  .venv/bin/python scripts/domain_adaptive_pretrain.py --epochs 5      # more epochs
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

# Add project root
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from sklearn.model_selection import train_test_split

from yt_depression_crawler.core.config import ensure_directories
from yt_depression_crawler.modeling.phobert.phobert_utils import set_seed

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME = "vinai/phobert-base"          # Starting checkpoint
CORPUS_FILE = PROJECT_DIR / "data" / "cleaned_comments.csv"  # 125K YouTube comments
OUTPUT_DIR = PROJECT_DIR / "models" / "phobert_domain_adapted"
METRICS_FILE = OUTPUT_DIR / "pretrain_metrics.json"

# Hyperparameters — conservative for overnight run on single GPU
MLM_PROBABILITY = 0.15                    # Standard BERT masking rate
MAX_LENGTH = 128                          # Truncate to this
BATCH_SIZE = 32                           # Per GPU (adjust for your VRAM)
EPOCHS = 2                                # 2 epochs on 125K — enough for domain warmup
LEARNING_RATE = 5e-5                      # Lower than fine-tune — this is continued pretraining
WARMUP_STEPS = 500
GRADIENT_ACCUMULATION = 2                 # Effective batch = BATCH_SIZE * 2 = 64
LOGGING_STEPS = 500
SAVE_STEPS = 1000                        # Save checkpoint every 1000 steps
SEED = 42
MAX_STEPS: int | None = None              # Override via CLI for quick test
VAL_SPLIT = 0.02                          # 2% for eval perplexity

logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────
class MLMDataset(Dataset):
    """Wraps tokenized texts for MLM."""
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Domain-adaptive PhoBERT MLM pretraining")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit steps (quick test)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--corpus", type=str, default=str(CORPUS_FILE))
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample N texts (quick run)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    ensure_directories()
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("Output directory: %s", OUTPUT_DIR)

    # ── 1. Load texts ──────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(args.corpus, dtype=str).fillna("")
    texts = df["comment_text"].str.strip().tolist()
    texts = [t for t in texts if len(t) >= 10]  # Filter very short
    if args.sample and args.sample < len(texts):
        import random
        random.seed(SEED)
        texts = random.sample(texts, args.sample)
    logger.info("Loaded %d texts (min 10 chars)", len(texts))

    # ── 2. Word segmentation (same as PhoBERT expects) ─────────────────
    try:
        from underthesea import word_tokenize
        from tqdm import tqdm as tqdm_seg
        logger.info("Running word segmentation on %d texts (this may take 30-60 min)...", len(texts))
        texts = [word_tokenize(t, format="text") for t in tqdm_seg(texts, desc="Segmentation")]
        logger.info("Segmentation complete")
    except Exception:
        logger.warning("underthesea not available, using raw texts")

    # ── 3. Train/val split ─────────────────────────────────────────────
    train_texts, val_texts = train_test_split(texts, test_size=VAL_SPLIT, random_state=SEED)
    logger.info("Train: %d, Val: %d", len(train_texts), len(val_texts))

    # ── 4. Tokenize ────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    logger.info("Tokenizing training set...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_special_tokens_mask=True,
    )
    logger.info("Tokenizing validation set...")
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_special_tokens_mask=True,
    )

    train_dataset = MLMDataset(train_encodings)
    val_dataset = MLMDataset(val_encodings)

    # ── 5. Data collator ───────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    # ── 6. Load model ──────────────────────────────────────────────────
    logger.info("Loading model: %s", MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    # ── 7. Training args ───────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=args.lr,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
    )

    # ── 8. Train ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting domain-adaptive pretraining...")
    logger.info("  Texts: %d train / %d val", len(train_texts), len(val_texts))
    logger.info("  Epochs: %d", args.epochs)
    logger.info("  Batch size: %d x %d accumulation = %d effective",
                args.batch_size, GRADIENT_ACCUMULATION, args.batch_size * GRADIENT_ACCUMULATION)
    logger.info("  Max steps: %s", args.max_steps or "full epochs")
    logger.info("  Steps per epoch: ~%d", len(train_dataset) // (args.batch_size * GRADIENT_ACCUMULATION))

    train_result = trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # ── 9. Final eval ──────────────────────────────────────────────────
    eval_result = trainer.evaluate()
    perplexity = math.exp(eval_result["eval_loss"]) if "eval_loss" in eval_result else float("nan")

    metrics = {
        "model_name": MODEL_NAME,
        "output_dir": str(OUTPUT_DIR),
        "corpus": str(args.corpus),
        "train_texts": len(train_texts),
        "val_texts": len(val_texts),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch_size": args.batch_size * GRADIENT_ACCUMULATION,
        "learning_rate": args.lr,
        "max_length": MAX_LENGTH,
        "mlm_probability": MLM_PROBABILITY,
        "final_train_loss": float(train_result.training_loss),
        "final_eval_loss": eval_result.get("eval_loss", float("nan")),
        "eval_perplexity": round(perplexity, 2),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("DOMAIN-ADAPTIVE PRETRAINING COMPLETE")
    logger.info("  Train loss: %.4f", metrics["final_train_loss"])
    logger.info("  Eval loss:  %.4f", metrics["final_eval_loss"])
    logger.info("  Perplexity: %.2f", metrics["eval_perplexity"])
    logger.info("  Model saved to: %s", OUTPUT_DIR)
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
