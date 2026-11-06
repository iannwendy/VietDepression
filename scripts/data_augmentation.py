"""Data Augmentation cho Vietnamese Depression Detection.

Multiple augmentation strategies:
1. PhoBERT Masked Language Model (contextual)
2. Synonym Replacement (Vietnamese word lists)
3. EDA (Easy Data Augmentation): insert, swap, delete
4. Random Character Noise (simulate typos)

Usage:
  .venv/bin/python scripts/data_augmentation.py [--mode all|synonym|eda|mlm]
  .venv/bin/python scripts/data_augmentation.py --depression-only
  .venv/bin/python scripts/data_augmentation.py --check-existing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DOCS_DIR = PROJECT_DIR / "docs"

# ── Vietnamese Synonym Lists ─────────────────────────────────────────
# Depression-related words with potential synonyms
VIETNAMESE_SYNONYMS: dict[str, list[str]] = {
    # Emotion words
    "buồn": ["chán", "sầu", "sai", "thất vọng", "đau lòng"],
    "cô đơn": ["một mình", "trống vắng", "heo hút", "lẻ loi"],
    "mệt mỏi": ["mệt", "chán nản", "kiệt sức", "uể oải"],
    "lo âu": ["lo lắng", "băn khoăn", "sợ hãi", "bất an"],
    "đau khổ": ["khổ sở", "đau lòng", "bi kịch"],
    "tuyệt vọng": ["mất hy vọng", "绝望"],
    "chán nản": ["chán", "mệt mỏi", "buồn chán"],

    # Mental state
    "stress": ["áp lực", "căng thẳng", " Strain"],
    "ám ảnh": ["quấn quanh", "day dứt", "mắc kẹt"],
    "sợ": ["e ngại", "kinh hoàng", "hoảng sợ"],
    "hoảng": ["sợ", "kinh hãi", "run sợ"],

    # Sleep/energy
    "mất ngủ": ["không ngủ được", "trằn trọc", "suy giấc ngủ"],
    "ngủ không được": ["mất ngủ", "trằn trọc"],

    # Social
    "bị bỏ rơi": ["cô đơn", "bị ruồng bỏ", "bị quên lãng"],
    "không ai hiểu": ["cô đơn", "lẻ loi", "không ai thông cảm"],

    # Positive words (for negative augmentation)
    "vui": ["hạnh phúc", "hân hoan", "mừng rỡ"],
    "hạnh phúc": ["vui vẻ", "sung sướng", "đủ đầy"],
    "tuyệt vời": ["tuyệt", "xuất sắc", "hoàn hảo"],
}

# Intensifiers
INTENSIFIERS = ["lắm", "quá", "vô", "kinh", "dữ", "lắm", "cực", "vô cùng"]

# Negation words
NEGATIONS = ["không", "chẳng", "đừng", "chớ", "chả", "chưa", "chẳng có"]

# ── Common Vietnamese punctuation/emoticons ───────────────────────────
EMOTICONS_POSITIVE = ["😊", "😄", "😁", "🙂", "❤️", "👍", "🎉", "😍", "🥰"]
EMOTICONS_NEGATIVE = ["😢", "😭", "😔", "😞", "💔", "😢", "🥺", "😩", "😰"]
EMOTICONS_NEUTRAL = ["😐", "🤔", "😅", "😂"]


class VietnameseAugmenter:
    """Vietnamese text augmentation without external APIs."""

    def __init__(self, seed: int = 42, phobert_model: Optional[str] = None):
        self.seed = seed
        random.seed(seed)

        self.phobert_model = None
        self.phobert_tokenizer = None
        self.device = None

        if phobert_model:
            self._load_phobert(phobert_model)

    def _load_phobert(self, model_path: str) -> None:
        """Load PhoBERT for masked language model augmentation."""
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            logging.info(f"Loading PhoBERT from {model_path}")
            self.phobert_tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False
            )
            self.phobert_model = AutoModelForMaskedLM.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.phobert_model.to(self.device)
            self.phobert_model.eval()
            logging.info(f"PhoBERT loaded on {self.device}")
        except Exception as e:
            logging.warning(f"Could not load PhoBERT: {e}")
            self.phobert_model = None

    # ── Method 1: Synonym Replacement ───────────────────────────────
    def synonym_replacement(self, text: str, n: int = 2) -> list[str]:
        """Replace up to n words with synonyms."""
        if n <= 0:
            return [text]

        words = text.split()
        if len(words) < 2:
            return [text]

        augmented = []
        replaced_indices = []

        for _ in range(n):
            new_words = words.copy()

            # Find words that have synonyms
            available = [
                i for i, w in enumerate(words)
                if w in VIETNAMESE_SYNONYMS and i not in replaced_indices
            ]

            if not available:
                break

            idx = random.choice(available)
            word = words[idx]
            synonyms = VIETNAMESE_SYNONYMS[word]

            # Avoid exact same word
            synonyms = [s for s in synonyms if s != word]
            if synonyms:
                replacement = random.choice(synonyms)
                new_words[idx] = replacement
                replaced_indices.append(idx)
                augmented.append(" ".join(new_words))

        return augmented

    # ── Method 2: Random Insertion ────────────────────────────────
    def random_insertion(self, text: str, n: int = 2) -> list[str]:
        """Randomly insert n intensifiers or filler words."""
        if n <= 0:
            return [text]

        words = text.split()
        if len(words) < 2:
            return [text]

        augmented = []
        for _ in range(n):
            new_words = words.copy()
            insert_word = random.choice(INTENSIFIERS)
            idx = random.randint(0, len(words) - 1)
            new_words.insert(idx, insert_word)
            augmented.append(" ".join(new_words))

        return augmented

    # ── Method 3: Random Swap ─────────────────────────────────────
    def random_swap(self, text: str, n: int = 1) -> list[str]:
        """Randomly swap n pairs of words."""
        if n <= 0:
            return [text]

        words = text.split()
        if len(words) < 2:
            return [text]

        augmented = []
        swapped_indices = set()

        for _ in range(n):
            new_words = words.copy()

            # Find pairs of indices that haven't been swapped
            available = [
                (i, j) for i in range(len(words))
                for j in range(i + 1, len(words))
                if i not in swapped_indices and j not in swapped_indices
            ]

            if not available:
                break

            i, j = random.choice(available)
            new_words[i], new_words[j] = new_words[j], new_words[i]
            swapped_indices.add(i)
            swapped_indices.add(j)
            augmented.append(" ".join(new_words))

        return augmented

    # ── Method 4: Random Deletion ─────────────────────────────────
    def random_deletion(self, text: str, p: float = 0.1) -> list[str]:
        """Randomly delete words with probability p."""
        if p <= 0:
            return [text]

        words = text.split()
        if len(words) < 2:
            return [text]

        new_words = [w for w in words if random.random() > p]

        # Ensure we don't delete everything
        if not new_words:
            return [text]

        return [" ".join(new_words)]

    # ── Method 5: Negation Insertion ──────────────────────────────
    def negation_insertion(self, text: str) -> list[str]:
        """Insert negations to flip sentiment (for data balancing)."""
        words = text.split()
        if len(words) < 2:
            return [text]

        augmented = []

        # Insert negation before first emotional word
        neg = random.choice(NEGATIONS)
        for i, word in enumerate(words):
            if word in VIETNAMESE_SYNONYMS:
                new_words = words.copy()
                new_words.insert(i, neg)
                augmented.append(" ".join(new_words))
                break

        return augmented

    # ── Method 6: PhoBERT MLM ────────────────────────────────────
    def phobert_mlm_augment(self, text: str, n: int = 2) -> list[str]:
        """Use PhoBERT masked language model to generate augmentations."""
        if self.phobert_model is None:
            return []

        try:
            import torch

            # Simple masking of random word
            words = text.split()
            if len(words) < 2:
                return []

            augmented = []
            indices_to_mask = random.sample(range(len(words)), min(n, len(words)))

            for mask_idx in indices_to_mask:
                masked_text = words.copy()
                original_word = masked_text[mask_idx]
                masked_text[mask_idx] = "[MASK]"

                # Tokenize
                input_text = " ".join(masked_text)
                inputs = self.phobert_tokenizer(
                    input_text, return_tensors="pt", truncation=True, max_length=128
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Find mask token position
                mask_token_id = self.phobert_tokenizer.mask_token_id
                mask_pos = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)

                if len(mask_pos[1]) == 0:
                    continue

                with torch.no_grad():
                    outputs = self.phobert_model(**inputs)
                    logits = outputs.logits

                # Get top predictions
                mask_logits = logits[0, mask_pos[1][0]]
                top_indices = torch.topk(mask_logits, k=5).indices

                for idx in top_indices[:2]:
                    new_word = self.phobert_tokenizer.decode([idx]).strip()
                    if new_word and new_word != original_word:
                        new_words = words.copy()
                        new_words[mask_idx] = new_word
                        augmented.append(" ".join(new_words))

            return augmented[:n]

        except Exception as e:
            logging.debug(f"MLM augmentation failed: {e}")
            return []

    # ── Method 7: Emoticon Augmentation ───────────────────────────
    def emoticon_augment(self, text: str, sentiment: str = "neutral") -> list[str]:
        """Add emoticons to text."""
        emoticons = {
            "positive": EMOTICONS_POSITIVE,
            "negative": EMOTICONS_NEGATIVE,
            "neutral": EMOTICONS_NEUTRAL,
        }
        options = emoticons.get(sentiment, EMOTICONS_NEUTRAL)

        augmented = []
        for emoji in random.sample(options, min(2, len(options))):
            augmented.append(f"{text} {emoji}")

        return augmented

    # ── Combined Augmentation ───────────────────────────────────────
    def augment(
        self,
        text: str,
        label: int,
        n_per_method: int = 1,
    ) -> list[tuple[str, int]]:
        """Generate augmentations for a text sample."""
        results = []
        seen = {text}  # Avoid duplicates

        def add_unique(text_aug: str) -> None:
            if text_aug and text_aug not in seen:
                seen.add(text_aug)
                results.append((text_aug, label))

        # Apply each method
        # 1. Synonym replacement
        for aug in self.synonym_replacement(text, n=n_per_method):
            add_unique(aug)

        # 2. Random insertion
        for aug in self.random_insertion(text, n=n_per_method):
            add_unique(aug)

        # 3. Random swap
        for aug in self.random_swap(text, n=n_per_method):
            add_unique(aug)

        # 4. Random deletion
        for aug in self.random_deletion(text, p=0.1):
            add_unique(aug)

        # 5. PhoBERT MLM (if available)
        if self.phobert_model:
            for aug in self.phobert_mlm_augment(text, n=n_per_method):
                add_unique(aug)

        return results


def load_data(
    input_file: Path,
    depression_only: bool = False,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load training data."""
    df = pd.read_csv(input_file, dtype=str).fillna("")
    df = df[df["comment_text"].str.strip().ne("")].copy()

    if depression_only:
        df = df[df["label"].astype(int) == 1].copy()
        logging.info(f"Filtered to depression samples: {len(df)}")

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logging.info(f"Sampled {sample_size} rows")

    return df


def generate_augmented_data(
    df: pd.DataFrame,
    n_augmentations: int = 3,
    phobert_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate augmented data."""
    augmenter = VietnameseAugmenter(seed=42, phobert_model=phobert_path)

    all_samples = []
    pbar = tqdm(df.iterrows(), total=len(df), desc="Augmenting")

    for _, row in pbar:
        text = str(row["comment_text"])
        label = int(row["label"])

        # Add original
        all_samples.append({"comment_text": text, "label": label, "augmented": False})

        # Generate augmentations
        augmentations = augmenter.augment(text, label, n_per_method=1)

        # Limit total augmentations per sample
        for aug_text, aug_label in augmentations[:n_augmentations]:
            if aug_text != text:  # Don't duplicate original
                all_samples.append({
                    "comment_text": aug_text,
                    "label": aug_label,
                    "augmented": True,
                })

        pbar.set_postfix({"total": len(all_samples)})

    result_df = pd.DataFrame(all_samples)
    logging.info(f"Generated {len(result_df)} samples from {len(df)} originals")
    logging.info(f"  - Original: {sum(~result_df['augmented'])}")
    logging.info(f"  - Augmented: {sum(result_df['augmented'])}")

    return result_df


def merge_with_train(
    augmented_df: pd.DataFrame,
    train_file: Path,
    output_file: Path,
    keep_duplicates: bool = False,
) -> dict:
    """Merge augmented data with existing training data."""
    train_df = pd.read_csv(train_file, dtype=str).fillna("")
    original_count = len(train_df)

    # Normalize for deduplication
    augmented_df["text_normalized"] = augmented_df["comment_text"].str.strip().str.lower()
    train_df["text_normalized"] = train_df["comment_text"].str.strip().str.lower()

    # Remove duplicates if requested
    if not keep_duplicates:
        existing_texts = set(train_df["text_normalized"])
        augmented_df = augmented_df[~augmented_df["text_normalized"].isin(existing_texts)].copy()

    # Merge
    augmented_to_add = augmented_df[["comment_text", "label", "augmented"]]
    if "weight" in train_df.columns:
        augmented_to_add["weight"] = 1.0  # Augmented samples get weight 1
    if "source" in train_df.columns:
        augmented_to_add["source"] = "augmented"

    merged_df = pd.concat([train_df, augmented_to_add], ignore_index=True)
    merged_df = merged_df.drop(columns=["text_normalized"], errors="ignore")

    # Save
    merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    stats = {
        "original_train_rows": original_count,
        "augmented_rows_added": len(augmented_df),
        "final_train_rows": len(merged_df),
        "duplicates_removed": len(augmented_df) - (len(merged_df) - original_count),
        "output_file": str(output_file),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Vietnamese text augmentation")
    parser.add_argument(
        "--input", type=str,
        default=str(DATA_DIR / "final_train.csv"),
        help="Input CSV file"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(DATA_DIR / "final_train_augmented.csv"),
        help="Output augmented CSV"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge with existing train data"
    )
    parser.add_argument(
        "--train-file", type=str,
        default=str(DATA_DIR / "final_train.csv"),
        help="Train file to merge with"
    )
    parser.add_argument(
        "--phobert", type=str,
        default=None,
        help="PhoBERT model path for MLM augmentation"
    )
    parser.add_argument(
        "--depression-only", action="store_true",
        help="Only augment depression samples (label=1)"
    )
    parser.add_argument(
        "--n-augment", type=int, default=3,
        help="Max augmentations per sample"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Sample size for augmentation (for testing)"
    )
    parser.add_argument(
        "--keep-duplicates", action="store_true",
        help="Keep duplicate texts from augmentation"
    )
    parser.add_argument(
        "--check-existing", action="store_true",
        help="Check existing augmented files"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # ── Check existing ───────────────────────────────────────────────
    if args.check_existing:
        aug_files = list(DATA_DIR.glob("*augment*"))
        print(f"Existing augmentation files: {len(aug_files)}")
        for f in aug_files:
            print(f"  - {f.name}")
        return

    # ── Load data ────────────────────────────────────────────────────
    logging.info(f"Loading data from {args.input}")
    df = load_data(
        Path(args.input),
        depression_only=args.depression_only,
        sample_size=args.sample_size,
    )

    # ── Generate augmentations ───────────────────────────────────────
    logging.info("Generating augmentations...")
    augmented_df = generate_augmented_data(
        df,
        n_augmentations=args.n_augment,
        phobert_path=args.phobert,
    )

    # ── Save augmented data ─────────────────────────────────────────
    augmented_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logging.info(f"Saved augmented data: {args.output}")

    # ── Merge with train ─────────────────────────────────────────────
    if args.merge:
        logging.info("Merging with existing train data...")
        stats = merge_with_train(
            augmented_df,
            Path(args.train_file),
            Path(args.output),
            keep_duplicates=args.keep_duplicates,
        )

        print(f"\n{'='*60}")
        print(f"AUGMENTATION COMPLETE")
        print(f"{'='*60}")
        print(f"Original train: {stats['original_train_rows']:,} rows")
        print(f"Augmented added: {stats['augmented_rows_added']:,} rows")
        print(f"Duplicates removed: {stats['duplicates_removed']:,}")
        print(f"Final train: {stats['final_train_rows']:,} rows")
        print(f"Output: {stats['output_file']}")

        # Label distribution
        final_df = pd.read_csv(stats['output_file'])
        print(f"\nLabel distribution:")
        print(final_df['label'].value_counts())

        # Source distribution if available
        if 'source' in final_df.columns:
            print(f"\nSource distribution:")
            print(final_df['source'].value_counts())

    else:
        print(f"\n{'='*60}")
        print(f"AUGMENTATION COMPLETE")
        print(f"{'='*60}")
        print(f"Generated: {len(augmented_df):,} samples")
        print(f"  - Original: {sum(~augmented_df['augmented']):,}")
        print(f"  - Augmented: {sum(augmented_df['augmented']):,}")
        print(f"Output: {args.output}")

        print(f"\nLabel distribution:")
        print(augmented_df['label'].value_counts())


if __name__ == "__main__":
    main()
