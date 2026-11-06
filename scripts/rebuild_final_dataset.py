"""Rebuild final dataset from merged gold set and weak labels.

Usage:
  PYTHONPATH="$PWD" .venv/bin/python scripts/rebuild_final_dataset.py
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

def main():
    print("=" * 60)
    print("REBUILDING FINAL DATASET FROM GOLD + WEAK LABELS")
    print("=" * 60)

    # Load merged gold set (Round 4 merged)
    gold_df = pd.read_csv(DATA_DIR / "train_gold.csv", dtype=str).fillna("")
    gold_df['label'] = gold_df['label'].astype(int)
    gold_df['source'] = 'human_gold'
    gold_df['weight'] = 3
    print(f"\nGold set: {len(gold_df)} samples")
    print(f"  - Normal (0): {sum(gold_df['label']==0)}")
    print(f"  - Depression (1): {sum(gold_df['label']==1)}")

    # Load auto-labeled comments
    auto_df = pd.read_csv(DATA_DIR / "auto_labeled_comments.csv", dtype=str).fillna("")
    print(f"\nAuto-labeled: {len(auto_df)} samples")

    # Filter weak_high_conf based on confidence
    weak_high = auto_df[auto_df['confidence'] == 'high'].copy()
    weak_high['source'] = 'weak_high_conf'
    weak_high['weight'] = 2
    print(f"Weak high conf: {len(weak_high)} samples")

    # Combine gold + weak
    final_df = pd.concat([gold_df, weak_high], ignore_index=True)
    print(f"\nCombined: {len(final_df)} samples")

    # Remove duplicates
    final_df['text_norm'] = final_df['comment_text'].str.lower().str.strip()
    final_df = final_df.drop_duplicates(subset=['text_norm'], keep='first')
    final_df = final_df.drop(columns=['text_norm'])
    print(f"After dedup: {len(final_df)} samples")

    # Clean label column - handle empty/invalid values
    final_df['label'] = pd.to_numeric(final_df['label'], errors='coerce')
    final_df = final_df.dropna(subset=['label'])  # Remove rows with invalid labels
    final_df['label'] = final_df['label'].astype(int)
    print(f"After label cleaning: {len(final_df)} samples")

    # Label distribution
    print(f"\nFinal distribution:")
    print(f"  - Normal (0): {sum(final_df['label']==0)}")
    print(f"  - Depression (1): {sum(final_df['label']==1)}")
    print(f"  - Ratio: 1:{sum(final_df['label']==0)/sum(final_df['label']==1):.1f}")

    # Stratified split (70/15/15)
    train, temp = train_test_split(
        final_df, test_size=0.30, random_state=42, stratify=final_df['label']
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=42, stratify=temp['label']
    )

    # Save
    train.to_csv(DATA_DIR / "final_train.csv", index=False)
    val.to_csv(DATA_DIR / "final_val.csv", index=False)
    test.to_csv(DATA_DIR / "final_test.csv", index=False)

    print(f"\nSaved splits:")
    print(f"  - Train: {len(train)} ({sum(train['label']==0)} normal, {sum(train['label']==1)} depression)")
    print(f"  - Val: {len(val)} ({sum(val['label']==0)} normal, {sum(val['label']==1)} depression)")
    print(f"  - Test: {len(test)} ({sum(test['label']==0)} normal, {sum(test['label']==1)} depression)")

    # Save full dataset
    final_df.to_csv(DATA_DIR / "final_dataset.csv", index=False)
    print(f"\nFull dataset: {DATA_DIR / 'final_dataset.csv'}")

    print("\n" + "=" * 60)
    print("DONE - Ready for model retraining!")
    print("=" * 60)


if __name__ == "__main__":
    main()
