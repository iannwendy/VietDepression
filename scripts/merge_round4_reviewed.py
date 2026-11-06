"""Merge Round 4 reviewed labels into gold set and training dataset.

Usage:
  PYTHONPATH="$PWD" .venv/bin/python scripts/merge_round4_reviewed.py
"""

import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DOCS_DIR = PROJECT_DIR / "docs"

def main():
    print("=" * 60)
    print("MERGING ROUND 4 REVIEWED LABELS")
    print("=" * 60)

    # Load Round 4 export
    round4_df = pd.read_csv(DOCS_DIR / "export_round4_active_learning.csv.csv")
    print(f"\nRound 4 export: {len(round4_df)} samples")
    print(f"Label distribution:\n{round4_df['final_label'].value_counts()}")

    # Filter to valid labels (normal, depression)
    valid_labels = round4_df[round4_df['final_label'].isin(['normal', 'depression'])].copy()
    print(f"\nValid labels (normal + depression): {len(valid_labels)}")

    # Convert labels: normal=0, depression=1
    valid_labels['label'] = valid_labels['final_label'].map({'normal': 0, 'depression': 1})

    # Load existing gold set
    gold_df = pd.read_csv(DATA_DIR / "train_gold.csv")
    print(f"\nExisting gold set: {len(gold_df)} samples")
    print(f"Existing gold distribution:\n{gold_df['label'].value_counts()}")

    # Create new gold entries from Round 4
    new_gold = valid_labels[['text', 'label']].copy()
    new_gold['source'] = 'round4_active_learning'
    new_gold['weight'] = 3  # Same weight as human_gold

    # Remove duplicates (check against existing gold)
    existing_texts = set(gold_df["comment_text"].str.lower().str.strip())
    new_gold['text_normalized'] = new_gold['text'].str.lower().str.strip()
    new_gold = new_gold[~new_gold['text_normalized'].isin(existing_texts)]
    new_gold = new_gold.drop(columns=['text_normalized'])

    print(f"\nNew gold samples (after dedup): {len(new_gold)}")
    print(f"New gold distribution:\n{new_gold['label'].value_counts()}")

    # Merge
    merged_gold = pd.concat([gold_df, new_gold], ignore_index=True)
    print(f"\nMerged gold set: {len(merged_gold)} samples")
    print(f"Merged distribution:\n{merged_gold['label'].value_counts()}")

    # Save merged gold
    merged_gold.to_csv(DATA_DIR / "train_gold.csv", index=False)
    print(f"\nSaved: train_gold.csv")

    # Now rebuild final dataset with merged gold
    print("\n" + "=" * 60)
    print("REBUILDING FINAL DATASET")
    print("=" * 60)

    # Load weak labels
    weak_df = pd.read_csv(DATA_DIR / "auto_labeled_comments.csv", dtype=str).fillna("")
    weak_df['label'] = weak_df['label'].astype(int)

    # Combine gold + weak (similar to original logic)
    # Gold samples get weight 3, weak_high_conf get weight 2
    gold_for_dataset = merged_gold.copy()
    gold_for_dataset = gold_for_dataset.rename(columns={'text': 'comment_text'})

    weak_high = weak_df[weak_df['source'] == 'weak_high_conf'].copy()

    # Final dataset: gold + weak_high_conf
    final_df = pd.concat([gold_for_dataset, weak_high], ignore_index=True)

    # Remove duplicates based on text
    final_df['text_normalized'] = final_df['comment_text'].str.lower().str.strip()
    final_df = final_df.drop_duplicates(subset=['text_normalized'], keep='first')
    final_df = final_df.drop(columns=['text_normalized'])

    print(f"\nFinal dataset: {len(final_df)} samples")
    print(f"Distribution:\n{final_df['label'].value_counts()}")

    # Save final dataset
    final_df.to_csv(DATA_DIR / "final_dataset.csv", index=False)
    print(f"\nSaved: {DATA_DIR / 'final_dataset.csv'}")

    # Stratified split
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(
        final_df, test_size=0.30, random_state=42, stratify=final_df['label']
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=42, stratify=temp['label']
    )

    train.to_csv(DATA_DIR / "final_train.csv", index=False)
    val.to_csv(DATA_DIR / "final_val.csv", index=False)
    test.to_csv(DATA_DIR / "final_test.csv", index=False)

    print(f"\nTrain: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Saved: final_train.csv, final_val.csv, final_test.csv")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Round 4 reviewed: {len(round4_df)} samples")
    print(f"  - normal: {len(round4_df[round4_df['final_label']=='normal'])}")
    print(f"  - depression: {len(round4_df[round4_df['final_label']=='depression'])}")
    print(f"  - excluded: {len(round4_df[round4_df['final_label']=='exclude'])}")
    print(f"  - uncertain: {len(round4_df[round4_df['final_label']=='uncertain'])}")
    print(f"\nNew gold samples added: {len(new_gold)}")
    print(f"Total gold set: {len(merged_gold)}")
    print(f"Total final dataset: {len(final_df)}")

    # Label distribution in final dataset
    print(f"\nFinal dataset label distribution:")
    print(f"  - Normal (0): {len(final_df[final_df['label']==0])}")
    print(f"  - Depression (1): {len(final_df[final_df['label']==1])}")

    return {
        'round4_reviewed': len(round4_df),
        'round4_normal': len(round4_df[round4_df['final_label']=='normal']),
        'round4_depression': len(round4_df[round4_df['final_label']=='depression']),
        'round4_excluded': len(round4_df[round4_df['final_label']=='exclude']),
        'new_gold_added': len(new_gold),
        'total_gold': len(merged_gold),
        'total_final_dataset': len(final_df),
    }


if __name__ == "__main__":
    main()
