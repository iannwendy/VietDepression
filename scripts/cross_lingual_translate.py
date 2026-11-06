#!/usr/bin/env python3
"""
Cross-lingual Translation Pipeline - Step 1: Download and Filter Reddit Depression Dataset

Downloads English Reddit depression data from HuggingFace, filters off-topic posts,
and balances the dataset for cross-lingual augmentation (Vietnamese).

Dataset: ShreyaR/DepressionDetection
Columns: clean_text, is_depression (binary 0/1)
Output: data/translated/reddit_dep_en.csv with columns: clean_text, is_depression
"""

import pandas as pd
import os

# Configuration
HF_DATASET = "ShreyaR/DepressionDetection"
OUTPUT_PATH = "data/translated/reddit_dep_en.csv"

# Off-topic keywords to exclude (case-insensitive)
OFF_TOPIC_KEYWORDS = [
    "minecraft", "fortnite", "gaming", "esports", "video game",
    "politics", "news", "election", "trump", "biden",
    "sports", "football", "basketball", "soccer"
]


def is_off_topic(text: str) -> bool:
    """Check if text contains off-topic keywords."""
    if pd.isna(text):
        return True
    text_lower = str(text).lower()
    return any(keyword in text_lower for keyword in OFF_TOPIC_KEYWORDS)


def download_dataset():
    """Download dataset from HuggingFace using the datasets library."""
    print(f"Loading dataset from HuggingFace: {HF_DATASET}")

    from datasets import load_dataset

    # Load dataset from HuggingFace
    dataset = load_dataset(HF_DATASET, split="train")

    # Convert to pandas DataFrame
    df = dataset.to_pandas()
    print(f"Downloaded {len(df)} samples")
    print(f"Dataset columns: {df.columns.tolist()}")
    print(f"Label distribution before filtering:")
    if 'is_depression' in df.columns:
        print(df['is_depression'].value_counts())
    elif 'label' in df.columns:
        print(df['label'].value_counts())

    return df


def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out off-topic posts."""
    print(f"\nFiltering off-topic posts...")

    # Determine which column is the label
    if 'is_depression' in df.columns:
        label_col = 'is_depression'
        text_col = 'clean_text'
    elif 'label' in df.columns:
        label_col = 'label'
        text_col = 'text'
    else:
        raise ValueError(f"Could not find label column. Available: {df.columns.tolist()}")

    # Identify depression and normal posts
    depression_df = df[df[label_col] == 1].copy()
    normal_df = df[df[label_col] == 0].copy()

    print(f"  Depression posts (label=1): {len(depression_df)}")
    print(f"  Normal posts (label=0): {len(normal_df)}")

    # Filter off-topic from normal posts
    normal_off_topic = normal_df[normal_df[text_col].apply(is_off_topic)]
    normal_filtered = normal_df[~normal_df[text_col].apply(is_off_topic)]

    print(f"  Normal off-topic posts excluded: {len(normal_off_topic)}")
    print(f"  Normal posts after filtering: {len(normal_filtered)}")

    # Target: 2000-2500 total samples with ~2:1 normal:depression ratio
    # Target total: 2250 (middle of range) => depression ~750, normal ~1500
    TARGET_TOTAL = 2250
    target_depression_count = TARGET_TOTAL // 3  # ~750
    target_normal_count = target_depression_count * 2  # ~1500

    # Sample depression posts if we have more than target
    if len(depression_df) > target_depression_count:
        depression_filtered = depression_df.sample(n=target_depression_count, random_state=42)
        print(f"  Depression posts sampled to {len(depression_filtered)} for target total")
    else:
        depression_filtered = depression_df.copy()
        print(f"  Using all {len(depression_filtered)} depression posts")

    # Sample normal posts to achieve 2:1 ratio
    if len(normal_filtered) > target_normal_count:
        normal_sampled = normal_filtered.sample(n=target_normal_count, random_state=42)
        print(f"  Normal posts sampled to {len(normal_sampled)} for 2:1 ratio")
    else:
        normal_sampled = normal_filtered
        print(f"  Using all {len(normal_sampled)} normal posts")

    # Combine and create output format
    result_df = pd.concat([depression_filtered, normal_sampled], ignore_index=True)

    # Rename columns to output format (matching spec: clean_text, is_depression)
    result_df = result_df.rename(columns={
        text_col: 'clean_text',
        label_col: 'is_depression'
    })

    # Select only required columns
    result_df = result_df[['clean_text', 'is_depression']]

    # Shuffle
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return result_df


def main():
    """Main pipeline."""
    print("=" * 60)
    print("Cross-lingual Translation Pipeline - Step 1")
    print("Downloading and Filtering Reddit Depression Dataset")
    print("=" * 60)

    # Download
    df = download_dataset()

    # Filter
    filtered_df = filter_dataset(df)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    filtered_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 60}")
    print(f"Output saved to: {OUTPUT_PATH}")
    print(f"Total samples: {len(filtered_df)}")
    print(f"Label distribution:")
    print(filtered_df['is_depression'].value_counts())
    print(f"Ratio (normal:depression): {filtered_df['is_depression'].value_counts()[0] / filtered_df['is_depression'].value_counts()[1]:.2f}:1")
    print(f"{'=' * 60}")

    # Verify output
    print("\nVerification:")
    print(f"  Rows: {len(filtered_df)}")
    print(f"  Columns: {filtered_df.columns.tolist()}")
    print("\nSample rows:")
    print(filtered_df.head(3).to_string())


if __name__ == "__main__":
    main()
