#!/usr/bin/env python3
"""
Merge pre-Round 6 gold samples (train_gold.csv, n=3,432) with Round 6 reviewed gold samples
(round6_reviewed_clean.csv, n=5,702) into final_dataset.csv (n=9,134), and verify or generate
stratified 70/15/15 splits:
  - final_train.csv (n=6,392: 5,130 Normal / 1,262 Depression)
  - final_val.csv   (n=1,371: 1,100 Normal / 271 Depression)
  - final_test.csv  (n=1,371: 1,100 Normal / 271 Depression)

By default, if the canonical split files already exist and pass 100% integrity and disjointness
checks, they are preserved unmodified so exact benchmark row order is never accidentally altered.
Pass --rebuild to regenerate splits from scratch using random_state=42.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
LABELED_DIR = PROJECT_DIR / "data" / "labeled"
ROUND6_DIR = PROJECT_DIR / "data" / "round6"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and verify Round 6 gold benchmark splits.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force regeneration of final_dataset.csv and stratified splits.",
    )
    args = parser.parse_args()

    train_gold_path = LABELED_DIR / "train_gold.csv"
    round6_clean_path = ROUND6_DIR / "round6_reviewed_clean.csv"
    final_dataset_path = LABELED_DIR / "final_dataset.csv"
    final_train_path = LABELED_DIR / "final_train.csv"
    final_val_path = LABELED_DIR / "final_val.csv"
    final_test_path = LABELED_DIR / "final_test.csv"

    df_pre6 = pd.read_csv(train_gold_path)
    df_r6 = pd.read_csv(round6_clean_path)

    print("=" * 72)
    print("GOLD BENCHMARK MERGE & INTEGRITY VERIFICATION")
    print("=" * 72)
    print(f"Pre-Round 6 Gold (train_gold.csv):          {len(df_pre6):,}")
    print(f"Round 6 Gold (round6_reviewed_clean.csv):   {len(df_r6):,}")

    # Verify if canonical files already exist and match
    if (
        not args.rebuild
        and final_dataset_path.exists()
        and final_train_path.exists()
        and final_val_path.exists()
        and final_test_path.exists()
    ):
        df_full = pd.read_csv(final_dataset_path)
        df_tr = pd.read_csv(final_train_path)
        df_va = pd.read_csv(final_val_path)
        df_te = pd.read_csv(final_test_path)

        tr_ids = set(df_tr["comment_text"].astype(str).str.strip())
        va_ids = set(df_va["comment_text"].astype(str).str.strip())
        te_ids = set(df_te["comment_text"].astype(str).str.strip())

        overlap_tr_va = len(tr_ids & va_ids)
        overlap_tr_te = len(tr_ids & te_ids)
        overlap_va_te = len(va_ids & te_ids)

        print(f"Merged Benchmark (final_dataset.csv):       {len(df_full):,}")
        print(
            f"  - Train (final_train.csv):                {len(df_tr):,} "
            f"(Normal={(df_tr['label']==0).sum():,}, Depression={(df_tr['label']==1).sum():,})"
        )
        print(
            f"  - Val   (final_val.csv):                  {len(df_va):,} "
            f"(Normal={(df_va['label']==0).sum():,}, Depression={(df_va['label']==1).sum():,})"
        )
        print(
            f"  - Test  (final_test.csv):                 {len(df_te):,} "
            f"(Normal={(df_te['label']==0).sum():,}, Depression={(df_te['label']==1).sum():,})"
        )
        print(
            f"Split Overlap Check (comment_text):         "
            f"train∩val={overlap_tr_va}, train∩test={overlap_tr_te}, val∩test={overlap_va_te}"
        )

        assert len(df_full) == len(df_tr) + len(df_va) + len(df_te), "Split count mismatch!"
        assert overlap_tr_va == 0 and overlap_tr_te == 0 and overlap_va_te == 0, "Split leakage detected!"
        print("Canonical splits verified 100% intact (preserved without modification).")
        return

    # Rebuild if requested
    common_cols = [c for c in df_pre6.columns if c in df_r6.columns]
    df_merged = pd.concat([df_pre6[common_cols], df_r6[common_cols]], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["comment_text"], keep="last").reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df_merged,
        test_size=0.30,
        stratify=df_merged["label"],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42,
    )

    df_merged.to_csv(final_dataset_path, index=False)
    train_df.to_csv(final_train_path, index=False)
    val_df.to_csv(final_val_path, index=False)
    test_df.to_csv(final_test_path, index=False)
    print(f"Rebuilt final_dataset.csv ({len(df_merged):,}) and 70/15/15 splits.")


if __name__ == "__main__":
    main()
