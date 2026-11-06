# Cross-Lingual Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add machine-translated English Reddit depression data to training set via NLLB-200, with full provenance tracking and quality controls.

**Architecture:** Download Reddit depression dataset from HuggingFace → Liberal filter → NLLB-200 translation (eng_Latn→vie) → Quality checks → Merge into training set only.

**Tech Stack:** Python, HuggingFace datasets, transformers (NLLB-200), pandas, SHA-256

---

## Global Constraints

- Translation: facebook/nllb-200-600M, source=eng_Latn, target=vie
- Source dataset: hugginglearners/reddit-depression-cleaned (CC0-1.0 license)
- Target: 2,000-2,500 translated samples after filtering
- Quality: 10% manual review (~200-300 samples)
- Data split: Translations added to TRAIN ONLY; val/test unchanged

---

## File Structure

```
data/
  translated/
    reddit_dep_en.csv          # Downloaded from HuggingFace
    reddit_dep_translated.csv   # Raw NLLB-200 translations
    reddit_dep_clean.csv        # Post-QC translations
    review_samples.csv         # 10% random samples for manual review
    train_merged.csv           # final_train + translations

scripts/
  cross_lingual_translate.py    # Main translation script
  translate_quality_check.py    # QC checks

data/labeled/
  final_train.csv              # Original training data (unchanged)
  final_val.csv                # Original validation (unchanged)
  final_test.csv               # Original test (unchanged)
```

---

## Task 1: Download and Filter Reddit Depression Dataset

**Files:**
- Create: `scripts/cross_lingual_translate.py` (partial)
- Create: `data/translated/reddit_dep_en.csv`

**Interfaces:**
- Produces: `data/translated/reddit_dep_en.csv` with columns `clean_text, is_depression`

- [ ] **Step 1: Create translation script with download function**

```python
# scripts/cross_lingual_translate.py

from pathlib import Path
import pandas as pd
from datasets import load_dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
TRANSLATED_DIR = PROJECT_DIR / "data" / "translated"

def download_reddit_depression_dataset():
    """Download Reddit Depression Dataset from HuggingFace."""
    print("Loading hugginglearners/reddit-depression-cleaned...")
    dataset = load_dataset("hugginglearners/reddit-depression-cleaned", split="train")
    df = dataset.to_pandas()
    
    # Rename columns to match our format
    df = df.rename(columns={"clean_text": "text", "is_depression": "label"})
    
    print(f"Loaded {len(df)} samples")
    print(f"Depression: {df['label'].sum()}, Normal: {(df['label'] == 0).sum()}")
    
    return df

def liberal_filter(df):
    """Liberal filter - keep all depression-labeled posts."""
    # Liberal: keep all with depression label
    # Exclude clearly off-topic posts (gaming, news, etc.)
    exclude_keywords = [
        "minecraft", "fortnite", "gaming", "esports", "video game",
        "politics", "news", "election", "trump", "biden",
        "sports", "football", "basketball", "soccer"
    ]
    
    def is_off_topic(text):
        text_lower = str(text).lower()
        return any(kw in text_lower for kw in exclude_keywords)
    
    # Keep all depression (label=1) and non-off-topic normal posts
    depression_posts = df[df['label'] == 1].copy()
    normal_posts = df[(df['label'] == 0) & (~df['text'].apply(is_off_topic))].copy()
    
    filtered = pd.concat([depression_posts, normal_posts]).reset_index(drop=True)
    
    # Balance: aim for ~2:1 normal:depression ratio
    n_dep = len(depression_posts)
    n_normal_target = min(len(normal_posts), n_dep * 2)
    normal_sampled = normal_posts.sample(n=n_normal_target, random_state=42)
    
    balanced = pd.concat([depression_posts, normal_sampled]).reset_index(drop=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"After filtering: {len(balanced)} samples")
    print(f"  Depression: {(balanced['label'] == 1).sum()}")
    print(f"  Normal: {(balanced['label'] == 0).sum()}")
    
    return balanced

def main():
    TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download and filter
    df = download_reddit_depression_dataset()
    df_filtered = liberal_filter(df)
    
    # Save
    output_path = TRANSLATED_DIR / "reddit_dep_en.csv"
    df_filtered.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run download script**

Run: `cd /Users/iannwendy/Documents/crawl_yt/youtube_depression_crawler && .venv/bin/python scripts/cross_lingual_translate.py`
Expected: Downloads ~2,000-3,000 samples, saves to data/translated/reddit_dep_en.csv

- [ ] **Step 3: Verify output**

```bash
wc -l data/translated/reddit_dep_en.csv
head -5 data/translated/reddit_dep_en.csv
```

- [ ] **Step 4: Commit**

```bash
git add data/translated/reddit_dep_en.csv scripts/cross_lingual_translate.py
git commit -m "feat: add Reddit depression dataset download and liberal filter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Implement NLLB-200 Translation

**Files:**
- Modify: `scripts/cross_lingual_translate.py` (add translation function)
- Create: `data/translated/reddit_dep_translated.csv`

**Interfaces:**
- Consumes: `data/translated/reddit_dep_en.csv`
- Produces: `data/translated/reddit_dep_translated.csv` with English-Vietnamese pairs + metadata

- [ ] **Step 1: Add translation function to script**

```python
# Add to scripts/cross_lingual_translate.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import hashlib
from datetime import datetime

# Global model instances (lazy loaded)
_tokenizer = None
_model = None

def get_device():
    """Detect available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_nllb_model():
    """Load NLLB-200-600M model."""
    global _tokenizer, _model
    if _tokenizer is None:
        print("Loading NLLB-200-600M...")
        device = get_device()
        print(f"Using device: {device}")
        
        _tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-600M")
        _model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-600M")
        _model = _model.to(device)
        _model.eval()
        print("Model loaded!")
    return _tokenizer, _model, device

def translate_text(text, tokenizer, model, device, max_length=128):
    """Translate single text from English to Vietnamese."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    forced_bos = tokenizer.lang_code_to_id["vie_Latn"]
    
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos,
        max_length=max_length,
        num_beams=3,
        length_penalty=0.6
    )
    
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated

def translate_batch(texts, tokenizer, model, device, batch_size=16):
    """Translate batch of texts."""
    translations = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, 
                          max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        forced_bos = tokenizer.lang_code_to_id["vie_Latn"]
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=128,
            num_beams=3,
            length_penalty=0.6
        )
        
        for output in outputs:
            translated = tokenizer.decode(output, skip_special_tokens=True)
            translations.append(translated)
    
    return translations

def translate_dataset(input_path, output_path):
    """Translate dataset and save with metadata."""
    df = pd.read_csv(input_path)
    
    # Load model
    tokenizer, model, device = load_nllb_model()
    
    # Translate
    print(f"Translating {len(df)} samples...")
    translations = translate_batch(df['text'].tolist(), tokenizer, model, device)
    
    # Add metadata
    df['original_text'] = df['text']
    df['text'] = translations
    df['text_hash'] = df['original_text'].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()
    )
    df['translation_model'] = 'nllb-200-600M'
    df['translation_version'] = '1.0'
    df['translation_date'] = datetime.now().isoformat()
    df['is_machine_translated'] = True
    df['source_dataset'] = 'reddit_depression_kaggle'
    df['source_label'] = df['label']
    
    # Rename for consistency
    df = df.rename(columns={'label': 'final_label'})
    df['label'] = df['final_label']  # Keep both for compatibility
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return df

# Add to main()
def main():
    TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)
    
    input_path = TRANSLATED_DIR / "reddit_dep_en.csv"
    translated_path = TRANSLATED_DIR / "reddit_dep_translated.csv"
    
    if not input_path.exists():
        download_reddit_depression_dataset()
        liberal_filter(pd.read_csv(input_path)).to_csv(input_path, index=False)
    
    translate_dataset(input_path, translated_path)
```

- [ ] **Step 2: Run translation**

Run: `.venv/bin/python scripts/cross_lingual_translate.py --translate`
Expected: Translates ~2,000-3,000 samples, saves to data/translated/reddit_dep_translated.csv

- [ ] **Step 3: Verify output**

```bash
wc -l data/translated/reddit_dep_translated.csv
head -3 data/translated/reddit_dep_translated.csv
```

Expected columns: text, original_text, label, text_hash, translation_model, translation_version, translation_date, is_machine_translated, source_dataset, source_label

- [ ] **Step 4: Commit**

```bash
git add data/translated/reddit_dep_translated.csv scripts/cross_lingual_translate.py
git commit -m "feat: add NLLB-200 translation pipeline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Quality Control Checks

**Files:**
- Create: `scripts/translate_quality_check.py`
- Modify: `data/translated/reddit_dep_clean.csv` (after QC)

**Interfaces:**
- Consumes: `data/translated/reddit_dep_translated.csv`
- Produces: `data/translated/reddit_dep_clean.csv`, `data/translated/review_samples.csv`

- [ ] **Step 1: Create quality check script**

```python
# scripts/translate_quality_check.py

import pandas as pd
from pathlib import Path
import random
import hashlib

PROJECT_DIR = Path(__file__).resolve().parents[1]
TRANSLATED_DIR = PROJECT_DIR / "data" / "translated"
LABELED_DIR = PROJECT_DIR / "data" / "labeled"

# Vietnamese negation words
NEGATIONS_VI = ["không", "chẳng", "đừng", "chớ", "chả", "chưa", "mất", "hết", "không còn"]
# English negation words
NEGATIONS_EN = ["not", "no", "never", "none", "neither", "nobody", "nothing", "nowhere", "hardly", "barely", "scarcely", "doesn't", "don't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "wasn't", "aren't", "weren't", "haven't", "hasn't", "hadn't"]

def has_negation(text, negations):
    """Check if text contains any negation words."""
    text_lower = str(text).lower()
    return any(f" {neg} " in f" {text_lower} " or f" {neg}." in f" {text_lower}." 
               for neg in negations)

def check_negation_preserved(row):
    """Check if negation is preserved in translation."""
    en_has_neg = has_negation(row['original_text'], NEGATIONS_EN)
    vi_has_neg = has_negation(row['text'], NEGATIONS_VI)
    return en_has_neg == vi_has_neg  # True if consistent

def check_length_ratio(row):
    """Check if translation length is reasonable."""
    len_en = len(str(row['original_text']))
    len_vi = len(str(row['text']))
    if len_en == 0:
        return False
    ratio = len_vi / len_en
    return 0.5 <= ratio <= 2.0  # Slightly wider than final 0.7-1.3

def check_overlap_with_holdout(df, labeled_dir):
    """Check for overlap with train/val/test sets."""
    # Load existing data
    train = pd.read_csv(labeled_dir / "final_train.csv")
    val = pd.read_csv(labeled_dir / "final_val.csv")
    test = pd.read_csv(labeled_dir / "final_test.csv")
    
    # Get hashes from existing data
    existing_hashes = set()
    for df_existing in [train, val, test]:
        if 'text_hash' in df_existing.columns:
            existing_hashes.update(df_existing['text_hash'].dropna().tolist())
    
    # Check Vietnamese text hashes
    if 'text_hash' in df.columns:
        overlap = df['text_hash'].isin(existing_hashes)
        return overlap.sum()
    
    # Check original text hashes
    en_hashes = df['original_text'].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()
    )
    overlap = en_hashes.isin(existing_hashes)
    return overlap.sum()

def run_quality_checks(input_path, output_path, review_output_path):
    """Run all quality checks and filter data."""
    df = pd.read_csv(input_path)
    print(f"Starting QC with {len(df)} samples...")
    
    # 1. Negation check
    print("Checking negation preservation...")
    df['negation_ok'] = df.apply(check_negation_preserved, axis=1)
    negation_fail = (~df['negation_ok']).sum()
    print(f"  Negation failures: {negation_fail}")
    
    # 2. Length ratio check
    print("Checking length ratios...")
    df['length_ok'] = df.apply(check_length_ratio, axis=1)
    length_fail = (~df['length_ok']).sum()
    print(f"  Length failures: {length_fail}")
    
    # 3. Overlap check
    print("Checking overlap with holdout sets...")
    overlap_count = check_overlap_with_holdout(df, LABELED_DIR)
    print(f"  Overlap found: {overlap_count}")
    
    # 4. Filter bad samples
    df['qc_passed'] = df['negation_ok'] & df['length_ok']
    df_clean = df[df['qc_passed']].copy()
    print(f"After QC: {len(df_clean)} samples ({len(df) - len(df_clean)} removed)")
    
    # 5. Select random samples for manual review (10%)
    review_size = max(200, int(len(df_clean) * 0.1))
    review_samples = df_clean.sample(n=review_size, random_state=42)
    
    # Save outputs
    df_clean.to_csv(output_path, index=False)
    review_samples.to_csv(review_output_path, index=False)
    
    print(f"\nSaved cleaned data: {output_path}")
    print(f"Saved review samples: {review_output_path}")
    print(f"\nQC Summary:")
    print(f"  Total input: {len(df)}")
    print(f"  Passed QC: {len(df_clean)}")
    print(f"  Removed: {len(df) - len(df_clean)}")
    print(f"  For review: {len(review_samples)}")
    
    return df_clean, review_samples

def main():
    TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)
    
    input_path = TRANSLATED_DIR / "reddit_dep_translated.csv"
    output_path = TRANSLATED_DIR / "reddit_dep_clean.csv"
    review_path = TRANSLATED_DIR / "review_samples.csv"
    
    run_quality_checks(input_path, output_path, review_path)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run quality checks**

Run: `.venv/bin/python scripts/translate_quality_check.py`
Expected: QC removes some samples, saves cleaned data and review samples

- [ ] **Step 3: Verify review samples format**

```bash
wc -l data/translated/review_samples.csv
head -1 data/translated/review_samples.csv
```

- [ ] **Step 4: Commit**

```bash
git add data/translated/reddit_dep_clean.csv data/translated/review_samples.csv scripts/translate_quality_check.py
git commit -m "feat: add translation quality control checks

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Merge with Training Data

**Files:**
- Create: `data/translated/train_merged.csv`
- Create: `scripts/merge_translated.py`

**Interfaces:**
- Consumes: `data/labeled/final_train.csv`, `data/translated/reddit_dep_clean.csv`
- Produces: `data/translated/train_merged.csv`

- [ ] **Step 1: Create merge script**

```python
# scripts/merge_translated.py

import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
TRANSLATED_DIR = PROJECT_DIR / "data" / "translated"
LABELED_DIR = PROJECT_DIR / "data" / "labeled"

def merge_training_data():
    """Merge clean training data with translated augmentation."""
    # Load original training data
    train = pd.read_csv(LABELED_DIR / "final_train.csv")
    print(f"Original train: {len(train)} samples")
    
    # Load translated data
    translated = pd.read_csv(TRANSLATED_DIR / "reddit_dep_clean.csv")
    print(f"Translated data: {len(translated)} samples")
    
    # Prepare translated data columns to match
    # Keep necessary columns from translation
    trans_cols = ['text', 'label', 'source_dataset', 'source_label', 
                  'text_hash', 'translation_model', 'translation_version',
                  'translation_date', 'is_machine_translated']
    translated_subset = translated[[c for c in trans_cols if c in translated.columns]].copy()
    
    # For original train, add translation columns as NaN
    for col in trans_cols:
        if col not in train.columns:
            train[col] = None
    
    # Mark training source
    translated_subset['augmentation_type'] = 'cross_lingual_translation'
    train['augmentation_type'] = 'original'
    
    # Combine
    combined = pd.concat([train, translated_subset], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"\nMerged train: {len(combined)} samples")
    print(f"  Original: {(combined['augmentation_type'] == 'original').sum()}")
    print(f"  Translated: {(combined['augmentation_type'] == 'cross_lingual_translation').sum()}")
    print(f"\nLabel distribution:")
    print(f"  Depression (1): {(combined['label'] == 1).sum()}")
    print(f"  Normal (0): {(combined['label'] == 0).sum()}")
    
    # Save
    output_path = TRANSLATED_DIR / "train_merged.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return combined

if __name__ == "__main__":
    merge_training_data()
```

- [ ] **Step 2: Run merge**

Run: `.venv/bin/python scripts/merge_translated.py`
Expected: Merges ~7,336 original + ~2,000 translated = ~9,300 total

- [ ] **Step 3: Verify merged data**

```bash
python -c "import pandas as pd; df=pd.read_csv('data/translated/train_merged.csv'); print(f'Total: {len(df)}'); print(df['augmentation_type'].value_counts())"
```

- [ ] **Step 4: Commit**

```bash
git add data/translated/train_merged.csv scripts/merge_translated.py
git commit -m "feat: merge translated data with training set

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Training Ablation Experiments

**Files:**
- Create: `scripts/train_cross_lingual_ablation.py`
- Results: `results/cross_lingual_*/`

**Interfaces:**
- Consumes: `data/labeled/final_train.csv`, `data/translated/train_merged.csv`
- Produces: Experiment results for 4 ablation conditions

- [ ] **Step 1: Create ablation training script**

```python
# scripts/train_cross_lingual_ablation.py

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess

PROJECT_DIR = Path(__file__).resolve().parents[1]
LABELED_DIR = PROJECT_DIR / "data" / "labeled"
TRANSLATED_DIR = PROJECT_DIR / "data" / "translated"
RESULTS_DIR = PROJECT_DIR / "results"

# Ablation conditions
CONDITIONS = {
    "A_clean_vi": "data/labeled/final_train.csv",
    "B_clean_va": "data/augmented_v2/final_train_augmented.csv",  # If exists
    "C_en_vi": "data/translated/train_merged.csv",
    "D_combined": "data/translated/train_merged.csv",  # Same as C for now
}

def run_experiment(condition_name, train_path, output_dir):
    """Run single experiment condition."""
    print(f"\n{'='*60}")
    print(f"Running condition: {condition_name}")
    print(f"Training data: {train_path}")
    print(f"{'='*60}")
    
    output_dir = RESULTS_DIR / f"cross_lingual_{condition_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training (using existing training script)
    cmd = [
        "python", "scripts/final_model_training.py",
        "--train-data", str(train_path),
        "--output-dir", str(output_dir),
        "--epochs", "3",
        "--seeds", "42"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "condition": condition_name,
        "train_path": str(train_path),
        "output_dir": str(output_dir),
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else "",
    }

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    for condition, train_path in CONDITIONS.items():
        if not Path(train_path).exists():
            print(f"Skipping {condition}: {train_path} not found")
            continue
        
        result = run_experiment(condition, train_path, RESULTS_DIR / condition)
        results.append(result)
    
    # Save results summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    summary_path = RESULTS_DIR / "cross_lingual_ablation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Ablation complete! Summary saved to:")
    print(f"  {summary_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run ablation experiments**

Run: `.venv/bin/python scripts/train_cross_lingual_ablation.py`
Expected: Trains 4 conditions, saves results to results/cross_lingual_*/

- [ ] **Step 3: Aggregate and compare results**

```python
# Compare results across conditions
import pandas as pd
from pathlib import Path

results_dir = Path("results")
conditions = ["A_clean_vi", "B_clean_va", "C_en_vi", "D_combined"]

for cond in conditions:
    cond_dir = results_dir / f"cross_lingual_{cond}"
    # Load and display metrics
    print(f"\n{cond}:")
    # (implementation depends on output format from training script)
```

- [ ] **Step 4: Commit results**

```bash
git add results/cross_lingual_*/
git commit -m "feat: add cross-lingual augmentation training results

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Manual Review of Translated Samples

**Files:**
- Review: `data/translated/review_samples.csv`
- Update: `data/translated/review_results.csv`

**Interfaces:**
- Consumes: `data/translated/review_samples.csv`
- Produces: Manual review annotations

- [ ] **Step 1: Prepare review interface**

Create a simple script to display samples for review:

```python
# scripts/review_translations.py

import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
TRANSLATED_DIR = PROJECT_DIR / "data" / "translated"

def display_samples_for_review(n=20):
    """Display random samples for manual review."""
    df = pd.read_csv(TRANSLATED_DIR / "review_samples.csv")
    
    print("="*80)
    print("TRANSLATION REVIEW - Check if translation preserves meaning")
    print("="*80)
    
    for i, row in df.head(n).iterrows():
        print(f"\n[Sample {i}] Label: {row['label']}")
        print(f"English (original):")
        print(f"  {row['original_text'][:300]}...")
        print(f"Vietnamese (translated):")
        print(f"  {row['text'][:300]}...")
        print("-"*80)

if __name__ == "__main__":
    display_samples_for_review()
```

- [ ] **Step 2: Run review (manual step)**

Run: `.venv/bin/python scripts/review_translations.py > data/translated/review_output.txt`
Open review_output.txt and manually review ~200-300 samples

- [ ] **Step 3: Document review findings**

```bash
# Create review summary
echo "Review findings:" > data/translated/review_summary.md
echo "- Total samples reviewed: ~200-300" >> data/translated/review_summary.md
echo "- Quality assessment: [excellent/good/acceptable/needs filtering]" >> data/translated/review_summary.md
echo "- Recommendations: [any filtering needed]" >> data/translated/review_summary.md
```

- [ ] **Step 4: Commit review results**

```bash
git add data/translated/review_*.txt data/translated/review_summary.md
git commit -m "docs: add manual review results for translated data

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Final Analysis and Paper Reporting

**Files:**
- Create: `results/cross_lingual_analysis.md`
- Update: `docs/paper_report.html` (ablation section)

- [ ] **Step 1: Generate analysis report**

```python
# scripts/analyze_cross_lingual_results.py

import pandas as pd
from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
TRANSLATED_DIR = PROJECT_DIR / "data" / "translated"

def generate_analysis():
    """Generate final analysis report."""
    
    # Load all ablation results
    conditions = ["A_clean_vi", "B_clean_va", "C_en_vi", "D_combined"]
    results = {}
    
    for cond in conditions:
        result_file = RESULTS_DIR / f"cross_lingual_{cond}" / "metrics.json"
        if result_file.exists():
            with open(result_file) as f:
                results[cond] = json.load(f)
    
    # Generate report
    report = []
    report.append("# Cross-Lingual Augmentation Analysis")
    report.append("")
    report.append("## Ablation Results")
    report.append("")
    report.append("| Condition | Train Size | F1-macro | F1-depression | Notes |")
    report.append("|-----------|------------|----------|---------------|-------|")
    
    for cond, data in results.items():
        train_size = data.get("train_size", "N/A")
        f1_macro = data.get("f1_macro", "N/A")
        f1_dep = data.get("f1_depression", "N/A")
        report.append(f"| {cond} | {train_size} | {f1_macro} | {f1_dep} | |")
    
    report.append("")
    report.append("## Translation Statistics")
    
    # Load translation stats
    trans_df = pd.read_csv(TRANSLATED_DIR / "reddit_dep_clean.csv")
    report.append(f"- Total translated: {len(trans_df)}")
    report.append(f"- Depression: {(trans_df['label']==1).sum()}")
    report.append(f"- Normal: {(trans_df['label']==0).sum()}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save
    output_path = RESULTS_DIR / "cross_lingual_analysis.md"
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    return report_text

if __name__ == "__main__":
    generate_analysis()
```

- [ ] **Step 2: Run analysis**

Run: `.venv/bin/python scripts/analyze_cross_lingual_results.py`

- [ ] **Step 3: Update paper report**

Add ablation results section to `docs/paper_report.html`:

```html
<h2>Cross-Lingual Augmentation Ablation</h2>
<p>
We evaluated machine-translated English depression data as an additional 
train-only cross-lingual augmentation condition. All translated instances 
retained source and translation provenance and were excluded from 
validation and test sets.
</p>
<table>
<tr><th>Condition</th><th>F1-macro</th><th>F1-depression</th></tr>
<tr><td>Clean Vietnamese</td><td>X.XX</td><td>X.XX</td></tr>
<tr><td>Clean + VA</td><td>X.XX</td><td>X.XX</td></tr>
<tr><td>Clean + En→Vi</td><td>X.XX</td><td>X.XX</td></tr>
<tr><td>Combined</td><td>X.XX</td><td>X.XX</td></tr>
</table>
```

- [ ] **Step 4: Commit final results**

```bash
git add results/cross_lingual_analysis.md docs/paper_report.html
git commit -m "docs: add cross-lingual augmentation analysis and paper update

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Download & filter Reddit dataset | `reddit_dep_en.csv`, `cross_lingual_translate.py` |
| 2 | NLLB-200 translation | `reddit_dep_translated.csv` |
| 3 | Quality control checks | `reddit_dep_clean.csv`, `review_samples.csv` |
| 4 | Merge with training data | `train_merged.csv` |
| 5 | Training ablation (4 conditions) | `results/cross_lingual_*/` |
| 6 | Manual review (~10%) | `review_output.txt`, `review_summary.md` |
| 7 | Analysis & paper reporting | `cross_lingual_analysis.md` |

**Total estimated time:**
- Translation: 30-60 min (GPU)
- QC + review: 30-45 min
- Training (4 conditions): 2-4 hours
- Analysis: 15-30 min
