# Cross-Lingual Augmentation Design

## Date: 2026-07-25
## Purpose: English → Vietnamese machine translation augmentation for depression detection

---

## 1. Overview

Add machine-translated English depression data as cross-lingual augmentation to improve model generalization. Only use English datasets with clear licensing and labels aligned with "depression-related personal disclosure".

---

## 2. Data Pipeline

### 2.1 Source Data

| Property | Value |
|----------|-------|
| Dataset | Reddit Depression Dataset (Kaggle) |
| URL | data.mendeley.com/datasets/xppzm3kv9g or Kaggle |
| Samples | ~2,500+ |
| Labels | depression/normal |
| License | Mendeley restricted / Kaggle open |

### 2.2 Filter Criteria (Liberal)

**Include:**
- All posts with depression label
- Posts from depression-related subreddits

**Exclude:**
- Off-topic (gaming, news, politics)
- Clearly non-depression content

**Expected output:** ~2,000-2,500 samples

### 2.3 Translation Pipeline

```
English text
    ↓
NLLB-200-600M (eng_Latn → vie)
    ↓
Vietnamese translation
```

| Property | Value |
|----------|-------|
| Model | facebook/nllb-200-600M |
| Source | eng_Latn |
| Target | vie |
| Device | CUDA/MPS (8GB VRAM min) |

---

## 3. Metadata Schema

Each translated sample includes:

```python
{
    "comment_text": str,          # Vietnamese translation
    "label": int,                  # 0=normal, 1=depression
    "source_dataset": str,        # "reddit_depression_kaggle"
    "source_label": int,          # Original label
    "text_hash": str,             # SHA-256 of English source
    "translation_model": str,    # "nllb-200-600M"
    "translation_version": str,  # "1.0"
    "translation_date": str,      # ISO timestamp
    "is_machine_translated": bool # True
}
```

---

## 4. Quality Controls

### 4.1 Automated Checks

| Check | Criteria | Action |
|-------|----------|--------|
| Negation | No change in negation words | Reject if changed |
| Length ratio | 0.7x - 1.3x of original | Flag if outside |
| Label preservation | Source label = target label | Reject if mismatch |
| Duplicate | Check vs train/val/test | Remove if overlap |

**Negation keywords (Vietnamese):**
- không, chưa, không bao giờ, chẳng, chả, đừng, chớ, mất, hết, không còn

### 4.2 Manual Review

- **Sample size:** 10% of translated data (~200-300 samples)
- **Method:** Random selection
- **Criteria:** Meaning preservation, natural Vietnamese, label alignment

---

## 5. Data Split Policy

| Split | Augmentation | Notes |
|-------|-------------|-------|
| Train | ✅ YES | Translated data added here only |
| Validation | ❌ NO | Unchanged (in-domain) |
| Test (in-domain) | ❌ NO | Unchanged (in-domain) |
| VSMEC (cross-domain) | ❌ NO | Unchanged |

---

## 6. Training Ablation

| Condition | Data | Description |
|----------|------|-------------|
| A | Clean Vietnamese | Original labeled data only |
| B | Clean + VA | Clean + Vietnamese augmentation |
| C | Clean + En→Vi | Clean + English→Vietnamese translations |
| D | Combined | Clean + VA + En→Vi |

---

## 7. Equipment

| Resource | Requirement |
|----------|-------------|
| GPU | NVIDIA GPU or MPS, 8GB VRAM recommended |
| Model size | NLLB-200-600M (~1.2GB) |
| Est. translation time | 30-60 minutes for 2,500 samples |
| Disk space | ~50MB additional |

---

## 8. Output Files

| File | Location | Description |
|------|----------|-------------|
| Translated data | `data/translated/reddit_dep_en_vi.csv` | English-Vietnamese pairs |
| Cleaned translations | `data/translated/reddit_dep_vi_clean.csv` | Post-QC translations |
| Review samples | `data/translated/review_samples.csv` | 10% for manual review |
| Training merge | `data/translated/train_merged.csv` | Combined with clean train |

---

## 9. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| "Translationese" effect | Medium | Conservative augmentation ratio |
| F1 decrease | Low-Medium | Ablation study will show |
| Label noise from source | Low | Liberal filter + manual review |

---

## 10. Paper Reporting

For the paper, report:

> "We evaluated machine-translated English depression data as an additional train-only cross-lingual augmentation condition. All translated instances retained source and translation provenance and were excluded from validation and test sets."
