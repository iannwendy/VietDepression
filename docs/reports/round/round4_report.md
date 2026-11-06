# Round 4 Active Learning Report

**Date:** 2026-07-04
**Round:** 4
**Status:** ✅ COMPLETED

---

## Round 4 Statistics

| Metric | Value |
|--------|-------|
| Total samples reviewed | 1,500 |
| Normal | 739 (49.3%) |
| Depression | 260 (17.3%) |
| Excluded | 452 (30.1%) |
| Uncertain | 49 (3.3%) |
| **New samples added** | **948 samples** |

---

## Dataset Impact

### Before Round 4
| Metric | Value |
|--------|-------|
| Gold set | ~2,072 samples |
| Final dataset | ~2,553 samples |

### After Round 4
| Metric | Value |
|--------|-------|
| Gold set | **3,020 samples** |
| Final dataset | **6,079 samples** |
| New samples added | 948 (699 normal, 249 depression) |

---

## Model Retraining Results

**Dataset:** 6,079 samples (train: 4,255 / val: 912 / test: 912)

| Model | In-domain F1 | Cross-domain F1 | Seeds |
|-------|-------------|----------------|-------|
| TF-IDF + LogReg | 0.8415 | 0.3780 | 1 |
| TF-IDF + LinearSVC | 0.8799 | 0.3574 | 1 |
| **PhoBERT (original)** | **0.8417 ± 0.0220** | **0.3850 ± 0.0219** | 3 |

### Before/After Comparison (PhoBERT)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| In-domain F1 | 0.8681 | 0.8417 | -0.0264 |
| Cross-domain F1 | 0.3727 | 0.3850 | **+0.0123** ✅ |

**Observation:** Cross-domain performance improved slightly with Round 4 data.

---

## Files Updated

| File | Status |
|------|--------|
| `data/train_gold.csv` | ✅ Updated (3,020 rows) |
| `data/final_dataset.csv` | ✅ Rebuilt (6,079 rows) |
| `data/final_train.csv` | ✅ Rebuilt (4,255 rows) |
| `data/final_val.csv` | ✅ Rebuilt (912 rows) |
| `data/final_test.csv` | ✅ Rebuilt (912 rows) |
| `README.md` | ✅ Updated metrics |
| `docs/PROGRESS_REPORT_2026-06-29.md` | ✅ Updated |

---

## Next Steps

- [x] ~~Complete PhoBERT retraining on new dataset~~
- [x] ~~Run full model evaluation (all 5 families)~~
- [x] ~~Update README.md with new metrics~~
- [ ] Update paper_report.html with Round 4 results
- [ ] Generate new visualizations

---

*Generated: 2026-07-04*
