# Round 6 v2 Final Results Summary

**Last Updated:** 2026-08-10 (Round 6 v2 complete retraining)

## Dataset Information

| Split | Samples | Normal | Depression | Depression Rate |
|-------|---------|--------|------------|-----------------|
| Train | 6,392 | 5,130 | 1,262 | 19.7% |
| Val | 1,371 | 1,100 | 271 | 19.8% |
| Test (In-Domain) | 1,371 | 1,100 | 271 | 19.8% |
| **Total** | **9,134** | **7,330** | **1,804** | **19.7%** |

**Cross-Domain Test (VSMEC):** 3,084 samples (balanced 50/50)

The validation and test sets are held out from training using stratified 70/15/15 split.

## In-Domain Evaluation Results (Test Set, n=1,371)

| Model | Accuracy | Precision-M | Recall-M | F1-Macro | F1-Depression |
|-------|----------|-------------|----------|----------|--------------|
| **PhoBERT (majority vote, 3 seeds)** | **0.8038** | 0.7041 | 0.7429 | **0.7187** | **0.5640** |
| PhoBERT (seed 42) | 0.8053 | 0.6981 | 0.7146 | 0.7055 | 0.5340 |
| PhoBERT (seed 123) | 0.7965 | 0.7005 | 0.7508 | 0.7172 | 0.5674 |
| PhoBERT (seed 2024) | 0.7943 | 0.6931 | 0.7328 | 0.7075 | 0.5481 |
| **TF-IDF + LinearSVC** | 0.8038 | 0.6956 | 0.7109 | 0.7025 | 0.5289 |
| TF-IDF + LogReg | 0.7936 | 0.6973 | 0.7476 | 0.7138 | 0.5626 |
| BiLSTM (majority vote) | 0.8062 | 0.0000 | 0.0000 | 0.6418 | 0.3991 |

### BiLSTM Per-Seed Results

| Seed | Accuracy | F1-Macro | F1-Depression |
|------|----------|----------|---------------|
| 42 | 0.8133 | 0.6585 | 0.4286 |
| 123 | 0.8009 | 0.6465 | 0.4129 |
| 2024 | 0.8045 | 0.6203 | 0.3558 |

## Cross-Domain Evaluation Results (VSMEC, n=3,084)

| Model | Accuracy | F1-Macro | F1-Depression |
|-------|----------|----------|---------------|
| **TF-IDF + LinearSVC** | **0.5107** | **0.3798** | **0.0948** |
| PhoBERT (majority vote) | 0.5104 | 0.3608 | 0.0612 |
| PhoBERT (seed 42) | 0.5130 | 0.3683 | 0.0719 |
| PhoBERT (seed 123) | 0.5162 | 0.3740 | 0.0794 |
| PhoBERT (seed 2024) | 0.5175 | 0.3762 | 0.0833 |
| TF-IDF + LogReg | 0.5052 | 0.3577 | 0.0498 |
| BiLSTM (majority vote) | 0.5003 | 0.3375 | 0.0090 |

**Generalization Gap (PhoBERT majority vote):** 0.7187 − 0.3608 = **0.3579 F1**

## Comparison Across Rounds

| Round | Dataset Size | PhoBERT In-Domain F1 | PhoBERT Cross-Domain F1 | Gap |
|-------|-------------|---------------------|------------------------|-----|
| Round 4 | ~4,000 | 0.8417 | 0.3850 | 0.4567 |
| Round 5 | ~7,336 | 0.7845 | 0.3598 | 0.4247 |
| **Round 6 v2** | **9,134** | **0.7187** | **0.3608** | **0.3579** |

**Note:** Round 4 numbers were computed on pre-repair data with potential label leakage. Round 5 and 6 use verified gold sets with no overlap.

## Active Learning Progress

| Round | Samples Annotated | Total Gold | Cross-Domain F1 |
|-------|-----------------|------------|-----------------|
| R1-R4 | ~2,072 | ~2,072 | 0.3850 |
| R5 | 1,360 | 3,432 | 0.3598 |
| R6 | 5,702 | 9,134 | 0.3608 |
| **Total** | **9,134** | **9,134** | — |

## Key Findings

1. **Cross-domain transfer remains limited** despite 6 rounds of active learning (9,134 gold samples)
2. **Simpler models generalize better** - TF-IDF + LinearSVC beats PhoBERT on cross-domain (0.3798 vs 0.3608)
3. **The gap is shrinking** from 0.4567 (R4) to 0.3579 (R6), but cross-domain F1 has not improved
4. **Data-centric limitation** - label definition divergence, text length mismatch, and linguistic register differences are the main barriers

## Model Checkpoints (Round 6 v2)

| Model | Location |
|-------|----------|
| PhoBERT (seed 42) | `models/round6_v2_retrained/phobert_seed_42/best_model/` |
| PhoBERT (seed 123) | `models/round6_v2_retrained/phobert_seed_123/best_model/` |
| PhoBERT (seed 2024) | `models/round6_v2_retrained/phobert_seed_2024/best_model/` |
| BiLSTM (seed 42/123/2024) | `models/round6_v2_retrained/bilstm_seed_{42,123,2024}/best_model.pt` |
| TF-IDF + LogReg | `models/round6_v2_retrained/tfidf_logreg_round6_v2.joblib` |
| TF-IDF + LinearSVC | `models/round6_v2_retrained/tfidf_linearsvc_round6_v2.joblib` |
| Canonical evaluation JSON | `models/round6_v2_retrained/evaluation_results.json` |
| Timestamped evaluation JSON | `results/round6_retrained_<timestamp>/evaluation_results.json` |
