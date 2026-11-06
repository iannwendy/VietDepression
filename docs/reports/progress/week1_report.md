# Week N Report — Trạng thái hiện tại (2026-06-26)

**Dự án:** Detection of Depression Signs in Vietnamese Social Media Text Using Deep Learning Models

**Trạng thái:** Tất cả Phase 1-3 hoàn thành; Phase 4 (paper write-up) đang được đồng bộ vào `docs/paper_report.html`.

---

## 1. Data

| Artifact | Size | Status |
|---|---|---|
| `data/cleaned_comments.csv` | 125,329 rows | ✅ crawled |
| `data/gold_review.csv` | **2,515 rows** (2,265 normal / 250 depression) | ✅ 3 review rounds merged |
| `data/final_dataset.csv` | **2,553 rows** (985 human_gold + 1,568 weak_high_conf) | ✅ balanced 2:1 |
| `data/final_train.csv` / `final_val.csv` / `final_test.csv` | **1,786 / 383 / 383** | ✅ stratified, 0 cross-domain leak |
| `data_unified/cross_domain_test.csv` | 3,084 rows (VSMEC) | ✅ held out |

**Quality gate (Round 3):** agreement 43.17%, Cohen's κ = -0.05, 66.3% of `depression_auto` weak labels rejected by blind reviewer — confirms reviewer độc lập.

## 2. Models (all trained on `final_train.csv`)

| Model | In-domain F1 | Cross-domain F1 | Notes |
|---|---|---|---|
| TF-IDF + LogReg | 0.8347 | 0.3917 | strong in-domain baseline |
| TF-IDF + LinearSVC | 0.8286 | 0.3820 | comparable to LogReg |
| BiLSTM (random) | 0.8357 | 0.4079 | paper Section 4.2 spec |
| BiLSTM (PhoBERT-frozen) | 0.8266 | 0.4352 | +2.7 pp cross-domain over random |
| **PhoBERT (3-seed)** | **0.8681 ± 0.0086** | **0.3727 ± 0.0242** | best in-domain |
| BERTopic-only | 0.5599 | 0.5030 | unsupervised; best cross-domain depression F1 |
| PhoBERT + BERTopic | 0.9501 | 0.3977 | not yet rerun on final_dataset |

## 3. DAPT counter-experiment (commit `c8bc7fe`)

3 seeds × 2 test sets × 2 models = 12 runs.

| Model | Final test F1 | VSMEC F1 |
|---|---|---|
| Original | 0.8681 ± 0.0086 | 0.3727 ± 0.0242 |
| DAPT | **0.8803 ± 0.0030** | 0.3620 ± 0.0188 |

**Direction reversed from pre-round-3:** DAPT now slightly improves
in-domain (+0.0122) at cost of marginal cross-domain loss (-0.0107);
neither delta is statistically significant at 3 seeds.

## 4. Commits in this batch (latest first)

```
c8bc7fe DAPT eval rerun on round-3 dataset + update paper §5.5 + README
0ddd82a P1a: Verify BERTopic artifacts + update paper §5.3 with actual numbers
1f85ff6 P2b: Apply weight column via WeightedRandomSampler in PhoBERT training
4ab5dbf P2a: Regression test for run_finetune() data path override
e3e818e P0a: Rerun TF-IDF + LogReg / LinearSVC on final_dataset (post round-3)
f09fddc Fix DAPT eval orchestrator to use final_train/val/test splits
35ca4b6 Merge round 3 review (1500 rows) → rebuild gold + final_dataset
e861333 Code health: split README headline tables, archive obsolete CSVs, add LinearSVC + tests
```

## 5. Tests

52 unit tests, all CPU, ~0.2s wall-clock. Coverage:

- `tests/test_cleaner.py` — 13 tests
- `tests/test_auto_labeler.py` — 11 tests
- `tests/test_gold_builder.py` — 5 tests
- `tests/test_phobert_postprocess.py` — 6 tests
- `tests/test_baseline_model.py` — 5 tests (LogReg + LinearSVC)
- `tests/test_dapt_eval_helpers.py` — 12 tests (incl. regression test for
  the DAPT orchestrator data-path bug)

## 6. Open / next

- BiLSTM with single-seed only — paper Table 5.1 notes "negligible
  variance" for classical baselines and BiLSTM, but a 3-seed sweep
  would tighten the comparison if time permits.
- PhoBERT + BERTopic row in Table 5.1 still uses pre-round-3 numbers
  (0.9501 / 0.3977); rerun deferred (low priority — the naive
  concatenation was shown to underperform in the pre-round-3 run).
- 5-seed sweep for DAPT counter-experiment would let us claim
  cross-domain significance (currently n=3 seeds, p≈0.34).

See [`docs/paper_report.html`](paper_report.html) for the full
manuscript and [`docs/ROADMAP_SAU_REVIEW.md`](ROADMAP_SAU_REVIEW.md)
for the closed-out roadmap.