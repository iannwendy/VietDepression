# VietDepression: Detecting Depression-Related Language in Vietnamese YouTube Comments

Official repository for **VietDepression**, a `9,134`-comment human-adjudicated Vietnamese YouTube benchmark and strictly inductive evaluation framework comparing weighted lexical triage, sparse linear classifiers, recurrent networks (`BiLSTM`), multilingual and Vietnamese formal/social-media transformers (`XLM-RoBERTa-base`, `PhoBERT-base`, `ViSoBERT`), inductive neural topic features (`BERTopic`), imbalanced, contrastive, and negatively correlated objectives (`LA-Focal`, `SupCon`, `NCNL`), and rubric-guided open-weight LLMs (`Qwen2.5-1.5B-Instruct`) across **14 benchmark configurations** in-domain and under cross-corpus prior-shift calibration on `UIT-VSMEC`.

---

## 🔗 Repository Navigation & Two-Tier Release Governance

To protect social-media posters against reverse-search re-identification in accordance with **CLPsych** and **eRisk** mental-health NLP privacy standards, the **Institutional Ethics Exemption** for non-interventional secondary analysis of de-identified public text under **Ton Duc Thang University** research ethics regulations, and the **YouTube API Services Terms of Service**, **VietDepression** strictly enforces a **Two-Tier Release Protocol** (zero raw YouTube comment strings or platform IDs are stored in public Git files):

| Tier & Resource | Path in Repository | Access Level |
|-----------------|--------------------|--------------|
| **Tier 1: De-Identified SHA-256 Split Manifests** | [`data/manifests/`](data/manifests/) (`final_{train,val,test,dataset}_manifest.csv`) | **Public (Open)** |
| **Tier 1: Manifest Integrity & Zero-Overlap Audit** | [`data/manifests/manifest_integrity_summary.json`](data/manifests/manifest_integrity_summary.json) | **Public (Open)** |
| **Tier 1: 335-Entry Lexicon & 22 Review Flags** | [`scripts/vietdepression_lexicon.py`](scripts/vietdepression_lexicon.py), [`docs/appendices/APPENDIX_KEYWORDS.md`](docs/appendices/APPENDIX_KEYWORDS.md) | **Public (Open)** |
| **Tier 1: Four-Rule Annotation Guidelines** | [`docs/ANNOTATION_GUIDELINES.md`](docs/ANNOTATION_GUIDELINES.md) | **Public (Open)** |
| **Tier 1: Training, Calibration & Audit Scripts** | [`scripts/`](scripts/), [`results/scientific_audit/`](results/scientific_audit/) | **Public (Open)** |
| **Tier 2: Normalized Text (`9,134` Gold Splits)** | Hydrated locally to `data/labeled/final_{train,val,test}.csv` (git-ignored) | **Controlled Academic DUA** ([`DATA_USE_AGREEMENT.md`](DATA_USE_AGREEMENT.md)) |

---

## 📊 Audited Benchmark Partitions (`9,134` Gold Comments; `0%` Overlap)

Every record in the `9,134`-comment benchmark carries a **100% human-adjudicated binary gold label** (`0` unreviewed rule labels, `0` pseudo-labels, `0` external rows; `300`-item double-annotated reliability audit: `93.33%` raw agreement, Cohen's $\kappa = 0.8667$):

| Partition | Public SHA-256 Manifest | Total | Positive (`1`) | Negative (`0`) | Analytical Role |
|-----------|-------------------------|------:|---------------:|---------------:|-----------------|
| **Train (`70%`)** | `data/manifests/final_train_manifest.csv` | `6,392` | `1,262` (`19.74%`) | `5,130` (`80.26%`) | Sole split used to fit all classifiers, TF-IDF, scalers & inductive BERTopic |
| **Validation (`15%`)** | `data/manifests/final_val_manifest.csv` | `1,371` | `271` (`19.77%`) | `1,100` (`80.23%`) | Hyperparameter tuning & early-stopping checkpoint selection |
| **In-Domain Test (`15%`)** | `data/manifests/final_test_manifest.csv` | `1,371` | `271` (`19.77%`) | `1,100` (`80.23%`) | Held-out in-domain evaluation (`0%` cross-split text overlap) |
| **Full Benchmark** | `data/manifests/final_dataset_manifest.csv` | **`9,134`** | **`1,804` (`19.75%`)** | **`7,330` (`80.25%`)** | Complete human-adjudicated gold benchmark |
| **UIT-VSMEC Full Proxy** | `data/manifests/uit_vsmec_proxy_manifest.csv` | `3,084` | `1,542` (`50.00%`) | `1,542` (`50.00%`) | External emotion-proxy transfer (`Sadness+Fear` vs. `Enjoyment`) |
| **UIT-VSMEC Sadness-Only** | `data/manifests/uit_vsmec_proxy_manifest.csv` | `2,691` | `1,149` (`42.70%`) | `1,542` (`57.30%`) | Construct-refined external proxy (`Sadness` vs. `Enjoyment`, Fear excluded) |

---

## 🏆 Headline Results (14 Benchmark Configurations — Strictly Inductive Protocol)

Multi-seed neural models (`seeds 42, 123, 2024`) report **Majority Vote** Macro-F1 with `2,000`-iteration non-parametric bootstrap `95%` CIs alongside 3-seed `Mean ± SD` ($\dagger$ denotes 5-fold stratified training CV for deterministic linear models), soft-vote `ROC-AUC` / `PR-AUC`, and uncalibrated (`\tau = 0.50`) external transfer:

| Model Configuration | In-Domain Macro-F1 `[95% CI]` | 3-Seed `Mean ± SD` | F1-Pos / Acc. | In-Domain ROC / PR-AUC | UIT-VSMEC ROC-AUC (Full / Sad-Only) | Uncal. Ext. Macro-F1 (Full / Sad-Only) |
|---------------------|-------------------------------|--------------------|---------------|------------------------|-------------------------------------|----------------------------------------|
| **ViSoBERT + LA-Focal** | **0.7518** `[0.7220, 0.7809]` | `0.7441 ± 0.0058` | **0.6011** / **0.8432** | 0.8462 / 0.6545 | 0.7438 / 0.7596 | 0.3508 / 0.3846 *(Bayes: **0.6433**)* |
| **ViSoBERT** | 0.7476 `[0.7176, 0.7748]` | `0.7450 ± 0.0068` | 0.5996 / 0.8344 | **0.8474** / 0.6578 | 0.7457 / 0.7547 | 0.3730 / 0.4094 *(Q-Rank: **0.6373**)* |
| **ViSoBERT + SupCon** | 0.7471 `[0.7172, 0.7751]` | **`0.7455 ± 0.0052`** | 0.5979 / 0.8352 | 0.8459 / **0.6613** | 0.7505 / **0.7601** | 0.3650 / 0.3989 *(Q-Rank: **0.6380**)* |
| **ViSoBERT + NCNL** | 0.7465 `[0.7150, 0.7749]` | `0.7437 ± 0.0037` | 0.5961 / 0.8359 | 0.8411 / 0.6516 | 0.6985 / 0.7061 | 0.3828 / 0.4180 |
| **PhoBERT + BERTopic (806-d)** | 0.7302 `[0.6974, 0.7606]` | `0.7236 ± 0.0073` | 0.5602 / 0.8373 | 0.8184 / 0.6111 | 0.5678 / 0.5670 | 0.3460 / 0.3794 |
| **PhoBERT-base** | 0.7265 `[0.6977, 0.7550]` | `0.7230 ± 0.0146` | 0.5756 / 0.8096 | 0.8356 / 0.6208 | 0.6992 / 0.7078 | 0.3620 / 0.3978 *(Q-Rank: **0.6189**)* |
| **TF-IDF + Logistic Regression** | 0.7138 `[0.6840, 0.7412]` | `0.7036 ± 0.0143`$\dagger$ | 0.5626 / 0.7936 | 0.8297 / 0.5954 | 0.6285 / 0.6248 | 0.3577 / 0.3909 |
| **TF-IDF + LinearSVC** | 0.7025 `[0.6721, 0.7327]` | `0.6994 ± 0.0164`$\dagger$ | 0.5289 / 0.8038 | 0.7927 / 0.5316 | 0.5869 / 0.5898 | 0.3798 / 0.4069 |
| **XLM-RoBERTa-base** | 0.6817 `[0.6539, 0.7086]` | `0.6769 ± 0.0192` | 0.5385 / 0.7462 | 0.8351 / 0.6162 | **0.7693** / 0.7583 | 0.4267 / 0.4480 *(Bayes: **0.6587**; Q-Rank: **0.6276**)* |
| **BiLSTM (Class-Weighted)** | 0.6507 `[0.6219, 0.6794]` | `0.6388 ± 0.0073` | 0.4639 / 0.7505 | 0.7671 / 0.5074 | 0.5640 / 0.5876 | 0.3824 / 0.4220 |
| **Lexical Weak Labeler** | 0.6448 `[0.6153, 0.6753]` | — *(Pilot $n=343$: `0.8092`)* | 0.4349 / 0.7688 | 0.6546 / 0.3321 | 0.5732 / 0.5774 | 0.3427 / 0.3731 |
| **Qwen2.5-1.5B (Rubric + 4-Shot)** | 0.5594 `[0.5310, 0.5859]` | — *(Zero-shot: `0.2867`)* | 0.3845 / 0.6287 | 0.6739 / 0.3428 | 0.6166 / 0.5943 | 0.3943 / 0.4119 |
| **Inductive BERTopic-Only (74-d)** | 0.5539 `[0.5210, 0.5865]` | — | 0.3730 / 0.6273 | 0.6475 / 0.2775 | 0.6083 / 0.6153 | **0.5025** / **0.5235** |
| **All-Negative Trivial Baseline** | 0.4452 | — | 0.0000 / 0.8023 | 0.5000 / 0.1977 | 0.5000 / 0.5000 | 0.3333 / 0.3643 |

- **Statistical Breakthrough Over Lexical & Multilingual Ceilings:** `ViSoBERT + LA-Focal` (`+3.80` points, paired bootstrap $p = 0.0045$, exact McNemar $p < 0.0001$), `ViSoBERT` (`+3.39`, $p = 0.0090$), `ViSoBERT + SupCon` (`+3.33`, $p = 0.0080$), and `ViSoBERT + NCNL` (`+3.28`, $p = 0.0145$) statistically surpass `TF-IDF + Logistic Regression` (`0.7138`), formal-register `PhoBERT-base` (`0.7265`, LA-Focal McNemar $p = 0.0007$), and generic multilingual `XLM-RoBERTa-base` (`0.6817`, `+6.59` to `+7.01` points despite `3×` more parameters, $p < 0.0001$).
- **Matched-Budget Active Learning (`400 → 6,392`):** Least-confidence uncertainty sampling improves test Macro-F1 by **`+4.58` to `+5.35` points** over matched-budget random selection at `25%–50%` pool budgets.
- **Label-Free Cross-Corpus Calibration (`UIT-VSMEC Sadness-Only`, Zero Target Labels):** Static Bayes ($\tau_{\text{Bayes}} = 0.1974$) lifts prior-decoupled `ViSoBERT + LA-Focal` from `0.3826` to **`0.6433`** (`+0.2607`) and `XLM-RoBERTa-base` to **`0.6587`**, while unsupervised Source-Rank Quantile calibration ($Q_{80.3\%}$) equalizes all five pretrained encoders within **`0.6189–0.6380`**.

---

## 🔬 Reproducing Results

### 1. Inspect Public Tier-1 Split Manifests Immediately (No DUA Required)
```bash
git clone https://github.com/iannwendy/VietDepression.git
cd VietDepression
pip install -r requirements.txt

# Audit repository privacy compliance and inspect public SHA-256 manifests
python3 scripts/build_and_verify_dua_manifests.py --audit-git
```

```python
import pandas as pd

# Verify split sizes, class priors, word-length distributions, and weak-labeler test F1 (0.6448)
manifest = pd.read_csv("data/manifests/final_dataset_manifest.csv")
print(manifest.groupby(["split", "label"]).size())
```

### 2. Full Model Training & Evaluation (With Tier-2 DUA Text)
1. Obtain the Tier-2 normalized text archive (`final_train.csv`, `final_val.csv`, `final_test.csv`, `final_dataset.csv`) under [`DATA_USE_AGREEMENT.md`](DATA_USE_AGREEMENT.md) and place the files in `data/labeled/` (git-ignored).
2. Cryptographically verify `100%` SHA-256 hash alignment and `0%` cross-split overlap:
   ```bash
   python3 scripts/build_and_verify_dua_manifests.py --verify-local
   ```
3. Run the strictly inductive training, statistical significance, active-learning, calibration, and error-taxonomy pipelines:
   ```bash
   PYTHONPATH="$PWD" python3 scripts/run_final_round6_evaluation.py
   PYTHONPATH="$PWD" python3 scripts/evaluate_bertopic_round6.py
   PYTHONPATH="$PWD" python3 scripts/run_visobert_and_ablations.py
   PYTHONPATH="$PWD" python3 scripts/run_top_venue_baselines.py
   PYTHONPATH="$PWD" python3 scripts/audit_error_taxonomy.py
   PYTHONPATH="$PWD" python3 scripts/update_kais_revision_figures.py
   ```
