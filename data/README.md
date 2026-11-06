# VietDepression Data Directory & Two-Tier Governance

**Official Repository:** `https://github.com/iannwendy/VietDepression`

---

## 🔒 Privacy & Two-Tier Access Policy

Because verbatim mental-health disclosures in public comments can be reverse-searched to identify individual posters, **VietDepression** strictly separates public de-identified metadata (**Tier 1**) from controlled-access normalized comment text (**Tier 2**) in accordance with CLPsych/eRisk ethics standards and [`DATA_USE_AGREEMENT.md`](../DATA_USE_AGREEMENT.md).

- **Zero raw YouTube comment text (`comment_text`), `comment_id`, or `video_id` CSVs are tracked in Git.** All local working CSVs under `data/labeled/`, `data/raw/`, `data/round*/`, `data/model_predictions/`, and `data/augmented_*/` are excluded via `.gitignore`.

---

## 📂 Directory Layout

### 1. `data/manifests/` — Tier 1 Public Release (Tracked in Git)
Contains de-identified cryptographic split manifests enabling immediate verification of split sizes, class balances, annotation-round provenance, lexical weak-score distributions, and length statistics without exposing raw text:

| Manifest File | Rows | Class Support | Description |
|---------------|-----:|---------------|-------------|
| `final_dataset_manifest.csv` | `9,134` | `1,804` P (`19.75%`) / `7,330` N (`80.25%`) | Complete human-adjudicated gold benchmark (`0%` duplicate SHA-256 hashes) |
| `final_train_manifest.csv` | `6,392` | `1,262` P (`19.74%`) / `5,130` N (`80.26%`) | Training split (`70%`) — sole split used for inductive fitting |
| `final_val_manifest.csv` | `1,371` | `271` P (`19.77%`) / `1,100` N (`80.23%`) | Validation split (`15%`) — checkpoint & hyperparameter selection |
| `final_test_manifest.csv` | `1,371` | `271` P (`19.77%`) / `1,100` N (`80.23%`) | Held-out in-domain test split (`15%`) |
| `uit_vsmec_proxy_manifest.csv` | `3,084` | `1,542` Sad+Fear / `1,542` Enjoy (`2,691` Sad-Only) | External emotion-proxy evaluation manifest (`duwuonline/UIT-VSMEC`) |
| `manifest_integrity_summary.json` | — | — | Machine-readable SHA-256 verification summary |

#### Manifest Schema (`final_*_manifest.csv`)
| Column | Type | Description |
|--------|------|-------------|
| `record_id` | `string` | Canonical benchmark identifier (`VD-TRN-0001` .. `VD-TST-1371`) |
| `split` | `string` | Partition (`train`, `val`, or `test`) |
| `text_sha256` | `string` | SHA-256 digest of Unicode NFC + whitespace-normalized UTF-8 comment text |
| `label` | `int` | Human-adjudicated binary gold label (`1` = depression-related distress, `0` = normal/non-distress) |
| `source` | `string` | Provenance batch (`human_gold_pre_round5`, `human_gold_round5`, `human_gold_round6`) |
| `annotation_round` | `string` | Curation round (`round4_or_earlier`, `round5`, `round6`) |
| `weak_score` | `int` | Additive 335-entry lexicon score $s(x)$ |
| `weak_label` | `string` | Provisional weak label (`depression_auto`, `normal_auto`, or `uncertain`) |
| `has_review_context_phrase` | `bool` | Exact match to the 22 contextual review-flag phrases |
| `need_review_composite` | `bool` | Composite five-condition `need_review` triage flag |
| `char_len` / `word_len` | `int` | Character and whitespace-delimited word counts |

### 2. `data/analysis/` — Aggregate Integrity & Labeling Reports (Tracked in Git)
Contains aggregate JSON audit summaries (`dataset_integrity_report_round6_v2.json`, `labeling_report.json`, `review_eval_report.json`) with zero raw comment text.

### 3. `data/labeled/` — Tier 2 Controlled Academic Access (Git-Ignored Local Directory)
Verified academic researchers who execute [`DATA_USE_AGREEMENT.md`](../DATA_USE_AGREEMENT.md) place the hydrated normalized-text CSVs (`final_train.csv`, `final_val.csv`, `final_test.csv`, `final_dataset.csv`) here and verify cryptographic integrity via:

```bash
python3 scripts/build_and_verify_dua_manifests.py --verify-local
```
