# VietDepression Academic Data Use Agreement (DUA) & Two-Tier Release Policy

In accordance with international mental-health NLP privacy standards established by **CLPsych** (Coppersmith et al., 2015) and **eRisk** (Losada et al., 2018), as well as institutional research ethics guidelines at **Ton Duc Thang University** and the **YouTube API Services Terms of Service**, the **VietDepression** benchmark is governed under a **Two-Tier Release Protocol** designed to prevent reverse-search re-identification of social-media posters while ensuring full scientific reproducibility.

---

## 1. Two-Tier Release Structure

### Tier 1 — Open Public Release (Included in This Repository)
Available immediately without restriction at `https://github.com/iannwendy/VietDepression`:
- **De-Identified Split Manifests (`data/manifests/`):**
  - `final_dataset_manifest.csv` (`9,134` gold records; `1,804` positive [`19.75%`], `7,330` negative [`80.25%`])
  - `final_train_manifest.csv` (`6,392` training records; `1,262` positive, `5,130` negative)
  - `final_val_manifest.csv` (`1,371` validation records; `271` positive, `1,100` negative)
  - `final_test_manifest.csv` (`1,371` held-out test records; `271` positive, `1,100` negative)
  - `uit_vsmec_proxy_manifest.csv` (`3,084` Full / `2,691` Sadness-Only external evaluation hashes)
  - `manifest_integrity_summary.json` (cryptographic audit of `0%` cross-split overlap)
  - *Note:* Each manifest provides `record_id`, `split`, `text_sha256` (SHA-256 over Unicode NFC-normalized text), `label`, `annotation_round`, `weak_score`, `weak_label`, `char_len`, and `word_len`—enabling direct verification of all dataset statistics, per-round provenance counts, length/score distributions (Fig. 2), and full-test weak-labeler metrics (Table 4) **without exposing raw comment text or YouTube IDs**.
- **Weak-Labeling Lexicon & Contextual Flags:** `scripts/vietdepression_lexicon.py` and `docs/appendices/APPENDIX_KEYWORDS.md` (`335` weighted entries + `22` contextual review flags).
- **Annotation Rubric & Guidelines:** `docs/ANNOTATION_GUIDELINES.md` (Four-Rule English Rubric) and `docs/appendices/manual_review/HUONG_DAN_REVIEW_LABEL_STUDIO.md`.
- **Training, Calibration & Statistical Audit Code:** All scripts under `scripts/` and audited result summaries under `results/scientific_audit/`.

### Tier 2 — Controlled Academic Release (Governed by Signed DUA)
Available free of charge to **verified academic and non-profit researchers** upon submission of the signed Data Use Agreement below:
- Normalized UTF-8 Vietnamese comment text (`comment_text`) hydrated into the canonical split files (`data/labeled/final_train.csv`, `data/labeled/final_val.csv`, `data/labeled/final_test.csv`, `data/labeled/final_dataset.csv`), matching `100%` of the `9,134` SHA-256 hashes in `data/manifests/` (`python3 scripts/build_and_verify_dua_manifests.py --verify-local`).
- Upstream author handles, channel profiles, and raw YouTube comment/video URLs are permanently excluded from distribution.

---

## 2. Terms and Conditions of Access (Tier 2)

By requesting and receiving the Tier-2 normalized text archive of **VietDepression**, the Principal Investigator (PI) and affiliated Recipient Institution agree to all of the following binding conditions:

1. **Non-Commercial Academic Research Only:** The dataset shall be used exclusively for non-commercial scientific research, academic benchmarking, and educational reproducibility in natural language processing and computational social science.
2. **Strict Prohibition on Redistribution:** The Recipient shall **not** upload, publish, mirror, or redistribute raw or normalized comment text in any public GitHub repository, Hugging Face dataset hub, Kaggle competition, open web server, or third-party LLM training API.
3. **Prohibition on Re-Identification & User Profiling:** The Recipient shall **not** attempt to reverse-search comment strings on YouTube or search engines, link comments to user accounts, infer personal identities, contact authors, or construct individual user-level psychological profiles.
4. **No Verbatim Quotation in Publications:** Published papers, preprints, slides, and error analyses must present **privacy-preserving paraphrased composites** (as demonstrated in Table 7 of the VietDepression manuscript and `docs/appendices/APPENDIX_ERRORS.md`) rather than verbatim comment strings.
5. **Non-Clinical Scope:** Labels in VietDepression represent an operational text-level annotation construct (depression-related distress language in isolated utterances) and **must not** be deployed for automated clinical diagnosis, medical triage, surveillance, or high-stakes decision-making without qualified human clinical oversight.
6. **Right-to-Deletion Compliance & Data Security:** Data files must be stored on access-controlled institutional storage (`data/labeled/` is git-ignored by default). If the VietDepression maintainers issue a hash-takedown notice for deleted upstream content, the Recipient agrees to purge the corresponding `text_sha256` rows promptly.

---

## 3. How to Request Tier-2 Academic Access

Verified faculty members, postdoctoral researchers, or supervised graduate/undergraduate researchers (countersigned by their academic advisor) should send an email from their **official institutional email address** (e.g., `.edu`, `.edu.vn`, `.ac.uk`, `.ac.jp`) to the Corresponding Author:

- **Corresponding Author:** Dr. Phuoc Tran (`tranthanhphuoc@tdtu.edu.vn`)
- **Affiliation:** Natural Language Processing and Knowledge Discovery Group, Faculty of Information Technology, Ton Duc Thang University, Ho Chi Minh City, Vietnam
- **Email Subject:** `[VietDepression DUA Request] - <Institution Name> - <PI Full Name>`

### Required DUA Declaration Block (Copy, Complete, and Sign via Institutional Email / PDF)

```text
VIETDEPRESSION ACADEMIC DATA USE AGREEMENT (DUA)

1. Principal Investigator / Advisor Full Name:
2. Academic Title / Position:
3. Institution & Department:
4. Official Institutional Email:
5. Institutional Webpage / Google Scholar / ORCID URL:
6. Co-Researchers / Supervised Students Covered by This Request (Names & Institutional Emails):
7. Intended Research Purpose (2–4 sentences):

Binding Attestation:
I certify that I am an academic researcher at the institution named above and that all listed
team members have read and agree to abide by the six (6) Terms and Conditions of the
VietDepression Data Use Agreement, including strict prohibitions on public redistribution,
reverse-search re-identification, user profiling, clinical diagnostic deployment without human
oversight, and verbatim quotation of user comments.

Signature (PI / Faculty Advisor): ___________________________   Date: _______________
```

### Cryptographic Verification After Receiving Tier-2 Files

Once the Tier-2 split files (`final_train.csv`, `final_val.csv`, `final_test.csv`, `final_dataset.csv`) are placed in `data/labeled/` (which is excluded from Git via `.gitignore`), run:

```bash
python3 scripts/build_and_verify_dua_manifests.py --verify-local
```

This verifies that all `9,134` records match the public SHA-256 manifests (`6,392` train / `1,371` val / `1,371` test) with `0%` cross-split text overlap prior to running model training and evaluation.
