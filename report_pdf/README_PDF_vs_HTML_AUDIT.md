# PDF vs HTML — Audit & Status (Updated 2026-06-28)

> **File audited:** `523H0002_523H0054_CDNC1_W2.pdf` (136 KB, 9 pages) ← original Week 2 PDF
> **Updated file:** `Report (1).pdf` (219 KB, 14 pages) ← new PDF (latest upload)
> **Reference:** `docs/paper_report.html` (~1,615 lines, full 6-chapter draft)
> **Last audit update:** 2026-06-28 (after Report (1).pdf uploaded)

---

## 0. Executive Summary — Status as of 2026-06-28

| Item | Original PDF | Report (1).pdf | HTML ground truth |
|---|---|---|---|
| 1.2.1 = "Overarching Aim" | ❌ "Primary Objectives" | ✅ "Overarching Aim" | ✅ |
| 1.2.3 = "Expected Outcomes" | ❌ "Expected Results" | ✅ "Expected Outcomes" | ✅ |
| 1.2.4 Scope 7 items (incl. Temporal) | ❌ 6 items | ✅ 7 items | ✅ |
| 1.2.4 Limitations as sub-list | ❌ Mixed with Scope | ❌ **Still missing** | ✅ |
| 1.2.2 Obj #4 sample counts (1,786/383/3,084) | ❌ No numbers | ✅ Numbers present | ✅ |
| 1.2.3 BERTopic 456 topics | ❌ Not stated | ✅ Stated | ✅ |
| 1.3 = "six chapters" (1.3.1–1.3.6) | ❌ "five chapters" | ✅ "five chapters" header, but actually 1.3.1–1.3.6 (inconsistency) | ✅ |
| Chapter 3 body content | ❌ Empty | 🟡 **§3.1–§3.4 only** (missing §3.5–§3.7) | ✅ Full §3.1–§3.7 |
| Chapter 4 body content | ❌ Placeholder | ❌ **Still placeholder** | ✅ §4.1–§4.3 |
| Chapter 5 body content | ❌ Placeholder | ❌ **Still placeholder** | ✅ §5.1–§5.5 |
| Chapter 6 body content | ❌ Placeholder | ❌ **Still placeholder** | ✅ §6.1–§6.3 |
| §2.1 Clinical (DSM-5 + PHQ-9) | ❌ Missing | ✅ Present | ✅ |
| §2.2 three-paradigm framing | ❌ 8 awkward subsections | ✅ Lexicon/TradML/DL | ✅ |
| §2.3 Vietnamese NLP + Table 2.1 | ❌ Orphan §2.2.8 | ✅ Table 1 (8 datasets) | ✅ |
| §2.4 Gap statement | ❌ Missing | ✅ Present | ✅ |
| Table 3.1 (Raw data schema) | ❌ Missing | ✅ Table 3 | ✅ |
| Table 3.2 (External datasets) | ❌ Missing | ✅ Table 4 | ✅ |
| Table 3.3 (Affect signal) | ❌ Missing | ✅ Table 5 | ✅ |
| Table 3.4 (Cross-tab) | ❌ Missing | ✅ Table 6 | ✅ |
| §3.5 Weak-labeling (formula, audit) | ❌ Missing | ❌ **Still missing** | ✅ |
| §3.6 Blind annotation (Cohen's κ) | ❌ Missing | ❌ **Still missing** | ✅ |
| §3.7 Final dataset (1,786/383/3,084 split) | ❌ Missing | ❌ **Still missing** | ✅ |
| Abstract numbers (post-round-3) | ❌ Old (0.9623 / 0.4318) | ❌ **Still old (0.9623 / 0.4318)** | ✅ Updated to 0.8681±0.0086 / 0.3727±0.0242 |
| §1.2.3 numbers (post-round-3) | ❌ Old | ❌ **Still old (0.9623 / 0.4318)** | ✅ Same numbers |
| §6.1 Related Works citations (Zhou, Kumar, …) | ❌ Missing | ❌ **§II does not exist; only §2.1 Clinical + §2.2 + §2.3 + §2.4** | ✅ |
| Reference [29] font glyph bug | ❌ "Bọ Y tế", "Tỏng quan vè" | ❌ **Still present** (page 14) | ✅ Fixed in HTML |
| §1.3.3 "11,703-sample intermediate" (old number) | n/a | ❌ Still present | ✅ Updated |
| §1.3.5 "461 discovered topics" (old number) | n/a | ❌ Still present | ✅ Updated to 456 |

**Net change from original PDF:** Report (1).pdf adds ~30% of the missing content (chapter structure fix, §1.2 restructure, §2.1–§2.4 background, partial Chapter 3 §3.1–§3.4 with 4 tables). But **8 critical items remain unfixed** before this is submission-ready.

---

## 1. Status: Items now DONE in Report (1).pdf ✅

These were flagged in the original audit and have been resolved:

### 1.1. §1.2.1 numbering → "Overarching Aim" ✅
PDF page 2: "1.2.1. Overarching Aim"

### 1.2. §1.2.3 → "Expected Outcomes" ✅
PDF page 3: "1.2.3. Expected Outcomes"

### 1.3. §1.2.4 Scope has all 7 items including Temporal ✅
PDF page 3–4 lists: Platform scope, Language scope, Comment type, Classification granularity, Ethical data handling, Model scope, **Temporal scope** (May–June 2026).

### 1.4. §1.2.2 Obj #4 sample counts ✅
PDF page 2: "the final training set (**1,786 samples, post round-3 review merge, stratified**). Evaluate each model on two independent test sets: an in-domain test set from the YouTube distribution (**383 samples**) and a fixed cross-domain test set (VSMEC, **3,084 samples**)"

### 1.5. §1.2.3 BERTopic 456 topics ✅
PDF page 3: "A BERTopic model comprising **456 topics (verified against `models/bertopic/bertopic_metrics.json`)**"

### 1.6. §1.3 split into 6 chapters ✅ (with one inconsistency)
- §1.3.1 Chapter 1: Introduction
- §1.3.2 Chapter 2: Background and Related Work
- §1.3.3 Chapter 3: Data Acquisition, Processing, and Annotation
- §1.3.4 Chapter 4: Model Training and Evaluation Methodology
- §1.3.5 Chapter 5: Results and Discussion
- §1.3.6 Chapter 6: Conclusion and Future Work

⚠️ **Minor inconsistency:** §1.3 opening line still says "**five chapters**" but the subsections list 6. This should be changed to "six chapters".

### 1.7. §II Background — major restructure ✅
- **§2.1 Depression: Clinical Conceptualization and Screening Instruments** (DSM-5 9 symptoms + PHQ-9 cutoff ≥10, sens/spec 88%)
- **§2.2 NLP Approaches** with three-paradigm framing (Lexicon-based / Traditional ML / Deep learning)
- **§2.3 Vietnamese NLP** with **Table 1** (8 external datasets)
- **§2.4 The Gap This Work Addresses** (3 gaps: dataset, blind protocol, cross-domain)

### 1.8. Chapter 3 partial content ✅
- §3.1 Data Acquisition Strategy (§3.1.1 Platform, §3.1.2 Keyword Design 5 groups, §3.1.3 YouTube API v3, §3.1.4 Crawler Architecture, §3.1.5 Raw Data Schema)
- §3.2 Data Preprocessing and Cleaning (§3.2.1 Normalization, §3.2.2 Spam Detection 4 heuristics, §3.2.3 Deduplication, §3.2.4 Min Length)
- §3.3 External Dataset Integration (§3.3.1 Rationale, §3.3.2 Dataset Catalog Table 4, §3.3.3 Affect Signal Mapping Table 5, §3.3.4 Cross-Tab Table 6)
- §3.4 Unified Corpus Construction (§3.4.1 Three-Tier)

Tables present: Table 3 (Raw data schema), Table 4 (External datasets with HF IDs), Table 5 (Affect signal derivation), Table 6 (Cross-tabulation).

---

## 2. Status: Items STILL MISSING in Report (1).pdf ❌

### 2.1. 🔴 Abstract numbers are still stale (pre-round-3)

**PDF Abstract (page 1):**
> "Results demonstrate that PhoBERT achieves an F1-macro of **0.9623** on the in-domain test set and **0.4318** on the cross-domain VSMEC test set, outperforming TF-IDF+SVM (**0.9312/0.4286**), BiLSTM (**0.8951/0.4272**), and BERTopic-only (0.5599/0.5030)."

**HTML ground truth (Table 5.1):**
> PhoBERT in-domain: 0.8681 ± 0.0086
> PhoBERT cross-domain: 0.3727 ± 0.0242
> BiLSTM random: 0.8145 ± 0.0244 / 0.4690 ± 0.0601
> BiLSTM PhoBERT-frozen: 0.8244 ± 0.0044 / 0.4344 ± 0.0008

**Required fix:** Replace all five numbers in the PDF Abstract with the post-round-3 numbers from `models/*/metrics.json`.

### 2.2. 🔴 §1.2.3 Expected Outcomes — PhoBERT numbers still stale

**PDF page 3:**
> "A comprehensive benchmark of five model architectures on two test domains, establishing the first published performance figures for Vietnamese depression detection—**PhoBERT F1-macro: 0.9623 (in-domain), 0.4318 (cross-domain)**."

**Required fix:** Same as 2.1 — use post-round-3 numbers.

### 2.3. 🔴 Chapter 3 missing §3.5, §3.6, §3.7

The PDF currently has Chapter 3 body = §3.1 + §3.2 + §3.3 + §3.4. But the HTML has 7 sections. Missing:

- **§3.5 Weak-Labeling via Weighted-Keyword Scoring** — most critical missing content. Includes:
  - §3.5.1 Lexicon Design (Table 3.6: 4 categories, 53+116+166+22 entries)
  - §3.5.2 Keyword Preparation and Accent Handling
  - §3.5.3 Scoring Function and Classification Thresholds (Table 3.7 + math formula)
  - §3.5.4 Review-Flagging Heuristics (5 conditions)
  - §3.5.5 Label Distribution Analysis (Table 3.8)
  - §3.5.6 Weak-Labeler Performance Audit (Table 3.9: precision 0.61, recall 0.98, F1-dep 0.75)

- **§3.6 Blind Human Annotation Protocol** — includes Cohen's Kappa = 0.63 (Batch 5) and −0.03 (Batch 8) finding, Table 3.10, Table 3.11 (1,607 gold samples)

- **§3.7 Final Training Dataset Assembly** — includes Table 3.12 (4 sources with weights), Table 3.13 (1,786 / 383 / 383 split)

### 2.4. 🔴 Chapter 4, 5, 6 are still placeholders

PDF page 13:
```
IV. MODEL TRAINING AND EVALUATION METHODOLOGY
V. RESULTS AND DISCUSSION
VI. CONCLUSION AND FUTURE WORK
```
All three are headings with **no body content**.

**Required fix:** Insert:
- **Chapter 4:** §4.1 Pipeline Overview + ASCII diagram + §4.2 Model Architectures (5 models) + §4.3 Evaluation Strategy
- **Chapter 5:** §5.1 Comparative Table 5.1 + §5.2 Generalization Gap Analysis (4 causes) + §5.3 BERTopic Results (Table 5.2) + §5.4 Error Analysis + §5.5 DAPT counter-experiment (Table 5.3)
- **Chapter 6:** §6.1 Summary of 5 Contributions + §6.2 Limitations + §6.3 Future Work (8 directions)

### 2.5. 🟠 §1.2.4 Limitations as sub-list still missing

PDF §1.2.4 has 7 Scope bullets, but **no separate Limitations sub-list**. The HTML §1.2.4 contains:
```
1.2.4. Scope and Limitations
   Scope: (a) Platform, (b) Language, (c) Comment type, (d) Classification,
          (e) Ethical data handling, (f) Model scope, (g) Temporal scope
   Limitations: 1. Gold set size, 2. Class imbalance, 3. Single annotator,
                4. Unresolved domain adaptation, 5. Single-platform scope
```

**Required fix:** Add a "**Limitations:**" sub-heading after the Scope bullets, then a 5-item numbered list.

### 2.6. 🟠 §3.4 missing §3.4.2 and §3.4.3

PDF has only §3.4.1 Three-Tier Corpus Architecture. HTML has:
- §3.4.1 Three-Tier Corpus Architecture
- §3.4.2 **Leakage Prevention Strategy** (5 measures)
- §3.4.3 **Corpus Manifest** (Table 3.5)

### 2.7. 🟠 §2.1 Related Works (Zhou, Kumar, Mishra, …) still not present

The original audit flagged this in §6.6. Report (1).pdf has §2.1 Clinical + §2.2 + §2.3 + §2.4, but **does not have a §2.5 Related Works** with the 12 cited papers (Zhou 2024, Kumar 2024, Mishra 2024, García-Díaz 2024, Le 2023, Gururangan 2020, Wang 2021, Li 2023, Rahman 2024, Thompson 2024, etc.).

This is debatable: the new PDF structure (§2.1 Clinical, §2.2 NLP, §2.3 Vietnamese, §2.4 Gap) actually **replaces** the related-works-by-paper format with a more thematic structure. If the team agrees with this restructure, the audit §6.6 can be closed by **officially retiring §2.1 Related Works** and adding the missing papers to the References list with a footnote that "see References for full citation list".

### 2.8. 🟡 Reference [29] font glyph bug still present

PDF page 14:
> "[29] **Bọ Y tế** Việt Nam [Vietnam Ministry of Health]. (2022). Báo cáo **Tỏng quan vè** Súc khỏe Tâm thàn tại Việt Nam [Overview Report on Mental Health in Vietnam]."

Three glyph errors:
- `Bọ` → `Bộ`
- `Tỏng` → `Tổng`
- `vè` → `về`
- `Súc` → `Sức`
- `Tâm thàn` → `Tâm thần`

(Per audit §8.1)

### 2.9. 🟡 §1.3.3 says "eleven thousand seven hundred three-sample intermediate"

PDF page 4: "...an earlier **11,703-sample intermediate** was superseded". The HTML does not use this number — verify against `docs/ROADMAP_SAU_REVIEW.md` and update.

### 2.10. 🟡 §1.3.5 says "461 discovered topics"

PDF page 4: "BERTopic results with **461 discovered topics** and five depression-relevant thematic clusters". HTML §5.3 has **456 topics** (verified against `models/bertopic/bertopic_metrics.json`).

**Required fix:** Change 461 → 456.

---

## 3. Cosmetic Issues (🟢)

| # | Location | Current | Should be | Status |
|---|---|---|---|---|
| 1 | §1.3 opening | "This report is organized into **five chapters**" | "**six chapters**" | ❌ Not fixed |
| 2 | §1.2.4 | No "Limitations" sub-heading | Add "Limitations:" sub-list | ❌ Not fixed |
| 3 | §3.4.1 item 3 | "42,952 texts from external datasets" — accurate | OK | ✅ |

---

## 4. Action Plan — Updated Priority Order

Based on Report (1).pdf status:

| Priority | Task | Severity | Effort | Note |
|---|---|---|---|---|
| 1 | Fix Abstract numbers (0.9623→0.8681, etc.) | 🔴 | 5 min | You said you are fixing this |
| 2 | Fix §1.2.3 Expected Outcomes numbers | 🔴 | 2 min | Same edit as #1 |
| 3 | Insert Chapter 3 §3.5, §3.6, §3.7 | 🔴 | 1–2 hours | Use HTML §3.5–§3.7 as source |
| 4 | Insert Chapters 4, 5, 6 body content | 🔴 | 2–3 hours | Use HTML §4, §5, §6 |
| 5 | Add §1.2.4 Limitations sub-list | 🟠 | 5 min | Use HTML §1.2.4 Limitations |
| 6 | Add §3.4.2, §3.4.3 | 🟠 | 10 min | Use HTML §3.4.2, §3.4.3 |
| 7 | Fix "five chapters" → "six chapters" | 🟢 | 1 min | Trivial typo |
| 8 | Fix §1.3.5 "461" → "456" | 🟡 | 1 min | Match HTML §5.3 |
| 9 | Verify "11,703-sample intermediate" number | 🟡 | 5 min | Check ROADMAP |
| 10 | Fix Reference [29] font glyph | 🟡 | 2 min | Bộ / Tổng / về / Sức / Tâm thần |

**Total estimated time to submission-ready:** ~4–6 hours of focused work.

**Recommended approach:** Once items 1+2 are fixed, regenerate the PDF from `docs/paper_report.html` (which now contains all the corrected content) using Chrome headless or `wkhtmltopdf`. This avoids manual patching of Ch 3–6.

---

## 5. Files Referenced

- Original PDF: `report_pdf/523H0002_523H0054_CDNC1_W2.pdf`
- Updated PDF: `report_pdf/Report (1).pdf` (14 pages, 219 KB)
- HTML source: `docs/paper_report.html` (1,615 lines)
- BERTopic metrics: `models/bertopic/bertopic_metrics.json` (456 topics, 48.30% outliers)
- DAPT results (current): `results/domain_adapted_eval_2026-06-26_181310/`
- Phase 2 final dataset: `phase2_report.json`
- Phase 3 comparison: `phase3_comparison_report.json`
- DAPT regression test: `tests/test_dapt_eval_helpers.py` (`TestRunFinetuneDataPath`)
- Label Studio batches: `label_studio_step5_review_key_MERGED.csv`, `label_studio_step8_active_learning_key_MERGED.csv`, `label_studio_round3_active_learning_key_MERGED.csv`

---

*End of audit. Last updated 2026-06-28 after `Report (1).pdf` was uploaded.*