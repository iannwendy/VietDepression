# PDF Correction Checklist - vs HTML Source & Actual Ground Truth

> **Generated:** 2026-07-04
> **PDF:** `Report (2).pdf` (23 pages, 437 KB)
> **Reference:** `docs/paper_report.html` (~1,734 lines)
> **Ground Truth:** JSON reports, dataset manifests, and actual script outputs

---

## 0. Executive Summary

| Category | Issues Found | Severity |
|----------|-------------|----------|
| Author/University Info | 2 | 🔴 CRITICAL |
| Chapter Content | 3 chapters missing | 🔴 CRITICAL |
| Dataset Sizes | 4 incorrect numbers | 🔴 CRITICAL |
| Model Results Table | Table 5.1 missing | 🔴 CRITICAL |
| Figures | 7 figures missing | 🔴 CRITICAL |
| References | Font issue + missing | 🟠 MODERATE |
| Other Numbers | 3 discrepancies | 🟡 MINOR |

---

## 1. AUTHOR & AFFILIATION ERRORS 🔴

### 1.1 Author Names

| Location | PDF says | Should be |
|----------|---------|-----------|
| Page 1 | `NGUYEN DUC ANH` | Single author only |
| Page 1 | `NGUYEN BAO MINH` | **Bao Minh Nguyen** |

**Issue:** PDF shows two authors but project is a single-author capstone by Bao Minh Nguyen.

### 1.2 University Affiliation

| Location | PDF says | Should be |
|----------|---------|-----------|
| Page 1 | `Ton Duc Thang University, Faculty of Information Technology` | `University of Information Technology, VNU-HCM` |
| Student IDs | `523H0002` and `523H0054` | Only `523H0054` |

**Issue:** Wrong university and only one student ID should appear.

### 1.3 Title Page Format

| Element | PDF | Should be |
|---------|-----|-----------|
| Author line | `NGUYEN DUC ANH` + `NGUYEN BAO MINH` | `Bao Minh Nguyen` |
| Affiliation | Ton Duc Thang | UIT, VNU-HCM |
| Email | `523h0002@student.tdtu.edu.vn` + `523h0054@...` | Only `523h0054@student.hcmiu.edu.vn` |
| Date | `June 21st` | `Academic Year 2025–2026` |

---

## 2. MISSING CHAPTER CONTENT 🔴

### 2.1 Chapters 4, 5, 6 Are Empty Placeholders

| Chapter | PDF Status | HTML Status |
|---------|-----------|-------------|
| IV. Model Training and Evaluation Methodology | ❌ EMPTY (just heading) | ✅ Full content |
| V. Results and Discussion | ❌ EMPTY (just heading) | ✅ Full content |
| VI. Conclusion and Future Work | ❌ EMPTY (just heading) | ✅ Full content |

**PDF Page 22:**
```
IV. MODEL TRAINING AND EVALUATION METHODOLOGY
V. RESULTS AND DISCUSSION
VI. CONCLUSION AND FUTURE WORK
References
```

**Required:** Insert full content from HTML sections 4, 5, 6.

---

## 3. DATASET SIZE ERRORS 🔴

### 3.1 Gold Set Size

| Source | Value | Status |
|--------|-------|--------|
| PDF (Abstract, §1.2.3) | 1,607 | ❌ OUTDATED |
| HTML (§1.2.3) | 1,607 | ✅ OK (pre-Round 4) |
| **Actual Round 4** | **3,020** | ✅ CURRENT |

**Fix locations in PDF:**
- Abstract: "1,607-sample independently annotated gold subset"
- §1.2.3 Expected Outcomes: "a 1,607-sample independently validated binary labels"

### 3.2 Final Dataset Size

| Source | Value | Status |
|--------|-------|--------|
| PDF (implicit via 1,786 train) | ~2,553 | ❌ OUTDATED |
| HTML (§3.7.3) | 1,786 / 383 / 383 | ✅ Pre-Round 4 |
| **Actual Round 4** | **4,255 / 912 / 912** | ✅ CURRENT |

**Fix locations in PDF:**
- All references to "1,786 samples" → "4,255 samples (Round 4)"
- All references to "383 samples" → "912 samples (Round 4)"
- All references to "2,553" → "6,079"

### 3.3 Training Split in Table 15

| PDF Page 21 | Value | Should be |
|-------------|-------|-----------|
| Train | 1,786 | **4,255** |
| Validation | 383 | **912** |
| Test (in-domain) | 383 | **912** |

### 3.4 Dataset Descriptions Throughout PDF

| Section | PDF Value | Should be |
|---------|---------|-----------|
| Abstract | "over 125,000 Vietnamese YouTube comments plus roughly 191,000 additional texts" | ✅ OK |
| §3.7.1 | "final_dataset.csv" references old numbers | Update all references |

---

## 4. MODEL RESULTS TABLE MISSING 🔴

### 4.1 Table 5.1 Not Present

The PDF does NOT contain Table 5.1 (Comparative Model Performance). This is the **most critical missing element**.

**PDF has NO comparison table for:**
- TF-IDF + LogReg
- TF-IDF + LinearSVC
- BiLSTM (random embedding)
- BiLSTM (PhoBERT-frozen)
- PhoBERT
- BERTopic-only
- PhoBERT + BERTopic
- PhoBERT + DAPT

### 4.2 Required Table 5.1 Content (from HTML)

```
Model                          | In-domain F1  | Cross-domain F1 | Seeds
-------------------------------|---------------|-----------------|------
TF-IDF + LogReg                | 0.8415        | 0.3780          | 1
TF-IDF + LinearSVC             | 0.8799        | 0.3574          | 1
BiLSTM (random embedding)      | 0.8145±0.0244 | 0.4690±0.0601   | 3
BiLSTM (PhoBERT-frozen)        | 0.8244±0.0044 | 0.4344±0.0008   | 3
PhoBERT (original)             | 0.8417±0.0220 | 0.3850±0.0219   | 3
BERTopic-only                  | 0.5599         | 0.5030           | -
PhoBERT + BERTopic             | 0.8497         | 0.4406           | -
PhoBERT + DAPT                 | 0.8803±0.0030 | 0.3620±0.0188   | 3

Post Round 4 Dataset (6,079 samples):
TF-IDF + LogReg                | 0.8415        | 0.3780          | 1
TF-IDF + LinearSVC             | 0.8799        | 0.3574          | 1
PhoBERT                        | 0.8417±0.0220 | 0.3850±0.0219   | 3

After Data Augmentation:
TF-IDF + LogReg (augmented)    | 0.9525        | 0.4280          | 1
TF-IDF + LinearSVC (augmented) | 0.9678        | 0.4006          | 1
BiLSTM (random, augmented)      | 0.6716±0.1900 | 0.4796±0.1035   | 3
PhoBERT (augmented)             | 0.9619        | 0.3993          | 1
BERTopic-only (augmented)       | 0.5864        | 0.5022          | -
PhoBERT + BERTopic (augmented) | 0.9377        | 0.5262          | -
```

### 4.3 PhoBERT Specific Numbers

| Metric | PDF Abstract | HTML & Actual |
|--------|-------------|---------------|
| In-domain F1 | 0.8681±0.0086 | ✅ CORRECT (pre-Round 4) |
| Cross-domain F1 | 0.3727±0.0242 | ✅ CORRECT (pre-Round 4) |
| **Post Round 4 in-domain** | Not mentioned | **0.8417±0.0220** |
| **Post Round 4 cross-domain** | Not mentioned | **0.3850±0.0219** |

---

## 5. MISSING FIGURES 🔴

### 5.1 Figures NOT in PDF (should be present)

| Figure | Description | HTML Reference |
|--------|------------|---------------|
| Figure 1.1 | Conceptual framework diagram | ✅ Referenced in §1.1 |
| Figure 2.1 | Method taxonomy tree | ✅ Referenced in §2.2 |
| Figure 3.1 | Weak-label distribution | ✅ Referenced in §3.5.5 |
| Figure 4.1 | Pipeline overview ASCII diagram | ✅ Present in §4.1 |
| Figure 5.1 | Generalization gap bar chart | ✅ Referenced in §5.2 |
| Figure 5.2 | BERTopic topics bar chart | ✅ Referenced in §5.3 |
| Figure 5.3 | Confusion matrices | ✅ Referenced in §5.4 |
| Figure 5.4 | DAPT counter-experiment scatter | ✅ Referenced in §5.5 |

### 5.2 Figures Actually in PDF

Based on extracted text, the PDF contains:
- Table figures (Table 1-15)
- Possibly weak-label pie chart (mentioned in §3.5.5)

**Missing all 7 publication-quality figures from the report_pdf/figures/ directory.**

---

## 6. REFERENCE ERRORS 🟠

### 6.1 Reference [29] Font Issue

| PDF Page 23 | Issue |
|-------------|-------|
| `[29] Bộ Y tế Việt Nam` | **Font rendering error** |

**Actual text should be:**
```
[29] Bộ Y tế Việt Nam [Vietnam Ministry of Health]. (2022). Báo cáo Tổng quan về Sức khỏe Tâm thần tại Việt Nam [Overview Report on Mental Health in Vietnam].
```

**Note:** Font encoding issue - Vietnamese characters not rendering properly.

### 6.2 Missing References

The HTML references section contains more references than the PDF. Verify all 30+ references are complete.

---

## 7. ADDITIONAL DISCREPANCIES 🟡

### 7.1 Abstract Mention of Comparison Models

| PDF Abstract says | Issue |
|-------------------|-------|
| "TF-IDF+SVM scored 0.9312/0.4286" | Numbers are from Phase 3, pre-Round 4 |
| "BiLSTM scored 0.8951/0.4272" | Numbers are from Phase 3 |
| "BERTopic-only baseline scored 0.5599/0.5030" | ✅ CORRECT |

**Should update to post-Round 4 values:**
- TF-IDF + LogReg: 0.8415 / 0.3780
- TF-IDF + LinearSVC: 0.8799 / 0.3574
- BiLSTM: 0.8145±0.0244 / 0.4690±0.0601

### 7.2 §1.2.3 Expected Outcomes - PhoBERT Numbers

| Location | PDF | Should be |
|---------|-----|-----------|
| Expected Outcomes | "PhoBERT F1-macro: 0.8681±0.0086 (in-domain), 0.3727±0.0242 (cross-domain)" | ✅ OK (pre-Round 4) |
| After Round 4 | Not mentioned | Add: "After Round 4 active learning: 0.8417±0.0220 (in-domain), 0.3850±0.0219 (cross-domain)" |

### 7.3 Table Numbering

| Element | PDF | HTML |
|---------|-----|------|
| Table 1 | External datasets | ✅ CORRECT |
| Tables 2-15 | Present | ✅ CORRECT |
| **Table 5.1** | **MISSING** | ✅ Present |
| Table 5.4 | MISSING | ✅ Present (Augmentation) |

---

## 8. SECTION STRUCTURE COMPARISON

### 8.1 Chapter 3 Completeness

| Section | PDF | HTML |
|---------|-----|------|
| §3.1 Data Acquisition | ✅ | ✅ |
| §3.2 Preprocessing | ✅ | ✅ |
| §3.3 External Datasets | ✅ | ✅ |
| §3.4 Unified Corpus | ✅ | ✅ |
| §3.5 Weak-Labeling | ✅ | ✅ |
| §3.6 Blind Annotation | ✅ | ✅ |
| §3.7 Final Dataset | ✅ | ✅ |

**Chapter 3 is COMPLETE in PDF.** ✅

### 8.2 Chapter 4 Completeness

| Section | PDF | HTML |
|---------|-----|------|
| §4.1 Pipeline Overview | ❌ MISSING | ✅ |
| §4.2 Model Architectures | ❌ MISSING | ✅ |
| §4.3 Evaluation Strategy | ❌ MISSING | ✅ |

### 8.3 Chapter 5 Completeness

| Section | PDF | HTML |
|---------|-----|------|
| §5.1 Comparative Model Performance | ❌ MISSING (Table 5.1) | ✅ |
| §5.2 Generalization Gap | ❌ MISSING | ✅ |
| §5.3 BERTopic Results | ❌ MISSING | ✅ |
| §5.4 Error Analysis | ❌ MISSING | ✅ |
| §5.5 DAPT Counter-Experiment | ❌ MISSING | ✅ |
| §5.6 Data Augmentation | ❌ MISSING | ✅ |

### 8.4 Chapter 6 Completeness

| Section | PDF | HTML |
|---------|-----|------|
| §6.1 Summary of Contributions | ❌ MISSING | ✅ |
| §6.2 Limitations | ❌ MISSING | ✅ |
| §6.3 Future Work | ❌ MISSING | ✅ |

---

## 9. ACTION ITEMS - PRIORITY ORDER

### Priority 1: CRITICAL (Must Fix Before Submission)

| # | Task | Location | Fix Required |
|---|------|---------|-------------|
| 1 | Fix author names | Page 1 | Single author: Bao Minh Nguyen |
| 2 | Fix university | Page 1 | UIT, VNU-HCM |
| 3 | Add Chapter 4 content | Pages 22+ | Copy from HTML §4.1-§4.3 |
| 4 | Add Chapter 5 content | Pages 22+ | Copy from HTML §5.1-§5.6 |
| 5 | Add Chapter 6 content | Pages 22+ | Copy from HTML §6.1-§6.3 |
| 6 | Add Table 5.1 | Chapter 5 | Insert full comparison table |
| 7 | Add Table 5.4 | Chapter 5 | Insert augmentation results |
| 8 | Add all 7 figures | Chapters 4-5 | Include from report_pdf/figures/ |

### Priority 2: HIGH (Should Fix)

| # | Task | Location | Fix Required |
|---|------|---------|-------------|
| 9 | Update dataset sizes | Throughout | 1,786→4,255, 383→912, 2,553→6,079 |
| 10 | Update gold set size | Abstract, §1.2.3 | 1,607→3,020 |
| 11 | Fix reference [29] | Page 23 | Font encoding fix |
| 12 | Update abstract model numbers | Abstract | Add Round 4 numbers |

### Priority 3: MEDIUM (Nice to Have)

| # | Task | Location | Fix Required |
|---|------|---------|-------------|
| 13 | Add Round 4 results paragraph | Chapter 5 | 0.3850 cross-domain improvement |
| 14 | Update title date | Page 1 | "Academic Year 2025–2026" |
| 15 | Verify all references | Pages 22-23 | Complete reference check |

---

## 10. FILES TO REFERENCE

### For Content:
- `docs/paper_report.html` - Complete source with all 6 chapters
- `docs/PROGRESS_REPORT_2026-07-04.md` - Round 4 results
- `docs/round4_report.md` - Round 4 details

### For Numbers:
- `docs/phase3_comparison_report.json` - Phase 3 results
- `models/all_augmented_results_20260704_104505.json` - Round 4 results
- `data_unified/manifest.json` - Dataset manifest

### For Figures:
- `report_pdf/figures/` - All 7 publication-quality figures

### For Validation:
- `docs/PROGRESS_REPORT_2026-07-04.md` - Final numbers
- `docs/PDF_CORRECTION_CHECKLIST.md` - This file

---

## 11. RECOMMENDATION

**The PDF needs a complete regeneration from the HTML source with:**

1. Author info corrected on title page
2. All 6 chapters with full content (not just headings)
3. All 7 figures included
4. Dataset sizes updated to Round 4 values
5. Table 5.1 and Table 5.4 inserted
6. Reference [29] font encoding fixed

**Easiest path:** Regenerate PDF directly from `docs/paper_report.html` after updating:
- Title page author info
- Dataset size numbers (find/replace 1,786→4,255, etc.)
- Adding Round 4 summary paragraph to Chapter 5

---

*Generated: 2026-07-04*
*Status: DRAFT - Needs verification against actual regenerated PDF*
