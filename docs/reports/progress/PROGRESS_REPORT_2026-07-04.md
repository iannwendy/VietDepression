# Báo Cáo Tiến Độ Kỹ Thuật — Ngày 04/07/2026

**Dự án:** Detection of Depression Signs in Vietnamese Social Media Text Using Deep Learning Models
**Người thực hiện:** Bao Minh Nguyen
**Cập nhật lần cuối:** 2026-07-04

---

## 📊 Tóm Tắt Trạng Thái

| Giai đoạn | Trạng thái |
|------------|------------|
| Phase 1: Data Collection & Annotation | ✅ **HOÀN THÀNH** |
| Phase 2: Dataset Construction | ✅ **HOÀN THÀNH** |
| Phase 3: Model Training & Evaluation | ✅ **HOÀN THÀNH** |
| Phase 4: Paper Write-up | ✅ **HOÀN THÀNH** |
| Round 4 Active Learning | ✅ **HOÀN THÀNH** |
| PDF Generation | 🔴 **CHƯA HOÀN THÀNH** |

---

## 📊 Round 4 Active Learning — HOÀN THÀNH

### Round 4 Review Statistics

| Metric | Value |
|--------|-------|
| Total samples reviewed | 1,500 |
| Normal | 739 (49.3%) |
| Depression | 260 (17.3%) |
| Excluded | 452 (30.1%) |
| Uncertain | 49 (3.3%) |
| **New samples added** | **948 samples** |

### Dataset Changes

| Dataset | Before Round 4 | After Round 4 | Delta |
|---------|---------------|---------------|-------|
| Gold set | 2,072 | **3,020** | +948 |
| Final dataset | 2,553 | **6,079** | +3,526 |
| Train | 1,786 | **4,255** | +2,469 |
| Val | 383 | **912** | +529 |
| Test | 383 | **912** | +529 |

---

## 📊 Kết Quả Model (Post Round-4)

### Table 1: Final Results

| Model | In-domain F1 | Cross-domain F1 | Seeds |
|-------|-------------|----------------|-------|
| TF-IDF + LogReg | 0.8415 | 0.3780 | 1 |
| TF-IDF + LinearSVC | 0.8799 | 0.3574 | 1 |
| BiLSTM (random) | 0.8145 ± 0.0244 | 0.4690 ± 0.0601 | 3 |
| BiLSTM (PhoBERT-frozen) | 0.8244 ± 0.0044 | 0.4344 ± 0.0008 | 3 |
| **PhoBERT (original)** | **0.8417 ± 0.0220** | **0.3850 ± 0.0219** | 3 |
| BERTopic-only | 0.5599 | 0.5030 | - |
| PhoBERT + BERTopic | 0.8497 | 0.4406 | - |
| PhoBERT + DAPT | 0.8803 ± 0.0030 | 0.3620 ± 0.0188 | 3 |

### Table 2: Before/After Round 4 Comparison (PhoBERT)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| In-domain F1 | 0.8681 | 0.8417 | -0.0264 |
| Cross-domain F1 | 0.3727 | **0.3850** | **+0.0123** ✅ |

**Key Finding:** Cross-domain F1 improved by +0.0123 with Round 4 data augmentation!

---

## ✅ NHỮNG GÌ ĐÃ HOÀN THÀNH

### Phase 1: Data Pipeline ✅
- [x] YouTube comment crawling (125,329 comments)
- [x] Text cleaning & normalization
- [x] Weak labeling via keyword scoring
- [x] Blind human annotation (Label Studio)
- [x] **Round 4 Active Learning** - 1,500 samples reviewed, 948 added

### Phase 2: Dataset Construction ✅
- [x] Gold set construction (3,020 samples)
- [x] Stratified train/val/test splits
- [x] Cross-domain leak check
- [x] **Final dataset: 6,079 samples**

### Phase 3: Model Training & Evaluation ✅
- [x] TF-IDF + LogReg baseline
- [x] TF-IDF + LinearSVC baseline
- [x] BiLSTM (random & PhoBERT-frozen) - 3 seeds
- [x] PhoBERT fine-tuning - 3 seeds
- [x] BERTopic topic modeling
- [x] PhoBERT + BERTopic hybrid
- [x] Domain-adaptive pretraining (DAPT) experiment
- [x] **52 unit tests passing**

### Phase 4: Paper Write-up ✅
- [x] Full paper draft (HTML) - all 6 chapters
- [x] 7 publication-quality figures
- [x] LaTeX abstract
- [x] **Updated with Round 4 results**

---

## 🔴 CÔNG VIỆC CÒN LẠI

### Priority 1: PDF Generation

| Task | Effort | Status |
|------|--------|--------|
| Generate PDF from HTML | 30 phút | ⏳ Pending |
| Verify numbers in PDF | 15 phút | ⏳ Pending |
| Fix any sync issues | 15 phút | ⏳ Pending |

**Commands:**
```bash
# Generate PDF using Chrome headless or wkhtmltopdf
# Or use browser print to PDF
```

### Priority 2: PDF Quality Check

| Task | Effort | Status |
|------|--------|--------|
| Verify all 6 chapters present | 15 phút | ⏳ Pending |
| Check table numbering | 10 phút | ⏳ Pending |
| Verify figure captions | 10 phút | ⏳ Pending |
| Fix Reference font issues | 15 phút | ⏳ Pending |

### Priority 3: Final Review

| Task | Effort | Status |
|------|--------|--------|
| Full PDF read-through | 30 phút | ⏳ Pending |
| Citation format (APA 7) | 15 phút | ⏳ Pending |
| Spelling check (tiếng Việt) | 15 phút | ⏳ Pending |
| Final backup | 5 phút | ⏳ Pending |

---

## 📁 FILES CHÍNH

| File | Mục đích |
|------|----------|
| `docs/paper_report.html` | Full paper draft (HTML source) |
| `report_pdf/Report (1).pdf` | Latest PDF version |
| `report_pdf/figures/` | All 7 generated figures |
| `README.md` | Headline results + data artifacts |
| `docs/round4_report.md` | Round 4 detailed report |
| `docs/PROGRESS_REPORT_2026-07-04.md` | This file |

---

## 🔗 COMMITS GẦN ĐÂY

```
cf2828e docs(paper): update Table 5.1 and abstract with Round 4 results
c1fba19 chore(round4): update datasets with Round 4 merged data
d39e3bd feat(round4): merge reviewed labels, rebuild dataset, retrain models
64cc1a4 fix(i18n): use getMetricLabel instead of undefined metricLabels
1d3bba7 feat(i18n): translate ModelComparison page
4e1a173 feat(i18n): translate History page
```

---

## ✅ CHECKLIST HOÀN THÀNH

### Must-have cho submission:

- [x] Phase 1: Data collection & annotation
- [x] Phase 2: Dataset construction
- [x] Phase 3: Model training & evaluation
- [x] Phase 4: Paper draft in HTML (all 6 chapters)
- [x] 7 publication-quality figures generated
- [x] DAPT counter-experiment completed
- [x] 52 unit tests passing
- [x] BiLSTM 3-seed evaluation
- [x] PhoBERT 3-seed evaluation
- [x] **Round 4 Active Learning** ✅
- [x] **Update paper with Round 4 results** ✅
- [ ] **Generate PDF from updated HTML**
- [ ] **PDF quality check**
- [ ] **Final review**

### Nice-to-have (không block submission):

- [ ] BiLSTM 5-seed sweep
- [ ] DAPT 5-seed sweep
- [ ] Figure 1 Pipeline Overview
- [ ] Figure 7 Annotation Protocol

---

## ⏰ ƯỚC TÍNH THỜI GIAN CÒN LẠI

| Task | Estimated Time |
|------|----------------|
| PDF generation | 30 phút |
| PDF quality check | 1 giờ |
| Final review | 1 giờ |
| **Tổng** | **~2.5 giờ** |

---

## 📋 PUSH COMMANDS

```bash
# Push to GitHub
git push origin main

# Push to GitLab
git push gitlab main

# Push cả hai
git push origin main && git push gitlab main
```

---

*Báo cáo này được tạo tự động.*
*Date: 2026-07-04*
