# Báo Cáo Tiến Độ Kỹ Thuật — Ngày 10/07/2026

**Dự án:** Detection of Depression Signs in Vietnamese Social Media Text Using Deep Learning Models
**Người thực hiện:** Bao Minh Nguyen
**Mã sinh viên:** 523H0054
**Cập nhật lần cuối:** 2026-07-10

---

## 📊 Tóm Tắt Trạng Thái

| Giai đoạn | Trạng thái |
|------------|------------|
| Phase 1: Data Collection & Annotation | ✅ **HOÀN THÀNH** |
| Phase 2: Dataset Construction | ✅ **HOÀN THÀNH** |
| Phase 3: Model Training & Evaluation | ✅ **HOÀN THÀNH** |
| Phase 4: Paper Write-up | ✅ **HOÀN THÀNH** |
| Round 4 Active Learning | ✅ **HOÀN THÀNH** (04/07/2026) |
| W3.pdf Content | ✅ **HOÀN CHỈNH** (32 trang) |
| HTML Title Page | ✅ **ĐÃ SỬA** |

---

## 1. So sánh W3.pdf vs HTML

### ✅ W3.pdf ĐÃ HOÀN CHỈNH (32 trang)

W3.pdf chứa **đầy đủ** nội dung:

| Section | W3.pdf | HTML |
|---------|--------|------|
| Chapter 1: Introduction | ✅ Complete | ✅ |
| Chapter 2: Background | ✅ Complete | ✅ |
| Chapter 3: Data (§3.1-§3.7) | ✅ Complete | ✅ |
| Chapter 4: Methodology | ✅ Complete | ✅ |
| Chapter 5: Results (§5.1-§5.6) | ✅ Complete | ✅ |
| Chapter 6: Conclusion | ✅ Complete | ✅ |
| References (30+ sources) | ✅ Complete | ✅ |

**Tables trong W3.pdf:**
- Table 1: External datasets
- Table 4: Dataset catalog
- Table 6: Cross-tabulation
- Table 7: Corpus manifest
- Table 8: Weak-labeling lexicon
- Table 9: Classification thresholds
- Table 10: Label distribution
- Table 11: Weak-labeler audit
- Table 12: Agreement analysis
- Table 13: Gold set construction
- Table 14: Training data sources
- Table 15: Dataset split config
- **Table 16**: Comparative Model Performance ✅
- Table 17: BERTopic depression topics
- Table 18: DAPT counter-experiment
- Table 19: Augmentation results

### ✅ HTML Title Page ĐÃ SỬA (10/07/2026)

**Trước khi sửa (W3.pdf):**
```
NGUYEN DUC ANH¹, NGUYEN BAO MINH²
¹[523H0002]
²[523H0054]
¹,²[Ton Duc Thang University], [Faculty of Information Technology]
¹[523h0002@student.tdtu.edu.vn]
²[523h0054@student.tdtu.edu.vn]
```

**Sau khi sửa (paper_report.html):**
```
Bao Minh Nguyen
Capstone Project — Department of Computer Science
University of Information Technology, VNU-HCM
Academic Year 2025–2026
Student ID: 523H0054 | Email: 523h0054@student.hcmiu.edu.vn
```

---

## 2. Những gì đã hoàn thành

### ✅ Phase 1: Data Pipeline

| Task | Status | Details |
|------|--------|---------|
| YouTube comment crawling | ✅ | 125,329 comments |
| Text cleaning | ✅ | Multi-stage pipeline |
| Weak labeling | ✅ | 335-keyword lexicon |
| Blind human annotation | ✅ | 1,750 samples |
| Round 4 Active Learning | ✅ | 1,500 reviewed, 948 added |

### ✅ Phase 2: Dataset Construction

| Dataset | Size (Post Round 4) |
|---------|---------------------|
| Gold set (train_gold.csv) | 3,020 |
| Final dataset (final_dataset.csv) | 6,079 |
| Train | 4,255 |
| Val | 912 |
| Test in-domain | 912 |
| Cross-domain (VSMEC) | 3,084 |

### ✅ Phase 3: Model Training

| Model | In-domain F1 | Cross-domain F1 |
|-------|-------------|----------------|
| TF-IDF + LogReg | 0.8415 | 0.3780 |
| TF-IDF + LinearSVC | 0.8799 | 0.3574 |
| BiLSTM (random) | 0.8145 ± 0.0244 | 0.4690 ± 0.0601 |
| BiLSTM (PhoBERT-frozen) | 0.8244 ± 0.0044 | 0.4344 ± 0.0008 |
| **PhoBERT** | **0.8417 ± 0.0220** | **0.3850 ± 0.0219** |
| BERTopic-only | 0.5599 | 0.5030 |
| PhoBERT + BERTopic | 0.8497 | 0.4406 |
| PhoBERT + DAPT | 0.8803 ± 0.0030 | 0.3620 ± 0.0188 |

### ✅ Data Augmentation

| Model | In-domain F1 | Cross-domain F1 |
|-------|-------------|----------------|
| PhoBERT (aug) | **0.9619** | 0.3993 |
| PhoBERT + BERTopic (aug) | 0.9377 | **0.5262** |

---

## 3. Hướng dẫn tạo PDF cuối cùng

### Bước 1: Regenerate PDF từ HTML đã sửa

```bash
# Mở HTML trong Chrome
open docs/paper_report.html

# Print to PDF (Cmd+P → Save as PDF)
```

### Bước 2: Kiểm tra output

- ✅ Title page: Bao Minh Nguyen, UIT, VNU-HCM, 523H0054
- ✅ All 6 chapters present
- ✅ Table 16 comparison table
- ✅ References

---

## 4. Files chính

| File | Mục đích |
|------|----------|
| `docs/paper_report.html` | HTML source (đã sửa title page) |
| `docs/PROGRESS_REPORT_2026-07-10.md` | File này |
| `report_pdf/figures/` | 7 publication-quality figures |
| `523H0002_523H0054_CDNC1_W3.pdf` | Current PDF (sai title page) |

---

## 5. Tổng kết

### ✅ ĐÃ HOÀN THÀNH

| Task | Date |
|------|------|
| Phase 1-3: Data, Models, Training | ✅ |
| Round 4 Active Learning | ✅ 04/07/2026 |
| HTML Title Page Fixed | ✅ 10/07/2026 |

### 📋 CÔNG VIỆC CUỐI CÙNG

**Regenerate PDF từ HTML** (~5 phút)
```bash
open docs/paper_report.html
# Cmd+P → Save as PDF
```

---

*Báo cáo này được tạo tự động dựa trên phân tích W3.pdf.*
*Date: 2026-07-10*
*Author: Bao Minh Nguyen (523H0054)*
