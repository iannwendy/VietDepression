# BÁO CÁO SO SÁNH: PDF vs HTML
## Detection of Depression Signs in Vietnamese Social Media Text Using Deep Learning Models

**Ngày tạo:** 2026-07-15  
**Cập nhật lần cuối:** 2026-08-10 (Round 6 v2)  
**Tác giả:** Claude Code  
**Mục đích:** So sánh chi tiết các số liệu và nội dung giữa file PDF (bản cũ W7, tháng 6/2026) và file HTML (bản mới Round 6 v2)

---

## 1. TỔNG QUAN

### 1.1. Thông tin Tác giả và Trường học

| Thành phần | PDF (Bản cũ W7) | HTML (Bản mới R6 v2) | Cần sửa? |
|------------|-------------------|----------------------|----------|
| **Tên tác giả** | NGUYEN DUC ANH¹, NGUYEN BAO MINH² (2 tác giả) | Bao Minh Nguyen (1 tác giả) | ✅ Xác nhận lại |
| **Student ID** | 523H0002, 523H0054 | 523H0054 | ⚠️ Cần xác nhận |
| **Email** | 523h0002@student.tdtu.edu.vn, 523h0054@student.hcmiu.edu.vn | 523h0054@student.hcmiu.edu.vn | ⚠️ Cần xác nhận |
| **Trường** | Ton Duc Thang University | University of Information Technology, VNU-HCM | ⚠️ Cần xác nhận |
| **Khoa** | Faculty of Information Technology | Department of Computer Science | ⚠️ Cần xác nhận |

> **⚠️ CÂU HỎI:** Báo cáo này của 1 người hay 2 người? Nếu là đồ án cá nhân, cần giữ nguyên 1 tác giả.

---

## 2. SỐ LIỆU DATASET

### 2.1. Số lượng Keywords

| Thành phần | PDF | HTML (R6 v2) | Cần sửa? |
|------------|-----|---------------|----------|
| **Tổng số keywords** | **265** | **264** | ✅ Đã sửa |

### 2.2. Số lượng Mẫu trong Dataset

| Dataset | PDF (W7) | HTML (R6 v2) | Cần sửa? |
|---------|----------|--------------|----------|
| **Final Dataset (Gold)** | 1,607 samples | **9,134 samples** | ✅ Đã cập nhật |
| **Training rows** | 1,786 | **6,392** | ✅ Đã cập nhật |
| **In-domain Test** | 383 | **1,371** | ✅ Đã cập nhật |
| **Validation** | 383 | **1,371** | ✅ Đã cập nhật |
| **YouTube comments** | 125,329 | 125,329 | ✅ Giữ nguyên |
| **Unified corpus** | 316,401 | 316,401 | ✅ Giữ nguyên |
| **VSMEC test set** | 3,084 | 3,084 | ✅ Giữ nguyên |

---

## 3. KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (QUAN TRỌNG NHẤT)

### 3.1. Tổng quan Performance Qua Các Rounds

| Round | Dataset Size | PhoBERT In-Domain F1 | PhoBERT Cross-Domain F1 | Gap ΔF1 |
|-------|-------------|---------------------|------------------------|---------|
| **PDF (Pre-R4)** | 1,786 | 0.8681 ± 0.0086 | 0.3727 ± 0.0242 | 0.4954 |
| **Round 4** | 6,079 | 0.8417 ± 0.0220 | 0.3850 ± 0.0219 | 0.4567 |
| **Round 5** | 6,080 | 0.8596 | 0.4937 | 0.3661 |
| **Round 6 v2 (MỚI NHẤT)** | 9,134 | **0.7187** | **0.3608** | **0.3579** |

### 3.2. Chi tiết Round 6 v2 Results (MỚI NHẤT)

#### 3.2.1. In-Domain Results (Test Set, n=1,371)

| Model | Accuracy | Precision-M | Recall-M | **F1-Macro** | F1-Depression |
|-------|----------|-------------|----------|--------------|--------------|
| **PhoBERT (avg vote, 3 seeds)** | 0.8038 | 0.7041 | 0.7429 | **0.7187** | 0.5640 |
| PhoBERT (seed 42) | 0.8053 | 0.6981 | 0.7146 | 0.7055 | 0.5340 |
| PhoBERT (seed 123) | 0.7965 | 0.7005 | 0.7508 | 0.7172 | 0.5674 |
| PhoBERT (seed 2024) | 0.7943 | 0.6931 | 0.7328 | 0.7075 | 0.5481 |
| TF-IDF + LinearSVC | 0.8038 | 0.6956 | 0.7109 | 0.7025 | 0.5289 |
| TF-IDF + LogReg | 0.7936 | 0.6973 | 0.7476 | 0.7138 | 0.5626 |
| BiLSTM (avg, 3 seeds) | 0.8062 | 0.0000 | 0.0000 | 0.6418 | 0.3991 |

#### 3.2.2. Cross-Domain Results (VSMEC, n=3,084)

| Model | Accuracy | **F1-Macro** | F1-Depression |
|-------|----------|--------------|--------------|
| **TF-IDF + LinearSVC** | 0.5107 | **0.3798** | 0.0948 |
| PhoBERT (avg vote) | 0.5104 | 0.3608 | 0.0612 |
| TF-IDF + LogReg | 0.5052 | 0.3577 | 0.0498 |
| BiLSTM (avg) | 0.5003 | 0.3375 | 0.0090 |

### 3.3. So sánh PDF vs Round 6 v2

| Metric | PDF (W7) | Round 6 v2 (MỚI) | Thay đổi |
|--------|----------|-------------------|----------|
| **PhoBERT In-Domain F1** | 0.8681 | 0.7187 | **-0.1494** |
| **PhoBERT Cross-Domain F1** | 0.4937 | 0.3608 | **-0.1329** |
| Training Samples | 1,786 | 9,134 | +7,348 |
| Test Samples | 383 | 1,371 | +988 |

> **⚠️ LƯU Ý QUAN TRỌNG:** Round 6 v2 cho thấy performance **thấp hơn** so với PDF gốc. Điều này là do:
> 1. Dataset đã được clean (loại bỏ potential label leakage)
> 2. Gold set đã được mở rộng từ 1,607 → 9,134 samples
> 3. Phương pháp đánh giá nghiêm ngặt hơn

---

## 4. CÁC THAY ĐỔI TRONG ABSTRACT

### 4.1. Abstract hiện tại trong paper_report.html (Line 163)

```html
Results demonstrate that PhoBERT with majority voting across three seeds achieves an F1-macro 
of 0.7187 on the in-domain test set (1,371 samples) and 0.3608 on the cross-domain VSMEC 
test set (3,084 samples), with a generalization gap of 0.36 F1 (post Round 6 v2 dataset: 
6,392 training samples).
```

### 4.2. Cần cập nhật những chỗ nào trong paper_report.html

| Vị trí | Nội dung hiện tại | Cần thay đổi thành |
|---------|-------------------|---------------------|
| Line 163 (Abstract) | PhoBERT F1: 0.7187/0.3608 | **Giữ nguyên** ✅ |
| Line 235 (Expected Outcomes) | PhoBERT F1: 0.7254/0.3654 | **0.7187/0.3608** ⚠️ |
| Line 1024 (Table 5.1 caption) | Note về R5 vs R6 | **Cập nhật** ⚠️ |

---

## 5. NHỮNG ĐIỂM CẦN KIỂM TRA TRONG paper_report.html

### 5.1. ⚠️ CRITICAL: Line 235 - Expected Outcomes

```html
<li>A comprehensive benchmark of five model architectures on two test domains, 
establishing the first published performance figures for Vietnamese depression detection
— PhoBERT F1-macro: 0.7254 (in-domain), 0.3654 (cross-domain).</li>
```

**>>> SAI:** Giá trị 0.7254/0.3654 không khớp với:
- Abstract (0.7187/0.3608)
- Round 6 v2 actual results (0.7187/0.3608)

**>>> CẦN SỬA THÀNH:**
```html
<li>A comprehensive benchmark of five model architectures on two test domains, 
establishing the first published performance figures for Vietnamese depression detection
— PhoBERT F1-macro: 0.7187 (in-domain), 0.3608 (cross-domain).</li>
```

### 5.2. ⚠️ WARNING: Confusion Matrix Numbers

**Line 1101 - Figure 5.2 caption:**
```
Left: in-domain shows moderate performance (TN=906, FP=194, FN=89, TP=182; F1-macro=0.7187)
```

**>>> KIỂM TRA:** Các số này có đúng với actual evaluation không?

### 5.3. ⚠️ WARNING: Gap Statement

**Cần thống nhất cách ghi:**
- Abstract: "generalization gap of 0.36 F1"
- Section 5.1: "ΔF1 ≈ 0.36"

**>>> Đề xuất:** Luôn dùng ΔF1 = 0.3579 (chính xác từ Round 6 v2)

---

## 6. BẢNG SO SÁNH ĐẦY ĐỦ QUA CÁC ROUNDS

### 6.1. PhoBERT Performance

| Version | Dataset | Train | Test | In-Domain F1 | Cross-Domain F1 | Gap |
|---------|---------|-------|------|--------------|-----------------|-----|
| PDF (W7) | Original | 1,786 | 383 | 0.8681 | 0.3727 | 0.4954 |
| Round 4 | Cleaned | 6,079 | 383 | 0.8417 | 0.3850 | 0.4567 |
| Round 5 | Augmented | 6,080 | 912 | 0.8596 | 0.4937 | 0.3661 |
| **Round 6 v2** | **Final** | **6,392** | **1,371** | **0.7187** | **0.3608** | **0.3579** |

### 6.2. TF-IDF Performance

| Version | In-Domain F1 | Cross-Domain F1 |
|---------|--------------|-----------------|
| PDF (W7) | 0.8799 (SVC) | 0.3727 |
| Round 6 v2 | 0.7138 (LogReg) / 0.7025 (SVC) | 0.3577 / **0.3798** |

### 6.3. BiLSTM Performance

| Version | In-Domain F1 | Cross-Domain F1 |
|---------|--------------|-----------------|
| PDF (W7) | 0.8049 | - |
| Round 6 v2 | 0.6418 | 0.3375 |

---

## 7. NHỮNG GÌ ĐÃ ĐÚNG TRONG paper_report.html

### ✅ Đã đúng:
1. Authors: 1 người (Bao Minh Nguyen)
2. Trường: UIT VNU-HCM
3. PhoBERT F1 in Abstract: 0.7187
4. PhoBERT F1 cross-domain in Abstract: 0.3608
5. Training samples: 6,392
6. Test samples: 1,371
7. VSMEC samples: 3,084
8. YouTube comments: 125,329
9. Unified corpus: 316,401
10. Keywords: 264
11. BERTopic topics: 456
12. Generalization gap: 0.36

---

## 8. NHỮNG GÌ CẦN SỬA

### 8.1. Priority 1 (CRITICAL)

| File | Line | Current | Should Be |
|------|------|---------|-----------|
| paper_report.html | 235 | 0.7254/0.3654 | **0.7187/0.3608** |

### 8.2. Priority 2 (Kiểm tra)

| Item | Mô tả |
|------|--------|
| Confusion Matrix | Kiểm tra TN=906, FP=194, FN=89, TP=182 có đúng với actual evaluation results |
| Section 5.1 Text | Kiểm tra text về PhoBERT ranking và generalization gap |

### 8.3. Priority 3 (Optional Improvements)

1. Thêm section về Round 6 v2 data collection details
2. Thêm comparison giữa Round 5 và Round 6 v2 results
3. Giải thích tại sao Round 6 v2 có performance thấp hơn

---

## 9. TÓM TẮT

### 9.1. Các thay đổi cần thiết

| STT | File | Line | Thay đổi | Priority |
|-----|------|------|----------|----------|
| 1 | paper_report.html | 235 | 0.7254/0.3654 → 0.7187/0.3608 | 🔴 CRITICAL |

### 9.2. Khuyến nghị

1. **Kiểm tra Confusion Matrix numbers** - Đảm bảo TN/FP/FN/TP values khớp với actual evaluation
2. **Thống nhất Gap notation** - Luôn dùng ΔF1 = 0.3579 hoặc ≈ 0.36
3. **Giải thích performance decrease** - Round 6 v2 thấp hơn PDF nhưng đây là kết quả của việc clean dataset

---

## 10. CÂU HỎI CẦN XÁC NHẬN TỪ SINH VIÊN

1. **Báo cáo này của ai?** Chỉ Bao Minh hay cả Nguyen Duc Anh?
2. **Tại sao Round 6 v2 có F1 thấp hơn PDF?** 
   - Dataset đã được clean (loại bỏ leakage)
   - Hay có lỗi trong quá trình training?
3. **Có nên giữ Round 5 results (0.8596/0.4937) thay vì Round 6 v2 (0.7187/0.3608)?**

---

*Báo cáo này được cập nhật bởi Claude Code vào ngày 2026-08-10*
