# BÁO CÁO SO SÁNH: PDF gốc vs paper_report.html

## Tổng quan

| Khía cạnh | PDF gốc (523H0002_523H0054_CDNC1_W7.pdf) | paper_report.html (hiện tại) |
|-----------|-----------------------------------------|------------------------------|
| **Số tác giả** | 2 người: Nguyen Duc Anh + Nguyen Bao Minh | 1 người: Bao Minh Nguyen |
| **Ngày nộp** | June 21st | Không ghi rõ |
| **Trường** | Ton Duc Thang University | University of Information Technology, VNU-HCM |
| **Số trang** | 34 trang | ~96 trang |

---

## 1. SỰ KHÁC BIỆT VỀ SỐ LIỆU (CRITICAL)

### 1.1 Headline Results

| Metric | PDF gốc | paper_report.html | Cần sửa? |
|--------|---------|-------------------|----------|
| PhoBERT In-domain F1 | **0.8596** | **0.7187** | ⚠️ Cần xác nhận |
| PhoBERT Cross-domain F1 | **0.4937** | **0.3608** | ⚠️ Cần xác nhận |
| Training samples | 6,080 | 6,392 | ✅ Đúng |
| Generalization Gap | 0.37 | 0.36 | ✅ Gần đúng |

### 1.2 Abstract Comparison

**PDF gốc Abstract:**
> "Results demonstrate that PhoBERT with majority voting across three seeds achieves an F1-macro of **0.8596** on the in-domain test set (**912 samples**) and **0.4937** on the cross-domain VSMEC test set (3,084 samples), with a generalization gap of 0.37 F1 (post Round 5 dataset: **6,080 training samples**)."

**paper_report.html Abstract:**
> "Results demonstrate that PhoBERT with majority voting across three seeds achieves an F1-macro of **0.7187** on the in-domain test set (**1,371 samples**) and **0.3608** on the cross-domain VSMEC test set (3,084 samples), with a generalization gap of 0.36 F1 (post Round 6 v2 dataset: 6,392 training samples)."

---

## 2. CÁC VẤN ĐỀ CẦN SỬA TRONG paper_report.html

### 2.1 ⚠️ CRITICAL: Authors Section

**PDF gốc:**
```
DETECTION OF DEPRESSION SIGNS IN
VIETNAMESE SOCIAL MEDIA TEXT USING
DEEP LEARNING MODELS
NGUYEN DUC ANH1, NGUYEN BAO MINH2

1[523H0002]
2[523H0054]
1,2[Ton Duc Thang University], [Faculty of Information Technology]
1[523h0002@student.tdtu.edu.vn]
2[523h0054@student.hcmiu.edu.vn]
```

**paper_report.html hiện tại:**
```html
<p class="author"><strong>Bao Minh Nguyen</strong></p>
<p class="affiliation">Capstone Project — Department of Computer Science<br>University of Information Technology, VNU-HCM<br>Academic Year 2025&ndash;2026</p>
<p class="author" style="font-size:11pt; margin-top:8pt;">Student ID: 523H0054 &nbsp;|&nbsp; Email: 523h0054@student.hcmiu.edu.vn</p>
```

**>>> CẦN SỬA:** Giữ nguyên 1 tác giả hoặc thêm Nguyen Duc Anh?

---

### 2.2 ⚠️ CRITICAL: Abstract Metrics

**Vấn đề:** Số liệu trong Abstract không khớp với PDF gốc

| Dòng trong paper_report.html | Giá trị hiện tại | Giá trị PDF gốc |
|------------------------------|------------------|-----------------|
| PhoBERT F1 in-domain | 0.7187 | **0.8596** |
| In-domain test samples | 1,371 | **912** |
| PhoBERT F1 cross-domain | 0.3608 | **0.4937** |
| Training samples | 6,392 | **6,080** |
| Gap statement | 0.36 | **0.37** |

**>>> CẦN SỬA:** Cập nhật Abstract với số liệu đúng từ Round cuối cùng

---

### 2.3 ⚠️ CRITICAL: Expected Outcomes (Section 1.2.2)

**Line 235 trong paper_report.html:**
```html
<li>A comprehensive benchmark of five model architectures on two test domains,
establishing the first published performance figures for Vietnamese depression detection
— PhoBERT F1-macro: 0.7254 (in-domain), 0.3654 (cross-domain).</li>
```

**>>> SAI:** Giá trị 0.7254/0.3654 không khớp với Abstract (0.7187/0.3608)

---

### 2.4 ⚠️ WARNING: Gold Set Size

**PDF gốc Table 13:**
- Total reviewed samples: 1,750
- Annotator marked uncertain: 59
- Annotator marked exclude: 82
- Duplicate text removed: 2
- **Final gold set size: 1,607** (163 depression, 1,444 normal)

**paper_report.html không đề cập rõ ràng về gold set ban đầu 1,607**

---

### 2.5 ⚠️ WARNING: Data Sources

**PDF gốc Section 3.7:**
- A: Human Gold (1,607 samples, weight=3)
- B: Weak Label High-Confidence (weight=2)
- C: PhoBERT Model Pseudo-labeling Confident (weight=1)
- D: External Negative Labels (weight=2)

**paper_report.html không mô tả chi tiết về multi-source integration này**

---

## 3. NHỮNG ĐIỂM ĐÃ ĐÚNG TRONG paper_report.html

### ✅ Đúng:
1. Research question và motivation tương tự
2. Cấu trúc 6 chapters
3. Mô tả về PhoBERT, TF-IDF, BiLSTM, BERTopic
4. Đề cập đến VSMEC là cross-domain dataset
5. Đề cập đến 4 yếu tố gây ra generalization gap
6. Đề cập đến 335 keyword trong weak-labeling lexicon

---

## 4. CÁC PHẦN THIẾU TRONG paper_report.html

### 4.1 Thiếu: Dataset Source Details
- Không đề cập 8 external datasets cụ thể
- Không mô tả source weighting scheme (A, B, C, D)

### 4.2 Thiếu: Annotation Protocol Details
- Cohen's Kappa values (0.63 và -0.03)
- Chi tiết về 5 review buckets
- Gold set construction details

### 4.3 Thiếu: Error Analysis chi tiết
- PDF có section về characteristic errors
- paper_report.html có nhưng không chi tiết bằng

### 4.4 Thiếu: Limitations section
- PDF có 5 limitations rõ ràng
- paper_report.html có nhưng cần kiểm tra chi tiết

---

## 5. HÀNH ĐỘNG CẦN THỰC HIỆN

### Priority 1: Xác nhận và cập nhật số liệu

| File | Vị trí | Nội dung cần sửa |
|------|--------|------------------|
| paper_report.html | Line 163 (Abstract) | PhoBERT F1: 0.8596 (in), 0.4937 (cross), 912 samples |
| paper_report.html | Line 235 | Expected outcome F1: 0.8596 (in), 0.4937 (cross) |

### Priority 2: Kiểm tra authorship

- [ ] Xác nhận: Báo cáo này của 1 người hay 2 người?
- [ ] Nếu 2 người: Cập nhật Title page và Author section

### Priority 3: Thêm chi tiết bị thiếu

- [ ] Thêm gold set construction details (1,607 samples)
- [ ] Thêm multi-source integration details
- [ ] Thêm Cohen's Kappa values

---

## 6. CÂU HỎI CẦN XÁC NHẬN TỪ SINH VIÊN

1. **Số liệu nào là đúng?** Round 5 (0.8596/0.4937) hay Round 6 v2 (0.7187/0.3608)?

2. **Báo cáo này của ai?** Chỉ Bao Minh hay cả Nguyen Duc Anh?

3. **Trường học?** Ton Duc Thang University hay UIT VNU-HCM?

4. **Dataset size?** 6,080 (R5) hay 6,392 (R6 v2)?

---

## 7. KHUYẾN NGHỊ

### Option A: Giữ paper_report.html và sửa lỗi

1. Cập nhật tất cả số liệu về Round 6 v2 (0.7187/0.3608)
2. Thêm chi tiết bị thiếu từ PDF gốc
3. Giữ nguyên 1 tác giả nếu đây là báo cáo cá nhân

### Option B: Tạo lại báo cáo mới dựa trên PDF

1. Giữ cấu trúc và format từ PDF gốc
2. Cập nhật số liệu mới nhất
3. Thêm các improvement từ Round 6

---

## 8. FILE ĐÍNH KÈM CẦN TẠO

1. `docs/reports/REVISION_REQUIRED.md` - Danh sách chi tiết các thay đổi cần thiết
2. `docs/reports/METRICS_COMPARISON.md` - Bảng so sánh metrics qua các rounds

---

*Generated: 2026-08-10*
*Comparison: PDF gốc W7 vs paper_report.html Round 6 v2*
