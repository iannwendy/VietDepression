# Week 1 Report — Tiến độ Dự án

**Dự án:** Detection of Depression Signs in Vietnamese Social Media Text Using Deep Learning Models


**Trạng thái:** Đúng tiến độ

---

## 1. Công việc đã hoàn thành

### 1.1 Thu thập dữ liệu YouTube

- Cấu hình YouTube Data API v3, thiết lập crawl với **50 keywords** tiếng Việt về chủ đề trầm cảm, áp lực, cô đơn, buồn bã:
  - Nhóm "trầm cảm": `trầm cảm`, `dấu hiệu trầm cảm`, `rối loạn lo âu`, `sống chung với trầm cảm`, ...
  - Nhóm "áp lực": `áp lực công việc`, `áp lực cuộc sống`, `stress`, `kiệt sức`, ...
  - Nhóm "cô đơn": `cô đơn`, `một mình`, `lạc lõng`, `không ai hiểu`, ...
  - Nhóm "buồn bã": `buồn`, `chán nản`, `mất động lực`, `tuyệt vọng`, ...
- Mỗi keyword: search tối đa 100 video, mỗi video lấy tối đa 100 comment
- Cấu hình: `SEARCH_ORDER=relevance`, `REGION_CODE=VN`, `RELEVANCE_LANGUAGE=vi`
- **Kết quả:** Crawl được **125.329 comment** từ video YouTube tiếng Việt
- Dữ liệu thô lưu tại `data/raw_comments.csv` (7 cột: `comment_id`, `video_id`, `video_title`, `keyword`, `comment_text`, `like_count`, `published_at`)

### 1.2 Làm sạch dữ liệu

- Chuẩn hóa Unicode (NFC), loại bỏ ký tự không in được
- Xóa comment trùng lặp, comment quá ngắn (< 10 ký tự), comment spam
- Tokenization cơ bản (tách câu, tách từ)
- Lưu kết quả vào `data/cleaned_comments.csv` — **125.329 dòng**

### 1.3 Thu thập bộ dữ liệu tiếng Việt bên ngoài

- Tải và xử lý **5 external datasets** từ HuggingFace và GitHub:
  - `thainq107/ntc-scv` — phân loại cảm xúc tiếng Việt (~25K dòng)
  - `tridm/UIT-VSFC` — sentiment analysis (~8K dòng)
  - `anotherpolarbear/vietnamese-sentiment-analysis` (~6K dòng)
  - `minhtoan/vietnamese-comment-sentiment` (~2K dòng)
  - `sepidmnorozy/Vietnamese_sentiment` (~1.6K dòng)
- Mục đích: bổ sung hard-negative (câu positive-affect, không phải trầm cảm) để chống overfit vào mọi câu buồn
- Trích xuất **VSMEC** làm cross-domain test set cố định (3.084 dòng, cân bằng 1.542/1.542)
  - KHÔNG BAO GIỜ đưa VSMEC vào train để tránh leakage

### 1.4 Xây dựng corpus thống nhất

- Hợp nhất toàn bộ text (YouTube + external) → `data_unified/corpus_text_all.csv` — **316.401 dòng**
- Tạo `data_unified/augmentation_negatives.csv` — **42.952 dòng** hard-negative (positive-affect do người gán nhãn)
- Tạo `data_unified/cross_domain_test.csv` — **3.084 dòng** VSMEC holdout

### 1.5 Weak-labeling tự động (keyword-based)

- Xây dựng bộ từ khóa depression/normal có trọng số:
  - Depression strong (trọng số +5): `trầm cảm`, `tự tử`, `muốn chết`, `tuyệt vọng`, ...
  - Depression medium (+3): `buồn`, `cô đơn`, `mất ngủ`, `áp lực`, `chán nản`, ...
  - Normal (-2): `vui`, `hạnh phúc`, `yêu đời`, `cảm ơn`, ...
- Gán nhãn tự động cho toàn bộ 125K comment YouTube:
  - `depression_auto`: 3.223 dòng
  - `normal_auto`: 23.695 dòng
  - `uncertain`: 98.410 dòng (đa số)
- Tạo `data/initial_train.csv` — **3.759 dòng** high-confidence weak label làm tập train ban đầu

### 1.6 Chuẩn bị review thủ công

- Viết script `prepare_label_studio_import.py` để tạo 2 bộ dữ liệu cho Label Studio:

### 1.7 Review thủ công 1.750 dòng

- Đã review toàn bộ 1.750 dòng trên Label Studio
- Kết quả:

| Nhãn | Step 5 | Step 8 | Tổng |
|------|--------|--------|------|
| normal | 575 | 871 | **1.446** |
| depression | 107 | 56 | **163** |
| uncertain | 17 | 42 | **59** |
| exclude | 51 | 31 | **82** |

- Nhận xét: depression rất hiếm trong mẫu (163/1.750 ≈ 9.3%), phù hợp với thực tế dữ liệu YouTube

### 1.8 Merge nhãn & Kiểm định chất lượng

- Viết script `phase1_merge_review.py` ghép `final_label` người-gán từ file export về file nguồn qua `row_id`
- Kết quả kiểm định:

| Chỉ số | Step 5 | Step 8 |
|--------|--------|--------|
| Tỉ lệ bất đồng (người vs máy) | 17.49% | **53.72%** |
| Cohen's Kappa | 0.63 | **-0.03** |
| Kết luận | Đồng thuận trung bình | **Gần như ngẫu nhiên** |

- **Phát hiện quan trọng:** Step 8 có disagreement 53.72% và kappa ≈ 0 — chứng minh review blind ĐÃ ĐỘC LẬP. Blind review là quyết định đúng đắn.

### 1.9 Rebuild gold set

- Kết hợp 2 nguồn → tạo `gold_review.csv` mới với nhãn người:
  - **1.607 dòng gold** (163 depression + 1.444 normal)
  - Chuẩn hóa text, loại bỏ trùng lặp
  - Nhãn `uncertain`/`exclude` → loại khỏi train
- Đánh giá baseline TF-IDF + LogReg trên gold mới:
  - **F1 depression = 0.38** (baseline đầu tiên trên gold người)
  - Weak-labeler: recall 98% nhưng precision chỉ 61% → keyword mù với câu buồn ngắn

### 1.10 Viết roadmap sau review

- Tạo `docs/ROADMAP_SAU_REVIEW.md` — vạch sẵn 4 phase từ sau review đến báo cáo
- Xác định các quyết định mở: PhoBERT vòng 2, hồi 113K pseudo-label, tỉ lệ cân bằng lớp

---

## 2. Kết quả đạt được

| Artifact | Trạng thái | Mô tả | Kích thước |
|----------|-----------|-------|------------|
| `raw_comments.csv` | ✅ | Dữ liệu thô từ YouTube API | 125.329 dòng |
| `cleaned_comments.csv` | ✅ | Dữ liệu đã làm sạch | 125.329 dòng |
| `auto_labeled_comments.csv` | ✅ | Weak-label tự động (keyword) | 125.328 dòng |
| `initial_train.csv` | ✅ | Train set ban đầu (weak high-confidence) | 3.759 dòng |
| `corpus_text_all.csv` | ✅ | Corpus thống nhất YouTube + external | 316.401 dòng |
| `augmentation_negatives.csv` | ✅ | Hard-negative external (positive-affect) | 42.952 dòng |
| `cross_domain_test.csv` | ✅ | Test set cố định VSMEC | 3.084 dòng |
| `review_samples.csv` | ✅ Updated | Đã cập nhật final_label từ người review | 750 dòng |
| `phobert_active_learning_samples.csv` | ✅ Updated | Đã cập nhật final_label từ người review | 1.000 dòng |
| `gold_review.csv` | ✅ Mới | Gold set sạch, nhãn người | 1.607 dòng |
| `label_studio_*_import.csv` | ✅ | File import blind cho Label Studio | 2 file |
| `label_studio_*_key.csv` | ✅ | File key để merge nhãn về sau | 2 file |
| `prepare_label_studio_import.py` | ✅ | Script sinh file import + key | — |
| `phase1_merge_review.py` | ✅ | Script Phase 1 (merge + quality check + gold) | — |
| `phase1_eval_report.json` | ✅ | Báo cáo Phase 1 | — |

---

## 3. Vấn đề gặp phải

### 3.1 Depressed class cực kỳ hiếm

- 163/1.750 mẫu review là depression (9.3%)
- Trong 125K comment YouTube, chỉ 3.223 được gán `depression_auto` (2.6%)
- Cần chiến lược cân bằng lớp khi build final dataset (dự kiến dùng external negatives)

### 3.2 Weak-labeler không đủ mạnh

- Recall 98% nhưng precision chỉ 61% → quá nhiều false positive
- Câu buồn ngắn ("chán quá", "mệt mỏi") bị gán nhầm depression
- Baseline TF-IDF + LogReg train trên weak-label có **F1 depression = 0.38** → không đủ cho production, cần deep learning

### 3.3 Domain shift YouTube vs VSMEC có tín hiệu sớm

- PhoBERT v1 (train trên weak-label) đạt F1=0.76 trên VSMEC — cao hơn baseline (0.61)
- Nhưng vẫn thấp hơn nhiều so với in-domain (~0.96) → domain shift là vấn đề thực sự
- Cần phân tích sâu hơn trong các tuần sau

### 3.4 Dữ liệu YouTube nhiễu

- Nhiều comment là lyrics bài hát, spam, quảng cáo
- Comment ngắn, nhiều teen slang, viết tắt ("ko", "dc", "j", "vkl")
- Khó phân biệt "buồn bình thường" và "trầm cảm thật sự"

---

## 4. Kế hoạch tuần 2

| Công việc | Dự kiến |
|-----------|---------|
| Train PhoBERT v2 trên gold người (1.607 dòng) | Đầu tuần |
| Dự đoán 125K comment với PhoBERT v2 để lấy pseudo-label | Đầu tuần |
| Build `final_dataset.csv` (kết hợp gold + weak high-conf + PhoBERT v2 + external negatives) | Giữa tuần |
| Cân bằng lớp, dedup, split train/val/test (70/15/15) | Giữa tuần |
| Train TF-IDF + SVM (baseline) | Cuối tuần |
| Bắt đầu build BiLSTM | Cuối tuần |

---

## 5. Quyết định cần chốt

- [x] ~~Nhãn uncertain/exclude → loại khỏi train~~ (đã chốt: loại)
- [x] ~~Dùng blind review cho Label Studio~~ (đã chốt: có, và đã chứng minh hiệu quả)
- [ ] Có dùng PhoBERT confident 113K làm pseudo-label không?
- [ ] Tỉ lệ human:pseudo trong final dataset?
- [ ] Có chạy PhoBERT vòng 2 (active learning) trước khi chốt final không?
- [ ] Tỉ lệ cân bằng lớp (1:1, 1:2, hay 1:3)?

---

— Hết tuần 1 —
