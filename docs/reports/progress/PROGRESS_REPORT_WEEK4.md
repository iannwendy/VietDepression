# BÁO CÁO TIẾN ĐỘ THỰC HIỆN CHUYÊN ĐỀ NGHIÊN CỨU
## Tuần 4: 06 – 12/07/2026

---

### Thông tin sinh viên
- **Sinh viên 1:** Nguyễn Đức Anh – MSSV: 523H0002
- **Sinh viên 2:** Nguyễn Bảo Minh – MSSV: 523H0054
- **GVHD:** TS. Trần Thanh Phước
- **Đề tài:** Phát hiện dấu hiệu trầm cảm trong văn bản tiếng Việt trên mạng xã hội sử dụng các mô hình học sâu

---

## I. KHỐI LƯỢNG THỰC HIỆN TUẦN 4

### 1. Hoàn thành Round 4 Active Learning

| Hạng mục | Kết quả |
|----------|---------|
| Mẫu review | 1,500 mẫu (739 normal, 260 depression, 452 excluded, 49 uncertain) |
| Mẫu mới thêm vào gold set | 948 mẫu |
| Gold set mới | 2,072 → **3,020 dòng** |
| Final dataset | 2,553 → **6,079 dòng** |
| Train/Val/Test split | 4,255 / 912 / 912 |

### 2. Retrain PhoBERT với Dataset mới

- Train PhoBERT fine-tune (3 seeds) trên dataset mới sau Round 4
- **Kết quả:**
  - F1 in-domain: 0.8417 ± 0.0220
  - F1 cross-domain (VSMEC): **0.3850 ± 0.0219** (+0.0123 so với trước Round 4)
- Duy trì 52 unit tests passing

---

## II. HOÀN THIỆN BÁO CÁO

### Công việc đã hoàn thành:
- ✅ Hoàn thiện đầy đủ 5 chương báo cáo
- ✅ Hoàn thiện Abstract (tiếng Việt + tiếng Anh)
- ✅ Cập nhật kết quả Round 4 vào báo cáo
- ✅ Đánh số bảng, chú thích hình, font phần Tài liệu tham khảo
- ✅ Kiểm tra định dạng trích dẫn APA 7th
- ✅ Kiểm tra chính tả toàn bộ báo cáo

---

## III. PHÁT TRIỂN WEB DEMO

### ✅ Frontend (React + TypeScript + TailwindCSS)

| Tính năng | Trạng thái | Mô tả |
|-----------|------------|-------|
| Dashboard | ✅ Hoàn thành | Hiển thị metrics tổng quan từ API thật |
| Single Prediction | ✅ Hoàn thành | Phân tích 1 đoạn text, gọi PhoBERT inference |
| Batch Prediction | ✅ Hoàn thành | Upload CSV để phân tích nhiều comments |
| Topic Analysis | ✅ Hoàn thành | Hiển thị topics từ BERTopic (API thật) |
| Statistics | ✅ Hoàn thành | Charts từ real data (confusion matrix, class distribution) |
| History | ✅ Hoàn thành | Lịch sử predictions từ SQLite database (persist) |
| Model Comparison | ✅ Hoàn thành | So sánh 9 models (TF-IDF, BiLSTM, PhoBERT, DAPT, aug) |
| Routing | ✅ Hoàn thành | React Router với 7 routes |
| i18n | ✅ Hoàn thành | Hỗ trợ tiếng Việt/tiếng Anh |

### ✅ Backend (FastAPI)

| Tính năng | Trạng thái | Mô tả |
|-----------|------------|-------|
| PhoBERT Engine | ✅ Hoàn thành | Load và inference round5 best_model thật |
| BERTopic Engine | ✅ Hoàn thành | Topic assignment từ trained model |
| Metrics Loader | ✅ Hoàn thành | Đọc metrics từ JSON files |
| DAPT Counter-Experiment | ✅ Hoàn thành | Hiển thị DAPT results với badges |
| Results Registry | ✅ Hoàn thành | Auto-update mechanism cho training rounds |
| SQLite Database | ✅ Hoàn thành | Prediction history persistence |
| Auto-save | ✅ Hoàn thành | Mỗi prediction tự động lưu vào DB |
| Hot-reload | ✅ Hoàn thành | POST /api/models/refresh để reload không cần restart |
| Health Check | ✅ Hoàn thành | `/api/health` endpoint |
| CORS Middleware | ✅ Hoàn thành | Cho phép cross-origin requests |

### ✅ Dev Environment

| Thành phần | Trạng thái | Mô tả |
|------------|------------|-------|
| Frontend Dev Server | ✅ Hoàn thành | Vite chạy trên port 3000 |
| Backend Dev Server | ✅ Hoàn thành | Uvicorn chạy trên port 8000 |
| Docker Setup | ✅ Hoàn thành | Multi-stage Dockerfile + docker-compose với volume mounts |
| GPU Support | ✅ Hoàn thành | Dockerfile hỗ trợ MPS (Apple Silicon) / CUDA |

---

## IV. ALIGNMENT VỚI ĐỀ TÀI NGHIÊN CỨU

| Research Component | Demo Feature | Status |
|-------------------|-------------|--------|
| PhoBERT + BERTopic Model | Model Comparison | ✅ Done |
| Vietnamese YouTube Comments | Batch Prediction | ✅ Done |
| Active Learning (Round 4) | Results Registry + Refresh | ✅ Done |
| Data Augmentation | Model Comparison (+aug variants) | ✅ Done |
| Cross-domain Evaluation (VSMEC) | Cross-Domain F1 column | ✅ Done |
| DAPT Counter-Experiment | DAPT badge + note | ✅ Done |
| BERTopic Topics | Topic Analysis Page | ✅ Done |
| Depression Detection | Prediction Page | ✅ Done |
| History Persistence | SQLite DB | ✅ Done |

---

## V. KẾ HOẠCH TUẦN 5 (13 – 19/07/2026)

### 1. Hoàn thiện Backend
- [ ] Implement BiLSTM Engine (hiện tại heavy, cần vocab.json + model.pt loading)
- [ ] Implement TF-IDF Engine
- [ ] Fix Scatter Chart trong ModelComparison (Recharts Scatter dùng pixel coords)
- [ ] Hoàn thiện YouTube Integration:
  - [ ] Endpoint paste YouTube video URL để fetch comments
  - [ ] Analyze all comments từ 1 video
  - [ ] Display video metadata (title, channel, view count)

### 2. Fix lỗi Web Demo
- [ ] Error Handling:
  - [ ] Loading states cho tất cả pages
  - [ ] Error boundaries / retry logic
  - [ ] API timeout handling
- [ ] Interactive Features:
  - [ ] Date range filter cho charts
  - [ ] Topic filter cho batch prediction results
- [ ] Deployment prep:
  - [ ] GPU support trong Docker (CUDA base image)
  - [ ] CI/CD pipeline
  - [ ] Production build optimization

### 3. Chuẩn bị kiểm tra giữa kỳ
- [ ] Review toàn bộ báo cáo lần cuối
- [ ] Prepare slides trình bày kiểm tra giữa kỳ
- [ ] Demo web demo hoạt động

---

## VI. TỔNG KẾT

| Hạng mục | Tuần 3 | Tuần 4 |
|----------|--------|--------|
| Gold set | 2,072 | **3,020** (+948) |
| Final dataset | 2,553 | **6,079** (+3,526) |
| PhoBERT F1 cross-domain | 0.3727 | **0.3850** (+0.0123) |
| Web Demo Features | 60% | **90%** |
| Unit Tests | 52 | 52 (passing) |

**Tỷ lệ hoàn thành đề tài: ~85%**

---

*Ngày lập báo cáo: 12/07/2026*
*Người lập: Nhóm sinh viên CDNC1*
