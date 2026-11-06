# Roadmap sau khi review thủ công xong

Tài liệu này vạch sẵn các bước **sau khi thành viên review trả về 2 file export từ Label Studio**
(đợt 8 active-learning + đợt 5 review). Mục tiêu: đi từ nhãn người-gán → `final_dataset.csv`
→ train dãy model so sánh → báo cáo. Bám đúng bước 9-15 trong pipeline của nhóm.

> Bối cảnh quan trọng đã phát hiện trong quá trình làm:
> - Recall của keyword weak-labeler trên distress người-gán chỉ **0.91%** → keyword mù với câu buồn ngắn.
> - Gold cũ cho baseline **accuracy 1.0 ảo** vì người review chấp nhận lại gợi ý hệ thống (96.3% trùng).
> - Vì vậy file review lần này là **blind** (chỉ `row_id` + `text`), buộc người gán độc lập.
> - Cross-domain test (VSMEC 3.084 dòng): baseline F1 **0.61**, PhoBERT-lần-1 F1 **0.76** — đây là điểm trung thực, không phải 1.0.

---

## Đầu vào khi bắt đầu roadmap

Thành viên review trả về:
- `label_studio_step8_active_learning_export.csv` (1.000 dòng, ưu tiên đợt 1)
- `label_studio_step5_review_export.csv` (750 dòng, đợt 2)

Mỗi dòng có `row_id` + nhãn người chọn (`depression` / `normal` / `uncertain` / `exclude`).
Hệ thống đã giữ sẵn `docs/*_key.csv` để ghép nhãn về dữ liệu gốc qua `row_id`.

---

## Phase 1 — Ghép nhãn & khôi phục gold SẠCH (mở khóa lại bước 5)

1. **Merge** export ↔ key qua `row_id` → ghép `final_label` người-gán về `review_samples.csv`
   và `phobert_active_learning_samples.csv`.
2. **Kiểm định chất lượng review** (để chứng minh không lặp lỗi cũ):
   - Đo **tỷ lệ bất đồng người vs máy**. Vì lần này blind nên PHẢI có bất đồng đáng kể;
     nếu vẫn ~96% trùng → review chưa độc lập, báo lại ngay trước khi đi tiếp.
   - Nếu ≥2 người cùng review một phần → tính **Cohen's kappa** (chỉ số nhóm cần cho báo cáo).
   - Thống kê tỷ lệ `uncertain` / `exclude`.
3. **Rebuild gold** (`gold_builder`) → `gold_review.csv` mới → **chạy lại** đánh giá weak-label
   + baseline trên gold. Lần này con số sẽ KHÔNG còn 1.0 ảo — điểm số trung thực đầu tiên cho báo cáo.

**Quyết định cần chốt:** nhãn `uncertain` / `exclude` → loại hẳn khỏi train (khuyến nghị).

---

## Phase 2 — Tạo `final_dataset.csv` (bước 9, đang thiếu)

Artifact khóa mở toàn bộ phần train so sánh. Thành phần đề xuất, theo thứ tự ưu tiên độ tin cậy:

| Nguồn | Nhãn | Vai trò |
|---|---|---|
| Human-reviewed (đợt 5 + 8) | gold | **trọng số cao nhất**, ưu tiên tuyệt đối |
| YouTube high-confidence weak | pseudo | nền dữ liệu lớn |
| PhoBERT confident (113K, prob≥0.9) | pseudo | *tùy chọn* — mở rộng nhưng có nhiễu |
| `data_unified/augmentation_negatives.csv` (42.952 external) | 0 | hard-negative chống "mọi câu buồn = trầm cảm" |

Sau đó: dedup theo text → cân bằng lớp → **split train/val/test in-domain**.
**Giữ `data_unified/cross_domain_test.csv` tách hẳn** — không bao giờ trộn vào train.

**3 quyết định cần chốt:**
- Có nhồi PhoBERT confident 113K vào không? (nhiều dữ liệu ↔ nhiều nhiễu)
- Có chạy **PhoBERT vòng 2** (train lại có thêm nhãn active-learning) trước khi chốt final không?
  Đây đúng tinh thần active-learning loop của pipeline.
- Tỷ lệ human:pseudo và cách cân bằng lớp.

---

## Phase 3 — Train dãy model so sánh (bước 10-13)

Tất cả train trên `final_dataset`, test trên **2 tập**: in-domain + cross-domain VSMEC cố định.

1. **TF-IDF + SVM** — hiện mới có TF-IDF + **LogReg**, *chưa phải SVM*. Thêm `LinearSVC`
   (tái dùng phần vectorizer word+char đã có trong `baseline_model.py`).
2. **BiLSTM** — **chưa tồn tại**, phải build mới (embedding tiếng Việt + BiLSTM + classifier).
   Phần code nặng nhất.
3. **PhoBERT (final)** — train lại trên `final_dataset` thật, không phải weak-label như "lần 1".
4. **PhoBERT + BERTopic** — chạy BERTopic trên `corpus_text_all.csv` (316K, đã có) để lấy topic,
   kết hợp đặc trưng chủ đề với PhoBERT.

---

## Phase 4 — So sánh & báo cáo (bước 14-15)

1. **Bảng tổng hợp** 5 model (baseline + 4) × 2 test set (in-domain / cross-domain):
   F1, precision, recall, confusion matrix. Bảng trung tâm của bài báo.
2. **Phân tích lỗi** — false positive/negative điển hình; tận dụng phát hiện đã có
   (baseline mù distress ngắn, recall keyword 0.91%).
3. Viết báo cáo dựa trên các report JSON tích lũy qua từng phase.

---

## Quyết định mở (sẽ hỏi lại khi file review về)

- **PhoBERT vòng 2?** có / không
- **Hồi 113K pseudo-label vào final?** có / không
- **Tỷ lệ human:pseudo** và cách cân bằng lớp
- **uncertain/exclude** loại khỏi train? (mặc định: có)

Khi nhận 2 file export, bắt đầu thẳng từ **Phase 1**.

---

## Trạng thái hiện tại (cập nhật 2026-06-26)

Toàn bộ Phase 1, 2, 3 đã hoàn thành. Phase 4 (báo cáo) đang được đồng bộ
vào `docs/paper_report.html`. Chi tiết:

### Phase 1 — ✅ DONE (commit `35ca4b6`)
- 3 vòng review merged: Step 5 (750) + Step 8 (1.000) + **Round 3 (1.500)**
- `gold_review.csv`: 1.607 → **2.515 rows** (2.265 normal / 250 depression)
- Round 3 quality check: agreement 43.17%, kappa = -0.05
  (reviewer độc lập — keyword weak-labeler bị reject 66.3%)
- Baseline F1 trên gold mới: 0.5334 (accuracy 0.6354) — KHÔNG 1.0 ảo

### Phase 2 — ✅ DONE
- `final_dataset.csv`: **2.553 rows** (985 human_gold + 1.568 weak_high_conf,
  balanced 2:1)
- `final_train.csv` / `final_val.csv` / `final_test.csv`:
  1.786 / 383 / 383 rows, stratified
- Cross-domain leak check: 0 overlap với `cross_domain_test.csv`

### Phase 3 — ✅ DONE (commits `e3e818e`, `1f85ff6`)
5 model families đã train + eval trên final_dataset:

| Model                       | In-domain F1 | Cross-domain F1 |
|-----------------------------|--------------|-----------------|
| TF-IDF + LogReg              | 0.8347       | 0.3917          |
| TF-IDF + LinearSVC           | 0.8286       | 0.3820          |
| BiLSTM (random embedding)    | 0.8357       | 0.4079          |
| BiLSTM (PhoBERT-frozen)      | 0.8266       | 0.4352          |
| **PhoBERT (3-seed mean±std)** | **0.8681 ± 0.0086** | **0.3727 ± 0.0242** |

DAPT counter-experiment (commit `c8bc7fe`, `f09fddc`):
- Original 0.8681 vs DAPT 0.8803 in-domain (DAPT slightly better,
  not significant, t = -1.84, p ≈ 0.21)
- Cross-domain essentially unchanged
- Direction REVERSED from pre-round-3 finding (DAPT used to degrade
  both test sets; bug in orchestrator trained on wrong splits)

### Phase 4 — IN PROGRESS
- `docs/paper_report.html` updated with new Table 5.1 (7 rows) +
  §5.5 DAPT counter-experiment (numbers + errata footnote for the
  bug)
- README + paper headlines all reference post-round-3 numbers

### Decisions resolved
- **Decision 1** (PhoBERT v2 retrain?): **NO**. final_dataset built without
  retraining PhoBERT v2.
- **Decision 2** (113K pseudo?): **NO**. Not included; round 3 added 985
  human_gold rows instead.
- **Decision 3** (human:pseudo ratio): human_gold weight=3, weak_high_conf
  weight=2, phobert_v2_confident weight=1. Consumed by
  `WeightedRandomSampler` in `phobert_train._build_train_loader`.
- **Decision 4** (uncertain/exclude?): **EXCLUDED** from train.
