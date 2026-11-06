# data/_archive/

Các file CSV/CSV-backup đã được superseded bởi phiên bản mới hơn.
**KHÔNG ĐỌC/TRỎ TỚI TỪ CODE** — chỉ giữ cho mục đích audit và reproduce
lại pipeline trước Phase 1.

| File | Superseded by | Ghi chú |
|---|---|---|
| `phobert_confident_predictions_v1_obsolete.csv` | `../phobert_confident_predictions.csv` | v1 dùng confidence threshold 0.90 + criterion cũ; v2 thêm sample weighting. `final_dataset.csv` chỉ dùng v2 (`source=phobert_v2_confident`). |
| `phobert_remaining_predictions_v1_obsolete.csv` | `../phobert_remaining_predictions.csv` | Output đợt 1 của `phobert_predict.predict_remaining_comments`; bị thay thế bởi đợt predict thứ 2 trước Phase 1. |
| `review_samples.backup_20260608_195538.csv` | `../review_samples.csv` | Snapshot review samples trước khi Label Studio export merge. |
| `review_samples.backup_before_phase1.csv` | `../review_samples.csv` | Snapshot ngay trước khi merge Step 5 + Step 8 human labels vào `final_label`. |
| `phobert_active_learning_samples.backup_before_phase1.csv` | `../phobert_active_learning_samples.csv` | Snapshot active-learning file (1.000 mẫu khó) trước khi reviewer điền `final_label`. |

**Lý do giữ lại:** reproducibility — nếu cần debug tại sao một sample bị
final_dataset loại ra, có thể quay lại snapshot cũ và re-merge.