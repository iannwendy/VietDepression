# Training Report - July 13, 2026

## M4 Pro MPS Training Results

### PhoBERT Model
- **Device**: Apple M4 Pro (MPS)
- **Training Time**: 7.3 minutes
- **Best Epoch**: 1
- **Best Val F1**: 0.9837
- **Test F1**: 0.9752
- **Test Accuracy**: 0.9894

### Baseline Model (TF-IDF + Logistic Regression)
- **Device**: CPU
- **Training Time**: ~2 seconds
- **Val F1**: 0.8373
- **Test F1**: 0.8373
- **Test Accuracy**: 0.8947

### Comparison

| Model | Val F1 | Test F1 | Test Acc | Training Time |
|-------|--------|---------|---------|--------------|
| PhoBERT (M4 Pro) | 0.9837 | 0.9752 | 0.9894 | 7.3 min |
| Baseline | 0.8373 | 0.8373 | 0.8947 | ~2 sec |

### PhoBERT Training History

| Epoch | Train Loss | Val F1 |
|-------|-----------|--------|
| 1 | 0.2769 | 0.9837 |
| 2 | 0.1021 | 0.9758 |
| 3 | 0.0339 | 0.9837 |

### Manual Test Results (PhoBERT)

| Text | Prediction | Expected | Correct |
|------|------------|----------|---------|
| "Tôi cảm thấy rất vui hôm nay!" | normal | normal | ✓ |
| "Cuộc sống này thật vô nghĩa..." | depression | depression | ✓ |
| "Chán quá, không biết làm gì..." | depression | normal | ✗ |
| "Tôi bị insomnia suốt tuần..." | depression | depression | ✓ |
| "Đi du lịch với gia đình..." | normal | normal | ✓ |
| "Mỗi ngày đều như một gánh nặng..." | depression | depression | ✓ |

**Accuracy**: 5/6 (83%)

### Files Updated

- `models/phobert_first/training_results.json`
- `models/tfidf_logreg.joblib`
- `models/baseline_metrics.json`

### Colab Setup Files Added

- `COLAB_SETUP.md` - Hướng dẫn chi tiết
- `colab_training.ipynb` - Notebook training
- `colab_helper.py` - Script helper
- `prepare_colab_data.py` - Data preparation

### Notes

- PhoBERT training successfully runs on Apple M4 Pro using MPS backend
- M4 Pro significantly faster than expected (~7 min vs estimated 20-30 min)
- All models are performing well with F1 > 0.97 for PhoBERT
