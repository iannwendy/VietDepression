# Web-Demo i18n Design Specification

## Overview
Add Vietnamese ↔ English language switching to the web-demo application using a hybrid clinical/research translation standard.

## Tech Stack
- **Library**: `react-i18next` (mature, well-documented, minimal bundle impact)
- **Pattern**: File-based translation with JSON locale files
- **Persistence**: `localStorage` for language preference

## File Structure
```
src/
├── i18n/
│   ├── index.ts              # i18n configuration
│   ├── locales/
│   │   ├── vi.json           # Vietnamese translations
│   │   └── en.json           # English translations
│   └── translationKeys.ts    # Centralized translation key constants
├── components/
│   └── layout/
│       └── LanguageSwitcher.tsx  # Segmented control component
```

## Language Switcher
- **Style**: Segmented control pill-shaped (VI | EN)
- **Position**: Header, right side
- **Behavior**: Click toggle directly switches languages
- **Active state**: Highlighted background for current language

## Translation Standards

### UI Labels (Clinical Standard)
| Key | VI | EN |
|-----|----|----|
| `nav.dashboard` | Trang chủ | Dashboard |
| `nav.prediction` | Phân tích văn bản | Text Analysis |
| `nav.batch` | Xử lý hàng loạt | Batch Processing |
| `nav.topics` | Phân tích chủ đề | Topic Analysis |
| `nav.statistics` | Thống kê | Statistics |
| `nav.history` | Lịch sử | History |
| `nav.compare` | So sánh mô hình | Model Comparison |
| `common.depression` | Trầm cảm | Major Depressive Disorder |
| `common.normal` | Bình thường | Non-Depressed |
| `risk.high` | Nguy cơ cao | High Risk |
| `risk.medium` | Nguy cơ trung bình | Moderate Risk |
| `risk.low` | Nguy cơ thấp | Low Risk |
| `button.analyze` | Phân tích | Analyze |
| `button.process` | Xử lý file | Process File |
| `button.download` | Tải xuống | Download |
| `button.remove` | Xóa | Remove |

### Extracted Topics (Research/NLP Standard)
| VI | EN |
|----|----|
| Cô đơn | Loneliness |
| Áp lực học tập | Academic Stress |
| Mất ngủ | Sleep Disturbance |
| Công việc | Work-Related Stress |
| Tự kỷ | Self-Isolation |

## Components to Modify
1. `Header.tsx` — add LanguageSwitcher
2. `Prediction.tsx` — translate UI + risk levels + explanations
3. `BatchPrediction.tsx` — translate labels + placeholders
4. `Topics.tsx` — translate topic names + descriptions
5. `Statistics.tsx` — translate chart labels + metrics
6. `History.tsx` — translate labels + placeholders
7. `ModelComparison.tsx` — translate metrics + titles
8. `Dashboard.tsx` — translate stats + descriptions

## Not Translated
- Vietnamese comment samples in mockData (input data)
- Technical model names: PhoBERT, BERTopic, vinai/phobert-base-v2

## Implementation Steps
1. Install `react-i18next` and `i18next`
2. Create i18n configuration (`src/i18n/index.ts`)
3. Create translation files (`vi.json`, `en.json`)
4. Create translation keys constants
5. Create LanguageSwitcher component
6. Integrate into Header
7. Update all page components to use translations
8. Test language switching
