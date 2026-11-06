# Web-Demo i18n Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Vietnamese ↔ English language switching to web-demo with clinical/research hybrid translation standard.

**Architecture:** React i18n using react-i18next with JSON locale files, persisted via localStorage. Language switcher component in header with segmented control style.

**Tech Stack:** react-i18next, i18next, TypeScript

---

## Global Constraints

- Use react-i18next v14.x (latest stable)
- Language preference persists in localStorage key: `language`
- Default language: Vietnamese (`vi`)
- All translation keys use dot notation (e.g., `nav.dashboard`)
- Do NOT translate: mockData content, PhoBERT, BERTopic, model names

---

## File Structure

```
web_demo/src/
├── i18n/
│   ├── index.ts                    # i18n configuration
│   ├── locales/
│   │   ├── vi.json                 # Vietnamese translations
│   │   └── en.json                 # English translations
│   └── keys.ts                     # Translation key constants
├── components/
│   └── layout/
│       └── LanguageSwitcher.tsx     # Segmented control component
```

Modified files:
- `web_demo/src/App.tsx` — wrap with I18nextProvider
- `web_demo/src/components/layout/Header.tsx` — add LanguageSwitcher
- `web_demo/src/pages/Dashboard.tsx` — translate UI
- `web_demo/src/pages/Prediction.tsx` — translate UI + risk levels
- `web_demo/src/pages/BatchPrediction.tsx` — translate labels
- `web_demo/src/pages/Topics.tsx` — translate topic names
- `web_demo/src/pages/Statistics.tsx` — translate chart labels
- `web_demo/src/pages/History.tsx` — translate labels
- `web_demo/src/pages/ModelComparison.tsx` — translate metrics

---

## Task 1: Install i18n Dependencies

**Files:**
- Modify: `web_demo/package.json`

**Interfaces:**
- Produces: `react-i18next` and `i18next` installed

- [ ] **Step 1: Install dependencies**

Run: `cd web_demo && npm install i18next react-i18next`

Expected: Successfully installed i18next@^24 and react-i18next@^15

- [ ] **Step 2: Commit**

```bash
cd web_demo && git add package.json package-lock.json && git commit -m "deps: add i18next and react-i18next"
```

---

## Task 2: Create Translation Key Constants

**Files:**
- Create: `web_demo/src/i18n/keys.ts`

**Interfaces:**
- Produces: `i18nKeys` object with all translation key paths as constants

- [ ] **Step 1: Create translation keys file**

```typescript
// web_demo/src/i18n/keys.ts
export const i18nKeys = {
  nav: {
    dashboard: 'nav.dashboard',
    prediction: 'nav.prediction',
    batch: 'nav.batch',
    topics: 'nav.topics',
    statistics: 'nav.statistics',
    history: 'nav.history',
    compare: 'nav.compare',
  },
  common: {
    depression: 'common.depression',
    normal: 'common.normal',
    confidence: 'common.confidence',
    risk: 'common.risk',
    topic: 'common.topic',
    loading: 'common.loading',
    error: 'common.error',
  },
  risk: {
    high: 'risk.high',
    medium: 'risk.medium',
    low: 'risk.low',
  },
  button: {
    analyze: 'button.analyze',
    process: 'button.process',
    download: 'button.download',
    remove: 'button.remove',
    upload: 'button.upload',
    search: 'button.search',
  },
  dashboard: {
    title: 'dashboard.title',
    subtitle: 'dashboard.subtitle',
    totalComments: 'dashboard.totalComments',
    currentModel: 'dashboard.currentModel',
    accuracy: 'dashboard.accuracy',
    macroF1: 'dashboard.macroF1',
    weightedF1: 'dashboard.weightedF1',
    trainingDate: 'dashboard.trainingDate',
    quickAnalysis: 'dashboard.quickAnalysis',
    batchProcessing: 'dashboard.batchProcessing',
  },
  prediction: {
    title: 'prediction.title',
    description: 'prediction.description',
    inputLabel: 'prediction.inputLabel',
    inputPlaceholder: 'prediction.inputPlaceholder',
    result: 'prediction.result',
    detectedTopic: 'prediction.detectedTopic',
    explanation: 'prediction.explanation',
    keywords: 'prediction.keywords',
  },
  batch: {
    title: 'batch.title',
    dropzone: 'batch.dropzone',
    format: 'batch.format',
    comment: 'batch.comment',
    processing: 'batch.processing',
  },
  topics: {
    title: 'topics.title',
    totalTopics: 'topics.totalTopics',
    topTopic: 'topics.topTopic',
    mostFrequent: 'topics.mostFrequent',
    coverage: 'topics.coverage',
    topKeywords: 'topics.topKeywords',
    examples: 'topics.examples',
    distribution: 'topics.distribution',
  },
  statistics: {
    title: 'statistics.title',
    predictionDist: 'statistics.predictionDist',
    topicDist: 'statistics.topicDist',
    confusionMatrix: 'statistics.confusionMatrix',
    metrics: 'statistics.metrics',
    precision: 'statistics.precision',
    recall: 'statistics.recall',
  },
  history: {
    title: 'history.title',
    searchPlaceholder: 'history.searchPlaceholder',
    filterAll: 'history.filterAll',
    filterDepression: 'history.filterDepression',
    filterNormal: 'history.filterNormal',
    noResults: 'history.noResults',
  },
  compare: {
    title: 'compare.title',
    bestModel: 'compare.bestModel',
    radar: 'compare.radar',
    accuracyCompare: 'compare.accuracyCompare',
    improvement: 'compare.improvement',
  },
  topicsResearch: {
    loneliness: 'topics.research.loneliness',
    academicStress: 'topics.research.academicStress',
    sleepDisturbance: 'topics.research.sleepDisturbance',
    workStress: 'topics.research.workStress',
    selfIsolation: 'topics.research.selfIsolation',
  },
} as const;

export type I18nKeys = typeof i18nKeys;
```

- [ ] **Step 2: Commit**

```bash
git add web_demo/src/i18n/keys.ts && git commit -m "feat(i18n): add translation key constants"
```

---

## Task 3: Create Vietnamese Translation File

**Files:**
- Create: `web_demo/src/i18n/locales/vi.json`

**Interfaces:**
- Produces: `vi.json` with all Vietnamese translations

- [ ] **Step 1: Create Vietnamese locale file**

```json
{
  "nav": {
    "dashboard": "Trang chủ",
    "prediction": "Phân tích văn bản",
    "batch": "Xử lý hàng loạt",
    "topics": "Phân tích chủ đề",
    "statistics": "Thống kê",
    "history": "Lịch sử",
    "compare": "So sánh mô hình"
  },
  "common": {
    "depression": "Trầm cảm",
    "normal": "Bình thường",
    "confidence": "Độ tin cậy",
    "risk": "Mức độ rủi ro",
    "topic": "Chủ đề",
    "loading": "Đang xử lý...",
    "error": "Đã xảy ra lỗi"
  },
  "risk": {
    "high": "Nguy cơ cao",
    "medium": "Nguy cơ trung bình",
    "low": "Nguy cơ thấp"
  },
  "button": {
    "analyze": "Phân tích",
    "process": "Xử lý file",
    "download": "Tải xuống",
    "remove": "Xóa",
    "upload": "Tải lên",
    "search": "Tìm kiếm"
  },
  "dashboard": {
    "title": "Mental Health AI",
    "subtitle": "Nền tảng phát hiện trầm cảm",
    "totalComments": "Tổng bình luận",
    "currentModel": "Mô hình hiện tại",
    "accuracy": "Độ chính xác",
    "macroF1": "Macro F1",
    "weightedF1": "Weighted F1",
    "trainingDate": "Ngày huấn luyện",
    "quickAnalysis": "Phân tích nhanh",
    "batchProcessing": "Xử lý hàng loạt"
  },
  "prediction": {
    "title": "Phân tích văn bản",
    "description": "Nhập văn bản để phân tích và phát hiện dấu hiệu trầm cảm",
    "inputLabel": "Văn bản đầu vào",
    "inputPlaceholder": "Nhập bình luận hoặc văn bản cần phân tích...",
    "result": "Kết quả",
    "detectedTopic": "Chủ đề được phát hiện",
    "explanation": "Giải thích từ AI",
    "keywords": "Từ khóa nổi bật"
  },
  "batch": {
    "title": "Xử lý hàng loạt",
    "dropzone": "Kéo file vào đây hoặc click để tải lên",
    "format": "Định dạng CSV mong đợi",
    "comment": "Bình luận",
    "processing": "Đang xử lý file..."
  },
  "topics": {
    "title": "Phân tích chủ đề",
    "totalTopics": "Tổng số chủ đề",
    "topTopic": "Chủ đề hàng đầu",
    "mostFrequent": "Phổ biến nhất",
    "coverage": "Độ phủ",
    "topKeywords": "Từ khóa hàng đầu",
    "examples": "Ví dụ bình luận",
    "distribution": "Phân bố chủ đề"
  },
  "statistics": {
    "title": "Thống kê & Phân tích",
    "predictionDist": "Phân bố dự đoán",
    "topicDist": "Phân bố chủ đề",
    "confusionMatrix": "Ma trận nhầm lẫn",
    "metrics": "Các chỉ số",
    "precision": "Độ chính xác",
    "recall": "Độ phủ"
  },
  "history": {
    "title": "Lịch sử dự đoán",
    "searchPlaceholder": "Tìm kiếm...",
    "filterAll": "Tất cả",
    "filterDepression": "Trầm cảm",
    "filterNormal": "Bình thường",
    "noResults": "Không tìm thấy kết quả"
  },
  "compare": {
    "title": "So sánh mô hình",
    "bestModel": "Mô hình hoạt động tốt nhất",
    "radar": "Biểu đồ radar hiệu suất",
    "accuracyCompare": "So sánh độ chính xác",
    "improvement": "Cải thiện"
  },
  "topics": {
    "research": {
      "loneliness": "Cô đơn",
      "academicStress": "Áp lực học tập",
      "sleepDisturbance": "Mất ngủ",
      "workStress": "Công việc",
      "selfIsolation": "Tự kỷ"
    }
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add web_demo/src/i18n/locales/vi.json && git commit -m "feat(i18n): add Vietnamese locale"
```

---

## Task 4: Create English Translation File

**Files:**
- Create: `web_demo/src/i18n/locales/en.json`

**Interfaces:**
- Produces: `en.json` with all English translations (clinical/research hybrid standard)

- [ ] **Step 1: Create English locale file**

```json
{
  "nav": {
    "dashboard": "Dashboard",
    "prediction": "Text Analysis",
    "batch": "Batch Processing",
    "topics": "Topic Analysis",
    "statistics": "Statistics",
    "history": "History",
    "compare": "Model Comparison"
  },
  "common": {
    "depression": "Major Depressive Disorder",
    "normal": "Non-Depressed",
    "confidence": "Confidence",
    "risk": "Risk Level",
    "topic": "Topic",
    "loading": "Processing...",
    "error": "An error occurred"
  },
  "risk": {
    "high": "High Risk",
    "medium": "Moderate Risk",
    "low": "Low Risk"
  },
  "button": {
    "analyze": "Analyze",
    "process": "Process File",
    "download": "Download",
    "remove": "Remove",
    "upload": "Upload",
    "search": "Search"
  },
  "dashboard": {
    "title": "Mental Health AI",
    "subtitle": "Depression Detection Platform",
    "totalComments": "Total Comments",
    "currentModel": "Current Model",
    "accuracy": "Accuracy",
    "macroF1": "Macro F1",
    "weightedF1": "Weighted F1",
    "trainingDate": "Training Date",
    "quickAnalysis": "Quick Analysis",
    "batchProcessing": "Batch Processing"
  },
  "prediction": {
    "title": "Text Analysis",
    "description": "Enter text to analyze and detect depression indicators",
    "inputLabel": "Input Text",
    "inputPlaceholder": "Enter a comment or text to analyze...",
    "result": "Result",
    "detectedTopic": "Detected Topic",
    "explanation": "AI Explanation",
    "keywords": "Highlighted Keywords"
  },
  "batch": {
    "title": "Batch Prediction",
    "dropzone": "Drop your file here or click to upload",
    "format": "Expected CSV format",
    "comment": "Comment",
    "processing": "Processing file..."
  },
  "topics": {
    "title": "Topic Analysis",
    "totalTopics": "Total Topics",
    "topTopic": "Top Topic",
    "mostFrequent": "Most Frequent",
    "coverage": "Coverage",
    "topKeywords": "Top Keywords",
    "examples": "Example Comments",
    "distribution": "Topic Distribution"
  },
  "statistics": {
    "title": "Statistics & Analytics",
    "predictionDist": "Prediction Distribution",
    "topicDist": "Topic Distribution",
    "confusionMatrix": "Confusion Matrix",
    "metrics": "Metrics",
    "precision": "Precision",
    "recall": "Recall"
  },
  "history": {
    "title": "Prediction History",
    "searchPlaceholder": "Search...",
    "filterAll": "All",
    "filterDepression": "Depression",
    "filterNormal": "Normal",
    "noResults": "No predictions found"
  },
  "compare": {
    "title": "Model Comparison",
    "bestModel": "Best Performing Model",
    "radar": "Performance Radar",
    "accuracyCompare": "Accuracy Comparison",
    "improvement": "Improvement"
  },
  "topics": {
    "research": {
      "loneliness": "Loneliness",
      "academicStress": "Academic Stress",
      "sleepDisturbance": "Sleep Disturbance",
      "workStress": "Work-Related Stress",
      "selfIsolation": "Self-Isolation"
    }
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add web_demo/src/i18n/locales/en.json && git commit -m "feat(i18n): add English locale"
```

---

## Task 5: Create i18n Configuration

**Files:**
- Create: `web_demo/src/i18n/index.ts`

**Interfaces:**
- Produces: Configured i18next instance, exported `useTranslation` hook, `changeLanguage` function

- [ ] **Step 1: Create i18n configuration**

```typescript
// web_demo/src/i18n/index.ts
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import vi from './locales/vi.json';
import en from './locales/en.json';

const LANGUAGE_KEY = 'language';
const DEFAULT_LANGUAGE = 'vi';

// Get saved language or default
const savedLanguage = localStorage.getItem(LANGUAGE_KEY) || DEFAULT_LANGUAGE;

i18n.use(initReactI18next).init({
  resources: {
    vi: { translation: vi },
    en: { translation: en },
  },
  lng: savedLanguage,
  fallbackLng: DEFAULT_LANGUAGE,
  interpolation: {
    escapeValue: false,
  },
});

// Save language preference when changed
i18n.on('languageChanged', (lng) => {
  localStorage.setItem(LANGUAGE_KEY, lng);
});

export default i18n;
export { changeLanguage } from 'i18next';
```

- [ ] **Step 2: Commit**

```bash
git add web_demo/src/i18n/index.ts && git commit -m "feat(i18n): add i18next configuration"
```

---

## Task 6: Create Language Switcher Component

**Files:**
- Create: `web_demo/src/components/layout/LanguageSwitcher.tsx`

**Interfaces:**
- Consumes: `useTranslation` hook from react-i18next
- Produces: LanguageSwitcher React component with segmented control

- [ ] **Step 1: Create LanguageSwitcher component**

```tsx
// web_demo/src/components/layout/LanguageSwitcher.tsx
import { useTranslation } from 'react-i18next';
import { changeLanguage } from '../../i18n';

const languages = [
  { code: 'vi', label: 'VI' },
  { code: 'en', label: 'EN' },
] as const;

export function LanguageSwitcher() {
  const { i18n } = useTranslation();
  const currentLang = i18n.language;

  return (
    <div className="flex items-center rounded-lg bg-slate-100 p-0.5 dark:bg-slate-800">
      {languages.map(({ code, label }) => (
        <button
          key={code}
          onClick={() => changeLanguage(code)}
          className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all duration-200 ${
            currentLang === code
              ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-white'
              : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add web_demo/src/components/layout/LanguageSwitcher.tsx && git commit -m "feat(i18n): add LanguageSwitcher component"
```

---

## Task 7: Integrate i18n into App.tsx

**Files:**
- Modify: `web_demo/src/App.tsx`

**Interfaces:**
- Consumes: `i18n` config from `./i18n`
- Produces: App wrapped with I18nextProvider

- [ ] **Step 1: Read current App.tsx**

```bash
cat web_demo/src/App.tsx
```

- [ ] **Step 2: Update App.tsx to import i18n**

Add at top of file:
```typescript
import './i18n';
```

- [ ] **Step 3: Commit**

```bash
git add web_demo/src/App.tsx && git commit -m "feat(i18n): import i18n configuration in App"
```

---

## Task 8: Add LanguageSwitcher to Header

**Files:**
- Modify: `web_demo/src/components/layout/Header.tsx`

**Interfaces:**
- Consumes: `LanguageSwitcher` component
- Produces: Header with language switcher on right side

- [ ] **Step 1: Read current Header.tsx**

```bash
cat web_demo/src/components/layout/Header.tsx
```

- [ ] **Step 2: Update Header.tsx**

Add import:
```typescript
import { LanguageSwitcher } from './LanguageSwitcher';
```

Add `<LanguageSwitcher />` before or after the nav links div, typically at the right end of the header.

- [ ] **Step 3: Commit**

```bash
git add web_demo/src/components/layout/Header.tsx && git commit -m "feat(i18n): add LanguageSwitcher to Header"
```

---

## Task 9: Translate Dashboard Page

**Files:**
- Modify: `web_demo/src/pages/Dashboard.tsx`

**Interfaces:**
- Consumes: `useTranslation` hook
- Produces: Dashboard with translated text

- [ ] **Step 1: Read Dashboard.tsx**

```bash
cat web_demo/src/pages/Dashboard.tsx
```

- [ ] **Step 2: Update Dashboard.tsx**

Add import:
```typescript
import { useTranslation } from 'react-i18next';
import { i18nKeys } from '../i18n/keys';
```

Replace hardcoded strings with:
```typescript
const { t } = useTranslation();
// ...
// Use t(i18nKeys.dashboard.title) instead of "Mental Health AI"
// Use t(i18nKeys.dashboard.totalComments) instead of "Tổng bình luận"
// etc.
```

- [ ] **Step 3: Commit**

```bash
git add web_demo/src/pages/Dashboard.tsx && git commit -m "feat(i18n): translate Dashboard page"
```

---

## Task 10: Translate Prediction Page

**Files:**
- Modify: `web_demo/src/pages/Prediction.tsx`

**Interfaces:**
- Consumes: `useTranslation` hook, risk level translations
- Produces: Prediction page with translated UI + risk levels

- [ ] **Step 1: Read Prediction.tsx**

```bash
cat web_demo/src/pages/Prediction.tsx
```

- [ ] **Step 2: Update Prediction.tsx**

Key translations:
- Risk levels: `t(i18nKeys.risk.high)`, `t(i18nKeys.risk.medium)`, `t(i18nKeys.risk.low)`
- Prediction results: `t(i18nKeys.common.depression)`, `t(i18nKeys.common.normal)`
- Topic translations: Use `t()` for detected topics

- [ ] **Step 3: Commit**

```bash
git add web_demo/src/pages/Prediction.tsx && git commit -m "feat(i18n): translate Prediction page"
```

---

## Task 11: Translate Remaining Pages

**Files:**
- Modify: `web_demo/src/pages/BatchPrediction.tsx`
- Modify: `web_demo/src/pages/Topics.tsx`
- Modify: `web_demo/src/pages/Statistics.tsx`
- Modify: `web_demo/src/pages/History.tsx`
- Modify: `web_demo/src/pages/ModelComparison.tsx`

**Interfaces:**
- Consumes: `useTranslation` hook
- Produces: All pages with translated text

- [ ] **Step 1: Read and update each file**

For each page:
1. Add import: `import { useTranslation } from 'react-i18next';` and `import { i18nKeys } from '../i18n/keys';`
2. Add `const { t } = useTranslation();`
3. Replace hardcoded strings with `t(i18nKeys.*)`

- [ ] **Step 2: Commit each file individually**

```bash
git add web_demo/src/pages/BatchPrediction.tsx && git commit -m "feat(i18n): translate BatchPrediction page"
git add web_demo/src/pages/Topics.tsx && git commit -m "feat(i18n): translate Topics page"
git add web_demo/src/pages/Statistics.tsx && git commit -m "feat(i18n): translate Statistics page"
git add web_demo/src/pages/History.tsx && git commit -m "feat(i18n): translate History page"
git add web_demo/src/pages/ModelComparison.tsx && git commit -m "feat(i18n): translate ModelComparison page"
```

---

## Task 12: Test Language Switching

**Interfaces:**
- Produces: Verified working language switcher

- [ ] **Step 1: Start dev server**

Run: `cd web_demo && npm run dev`

- [ ] **Step 2: Verify Vietnamese is default**

Open browser, verify all text shows Vietnamese.

- [ ] **Step 3: Click EN on language switcher**

Verify all text switches to English.

- [ ] **Step 4: Refresh page**

Verify language preference persists (stays on English).

- [ ] **Step 5: Click VI to switch back**

Verify Vietnamese displays correctly.

---

## Spec Coverage Checklist

| Requirement | Task |
|-------------|------|
| Install react-i18next | Task 1 |
| Create translation keys | Task 2 |
| Create vi.json | Task 3 |
| Create en.json | Task 4 |
| Create i18n config | Task 5 |
| Create LanguageSwitcher | Task 6 |
| Integrate into App | Task 7 |
| Add to Header | Task 8 |
| Translate Dashboard | Task 9 |
| Translate Prediction | Task 10 |
| Translate all pages | Task 11 |
| Test switching | Task 12 |
