# Visualization Proposal — Vietnamese Depression Detection Paper

> **Date:** 2026-06-28
> **Source:** Review of `docs/paper_report.html` (1,615 lines) and `report_pdf/Report (1).pdf` (14 pages)
> **Purpose:** Identify visualization opportunities to enrich the paper with publication-quality figures and tables.

---

## 0. Tổng quan

Báo cáo hiện tại **gần như chỉ có bảng** (tables), **không có figures** nào. Điều này khiến paper trông "khô khan" và khó truyền tải insight trực quan. Đề xuất bổ sung **7 figures + 0 tables mới** theo APA 7.0 / academic convention.

**Quy ước đặt tên (theo APA 7.0 + best practice academic):**
- Format: `fig-<chapter>-<number>.<extension>` (lowercase, dash-separated)
- Số thứ tự theo **chapter xuất hiện đầu tiên** (không reset theo chapter trong HTML, nhưng PDF có thể dùng `Figure 3.1, 3.2` style)
- Lưu trong `report_pdf/figures/` (đặt cùng folder với PDF output)
- Định dạng: **PNG** (cho paper Word/PDF) hoặc **SVG** (cho LaTeX/web — vector, scalable)
- Resolution: **300 DPI** cho publication
- Color palette: **colorblind-safe** (theo skill `academic-paper` Phase 4 visualization_agent)

---

## 1. Hình ảnh đề xuất — Chi tiết

### 🎯 Figure 1 — Data Pipeline Overview (System Architecture)
**Đặt tên:** `fig-01-pipeline-overview.png`
**Chapter xuất hiện:** §1.3.3 (Chapter 3) — sau khi insert body
**Loại:** Horizontal flow diagram (boxes + arrows)
**Mục đích:** Minh họa toàn bộ 7-stage pipeline (Crawl → Clean → External → Corpus → Weak-label → Annotate → Train)
**Nguồn:** Code path trong repo (đã có ASCII diagram trong HTML §4.1, line 924-945)

**Thiết kế:**
```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Stage 1 │    │  Stage 2 │    │  Stage 3 │    │  Stage 4 │    │  Stage 5 │
│  YouTube │ →  │  Clean   │ →  │ External │ →  │ Unified  │ →  │ Weak-    │
│  Crawl   │    │  & Dedupe│    │  Datasets│    │  Corpus  │    │  Label   │
│ 125K     │    │  125K    │    │  191K    │    │  316K    │    │  3-tier  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                                    ↓
┌──────────┐    ┌──────────┐
│  Stage 7 │    │  Stage 6 │
│  Train   │ ←  │ Blind    │
│ 5 models │    │ Annotate │
│          │    │ 1,750 → 1,607 gold │
└──────────┘    └──────────┘
```

**Tool:** draw.io (miễn phí) hoặc PowerPoint
**Style:** Boxes có màu theo phase, mũi tên đậm, label rõ ràng

---

### 🎯 Figure 2 — Weak-Labeler Score Distribution
**Đặt tên:** `fig-02-weak-label-distribution.png`
**Chapter xuất hiện:** §3.5.5 (Weak-labeling label distribution)
**Loại:** Stacked horizontal bar chart (3 tiers × 2 confidence levels)
**Mục đích:** Minh họa phân phối labels từ Table 3.8 (3,223 depression_auto, 23,695 normal_auto, 98,410 uncertain)
**Nguồn data:** Weak-labeler output trên 125,329 comments (Table 3.8 trong HTML §3.5.5)

**Thiết kế:**
```
depression_auto:  ▓▓▓░░░░░░░░ 3,223 (2.57%)
  high confidence:▓░░░░░░░░░░   779 (0.62%)
  medium conf:    ▓▓▓░░░░░░░   2,444 (1.95%)
normal_auto:     ▓▓▓▓▓▓▓▓░░░ 23,695 (18.91%)
  high conf:      ▓▓▓░░░░░░░   3,449 (2.75%)
  medium conf:    ▓▓▓▓▓▓░░░░  20,246 (16.16%)
uncertain:       ▓▓▓▓▓▓▓▓▓▓▓ 98,410 (78.52%)
```

**Tool:** matplotlib (Python) — sẽ viết script `scripts/plot_label_distribution.py`
**Style:** Colorblind-safe palette (orange/blue/grey), percentage labels ở cuối mỗi bar

---

### 🎯 Figure 3 — Confusion Matrix Heatmap (5 models × 2 test sets = 10 heatmaps)
**Đặt tên:** `fig-03-confusion-matrices.png` (panel of 10) hoặc `fig-03a-03j-cm-*.png` (10 ảnh riêng)
**Chapter xuất hiện:** §5.4 (Error Analysis) hoặc Appendix
**Loại:** 2×5 grid of confusion matrix heatmaps
**Mục đích:** Minh họa error patterns (FP/FN) cho 5 models × 2 test sets
**Nguồn data:** Confusion matrices từ `models/*/metrics.json`

**Thiết kế:**
```
                    In-domain (383)        Cross-domain (3,084)
TF-IDF+LogReg:      [TP 107  FN 20]        [TP 12   FN 1530]
                    [FP 38   TN 218]       [FP 0    TN 1542]
BiLSTM random:      [...]
BiLSTM PhoBERT:     [...]
PhoBERT:            [TP 100  FN 27]        [TP 17   FN 1525]
                    [FP 17   TN 239]       [FP 0    TN 1542]
BERTopic:           [near-random]
PhoBERT+BERTopic:   [TP 99   FN 28]
```

**Tool:** matplotlib + seaborn heatmap
**Style:** Blues colormap, value annotations, normalized rows

---

### 🎯 Figure 4 — Generalization Gap Visualization
**Đặt tên:** `fig-04-generalization-gap.png`
**Chapter xuất hiện:** §5.2 (Analysis of the Generalization Gap) — ĐÂY LÀ FIGURE QUAN TRỌNG NHẤT
**Loại:** Grouped bar chart với error bars (in-domain vs cross-domain cho 5 models)
**Mục đích:** Minh họa rõ ràng 0.50-0.53 F1 gap giữa in-domain và cross-domain
**Nguồn data:** Table 5.1 (HTML §5.1)

**Thiết kế:**
```
F1-macro
1.0 ┤
    │  ▓▓▓▓          (in-domain)
0.8 ┤  ▓▓▓▓  ▓▓▓▓
    │  ▓▓▓▓  ▓▓▓▓  ▓▓▓▓
0.6 ┤  ▓▓▓▓  ▓▓▓▓  ▓▓▓▓
    │  ▓▓▓▓  ▓▓▓▓  ▓▓▓▓  ▓▓▓▓  ▓▓▓▓
0.4 ┤  ▓▓▓▓  ▓▓▓▓  ▓▓▓▓  ▓▓▓▓  ▓▓▓▓
    │  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  (cross-domain)
0.2 ┤  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░
    │  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░
0.0 ┴─────────────────────────────
     TFIDF  BiLST  BiLST  Pho   Pho+
     LogR   rand   phob   BERT  BERTopic
            ←gap→ ←0.50 F1-macro→
```

**Tool:** matplotlib bar chart với error bars
**Style:** Blue (in-domain) vs orange (cross-domain), gap annotation arrows

---

### 🎯 Figure 5 — BERTopic Topic Distribution (Bivariate Chart)
**Đặt tên:** `fig-05-bertopic-topics.png`
**Chapter xuất hiện:** §5.3 (BERTopic Results)
**Loại:** Horizontal bar chart (top 20 topics by size) với color-coding theo depression-relevance
**Mục đích:** Minh họa topic distribution + identify depression-related clusters
**Nguồn data:** `models/bertopic/bertopic_metrics.json` (456 topics, 48.30% outliers)

**Thiết kế:**
```
Topic 7  (sleep/doctor)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 1,687
Topic 14 (sadness/music) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓   1,214
Topic 19 (burnout/work)  ▓▓▓▓▓▓▓▓▓▓▓▓      1,052
Topic 27 (loneliness)    ▓▓▓▓▓▓▓▓▓          844
Topic 33 (clinical de.)  ▓▓▓▓▓▓▓▓           749
Topic 0  (restaurant)    ▓▓▓▓▓▓▓            618
...
Topic -1 (outlier)       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 149,650 (48.30%)
```

**Tool:** matplotlib horizontal bar chart
**Style:** Depression-related topics màu đỏ/cam, normal topics màu xám, outliers riêng

---

### 🎯 Figure 6 — DAPT Counter-Experiment Visualization
**Đặt tên:** `fig-06-dapt-counter-experiment.png`
**Chapter xuất hiện:** §5.5 (DAPT)
**Loại:** Scatter plot với error bars (per-seed points) + connecting lines
**Mục đích:** Minh họa directional reversal của DAPT effect
**Nguồn data:** `results/domain_adapted_eval_2026-06-26_181310/`

**Thiết kế:**
```
F1-macro
0.95 ┤       ● PhoBERT+DAPT (seed 42,123,2024)
0.90 ┤   ●─PhoBERT original
     │   ●─
0.85 ┤   ●
     │       ●──●  (in-domain)
0.80 ┤       ●
     │   ●───●      ▲ DAPT slight gain
0.75 ┤
     │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
0.40 ┤   ● ── PhoBERT (cross-domain)
0.38 ┤       ● ── PhoBERT+DAPT
0.35 ┤
     └──────────────────────────
     Δ in-domain = +0.0122 (not sig, p=0.21)
     Δ cross-domain = -0.0107 (not sig, p=0.34)
```

**Tool:** matplotlib scatter + lines
**Style:** Blue (original) vs red (DAPT), connecting lines giữa các seed, error bars std

---

### 🎯 Figure 7 — Annotation Protocol Diagram
**Đặt tên:** `fig-07-annotation-protocol.png`
**Chapter xuất hiện:** §3.6 (Blind Human Annotation)
**Loại:** Workflow diagram (comparing old contaminated vs new blind protocol)
**Mục đích:** Minh họa "annotation contamination" finding + blind protocol solution
**Nguồn:** HTML §3.6.1 (motivation paragraph)

**Thiết kế:**
```
Old (Contaminated):
   Comment + machine label
       ↓
   Annotator sees BOTH
       ↓
   96.3% agree → F1 = 1.0 (FAKE!)

New (Blind Protocol):
   Comment only (+ row_id)
       ↓
   Annotator judges independently
       ↓
   Cohen's κ = 0.63 (Batch 5)
   Cohen's κ = -0.03 (Batch 8, active learning)
       ↓
   Genuine independent labels
```

**Tool:** draw.io / PowerPoint
**Style:** Side-by-side comparison, color-coded (red = old, green = new)

---

## 2. Tables đề xuất bổ sung (không bắt buộc)

| ID | Tên | Mục đích | Vị trí chèn |
|---|---|---|---|
| Table A | Keyword frequency distribution | Top 20 most-matched keywords trong weak-labeler (cảm ơn 14,185, hay quá 4,621, etc.) | §3.5.5 |
| Table B | Cohen's Kappa per bucket | Chi tiết disagreement rate cho 5 review buckets (depression_high, normal_high, uncertain, need_review, boundary) | §3.6.4 |
| Table C | Confusion matrix details | Full TP/FP/FN/TN cho 5 models × 2 test sets (10 matrices) | §5.4 hoặc Appendix |

---

## 3. Hình ảnh có sẵn — KHÔNG cần tạo mới

| Đã có trong HTML | Trạng thái |
|---|---|
| Figure 4.1 — Pipeline ASCII diagram (HTML §4.1) | ✅ Đã có trong PDF (page 13) |
| Table 5.1, 5.2, 5.3 | ✅ Đã có |
| Tables 1-6 (Chapter 3) | ✅ Đã có |

---

## 4. Tóm tắt — Figures ưu tiên

### 🔴 Ưu tiên cao (must-have cho paper submission)

| # | Figure | Effort | Impact |
|---|---|---|---|
| 4 | **Generalization Gap** (Figure 4) | 1-2h matplotlib script | Cao — visualization chính của paper |
| 5 | **BERTopic Topics** (Figure 5) | 1-2h matplotlib | Cao — minh họa thematic structure |
| 2 | **Weak-label distribution** (Figure 2) | 1h matplotlib | Trung bình — class imbalance insight |

### 🟠 Ưu tiên trung bình

| # | Figure | Effort | Impact |
|---|---|---|---|
| 1 | **Pipeline Overview** (Figure 1) | 2-3h draw.io | Cao — system architecture |
| 3 | **Confusion Matrices** (Figure 3) | 2-3h matplotlib | Trung bình — error analysis |
| 6 | **DAPT Counter-Experiment** (Figure 6) | 1-2h matplotlib | Trung bình |

### 🟡 Optional

| # | Figure | Effort | Impact |
|---|---|---|---|
| 7 | **Annotation Protocol** (Figure 7) | 2h draw.io | Trung bình — methodology detail |

---

## 5. Lệnh tạo figures (khi cài matplotlib)

```bash
cd /Users/iannwendy/Documents/crawl_yt/youtube_depression_crawler

# Cài matplotlib + seaborn
pip3 install --user --break-system-packages matplotlib seaborn

# Script gợi ý (sẽ viết nếu bạn confirm):
python3 scripts/plot_label_distribution.py       → fig-02
python3 scripts/plot_generalization_gap.py       → fig-04
python3 scripts/plot_bertopic_topics.py           → fig-05
python3 scripts/plot_dapt_experiment.py          → fig-06
python3 scripts/plot_confusion_matrices.py        → fig-03
```

Output: `report_pdf/figures/fig-NN-name.png` (300 DPI, colorblind-safe)

---

## 6. Đặt tên files — Convention chi tiết

Theo APA 7.0 / ICLR / NeurIPS conventions:

```
report_pdf/figures/
├── fig-01-pipeline-overview.png        (System architecture)
├── fig-02-weak-label-distribution.png   (Class distribution)
├── fig-03-confusion-matrices.png       (Error analysis)
├── fig-04-generalization-gap.png       (Main finding)
├── fig-05-bertopic-topics.png          (Topic distribution)
├── fig-06-dapt-counter-experiment.png  (DAPT effect)
└── fig-07-annotation-protocol.png      (Methodology)
```

**Trong paper caption, dùng format:**
- `Figure 1. Pipeline overview of the seven-stage data processing pipeline.`
- `Figure 4. Generalization gap across five model architectures.`
- Số thứ tự **reset theo chapter** nếu theo APA 7.0 strict (Figure 1.1, 1.2...) hoặc **liên tục** (Figure 1, 2, 3...) nếu theo ICLR/NeurIPS convention.

**Recommendation:** Dùng **continuous numbering** (Figure 1, 2, ...) cho dễ reference và consistent với Table 1, 2, ...

---

## 7. Tóm tắt — Bạn cần làm gì tiếp theo?

1. **Chọn figures muốn tạo** (recommend bắt đầu với Figure 4 + Figure 5 — highest impact)
2. **Cài matplotlib + seaborn** trên máy (1 lệnh pip)
3. **Tôi viết scripts Python** để generate các figures từ JSON metrics files đã có
4. **Insert figures vào HTML** tại đúng vị trí chapter
5. **Regenerate PDF** từ HTML

Bạn muốn tôi bắt đầu với Figure 4 (Generalization Gap) — visualization quan trọng nhất — không?

---

*End of visualization proposal. Last updated 2026-06-28.*