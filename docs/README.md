# Docs Organization

This directory contains all documentation, reports, datasets, and scripts for the Vietnamese Depression Detection project.

## Directory Structure

```
docs/
├── ANNOTATION_GUIDELINES.md       # Public Four-Rule Annotation Rubric & Adjudication Protocol
├── _archive/                      # Archived/backup files
│   └── figures_backup_20260718/  # Old figure backups
│
├── active_learning/               # Active learning reports & round metadata
│   │                              # (Raw Label Studio CSV exports are git-ignored under DATA_USE_AGREEMENT.md)
│   ├── round3/                   # Round 3 selection reports
│   ├── round4/                   # Round 4 selection reports
│   ├── round5/                   # Round 5 selection reports
│   └── steps/
│       ├── step5_review/        # Step 5 review metadata
│       └── step8_active_learning/ # Step 8 active-learning metadata
│
├── appendices/                    # Paper appendices (privacy-preserving paraphrased composites)
│   ├── APPENDIX_*.md            # Main appendices (errors, hyperparameters, keywords)
│   ├── manual_review/           # Label Studio review guides
│   └── review_decision/         # Review decision checklists
│
├── final_fig/                    # Final figures for paper (h1-h9)
│
├── phases/                       # Phase evaluation reports & scripts
│   ├── phase1_eval_report.json
│   ├── phase2_report.json
│   ├── phase3/                  # Phase 3 model training scripts
│   ├── phase3_comparison_report.json
│   └── phase3_phobert_bertopic_metrics.json
│
├── report_pdf/                   # Figures for PDF report
│
├── reports/                      # Progress, round, and training reports
│   ├── FINAL_RESULTS_SUMMARY.md
│   ├── PDF_vs_HTML_comparison_report.md
│   ├── ROADMAP_SAU_REVIEW.md
│   ├── VISUALIZATION_PROPOSAL.md
│   ├── progress/                # Weekly progress reports (YYYY-MM-DD)
│   ├── round/                   # Round selection reports
│   └── training/               # Training reports
│
├── scripts/                      # Python utility scripts
│   ├── merge_round4_active_learning.py
│   ├── merge_round5_active_learning.py
│   ├── phase1_merge_review.py
│   ├── phase2_build_final_dataset.py
│   ├── prepare_label_studio_import.py
│   ├── prepare_round3_active_learning.py
│   └── prepare_round4_active_learning.py
│
├── superpowers/                  # Claude superpowers plans & specs
│   ├── plans/
│   └── specs/
│
└── paper_report.html             # Paper HTML report
```

## File Naming Conventions

### Reports (MD files)
- Progress reports: `PROGRESS_REPORT_YYYY-MM-DD.md`
- Round reports: `round{N}_report.md` or `round{N}_selection_report.json`
- Training reports: `TRAINING_REPORT_YYYY-MM-DD.md`

### Active Learning Data (Local Tier-2 DUA CSVs — Git-Ignored)
> Raw Label Studio imports/exports containing verbatim YouTube comments are restricted locally and excluded from public Git tracking (`DATA_USE_AGREEMENT.md`). Public de-identified split manifests (`record_sha256`, `label`) are published in `data/manifests/`.
- Import files (local): `label_studio_round{N}_active_learning_import.csv`
- Key files (local): `label_studio_round{N}_active_learning_key.csv`
- Merged files (local): `*_MERGED.csv`

### Phase Reports (JSON)
- Phase evaluations: `phase{N}_*_report.json`
- Metrics: `phase{N}_*_metrics.json`

## Last Reorganized

2026-11-06: Synced with Two-Tier DUA governance (`DATA_USE_AGREEMENT.md`) and public annotation guidelines (`docs/ANNOTATION_GUIDELINES.md`).
