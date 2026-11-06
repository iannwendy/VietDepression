# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-14

### Added
- Paper finalization: `paper_report.html` with complete sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion)
- Appendix A: Complete keyword lists for depression-medium and depression-strong lexicons
- Appendix B: Model hyperparameters for all trained models (PhoBERT, BiLSTM, TF-IDF)
- Appendix C: Error examples with detailed analysis of False Positive and False Negative cases
- Round 5 active learning results (1,533 samples selected using probability_closest_to_0.5 strategy)
- Round 5 selection report and merge scripts

### Changed
- Final dataset expanded to 6,079 samples across all rounds (4,255 train / 912 val / 912 test)
- Headline F1 updated: 0.84 in-domain, 0.39 cross-domain (VSMEC)
- Generalization gap confirmed at ~0.50 F1-macro across all supervised models

### Fixed
- Annotation contamination protocol to prevent label leakage
- Updated evaluation metrics with multi-seed validation (seeds: 42, 123, 2024)

## [1.0.0] - 2026-07-13

### Added
- Round 5 active learning with PhoBERT fine-tuning
- Complete academic paper with methodology and results
- Annotation contamination protocol documentation

### Changed
- Final dataset expanded to 6,079 samples across all rounds
- Improved active learning selection criteria

### Fixed
- Annotation contamination protocol to prevent label leakage
