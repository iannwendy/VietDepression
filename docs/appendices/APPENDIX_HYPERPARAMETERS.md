# Appendix B: Model Hyperparameters

This appendix documents all hyperparameters used for training and evaluation in the Vietnamese YouTube Depression Detection system.

---

## B.1 Baseline Models (TF-IDF + Logistic Regression / LinearSVC)

### B.1.1 Feature Extraction

| Parameter | Value | Description |
|-----------|-------|-------------|
| Word TF-IDF Analyzer | `word` | Word-level tokenization |
| Word N-gram Range | `(1, 2)` | Unigrams and bigrams |
| Char TF-IDF Analyzer | `char_wb` | Character-level with word boundaries |
| Char N-gram Range | `(3, 5)` | 3-5 character n-grams |
| Min Document Frequency | `2` | Ignore terms appearing in fewer than 2 documents |
| Max Features | `80,000` | Maximum vocabulary size per TF-IDF vectorizer |
| Lowercase | `True` | Convert text to lowercase before vectorization |

### B.1.2 Logistic Regression Classifier

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Iterations | `1,000` | Maximum optimization iterations |
| Class Weight | `balanced` | Automatically adjust weights inversely proportional to class frequencies |
| Solver | `liblinear` | Library for large-scale linear classification |
| Random State | `42` | Seed for reproducibility |

### B.1.3 LinearSVC Classifier

| Parameter | Value | Description |
|-----------|-------|-------------|
| C (Regularization) | `1.0` | Inverse of regularization strength |
| Class Weight | `balanced` | Automatically adjust weights inversely proportional to class frequencies |
| Max Iterations | `2,000` | Maximum optimization iterations |
| Random State | `42` | Seed for reproducibility |

---

## B.2 PhoBERT Model

### B.2.1 Model Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| Pretrained Model | `vinai/phobert-base` | Vietnamese BERT model from VinAI Research |
| Word Segmentation | `True` | Use PhoBERT's BPE tokenizer with Underthesea |

### B.2.2 Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Sequence Length | `128` tokens | Maximum input length after tokenization |
| Batch Size (Training) | `8` | Training batch size per step |
| Batch Size (Evaluation) | `16` | Validation/test batch size |
| Batch Size (Prediction) | `32` | Inference batch size |
| Epochs | `3` | Number of training epochs |
| Learning Rate | `2e-5` | AdamW initial learning rate |
| Weight Decay | `0.01` | L2 regularization coefficient |
| Warmup Ratio | `0.06` | Fraction of training steps for linear warmup |
| Gradient Clipping | `1.0` | Maximum gradient norm for clipping |
| Early Stopping Patience | `2` | Epochs to wait before stopping if no improvement |
| Random Seed | `42` | Seed for reproducibility |
| Device | `auto` | Device selection (cpu/cuda/mps/auto) |

### B.2.3 Active Learning & Pseudo-Labeling

| Parameter | Value | Description |
|-----------|-------|-------------|
| Pseudo-Label Threshold | `0.90` | Minimum probability to accept pseudo-label |
| Active Learning Threshold | `0.70` | Minimum probability threshold for uncertainty sampling |
| Active Learning Max Samples | `1,000` | Maximum samples selected per active learning round |
| Prediction Chunk Size | `2,000` | Number of samples processed per prediction batch |

---

## B.3 BiLSTM Models

### B.3.1 Architecture Configuration

| Parameter | BiLSTM-Random | BiLSTM-PhoBERT | Description |
|-----------|---------------|----------------|-------------|
| Vocabulary Size | 15,000 | 15,000 | Maximum vocabulary size |
| Embedding Dimension | 128 | 128 (projected from 768) | Word embedding dimension |
| Hidden Dimension | 128 | 128 | LSTM hidden state dimension |
| Number of Layers | 2 | 2 | Stacked LSTM layers |
| Dropout Rate | 0.5 | 0.5 | Dropout between layers |
| Number of Classes | 2 | 2 | Binary classification |
| Classifier Hidden | 64 | 64 | Intermediate classifier dimension |
| Pretrained Embeddings | None (random init) | PhoBERT word embeddings (frozen) | - |

### B.3.2 Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 8 | Number of training epochs |
| Batch Size | 32 | Training and evaluation batch size |
| Learning Rate | 1e-3 | Adam optimizer learning rate |
| Max Sequence Length | 100 tokens | Maximum tokenized sequence length |
| Gradient Clipping | 1.0 | Maximum gradient norm for clipping |
| Loss Function | Cross-Entropy (unweighted) | No class weighting per paper specification |
| Random Seed | 42 | Seed for reproducibility |

### B.3.3 Loss Function Note

Per the paper specification, BiLSTM models use **unweighted cross-entropy loss** without class balancing. This differs from the TF-IDF baselines which use `class_weight='balanced'`.

---

## B.4 BERTopic Configuration

### B.4.1 Topic Modeling Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| Minimum Topic Size | 50 | Minimum documents per topic |
| Number of Topics | `auto` | Automatically determine optimal number |
| Embedding Model | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence transformer for Vietnamese |
| Top N Words | 10 | Number of words to display per topic |

### B.4.2 UMAP Dimensionality Reduction

| Parameter | Value | Description |
|-----------|-------|-------------|
| N Neighbors | 15 | Local neighborhood size |
| Minimum Distance | 0.0 | Minimum distance between embedded points |

### B.4.3 HDBSCAN Clustering

| Parameter | Value | Description |
|-----------|-------|-------------|
| Minimum Cluster Size | 50 | Minimum documents to form a cluster |
| Minimum Samples | 10 | Core point density parameter |

### B.4.4 Memory Optimization

| Parameter | Value | Description |
|-----------|-------|-------------|
| Calculate Probabilities | `False` | Disable probability calculations for memory efficiency |

---

## B.5 Data Split Ratios

| Parameter | Value | Description |
|-----------|-------|-------------|
| Training Ratio | 0.70 (70%) | Fraction of data for training |
| Validation Ratio | 0.15 (15%) | Fraction of data for validation |
| Test Ratio | 0.15 (15%) | Fraction of data for testing |

---

## B.6 Weak Labeling Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Initial Depression Samples | 10,000 | Maximum depression samples for initial training |
| Initial Normal Samples | 10,000 | Maximum normal samples for initial training |
| Minimum Characters | 10 | Minimum comment length for inclusion |
| Maximum Characters | 500 | Maximum comment length for inclusion |
| Review Samples Per Bucket | 150 | Samples per confidence bucket for human review |
| Random Seed | 42 | Seed for reproducibility |

---

## B.7 Keyword Scoring Weights

| Parameter | Value | Description |
|-----------|-------|-------------|
| Depression Strong Weight | 5 | Weight for strong depression keywords |
| Depression Medium Weight | 3 | Weight for medium depression keywords |
| Normal Weight | -2 | Weight for normal/non-depression keywords |
| Depression Auto Threshold | 5 | Minimum score for auto-labeling as depression |
| Normal Auto Threshold | -2 | Maximum score for auto-labeling as normal |
| High Confidence Depression | 8 | Score threshold for high-confidence depression |
| High Confidence Normal | -4 | Score threshold for high-confidence normal |

---

*Document generated from `yt_depression_crawler/core/config.py` and model implementation files.*
