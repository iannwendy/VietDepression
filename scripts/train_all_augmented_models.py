"""Train ALL 5 model families on augmented dataset.
Models: TF-IDF + LogReg, TF-IDF + LinearSVC, BiLSTM (random), BiLSTM (PhoBERT-frozen), PhoBERT

Usage:
  PYTHONPATH="$PWD" .venv/bin/python scripts/train_all_augmented_models.py
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Disable proxy
import os
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
AUG_DIR = DATA_DIR / "augmented_version"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data(use_augmented=False):
    """Load dataset - either original or augmented."""
    if use_augmented:
        logger.info("Loading augmented data...")
        train = pd.read_csv(AUG_DIR / "final_train_aug.csv", dtype=str).fillna("")
        val = pd.read_csv(AUG_DIR / "final_val_aug.csv", dtype=str).fillna("")
    else:
        train = pd.read_csv(DATA_DIR / "final_train.csv", dtype=str).fillna("")
        val = pd.read_csv(DATA_DIR / "final_val.csv", dtype=str).fillna("")

    test = pd.read_csv(DATA_DIR / "final_test.csv", dtype=str).fillna("")
    cross = pd.read_csv(PROJECT_DIR / "data_unified" / "cross_domain_test.csv", dtype=str).fillna("")

    for df in [train, val, test, cross]:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)

    return train, val, test, cross


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_depression": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


# ── TF-IDF Models ───────────────────────────────────────────────────

def train_tfidf_logreg(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, cross_texts, cross_labels):
    """TF-IDF + Logistic Regression."""
    logger.info("=" * 50)
    logger.info("MODEL: TF-IDF + LogReg (Augmented)")
    logger.info("=" * 50)

    # Fit TF-IDF
    word_tfidf = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2), max_features=80000,
        sublinear_tf=True, min_df=2
    )
    char_tfidf = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 5), max_features=80000,
        sublinear_tf=True, min_df=2
    )

    logger.info("  Fitting TF-IDF vectorizers...")
    from scipy.sparse import hstack
    X_train_word = word_tfidf.fit_transform(train_texts)
    X_train_char = char_tfidf.fit_transform(train_texts)
    X_train = hstack([X_train_word, X_train_char])

    X_test_word = word_tfidf.transform(test_texts)
    X_test_char = char_tfidf.transform(test_texts)
    X_test = hstack([X_test_word, X_test_char])

    X_cross_word = word_tfidf.transform(cross_texts)
    X_cross_char = char_tfidf.transform(cross_texts)
    X_cross = hstack([X_cross_word, X_cross_char])

    # Train
    logger.info("  Training Logistic Regression...")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(X_train, train_labels)

    # Predict
    test_preds = clf.predict(X_test)
    cross_preds = clf.predict(X_cross)

    test_metrics = compute_metrics(test_labels, test_preds)
    cross_metrics = compute_metrics(cross_labels, cross_preds)

    logger.info(f"  In-domain F1: {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Cross-domain F1: {cross_metrics['f1_macro']:.4f}")

    return {"test": test_metrics, "cross_domain": cross_metrics}


def train_tfidf_svc(train_texts, train_labels, test_texts, test_labels, cross_texts, cross_labels):
    """TF-IDF + LinearSVC."""
    logger.info("=" * 50)
    logger.info("MODEL: TF-IDF + LinearSVC (Augmented)")
    logger.info("=" * 50)

    word_tfidf = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2), max_features=80000,
        sublinear_tf=True, min_df=2
    )
    char_tfidf = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 5), max_features=80000,
        sublinear_tf=True, min_df=2
    )

    from scipy.sparse import hstack
    X_train_word = word_tfidf.fit_transform(train_texts)
    X_train_char = char_tfidf.fit_transform(train_texts)
    X_train = hstack([X_train_word, X_train_char])

    X_test_word = word_tfidf.transform(test_texts)
    X_test_char = char_tfidf.transform(test_texts)
    X_test = hstack([X_test_word, X_test_char])

    X_cross_word = word_tfidf.transform(cross_texts)
    X_cross_char = char_tfidf.transform(cross_texts)
    X_cross = hstack([X_cross_word, X_cross_char])

    logger.info("  Training LinearSVC...")
    clf = LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)
    clf.fit(X_train, train_labels)

    test_preds = clf.predict(X_test)
    cross_preds = clf.predict(X_cross)

    test_metrics = compute_metrics(test_labels, test_preds)
    cross_metrics = compute_metrics(cross_labels, cross_preds)

    logger.info(f"  In-domain F1: {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Cross-domain F1: {cross_metrics['f1_macro']:.4f}")

    return {"test": test_metrics, "cross_domain": cross_metrics}


# ── BiLSTM Models ─────────────────────────────────────────────────

class TextDatasetLSTM(Dataset):
    def __init__(self, texts, labels, vocab, word2idx, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).lower().split()
        indices = [self.word2idx.get(w, 0) for w in text[:self.max_len]]
        indices = indices + [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class BiLSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=2, use_phobert=False, phobert_model=None):
        super().__init__()
        self.use_phobert = use_phobert
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(hidden_dim * 2, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc_out = torch.nn.Linear(64, num_classes)

        # PhoBERT frozen embedding projection
        if use_phobert and phobert_model is not None:
            self.phobert = phobert_model
            self.proj = torch.nn.Linear(768, embed_dim)
            for param in self.phobert.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.use_phobert:
            with torch.no_grad():
                outputs = self.phobert(**{k: v for k, v in x.items() if k != 'labels'})
                emb = outputs.last_hidden_state[:, 0, :]
            emb = self.proj(emb)
        else:
            emb = self.embedding(x)

        lstm_out, _ = self.lstm(emb)
        # Take last hidden state
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = torch.relu(self.fc2(out))
        out = self.dropout(out)
        return self.fc_out(out)


def build_vocab(texts, min_freq=2):
    """Build vocabulary from texts."""
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(str(text).lower().split())
    vocab = ['<pad>', '<unk>'] + [w for w, c in counter.items() if c >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_bilstm(
    train_texts, train_labels, val_texts, val_labels,
    test_texts, test_labels, cross_texts, cross_labels,
    use_phobert=False, phobert_path=None, seed=42
):
    """Train BiLSTM model."""
    model_name = "BiLSTM (PhoBERT-frozen)" if use_phobert else "BiLSTM (random)"
    logger.info("=" * 50)
    logger.info(f"MODEL: {model_name} (Augmented)")
    logger.info("=" * 50)

    set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    phobert_model = None
    if use_phobert and phobert_path:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(phobert_path), use_fast=False)
        phobert_model = AutoModel.from_pretrained(str(phobert_path))
        phobert_model.to(device)
        phobert_model.eval()
        logger.info(f"  Loaded PhoBERT from {phobert_path}")

    max_len = 100
    batch_size = 32

    if use_phobert:
        # For PhoBERT, use PhoBERT tokenizer
        class PhoBERTDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = str(self.texts[idx])
                enc = self.tokenizer(text, truncation=True, padding='max_length',
                                     max_length=self.max_len, return_tensors='pt')
                return {
                    'input_ids': enc['input_ids'].squeeze(),
                    'attention_mask': enc['attention_mask'].squeeze(),
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }

        train_dataset = PhoBERTDataset(train_texts, train_labels, tokenizer, max_len)
        val_dataset = PhoBERTDataset(val_texts, val_labels, tokenizer, max_len)
        test_dataset = PhoBERTDataset(test_texts, test_labels, tokenizer, max_len)
        cross_dataset = PhoBERTDataset(cross_texts, cross_labels, tokenizer, max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        cross_loader = DataLoader(cross_dataset, batch_size=batch_size)

        vocab_size = 1  # Not used for PhoBERT
    else:
        # Build vocab for random embedding
        vocab, word2idx = build_vocab(train_texts + val_texts)
        vocab_size = len(vocab)
        logger.info(f"  Vocabulary size: {vocab_size}")

        train_dataset = TextDatasetLSTM(train_texts, train_labels, vocab, word2idx, max_len)
        val_dataset = TextDatasetLSTM(val_texts, val_labels, vocab, word2idx, max_len)
        test_dataset = TextDatasetLSTM(test_texts, test_labels, vocab, word2idx, max_len)
        cross_dataset = TextDatasetLSTM(cross_texts, cross_labels, vocab, word2idx, max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        cross_loader = DataLoader(cross_dataset, batch_size=batch_size)

    # Model
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=128,
        use_phobert=use_phobert,
        phobert_model=phobert_model
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.0]).to(device))  # Class weights

    # Training
    best_val_f1 = 0
    best_state = None
    epochs = 8

    logger.info("  Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if use_phobert:
                labels = batch['labels'].to(device)  # Save labels BEFORE modifying batch
                batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            else:
                x, labels = batch
                x = x.to(device)
                labels = labels.to(device)

            outputs = model(x if not use_phobert else batch)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                if use_phobert:
                    val_labels_list.extend(batch['labels'].tolist())  # Save labels BEFORE moving batch
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(batch)
                else:
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    val_labels_list.extend(y.tolist())
                    outputs = model(x)
                preds = torch.argmax(outputs, dim=-1).cpu().tolist()
                val_preds.extend(preds)

        val_f1 = f1_score(val_labels_list, val_preds, average='macro', zero_division=0)
        logger.info(f"    Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best and evaluate
    model.load_state_dict(best_state)
    model.eval()

    def predict(loader):
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                if use_phobert:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(batch)
                else:
                    x, _ = batch
                    x = x.to(device)
                    outputs = model(x)
                preds.extend(torch.argmax(outputs, dim=-1).cpu().tolist())
                labels.extend(batch['labels'].tolist() if use_phobert else batch[1].tolist())
        return labels, preds

    test_labels_out, test_preds = predict(test_loader)
    cross_labels_out, cross_preds = predict(cross_loader)

    test_metrics = compute_metrics(test_labels_out, test_preds)
    cross_metrics = compute_metrics(cross_labels_out, cross_preds)

    logger.info(f"  In-domain F1: {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Cross-domain F1: {cross_metrics['f1_macro']:.4f}")

    return {"test": test_metrics, "cross_domain": cross_metrics}


# ── Main ────────────────────────────────────────────────────────────

def main():
    # Load augmented data
    train_df, val_df, test_df, cross_df = load_data(use_augmented=True)

    # Combine train + val
    train_texts = train_df["comment_text"].tolist() + val_df["comment_text"].tolist()
    train_labels = train_df["label"].astype(int).tolist() + val_df["label"].astype(int).tolist()

    test_texts = test_df["comment_text"].tolist()
    test_labels = test_df["label"].astype(int).tolist()
    cross_texts = cross_df["comment_text"].tolist()
    cross_labels = cross_df["label"].astype(int).tolist()

    logger.info(f"Fit samples: {len(train_texts)}")
    logger.info(f"Test samples: {len(test_texts)}")
    logger.info(f"Cross-domain samples: {len(cross_texts)}")
    logger.info(f"Label distribution: {sum(train_labels)} depression, {len(train_labels) - sum(train_labels)} normal")

    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. TF-IDF + LogReg
    try:
        results["TF-IDF + LogReg"] = train_tfidf_logreg(
            train_texts, train_labels,
            val_df["comment_text"].tolist(), val_df["label"].astype(int).tolist(),
            test_texts, test_labels, cross_texts, cross_labels
        )
    except Exception as e:
        logger.error(f"TF-IDF LogReg failed: {e}")

    # 2. TF-IDF + LinearSVC
    try:
        results["TF-IDF + LinearSVC"] = train_tfidf_svc(
            train_texts, train_labels, test_texts, test_labels, cross_texts, cross_labels
        )
    except Exception as e:
        logger.error(f"TF-IDF LinearSVC failed: {e}")

    # 3. BiLSTM (random) - 3 seeds
    for seed in [42, 123, 2024]:
        try:
            key = f"BiLSTM (random) seed={seed}"
            results[key] = train_bilstm(
                train_texts, train_labels,
                val_df["comment_text"].tolist(), val_df["label"].astype(int).tolist(),
                test_texts, test_labels, cross_texts, cross_labels,
                use_phobert=False, seed=seed
            )
        except Exception as e:
            logger.error(f"BiLSTM random seed={seed} failed: {e}")

    # 4. BiLSTM (PhoBERT-frozen) - 3 seeds
    phobert_path = MODEL_DIR / "phobert_base_local"
    for seed in [42, 123, 2024]:
        try:
            key = f"BiLSTM (PhoBERT-frozen) seed={seed}"
            results[key] = train_bilstm(
                train_texts, train_labels,
                val_df["comment_text"].tolist(), val_df["label"].astype(int).tolist(),
                test_texts, test_labels, cross_texts, cross_labels,
                use_phobert=True, phobert_path=phobert_path, seed=seed
            )
        except Exception as e:
            logger.error(f"BiLSTM PhoBERT-frozen seed={seed} failed: {e}")

    # Save results
    output = {
        "timestamp": timestamp,
        "models": results,
        "summary": {}
    }

    # Add summary with mean ± std for BiLSTM
    for key in results:
        output["summary"][key] = {
            "in_domain_f1": results[key]["test"]["f1_macro"],
            "cross_domain_f1": results[key]["cross_domain"]["f1_macro"],
        }

    # Compute BiLSTM means
    random_f1s = [results[k]["test"]["f1_macro"] for k in results if "BiLSTM (random)" in k]
    random_cross = [results[k]["cross_domain"]["f1_macro"] for k in results if "BiLSTM (random)" in k]
    frozen_f1s = [results[k]["test"]["f1_macro"] for k in results if "BiLSTM (PhoBERT-frozen)" in k]
    frozen_cross = [results[k]["cross_domain"]["f1_macro"] for k in results if "BiLSTM (PhoBERT-frozen)" in k]

    if random_f1s:
        output["summary"]["BiLSTM (random) mean"] = {
            "in_domain_f1": f"{np.mean(random_f1s):.4f} ± {np.std(random_f1s):.4f}",
            "cross_domain_f1": f"{np.mean(random_cross):.4f} ± {np.std(random_cross):.4f}",
        }
    if frozen_f1s:
        output["summary"]["BiLSTM (PhoBERT-frozen) mean"] = {
            "in_domain_f1": f"{np.mean(frozen_f1s):.4f} ± {np.std(frozen_f1s):.4f}",
            "cross_domain_f1": f"{np.mean(frozen_cross):.4f} ± {np.std(frozen_cross):.4f}",
        }

    output_file = MODEL_DIR / f"all_augmented_results_{timestamp}.json"
    output_file.write_text(json.dumps(output, indent=2))
    logger.info(f"\nResults saved: {output_file}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS (All Models on Augmented Dataset)")
    logger.info("=" * 60)
    for name, r in results.items():
        logger.info(f"{name}:")
        logger.info(f"  In-domain F1:     {r['test']['f1_macro']:.4f}")
        logger.info(f"  Cross-domain F1:  {r['cross_domain']['f1_macro']:.4f}")

    if random_f1s:
        logger.info(f"\nBiLSTM (random) mean: In={np.mean(random_f1s):.4f}, Cross={np.mean(random_cross):.4f}")
    if frozen_f1s:
        logger.info(f"BiLSTM (PhoBERT-frozen) mean: In={np.mean(frozen_f1s):.4f}, Cross={np.mean(frozen_cross):.4f}")

    return results


if __name__ == "__main__":
    main()
