"""Quick evaluation and save results for Round 5 retrained models."""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "round5_retrained"
RESULTS_DIR = PROJECT_DIR / "results"

TEST_FILE = PROJECT_DIR / "data" / "labeled" / "final_test.csv"
VAL_FILE = PROJECT_DIR / "data" / "labeled" / "final_val.csv"
TRAIN_FILE = PROJECT_DIR / "data" / "labeled" / "final_train.csv"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_depression": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

# Load data
test_df = pd.read_csv(TEST_FILE)
test_texts = test_df["comment_text"].values
test_labels = test_df["label"].values

print(f"Test: {len(test_df)} samples | Labels: {dict(pd.Series(test_labels).value_counts())}")

all_results = {"in_domain": {}, "dataset_info": {}}

# TF-IDF + LogReg
print("\n1. TF-IDF + LogReg")
pipeline = joblib.load(MODEL_DIR / "tfidf_logreg_round5.joblib")
preds = pipeline.predict(test_texts)
metrics = compute_metrics(test_labels, preds)
all_results["in_domain"]["tfidf_logreg"] = metrics
print(f"   Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

# TF-IDF + LinearSVC
print("\n2. TF-IDF + LinearSVC")
pipeline = joblib.load(MODEL_DIR / "tfidf_linearsvc_round5.joblib")
preds = pipeline.predict(test_texts)
metrics = compute_metrics(test_labels, preds)
all_results["in_domain"]["tfidf_svc"] = metrics
print(f"   Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

# PhoBERT
print("\n3. PhoBERT (3 seeds)")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

class PhoBERTDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = tokenizer(str(self.texts[idx]), padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoding.items()}

phobert_results = []
for seed in [42, 123, 2024]:
    model_dir = MODEL_DIR / f"phobert_seed_{seed}"
    if model_dir.exists():
        print(f"   Seed {seed}...")
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir / "best_model"))
        model.to(device)
        model.eval()

        dataset = PhoBERTDataset(test_texts)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch in loader:
                outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs[:, 1].cpu().numpy())

        metrics = compute_metrics(test_labels, all_preds)
        phobert_results.append({"seed": seed, "metrics": metrics, "predictions": all_preds, "probabilities": all_probs})
        all_results["in_domain"][f"phobert_seed{seed}"] = {k: v for k, v in metrics.items() if k not in ["predictions", "probabilities", "confusion_matrix"]}
        print(f"      Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

# Average PhoBERT
avg_probs = np.mean([r["probabilities"] for r in phobert_results], axis=0)
avg_preds = (avg_probs >= 0.5).astype(int)
avg_metrics = compute_metrics(test_labels, avg_preds)
all_results["in_domain"]["phobert_avg"] = {k: v for k, v in avg_metrics.items() if k != "confusion_matrix"}
print(f"   Average: Acc={avg_metrics['accuracy']:.4f}, F1={avg_metrics['f1_macro']:.4f}")

# BiLSTM
print("\n4. BiLSTM (3 seeds)")

class Vocabulary:
    def __init__(self):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
    def __len__(self):
        return len(self.itos)
    def text_to_sequence(self, text):
        return [self.stoi.get(w, 1) for w in text.split()]

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))

class BiLSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        seq = self.vocab.text_to_sequence(str(self.texts[idx]))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, torch.tensor(labels)

bilstm_results = []
for seed in [42, 123, 2024]:
    model_path = MODEL_DIR / f"bilstm_seed_{seed}" / "best_model.pt"
    if model_path.exists():
        print(f"   Seed {seed}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        vocab = checkpoint["vocab"]

        model = BiLSTMClassifier(len(vocab))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        dataset = BiLSTMDataset(test_texts, test_labels, vocab)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

        all_preds = []
        with torch.no_grad():
            for texts_batch, _ in loader:
                outputs = model(texts_batch.to(device))
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

        metrics = compute_metrics(test_labels, all_preds)
        bilstm_results.append({"seed": seed, "metrics": metrics})
        all_results["in_domain"][f"bilstm_seed{seed}"] = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        print(f"      Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

# Average BiLSTM
if bilstm_results:
    avg_metrics = {
        "accuracy": np.mean([r["metrics"]["accuracy"] for r in bilstm_results]),
        "f1_macro": np.mean([r["metrics"]["f1_macro"] for r in bilstm_results]),
        "f1_depression": np.mean([r["metrics"]["f1_depression"] for r in bilstm_results]),
    }
    all_results["in_domain"]["bilstm_avg"] = avg_metrics
    print(f"   Average: Acc={avg_metrics['accuracy']:.4f}, F1={avg_metrics['f1_macro']:.4f}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = RESULTS_DIR / f"round5_final_results_{timestamp}"
results_dir.mkdir(parents=True, exist_ok=True)

all_results["timestamp"] = datetime.now().isoformat()
all_results["dataset_info"] = {
    "train_size": len(pd.read_csv(TRAIN_FILE)),
    "val_size": len(pd.read_csv(VAL_FILE)),
    "test_size": len(test_df),
}

results_file = results_dir / "evaluation_results.json"
with open(results_file, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# Print summary
print("\n" + "="*70)
print("ROUND 5 FINAL RESULTS (FIXED DATA)")
print("="*70)
print(f"{'Model':<25} | {'Acc':>7} | {'Prec-M':>7} | {'Rec-M':>7} | {'F1-M':>7} | {'F1-D':>7}")
print("-"*70)
for name, m in all_results["in_domain"].items():
    if isinstance(m, dict) and "accuracy" in m:
        print(f"{name:<25} | {m['accuracy']:>7.4f} | {m.get('precision_macro', 0):>7.4f} | {m.get('recall_macro', 0):>7.4f} | {m['f1_macro']:>7.4f} | {m.get('f1_depression', 0):>7.4f}")

print(f"\nResults saved to: {results_file}")
