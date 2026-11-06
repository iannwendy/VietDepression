"""Phase 3 — Evaluate all 5 models and produce comparison report."""
import json, sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from collections import Counter

from yt_depression_crawler.modeling.phobert.phobert_utils import (
    get_device, prepare_many_texts, PhoBertDataset,
)
from yt_depression_crawler.core.config import PHOBERT_MAX_LENGTH

SEED = 42
FINAL_TRAIN = Path("data/final_train.csv")
FINAL_TEST = Path("data/final_test.csv")
CROSS_DOMAIN = Path("data_unified/cross_domain_test.csv")
BERTOPIC_PKL = Path("models/bertopic/bertopic_model.pkl")
PHOBERT_V2 = Path("models/phobert_second")
PHOBERT_FINAL = Path("models/phase3/phobert_final")
DEVICE = get_device()
print(f"Device: {DEVICE}")

train_df = pd.read_csv(FINAL_TRAIN, dtype={"comment_text": str}).fillna("")
train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").astype(int)
test_df = pd.read_csv(FINAL_TEST, dtype={"comment_text": str}).fillna("")
test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce").astype(int)
cross_df = pd.read_csv(CROSS_DOMAIN, dtype={"comment_text": str}).fillna("")
cross_df["label"] = pd.to_numeric(cross_df["label"], errors="coerce").astype(int)

print(f"Train: {len(train_df)}, Test: {len(test_df)}, Cross: {len(cross_df)}")

def metrics(y_true, y_pred):
    labels = [0, 1]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_depression": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

# ═══ Model 1: TF-IDF + SVM ═══
print("\n=== Model 1: TF-IDF + SVM ===")
m1 = Pipeline(steps=[
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, max_features=80000, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=80000, lowercase=True)),
    ])),
    ("clf", LinearSVC(max_iter=2000, class_weight="balanced", random_state=SEED, dual=False)),
])
m1.fit(train_df["comment_text"], train_df["label"])
r1_in = metrics(test_df["label"], m1.predict(test_df["comment_text"]))
r1_cross = metrics(cross_df["label"], m1.predict(cross_df["comment_text"]))
print(f"  In-domain F1: {r1_in['f1_macro']}, Cross-domain F1: {r1_cross['f1_macro']}")

# ═══ Model 2: BiLSTM ═══
print("\n=== Model 2: BiLSTM ===")
class BiLSTM:
    def __init__(self):
        self.w2i = {"<PAD>": 0, "<UNK>": 1}
        self.mlen = 100
    def build(self, texts):
        c = Counter()
        for t in texts:
            for tok in t.lower().split(): c[tok] += 1
        for w, _ in c.most_common(14998):
            self.w2i[w] = len(self.w2i)
    def enc(self, texts):
        d = np.zeros((len(texts), self.mlen), dtype=np.int64)
        for i, t in enumerate(texts):
            for j, tok in enumerate(t.lower().split()[:self.mlen]):
                d[i, j] = self.w2i.get(tok, 1)
        return d
    def fit(self, texts, labels):
        self.build(texts)
        X = torch.tensor(self.enc(texts), dtype=torch.long)
        y = torch.tensor(np.array(labels, dtype=np.int64), dtype=torch.long)
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=32, shuffle=True)
        self.m = torch.nn.ModuleDict({
            "emb": torch.nn.Embedding(len(self.w2i), 128, padding_idx=0),
            "lstm": torch.nn.LSTM(128, 128, 2, batch_first=True, bidirectional=True, dropout=0.5),
            "clf": torch.nn.Sequential(
                torch.nn.Linear(256, 64), torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(64, 2)),
        }).to(DEVICE)
        opt = torch.optim.Adam(self.m.parameters(), lr=0.001)
        lf = torch.nn.CrossEntropyLoss()
        self.m.train()
        for ep in range(8):
            tl = 0.0
            for bx, by in dl:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                opt.zero_grad()
                emb = self.m["emb"](bx)
                lo, _ = self.m["lstm"](emb)
                logits = self.m["clf"](lo[:, -1, :])
                loss = lf(logits, by)
                loss.backward()
                opt.step()
                tl += loss.item()
        return self
    def predict(self, texts):
        self.m.eval()
        X = torch.tensor(self.enc(texts), dtype=torch.long)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                b = X[i:i+32].to(DEVICE)
                emb = self.m["emb"](b)
                lo, _ = self.m["lstm"](emb)
                logits = self.m["clf"](lo[:, -1, :])
                preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        return preds

bilstm = BiLSTM()
bilstm.fit(train_df["comment_text"].tolist(), train_df["label"].tolist())
r2_in = metrics(test_df["label"], bilstm.predict(test_df["comment_text"]))
r2_cross = metrics(cross_df["label"], bilstm.predict(cross_df["comment_text"]))
print(f"  In-domain F1: {r2_in['f1_macro']}, Cross-domain F1: {r2_cross['f1_macro']}")

# ═══ Model 3: PhoBERT final ═══
print("\n=== Model 3: PhoBERT ===")
tok3 = AutoTokenizer.from_pretrained(str(PHOBERT_FINAL), use_fast=False)
mdl3 = AutoModelForSequenceClassification.from_pretrained(str(PHOBERT_FINAL)).to(DEVICE)
mdl3.eval()
def pred3(texts):
    prep = prepare_many_texts(list(texts))
    ds = PhoBertDataset(prep, None, tok3, PHOBERT_MAX_LENGTH)
    dl = DataLoader(ds, batch_size=32)
    preds = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds.extend(torch.argmax(mdl3(**batch).logits, dim=-1).cpu().tolist())
    return preds
r3_in = metrics(test_df["label"], pred3(test_df["comment_text"]))
r3_cross = metrics(cross_df["label"], pred3(cross_df["comment_text"]))
print(f"  In-domain F1: {r3_in['f1_macro']}, Cross-domain F1: {r3_cross['f1_macro']}")

# ═══ Model 4: BERTopic-only ═══
print("\n=== Model 4: BERTopic-only ===")
with open(BERTOPIC_PKL, "rb") as f:
    tm = pickle.load(f)
emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
def topic_feat(texts):
    tl = list(texts)
    e = emb.encode(tl, show_progress_bar=True, batch_size=64)
    topics, probs = tm.transform(tl, embeddings=e)
    return np.column_stack([
        np.array(topics, dtype=np.float64).reshape(-1, 1),
        np.array(probs if probs is not None else [0]*len(tl), dtype=np.float64).reshape(-1, 1),
    ])
X4 = topic_feat(train_df["comment_text"])
sc4 = StandardScaler().fit(X4)
clf4 = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
clf4.fit(sc4.transform(X4), train_df["label"])
r4_in = metrics(test_df["label"], clf4.predict(sc4.transform(topic_feat(test_df["comment_text"]))))
r4_cross = metrics(cross_df["label"], clf4.predict(sc4.transform(topic_feat(cross_df["comment_text"]))))
print(f"  In-domain F1: {r4_in['f1_macro']}, Cross-domain F1: {r4_cross['f1_macro']}")

# ═══ Model 5: PhoBERT + BERTopic ═══
print("\n=== Model 5: PhoBERT + BERTopic (PROPOSED) ===")
tok5 = AutoTokenizer.from_pretrained(str(PHOBERT_V2), use_fast=False)
pho5 = AutoModel.from_pretrained(str(PHOBERT_V2)).to(DEVICE)
pho5.eval()
def pho_emb(texts):
    prep = prepare_many_texts(list(texts))
    ds = PhoBertDataset(prep, None, tok5, PHOBERT_MAX_LENGTH)
    dl = DataLoader(ds, batch_size=16)
    embs = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            embs.append(pho5(**batch).last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embs)

print("  Extracting PhoBERT features...")
p_train = pho_emb(train_df["comment_text"])
t_train = topic_feat(train_df["comment_text"])
X5 = np.hstack([p_train, t_train])
sc5 = StandardScaler().fit(X5)
clf5 = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
clf5.fit(sc5.transform(X5), train_df["label"])

print("  Evaluating...")
p_test = pho_emb(test_df["comment_text"])
t_test = topic_feat(test_df["comment_text"])
r5_in = metrics(test_df["label"], clf5.predict(sc5.transform(np.hstack([p_test, t_test]))))
p_cross = pho_emb(cross_df["comment_text"])
t_cross = topic_feat(cross_df["comment_text"])
r5_cross = metrics(cross_df["label"], clf5.predict(sc5.transform(np.hstack([p_cross, t_cross]))))
print(f"  In-domain F1: {r5_in['f1_macro']}, Cross-domain F1: {r5_cross['f1_macro']}")

# ═══ COMPARISON ═══
results = [
    ("TF-IDF + SVM", r1_in, r1_cross),
    ("BiLSTM", r2_in, r2_cross),
    ("PhoBERT", r3_in, r3_cross),
    ("BERTopic-only", r4_in, r4_cross),
    ("PhoBERT + BERTopic", r5_in, r5_cross),
]

print("\n" + "=" * 80)
print("FINAL COMPARISON — ALL 5 MODELS")
print("=" * 80)

print(f"\n{'Model':<25s} {'In-domain':>30s} {'Cross-domain':>30s}")
print(f"{'':25s} {'F1_macro':>10s} {'F1_w':>8s} {'F1_dep':>8s} {'F1_macro':>10s} {'F1_w':>8s} {'F1_dep':>8s}")
print("-" * 80)
for name, r_in, r_cross in results:
    print(f"{name:<25s} {r_in['f1_macro']:>10.4f} {r_in['f1_weighted']:>8.4f} {r_in['f1_depression']:>8.4f} {r_cross['f1_macro']:>10.4f} {r_cross['f1_weighted']:>8.4f} {r_cross['f1_depression']:>8.4f}")

print(f"\n{'Model':<25s} {'Acc':>8s} {'Prec_m':>8s} {'Rec_m':>8s} {'F1_m':>8s} {'Acc':>8s} {'Prec_m':>8s} {'Rec_m':>8s} {'F1_m':>8s}")
print(f"{'':25s} {'(in)':>8s} {'(in)':>8s} {'(in)':>8s} {'(in)':>8s} {'(cross)':>8s} {'(cross)':>8s} {'(cross)':>8s} {'(cross)':>8s}")
print("-" * 80)
for name, r_in, r_cross in results:
    print(f"{name:<25s} {r_in['accuracy']:>8.4f} {r_in['precision_macro']:>8.4f} {r_in['recall_macro']:>8.4f} {r_in['f1_macro']:>8.4f} {r_cross['accuracy']:>8.4f} {r_cross['precision_macro']:>8.4f} {r_cross['recall_macro']:>8.4f} {r_cross['f1_macro']:>8.4f}")

# Save
report = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "phase": 3,
    "models": [
        {"model": n, "in_domain": ri, "cross_domain": rc}
        for n, ri, rc in results
    ],
}
Path("docs/phase3_comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(f"\nReport saved to docs/phase3_comparison_report.json")
print("PHASE 3 COMPLETE")
