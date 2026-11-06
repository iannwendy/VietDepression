#!/usr/bin/env python3
"""
Generate 6 Publication-Grade 300-DPI Figures for the Revised VietDepression Manuscript.
Directly addresses Reviewer 1 Comment 11 and Reviewer 2 Minor Comment 3.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "labeled"
UNIFIED_DIR = BASE_DIR / "data_unified"
MODEL_DIR = BASE_DIR / "models" / "round6_v2_retrained"
AUDIT_DIR = BASE_DIR / "results" / "scientific_audit"
FIG_DIR = AUDIT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(BASE_DIR))
from scripts.vietdepression_lexicon import label_comment_detailed


def _vietnamese_tokenize(text: str):
    return str(text).split()


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def make_fig1_workflow():
    """Figure 1: 6-Stage VietDepression Pipeline & Strictly Inductive Evaluation Protocol."""
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def draw_box(x, y, w, h, title, lines, facecolor="#F4F7FB", edgecolor="#2B5B84", title_color="#1A365D"):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.2",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top",
                fontsize=8.2, fontweight="bold", color=title_color)
        for idx, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 4.7 - idx * 2.55, line, ha="center", va="top",
                    fontsize=7.0, color="#2D3748")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.7", lw=1.5, color="#2B5B84")
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.0, label, ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color="#2B5B84")

    # Top row: Stages 1 -> 2 -> 3
    draw_box(
        2, 37, 28, 16,
        "Stage 1: YouTube API Retrieval",
        [
            "• 264 depression/distress queries",
            "• 2,906 unique Vietnamese videos",
            "• Raw inventory: 125,329 comments",
            "• Schema: id, video_id, text, time",
        ],
        facecolor="#EBF8FF", edgecolor="#2B6CB0", title_color="#1A365D"
    )
    draw_arrow(30.6, 45, 35.4, 45)

    draw_box(
        36, 37, 28, 16,
        "Stage 2: NFC & Quality Filtering",
        [
            "• Preserve diacritics, case & emoji",
            "• Remove spam, URLs, <5 chars, dupes",
            "• Removed: 5,680 records (4.53%)",
            "• Filtered pool: 119,649 comments",
        ],
        facecolor="#EBF8FF", edgecolor="#2B6CB0", title_color="#1A365D"
    )
    draw_arrow(64.6, 45, 69.4, 45)

    draw_box(
        70, 37, 28, 16,
        "Stage 3: Weighted Lexicon Scoring",
        [
            "• 335 entries (+5 / +3 / -2) + 22 flags",
            "• Pos: 3,223 (2.57%) | Neg: 23,695 (18.91%)",
            "• Uncertain [-1..4]: 98,410 (78.52%)",
            "• Composite need_review: 120,334 (96.02%)",
        ],
        facecolor="#FEFCBF", edgecolor="#B7791F", title_color="#744210"
    )

    # Vertical arrow from Stage 3 down to Stage 4
    draw_arrow(84, 36.4, 84, 31.2)

    # Middle-bottom row: Stage 4 <- Stage 5 -> Stage 6
    draw_box(
        70, 14, 28, 16.5,
        "Stage 4: Blind Human Annotation",
        [
            "• Label Studio (machine scores hidden)",
            "• 4 trained native VN annotators",
            "• 4-rule perspective & polysemy guide",
            "• Double-annotated QC (n=300): κ = 0.87",
        ],
        facecolor="#E6FFFA", edgecolor="#2C7A7B", title_color="#234E52"
    )
    draw_arrow(69.4, 22.2, 64.6, 22.2)

    draw_box(
        36, 14, 28, 16.5,
        "Stage 5: 6-Round Active Learning",
        [
            "• Uncertainty u(x) = 1 - |2p(x) - 1|",
            "• R1-R4: 2,072 | R5: +1,360 | R6: +5,702",
            "• Final Gold Benchmark: 9,134 comments",
            "• 1,804 Positive (19.75%) / 7,330 Normal",
        ],
        facecolor="#E6FFFA", edgecolor="#2C7A7B", title_color="#234E52"
    )
    draw_arrow(35.4, 22.2, 30.6, 22.2)

    draw_box(
        2, 14, 28, 16.5,
        "Stage 6: Zero-Leakage Splits",
        [
            "• Train (70%): 6,392 (1,262 P / 5,130 N)",
            "• Val (15%): 1,371 (271 P / 1,100 N)",
            "• In-Domain Test (15%): 1,371 (271 P / 1,100 N)",
            "• 0% exact-text overlap across splits",
        ],
        facecolor="#EDF2F7", edgecolor="#4A5568", title_color="#1A202C"
    )

    # Bottom banner: Inductive Representation & Dual Evaluation Protocol
    draw_arrow(16, 13.4, 16, 9.6)
    draw_box(
        2, 1.0, 96, 8.2,
        "Strictly Inductive Model Training & Dual Evaluation Protocol (Zero Test or External Exposure)",
        [
            "• Fitted strictly on Train (n = 6,392): TF-IDF (Word 1-2g + Char 3-5g), BiLSTM, PhoBERT-base, ViSoBERT, Inductive BERTopic (36 topics + 1 outlier slot)",
            "• Evaluated with 2,000-Bootstrap 95% CIs, Exact McNemar & Paired Bootstrap tests on In-Domain Test (n = 1,371) & UIT-VSMEC (n = 3,084 Full / 2,691 Sadness)",
        ],
        facecolor="#FFF5F5", edgecolor="#C53030", title_color="#742A2A"
    )

    fig.tight_layout()
    out = FIG_DIR / "fig1_new.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def make_fig2_dataset_characterization():
    """Figure 2: (a) Weak-labeler score distribution by gold class; (b) Comment length by gold class."""
    train_df = pd.read_csv(DATA_DIR / "final_train.csv")
    val_df = pd.read_csv(DATA_DIR / "final_val.csv")
    test_df = pd.read_csv(DATA_DIR / "final_test.csv")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    scores = []
    word_lens = []
    for txt in full_df["comment_text"].fillna("").astype(str):
        det = label_comment_detailed(txt)
        scores.append(det.depression_score)
        word_lens.append(len(txt.split()))
    full_df["lex_score"] = np.clip(scores, -8, 20)
    full_df["word_len"] = np.clip(word_lens, 1, 250)

    neg_scores = full_df[full_df["label"] == 0]["lex_score"].values
    pos_scores = full_df[full_df["label"] == 1]["lex_score"].values

    neg_words = full_df[full_df["label"] == 0]["word_len"].values
    pos_words = full_df[full_df["label"] == 1]["word_len"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    # Panel (a): Weak-label score distribution
    bins_s = np.arange(-8.5, 20.5, 1.0)
    ax1.hist(neg_scores, bins=bins_s, density=True, alpha=0.62, color="#3182CE", edgecolor="black", linewidth=0.4,
             label="Gold Negative (Normal, n = 7,330)")
    ax1.hist(pos_scores, bins=bins_s, density=True, alpha=0.62, color="#E53E3E", edgecolor="black", linewidth=0.4,
             label="Gold Positive (Depression, n = 1,804)")
    ax1.axvline(-2, color="#2B6CB0", linestyle="--", linewidth=1.5, label="Normal threshold (s ≤ -2: 18.91% pool)")
    ax1.axvline(5, color="#C53030", linestyle="--", linewidth=1.5, label="Depression threshold (s ≥ 5: 2.57% pool)")
    ax1.axvspan(-1.5, 4.5, color="#ECC94B", alpha=0.18, label="Uncertain abstention zone (-1 ≤ s ≤ 4: 78.52%)")
    ax1.set_title("(a) Lexical Weak-Labeler Score Distribution s(x)")
    ax1.set_xlabel("Additive Lexicon Score s(x) [clipped to [-8, 20]]")
    ax1.set_ylabel("Empirical Density (within gold class)")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.6)

    # Panel (b): Comment word length distribution by gold class
    bins_w = np.linspace(0, 220, 35)
    ax2.hist(neg_words, bins=bins_w, density=True, alpha=0.58, color="#3182CE", edgecolor="black", linewidth=0.4,
             label="Normal (mean = 60.0 ± 69.3w, median = 37w)")
    ax2.hist(pos_words, bins=bins_w, density=True, alpha=0.58, color="#E53E3E", edgecolor="black", linewidth=0.4,
             label="Depression (mean = 75.4 ± 85.8w, median = 44w)")
    ax2.axvline(37, color="#2B6CB0", linestyle=":", linewidth=1.6)
    ax2.axvline(44, color="#C53030", linestyle=":", linewidth=1.6)
    ax2.set_title("(b) Comment Length Distribution by Gold Class (n = 9,134)")
    ax2.set_xlabel("Comment Length (whitespace-delimited words, clipped at 220)")
    ax2.set_ylabel("Empirical Density")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.0)

    fig.tight_layout()
    out = FIG_DIR / "fig2_new.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def make_fig3_hybrid_and_topics():
    """Figure 3: (a) Clean 806-dim Hybrid Vector Schematic; (b) Inductive BERTopic Clusters & Positive Rates."""
    train_df = pd.read_csv(DATA_DIR / "final_train.csv")
    bt_model = joblib.load(BASE_DIR / "models" / "bertopic" / "bertopic_model.pkl")
    train_topics = np.array(bt_model.topics_, dtype=int)
    train_labels = train_df["label"].values

    topic_labels_readable = {
        -1: "Topic -1: Outlier Cluster (Unassigned)",
        0: "Topic 0: Depression, Medication & Therapy",
        1: "Topic 1: Chronic Insomnia & Sleep Loss",
        2: "Topic 2: Family & Parental Conflict",
        3: "Topic 3: Video Appreciation & Peer Support",
        4: "Topic 4: Undiacritized General Comments",
        5: "Topic 5: Nighttime Crying & Loneliness",
        6: "Topic 6: School, Exams & Academic Stress",
        7: "Topic 7: Healing, Music & Encouragement",
    }

    rows = []
    for tid in [-1, 0, 1, 2, 3, 4, 5, 6, 7]:
        mask = (train_topics == tid)
        cnt = int(np.sum(mask))
        pos_rate = float(np.mean(train_labels[mask]) * 100.0) if cnt > 0 else 0.0
        rows.append((topic_labels_readable[tid], cnt, pos_rate))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.7), gridspec_kw={"width_ratios": [1.08, 1.12]})

    # Panel (a): Clean schematic of 806-dim Inductive Hybrid Architecture
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axis("off")
    ax1.set_title("(a) Inductive Hybrid Representation (z(x) ∈ ℝ⁸⁰⁶)")

    def sbox(x, y, w, h, title, sub, fc, ec):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4",
                           facecolor=fc, edgecolor=ec, linewidth=1.4)
        ax1.add_patch(p)
        ax1.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1A202C")
        ax1.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=6.6, color="#2D3748")

    sbox(4, 78, 92, 15, "Normalized Comment Text x (underthesea segmented)", "Strictly Inductive Evaluation (0% Test or VSMEC Exposure in Topic Fit)", "#EDF2F7", "#4A5568")
    ax1.annotate("", xy=(25, 67), xytext=(38, 77.5), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#2B6CB0"))
    ax1.annotate("", xy=(75, 67), xytext=(62, 77.5), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#B7791F"))

    sbox(2, 46, 45, 20, "Frozen PhoBERT-base", "h(x) ∈ ℝ⁷⁶⁸ (CLS pooling)\nFine-tuned per seed (42, 123, 2024)", "#EBF8FF", "#2B6CB0")
    sbox(53, 46, 45, 20, "Inductive Train-Only BERTopic", "OneHot(t(x)) ∈ {0,1}³⁷ + q(x) ∈ [0,1]\nSlot 0: Outlier (-1) | Slots 1..36: Topics", "#FEFCBF", "#B7791F")

    ax1.annotate("", xy=(42, 34), xytext=(24.5, 45.5), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#2B6CB0"))
    ax1.annotate("", xy=(58, 34), xytext=(75.5, 45.5), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#B7791F"))

    sbox(4, 19, 92, 14.5, "Concatenated Vector z(x) = [h(x); OneHot(t(x)); q(x)] ∈ ℝ⁸⁰⁶", "768-d Contextual + 37-d Topic Indicator (incl. Outlier) + 1-d Topic Confidence", "#E6FFFA", "#2C7A7B")
    ax1.annotate("", xy=(50, 13.5), xytext=(50, 18.5), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#C53030"))
    sbox(4, 1, 92, 12, "StandardScaler + L2 Logistic Regression (C = 1.0, Balanced)", "Soft-Vote & Majority-Vote Aggregation across 3 Seeds", "#FFF5F5", "#C53030")

    # Panel (b): Top Inductive Topics & Positive Class Rates in final_train.csv (n=6,392)
    labels_y = [f"{r[0]} (n={r[1]})" for r in rows][::-1]
    pos_rates = [r[2] for r in rows][::-1]
    colors_b = ["#C53030" if v >= 25.0 else ("#DD6B20" if v >= 19.74 else "#3182CE") for v in pos_rates]

    y_pos = np.arange(len(labels_y))
    bars = ax2.barh(y_pos, pos_rates, color=colors_b, edgecolor="black", linewidth=0.5, height=0.65, alpha=0.88)
    ax2.axvline(19.74, color="#1A202C", linestyle="--", linewidth=1.3, label="Train Base Positive Rate (19.74%)")
    for bar, val in zip(bars, pos_rates):
        ax2.text(val + 0.8, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", ha="left", fontsize=8.0, fontweight="bold")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_y, fontsize=8.0)
    ax2.set_xlim(0, max(pos_rates) + 9.0)
    ax2.set_xlabel("Empirical Gold Positive-Class Rate (% Depression in Train)")
    ax2.set_title("(b) Positive-Class Rate across Inductive BERTopic Clusters (n = 6,392)")
    ax2.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    out = FIG_DIR / "fig3_new.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def make_fig4_roc_pr_and_forest(visobert_data: dict, eval_data: dict):
    """Figure 4: (a) In-Domain Precision-Recall Curves; (b) Macro-F1 Forest Plot with 95% Bootstrap CIs (Original Models Only)."""
    curves = visobert_data["curves_in_domain"]
    y_test = np.array(curves["y_test"], dtype=int)
    p_phobert = np.array(curves["phobert_soft_prob"], dtype=float)
    p_logreg = np.array(curves["tfidf_logreg_prob"], dtype=float)
    p_lex = np.array(curves["lexicon_prob"], dtype=float)

    # Load Hybrid PhoBERT+BERTopic probabilities on in-domain test
    hyb_df = pd.read_csv(BASE_DIR / "results" / "phobert_bertopic_predictions_in_domain.csv")
    p_hyb = hyb_df["prob_depression"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw={"width_ratios": [1.05, 1.15]})

    # Panel (a): Precision-Recall Curves on In-Domain Test (n=1,371) for Original Models
    model_curves = [
        ("PhoBERT-base (Soft Vote)", p_phobert, "#C53030", "-", 2.1, eval_data["in_domain"]["phobert_soft_vote"]["pr_auc"]),
        ("PhoBERT + BERTopic Hybrid", p_hyb, "#2C7A7B", "--", 1.9, eval_data["in_domain"]["phobert_bertopic"]["pr_auc"]),
        ("TF-IDF + LogReg (L2, C=1.0)", p_logreg, "#D69E2E", "-.", 1.9, eval_data["in_domain"]["tfidf_logreg"]["pr_auc"]),
        ("Lexicon Weak-Labeler (All 1,371)", p_lex, "#718096", ":", 1.6, eval_data["in_domain"]["weak_labeler_lexicon"]["pr_auc"]),
    ]
    for label, probs, col, ls, lw, pr_auc in model_curves:
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ax1.plot(rec, prec, color=col, linestyle=ls, linewidth=lw, label=f"{label} (PR-AUC = {pr_auc:.4f})")

    base_rate = float(np.mean(y_test))
    ax1.axhline(base_rate, color="black", linestyle=":", linewidth=1.0, label=f"No-Skill Baseline (Prevalence = {base_rate:.4f})")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.02)
    ax1.set_xlabel("Positive-Class Recall (Depression)")
    ax1.set_ylabel("Positive-Class Precision (Depression)")
    ax1.set_title("(a) In-Domain Precision–Recall Curves (n = 1,371)")
    ax1.legend(loc="upper right", frameon=True, fontsize=7.6)

    # Panel (b): Forest Plot of In-Domain Macro-F1 with 95% Bootstrap CIs (B = 2,000) — Original Models Only
    forest_items = [
        ("PhoBERT + BERTopic (Maj. Vote)", eval_data["in_domain"]["phobert_bertopic_majority_vote"]["f1_macro"],
         eval_data["bootstrap_95_ci_in_domain"]["phobert_bertopic"]["f1_macro"]["ci95_lower"],
         eval_data["bootstrap_95_ci_in_domain"]["phobert_bertopic"]["f1_macro"]["ci95_upper"], "#2C7A7B"),
        ("PhoBERT-base (Maj. Vote)", eval_data["in_domain"]["phobert_majority_vote"]["f1_macro"],
         eval_data["bootstrap_95_ci_in_domain"]["phobert_majority_vote"]["f1_macro"]["ci95_lower"],
         eval_data["bootstrap_95_ci_in_domain"]["phobert_majority_vote"]["f1_macro"]["ci95_upper"], "#C53030"),
        ("PhoBERT-base (Soft Vote)", eval_data["in_domain"]["phobert_soft_vote"]["f1_macro"],
         eval_data["bootstrap_95_ci_in_domain"]["phobert_soft_vote"]["f1_macro"]["ci95_lower"],
         eval_data["bootstrap_95_ci_in_domain"]["phobert_soft_vote"]["f1_macro"]["ci95_upper"], "#E53E3E"),
        ("PhoBERT + BERTopic (Soft Vote)", eval_data["in_domain"]["phobert_bertopic"]["f1_macro"],
         eval_data["bootstrap_95_ci_in_domain"]["phobert_bertopic"]["f1_macro"]["ci95_lower"],
         eval_data["bootstrap_95_ci_in_domain"]["phobert_bertopic"]["f1_macro"]["ci95_upper"], "#319795"),
        ("TF-IDF + LogReg (L2, C=1.0)", eval_data["in_domain"]["tfidf_logreg"]["f1_macro"],
         eval_data["bootstrap_95_ci_in_domain"]["tfidf_logreg"]["f1_macro"]["ci95_lower"],
         eval_data["bootstrap_95_ci_in_domain"]["tfidf_logreg"]["f1_macro"]["ci95_upper"], "#D69E2E"),
        ("TF-IDF + LinearSVC (C=1.0)", eval_data["in_domain"]["tfidf_svc"]["f1_macro"],
         eval_data["bootstrap_95_ci_in_domain"]["tfidf_svc"]["f1_macro"]["ci95_lower"],
         eval_data["bootstrap_95_ci_in_domain"]["tfidf_svc"]["f1_macro"]["ci95_upper"], "#DD6B20"),
        ("BiLSTM Weighted (Maj. Vote)", eval_data["in_domain"]["bilstm_weighted_majority_vote"]["f1_macro"],
         eval_data["bootstrap_95_ci_in_domain"]["bilstm_weighted_majority_vote"]["f1_macro"]["ci95_lower"],
         eval_data["bootstrap_95_ci_in_domain"]["bilstm_weighted_majority_vote"]["f1_macro"]["ci95_upper"], "#805AD5"),
        ("Lexicon Weak-Labeler (All 1,371)", eval_data["in_domain"]["weak_labeler_lexicon"]["f1_macro"],
         eval_data["bootstrap_95_ci_in_domain"]["weak_labeler_lexicon"]["f1_macro"]["ci95_lower"],
         eval_data["bootstrap_95_ci_in_domain"]["weak_labeler_lexicon"]["f1_macro"]["ci95_upper"], "#718096"),
        ("BiLSTM Unweighted (Maj. Vote)", eval_data["in_domain"]["bilstm_majority_vote"]["f1_macro"],
         eval_data["bootstrap_95_ci_in_domain"]["bilstm_majority_vote"]["f1_macro"]["ci95_lower"],
         eval_data["bootstrap_95_ci_in_domain"]["bilstm_majority_vote"]["f1_macro"]["ci95_upper"], "#9F7AEA"),
        ("Inductive BERTopic-Only", eval_data["in_domain"]["bertopic_only"]["f1_macro"],
         0.5210, 0.5865, "#4A5568"),
    ][::-1]

    y_idx = np.arange(len(forest_items))
    for i, (name, val, lo, hi, col) in enumerate(forest_items):
        ax2.errorbar(val, i, xerr=[[val - lo], [hi - val]], fmt="o", color=col, ecolor=col,
                     elinewidth=2.0, capsize=4, markersize=6)
        ax2.text(hi + 0.005, i, f"{val:.4f} [{lo:.3f}, {hi:.3f}]", va="center", ha="left", fontsize=7.6, fontweight="bold")

    ax2.axvline(0.4452, color="#A0AEC0", linestyle="--", linewidth=1.3, label="Trivial Baseline (0.4452)")
    ax2.set_yticks(y_idx)
    ax2.set_yticklabels([x[0] for x in forest_items], fontsize=8.0)
    ax2.set_xlim(0.42, 0.86)
    ax2.set_xlabel("In-Domain Macro-F1 (with 2,000-Bootstrap 95% Confidence Intervals)")
    ax2.set_title("(b) Model Comparison & 95% Bootstrap CIs (n = 1,371)")
    ax2.legend(loc="upper left", frameon=True, fontsize=7.6)

    fig.tight_layout()
    out = FIG_DIR / "fig4_new.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def make_fig5_confusion_matrices(visobert_data: dict, eval_data: dict):
    """Figure 5: Reconciled Confusion Matrices (PhoBERT In-Domain, TF-IDF+LogReg In-Domain, PhoBERT External)."""
    cm_pho_in = np.array(eval_data["in_domain"]["phobert_majority_vote"]["confusion_matrix"])
    cm_lr_in = np.array(eval_data["in_domain"]["tfidf_logreg"]["confusion_matrix"])
    cm_pho_ext = np.array(eval_data["cross_domain"]["phobert_majority_vote"]["confusion_matrix"])

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))

    panels = [
        (axes[0], cm_pho_in, "(a) PhoBERT In-Domain (Maj. Vote)\nF1-Macro = 0.7265 | Acc = 0.8096", "Blues", ["Normal (0)", "Depression (1)"]),
        (axes[1], cm_lr_in, "(b) TF-IDF + LogReg In-Domain\nF1-Macro = 0.7138 | Acc = 0.7936", "Oranges", ["Normal (0)", "Depression (1)"]),
        (axes[2], cm_pho_ext, "(c) PhoBERT External (τ = 0.50)\nF1-Macro = 0.3620 | Acc = 0.5107", "Greens", ["Enjoyment (0)", "Sad/Fear (1)"]),
    ]

    for ax, cm, title, cmap, tick_labels in panels:
        im = ax.imshow(cm, cmap=cmap, alpha=0.85)
        ax.set_title(title, fontsize=9.2, fontweight="bold", pad=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"Pred\n{l}" for l in tick_labels], fontsize=8.0)
        ax.set_yticklabels([f"True\n{l}" for l in tick_labels], fontsize=8.0)
        row_sums = cm.sum(axis=1, keepdims=True)
        for i in range(2):
            for j in range(2):
                cnt = cm[i, j]
                pct = (cnt / row_sums[i, 0]) * 100.0
                text_color = "white" if cnt > (cm.max() * 0.55) else "#1A202C"
                ax.text(j, i, f"{cnt:,}\n({pct:.1f}%)", ha="center", va="center",
                        fontsize=9.2, fontweight="bold", color=text_color)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#4A5568")

    fig.tight_layout()
    out = FIG_DIR / "fig5_new.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def make_fig6_ablation_and_calibration(visobert_data: dict, eval_data: dict):
    """Figure 6: (a) Class-Weighting Ablation (y-axis starting at 0.0, Original 3 Models); (b) Prior-Shift Threshold Calibration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5))

    # Panel (a): Class-Weighting Ablation (y-axis starting at 0.0 per Reviewer 1 Comment 11)
    models_abl = ["PhoBERT-base\n(Maj. Vote)", "TF-IDF +\nLogReg", "BiLSTM\n(Maj. Vote)"]
    unw_vals = [
        visobert_data["phobert_unweighted"]["in_domain"]["majority_vote"]["f1_macro"],
        visobert_data["tfidf_logreg_unweighted"]["f1_macro"],
        eval_data["in_domain"]["bilstm_majority_vote"]["f1_macro"],
    ]
    w_vals = [
        eval_data["in_domain"]["phobert_majority_vote"]["f1_macro"],
        eval_data["in_domain"]["tfidf_logreg"]["f1_macro"],
        eval_data["in_domain"]["bilstm_weighted_majority_vote"]["f1_macro"],
    ]

    x = np.arange(len(models_abl))
    width = 0.34
    b1 = ax1.bar(x - width / 2, unw_vals, width, label="Unweighted Loss (w₁ = 1.0)", color="#A0AEC0", edgecolor="black", linewidth=0.6)
    b2 = ax1.bar(x + width / 2, w_vals, width, label="Class-Weighted Loss (w₁ = 4.065)", color="#2B6CB0", edgecolor="black", linewidth=0.6)

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.4f}", ha="center", va="bottom", fontsize=8.0)
    for bar, u in zip(b2, unw_vals):
        h = bar.get_height()
        gain = h - u
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.4f}\n({gain:+.4f})", ha="center", va="bottom",
                 fontsize=8.0, fontweight="bold", color="#1A365D")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models_abl, fontsize=8.8)
    ax1.set_ylim(0.0, 0.92)
    ax1.set_ylabel("In-Domain Macro-F1 (y-axis starts at 0.0)")
    ax1.set_title("(a) Class-Weighting Ablation on In-Domain Test (n = 1,371)")
    ax1.legend(loc="upper right", frameon=True)

    # Panel (b): Cross-Domain Prior-Shift Threshold Calibration (y-axis starts at 0.0)
    sweep = eval_data["threshold_calibration_analysis"]["phobert_soft_vote"]["threshold_sweep"]
    thrs, f1_in, f1_vf, f1_vs = [], [], [], []
    for k, v in sorted(sweep.items(), key=lambda item: item[1]["threshold"]):
        thrs.append(v["threshold"])
        f1_in.append(v["in_domain_test"]["f1_macro"])
        f1_vf.append(v["vsmec_full"]["f1_macro"])
        f1_vs.append(v["vsmec_sadness_only"]["f1_macro"])

    ax2.plot(thrs, f1_in, "o-", color="#2B6CB0", linewidth=2.0, label="In-Domain YouTube Test (n = 1,371)")
    ax2.plot(thrs, f1_vs, "s-", color="#2C7A7B", linewidth=2.0, label="UIT-VSMEC Sadness-Only (n = 2,691)")
    ax2.plot(thrs, f1_vf, "^--", color="#C53030", linewidth=1.8, label="UIT-VSMEC Full Sadness+Fear (n = 3,084)")
    ax2.axvline(0.1974, color="#2C7A7B", linestyle=":", linewidth=1.6, label="Bayes Prior Threshold (τ = π_train = 0.1974)")
    ax2.axvline(0.5000, color="#718096", linestyle="--", linewidth=1.3, label="Default Uncalibrated Threshold (τ = 0.50)")

    ax2.set_ylim(0.0, 0.88)
    ax2.set_xlim(0.02, 0.58)
    ax2.set_xlabel("Classification Decision Threshold τ on P(y = 1 | x)")
    ax2.set_ylabel("Macro-F1 Score (y-axis starts at 0.0)")
    ax2.set_title("(b) PhoBERT Prior-Shift Threshold Calibration Across Domains")
    ax2.legend(loc="lower left", frameon=True, fontsize=7.5)

    fig.tight_layout()
    out = FIG_DIR / "fig6_new.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    make_fig1_workflow()
    make_fig2_dataset_characterization()
    make_fig3_hybrid_and_topics()

    vis_path = MODEL_DIR / "visobert_and_ablations_results.json"
    eval_path = MODEL_DIR / "evaluation_results.json"
    if vis_path.exists() and eval_path.exists():
        with open(vis_path, "r", encoding="utf-8") as f:
            vis_data = json.load(f)
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        make_fig4_roc_pr_and_forest(vis_data, eval_data)
        make_fig5_confusion_matrices(vis_data, eval_data)
        make_fig6_ablation_and_calibration(vis_data, eval_data)
    else:
        print("Waiting for visobert_and_ablations_results.json to generate Figs 4-6...")


if __name__ == "__main__":
    main()
