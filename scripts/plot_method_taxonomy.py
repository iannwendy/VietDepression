"""Generate Figure C2-1: Methodological taxonomy of NLP approaches to
depression detection from text.

This figure organises the literature surveyed in §2.2 into a three-branch
taxonomy (lexicon-based / traditional ML / deep learning) with the project's
own PhoBERT contribution highlighted at the deep-learning leaf.

Outputs:
  report_pdf/figures/fig-c2-01-method-taxonomy.png  (300 DPI)
  report_pdf/figures/fig-c2-01-method-taxonomy.svg  (vector)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "report_pdf" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-safe palette
COLOR_LEXICON = "#999999"        # neutral grey
COLOR_CLASSIC = "#56B4E9"        # sky blue
COLOR_DEEP = "#0072B2"           # blue
COLOR_DEEP_DOMAIN = "#D55E00"    # vermillion (domain-adapted)
COLOR_VIETNAMESE = "#009E73"     # bluish green (this work)
COLOR_TEXT = "#1a1a1a"
COLOR_MUTED = "#555555"


def _node(ax, x, y, w, h, text, fill, edge="black", fontsize=9, fontweight="normal", alpha=0.20):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=fill,
        alpha=alpha,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=COLOR_TEXT,
    )


def _branch(ax, x1, y1, x2, y2, color, lw=1.3):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-",
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arrow)


def render(out_png: Path, out_svg: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.6), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # ----- Title -----
    ax.text(
        6.0,
        6.55,
        "Methodological taxonomy of NLP approaches to depression detection",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    ax.text(
        6.0,
        6.2,
        "from lexicon-based methods to deep contextual models (PhoBERT, this work)",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLOR_MUTED,
        style="italic",
    )

    # ----- Root node -----
    _node(
        ax,
        6.0,
        5.55,
        5.6,
        0.55,
        "Depression detection from user-generated text",
        "#FFFFFF",
        edge="black",
        fontsize=10.5,
        fontweight="bold",
        alpha=1.0,
    )

    # ----- Branch 1: Lexicon-based -----
    _node(
        ax,
        2.0,
        4.55,
        2.8,
        0.5,
        "1.  Lexicon-based",
        "#FFFFFF",
        edge=COLOR_LEXICON,
        fontsize=10.5,
        fontweight="bold",
        alpha=1.0,
    )
    _node(
        ax,
        2.0,
        3.85,
        2.8,
        0.45,
        "LIWC, ANEW\n(Pennebaker 2015)",
        COLOR_LEXICON,
        fontsize=8.5,
    )
    _node(
        ax,
        2.0,
        3.20,
        2.8,
        0.45,
        "Aggregate emotion scores;\ncontext-blind",
        COLOR_LEXICON,
        fontsize=8.5,
    )
    _node(
        ax,
        2.0,
        2.50,
        2.8,
        0.40,
        "✗  No negation/sarcasm\n✗  Language-specific",
        "#FFFFFF",
        edge=COLOR_LEXICON,
        fontsize=8.0,
    )

    # ----- Branch 2: Traditional ML -----
    _node(
        ax,
        6.0,
        4.55,
        2.8,
        0.5,
        "2.  Traditional ML",
        "#FFFFFF",
        edge=COLOR_CLASSIC,
        fontsize=10.5,
        fontweight="bold",
        alpha=1.0,
    )
    _node(
        ax,
        6.0,
        3.85,
        2.8,
        0.45,
        "TF-IDF + SVM / LR / RF\n(Coppersmith 2014; CLPsych)",
        COLOR_CLASSIC,
        fontsize=8.5,
    )
    _node(
        ax,
        6.0,
        3.20,
        2.8,
        0.45,
        "Stylometric + POS features;\nshallow semantics",
        COLOR_CLASSIC,
        fontsize=8.5,
    )
    _node(
        ax,
        6.0,
        2.50,
        2.8,
        0.40,
        "✓ Strong baselines\n✗ Manual feature engineering",
        "#FFFFFF",
        edge=COLOR_CLASSIC,
        fontsize=8.0,
    )

    # ----- Branch 3: Deep Learning -----
    _node(
        ax,
        10.0,
        4.55,
        2.8,
        0.5,
        "3.  Deep Learning",
        "#FFFFFF",
        edge=COLOR_DEEP,
        fontsize=10.5,
        fontweight="bold",
        alpha=1.0,
    )

    # Sub-branch: Generic pretrained
    _node(
        ax,
        8.7,
        3.85,
        2.4,
        0.45,
        "Generic pretrained\nBERT, RoBERTa",
        COLOR_DEEP,
        fontsize=8.5,
    )
    _node(
        ax,
        8.7,
        3.20,
        2.4,
        0.45,
        "Strong general semantics;\nout-of-domain mental health",
        COLOR_DEEP,
        fontsize=8.0,
    )

    # Sub-branch: Domain-adapted
    _node(
        ax,
        11.3,
        3.85,
        2.4,
        0.45,
        "Domain-adapted\nMentalBERT, PsychBERT",
        COLOR_DEEP_DOMAIN,
        fontsize=8.5,
    )
    _node(
        ax,
        11.3,
        3.20,
        2.4,
        0.45,
        "Reddit / Twitter tuning;\nEnglish-only",
        COLOR_DEEP_DOMAIN,
        fontsize=8.0,
    )

    # Sub-branch: Vietnamese (this work)
    _node(
        ax,
        10.0,
        2.50,
        2.8,
        0.50,
        "Vietnamese\nPhoBERT  ←  THIS WORK",
        COLOR_VIETNAMESE,
        edge=COLOR_VIETNAMESE,
        fontsize=9.0,
        fontweight="bold",
        alpha=0.30,
    )
    _node(
        ax,
        10.0,
        1.85,
        2.8,
        0.40,
        "20GB monolingual corpus;\nVietnamese tokenizer",
        COLOR_VIETNAMESE,
        fontsize=8.0,
    )
    _node(
        ax,
        10.0,
        1.30,
        2.8,
        0.40,
        "✓ In-domain F1 = 0.9623\n✗ Cross-domain F1 = 0.4318",
        "#FFFFFF",
        edge=COLOR_VIETNAMESE,
        fontsize=8.0,
        fontweight="bold",
    )

    # ----- Connector lines (root → branches) -----
    _branch(ax, 6.0, 5.30, 2.0, 4.80, color=COLOR_LEXICON, lw=1.5)
    _branch(ax, 6.0, 5.30, 6.0, 4.80, color=COLOR_CLASSIC, lw=1.5)
    _branch(ax, 6.0, 5.30, 10.0, 4.80, color=COLOR_DEEP, lw=1.5)

    # ----- Within-branch vertical lines -----
    # Branch 1
    _branch(ax, 2.0, 4.30, 2.0, 4.05, color=COLOR_LEXICON)
    _branch(ax, 2.0, 3.62, 2.0, 3.40, color=COLOR_LEXICON)
    _branch(ax, 2.0, 2.98, 2.0, 2.70, color=COLOR_LEXICON)

    # Branch 2
    _branch(ax, 6.0, 4.30, 6.0, 4.05, color=COLOR_CLASSIC)
    _branch(ax, 6.0, 3.62, 6.0, 3.40, color=COLOR_CLASSIC)
    _branch(ax, 6.0, 2.98, 6.0, 2.70, color=COLOR_CLASSIC)

    # Branch 3: split into 3 sub-branches at y=3.40
    _branch(ax, 10.0, 4.30, 10.0, 3.62, color=COLOR_DEEP, lw=1.5)
    _branch(ax, 10.0, 3.62, 8.7, 3.62, color=COLOR_DEEP)
    _branch(ax, 10.0, 3.62, 11.3, 3.62, color=COLOR_DEEP_DOMAIN)
    _branch(ax, 10.0, 3.62, 10.0, 2.75, color=COLOR_VIETNAMESE, lw=1.5)

    _branch(ax, 8.7, 3.62, 8.7, 3.40, color=COLOR_DEEP)
    _branch(ax, 8.7, 2.98, 8.7, 2.75, color=COLOR_DEEP)
    _branch(ax, 11.3, 3.62, 11.3, 3.40, color=COLOR_DEEP_DOMAIN)
    _branch(ax, 11.3, 2.98, 11.3, 2.75, color=COLOR_DEEP_DOMAIN)
    _branch(ax, 10.0, 2.25, 10.0, 2.05, color=COLOR_VIETNAMESE)
    _branch(ax, 10.0, 1.65, 10.0, 1.50, color=COLOR_VIETNAMESE)

    # ----- Legend (bottom) -----
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_LEXICON, alpha=0.30, edgecolor=COLOR_LEXICON, label="Lexicon-based"),
        mpatches.Patch(facecolor=COLOR_CLASSIC, alpha=0.30, edgecolor=COLOR_CLASSIC, label="Traditional ML"),
        mpatches.Patch(facecolor=COLOR_DEEP, alpha=0.30, edgecolor=COLOR_DEEP, label="Deep (generic)"),
        mpatches.Patch(facecolor=COLOR_DEEP_DOMAIN, alpha=0.30, edgecolor=COLOR_DEEP_DOMAIN, label="Deep (domain-adapted)"),
        mpatches.Patch(facecolor=COLOR_VIETNAMESE, alpha=0.40, edgecolor=COLOR_VIETNAMESE, label="This work (PhoBERT)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=5,
        frameon=True,
        fontsize=8.5,
        framealpha=0.95,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_svg}")


if __name__ == "__main__":
    stem = "fig-c2-01-method-taxonomy"
    render(
        FIGURES_DIR / f"{stem}.png",
        FIGURES_DIR / f"{stem}.svg",
    )