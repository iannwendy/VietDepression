"""Generate Figure C1-1: Conceptual framework — clinical reality, ecological
signal, and ML pipeline as three stacked layers motivating this work.

This figure motivates Chapter 1 by visualising the central argument of §1.1:
(1) depression is a high-burden condition constrained by structural barriers
(2) social media offers an ecological linguistic signal source
(3) deep-learning pipelines can operationalise this signal at scale

Outputs:
  report_pdf/figures/fig-c1-01-conceptual-framework.png  (300 DPI)
  report_pdf/figures/fig-c1-01-conceptual-framework.svg  (vector)
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

# Colorblind-safe palette (Okabe-Ito / Wong 2011)
COLOR_CLINICAL = "#D55E00"   # vermillion
COLOR_SIGNAL = "#0072B2"     # blue
COLOR_ML = "#009E73"         # bluish green
COLOR_THIS_WORK = "#E69F00"  # orange (highlight)
COLOR_TEXT = "#1a1a1a"
COLOR_MUTED = "#555555"


def _box(ax, x, y, w, h, text, fill, edge="black", fontsize=10, fontweight="normal"):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=fill,
        alpha=0.18,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=COLOR_TEXT,
        wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2, label=None, color="black", lw=1.4):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arrow)
    if label is not None:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLOR_MUTED,
            style="italic",
        )


def render(out_png: Path, out_svg: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # ----- Title -----
    ax.text(
        5.0,
        6.6,
        "Conceptual framework: from clinical depression to scalable NLP screening",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=COLOR_TEXT,
    )

    # ----- Layer 1: Clinical reality -----
    ax.text(
        0.25,
        5.55,
        "Layer 1  ·  Clinical reality",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLOR_CLINICAL,
    )
    _box(
        ax,
        0.5,
        4.55,
        2.5,
        0.85,
        "Major Depressive\nDisorder (MDD)\n7.5% of global YLDs",
        "#D55E00",
        fontsize=9,
        fontweight="bold",
    )
    _box(
        ax,
        3.25,
        4.55,
        2.5,
        0.85,
        "DSM-5 criteria\n(9 symptoms)",
        "#D55E00",
        fontsize=9,
    )
    _box(
        ax,
        6.0,
        4.55,
        3.5,
        0.85,
        "PHQ-9 screening\n(seeking self-report)",
        "#D55E00",
        fontsize=9,
    )
    ax.text(
        5.0,
        4.25,
        "Stigma  ·  workforce shortage (0.2 psychiatrists / 100K in VN)  ·  self-report bias",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLOR_MUTED,
        style="italic",
    )

    # ----- Layer 2: Ecological signal -----
    ax.text(
        0.25,
        3.55,
        "Layer 2  ·  Ecological signal source",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLOR_SIGNAL,
    )
    _box(
        ax,
        1.5,
        2.55,
        3.5,
        0.85,
        "YouTube comments\n(2B+ MAU; Vietnamese\n125K crawled)",
        "#0072B2",
        fontsize=9,
        fontweight="bold",
    )
    _box(
        ax,
        5.25,
        2.55,
        3.25,
        0.85,
        "Ecologically valid:\nspontaneous, unprompted,\nembedded in context",
        "#0072B2",
        fontsize=9,
    )
    ax.text(
        5.0,
        2.25,
        "No diacritics  ·  teen slang  ·  code-switching  ·  short  ·  noisy",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLOR_MUTED,
        style="italic",
    )

    # ----- Layer 3: ML pipeline (this work) -----
    ax.text(
        0.25,
        1.55,
        "Layer 3  ·  ML pipeline  (this work)",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLOR_ML,
    )
    _box(
        ax,
        0.4,
        0.55,
        1.55,
        0.85,
        "Weak-label\n(335-keyword\nlexicon)",
        "#009E73",
        fontsize=8.5,
    )
    _box(
        ax,
        2.05,
        0.55,
        1.55,
        0.85,
        "Blind human\nannotation\n(1,607 gold)",
        "#009E73",
        fontsize=8.5,
    )
    _box(
        ax,
        3.7,
        0.55,
        1.55,
        0.85,
        "Train\nTF-IDF / BiLSTM /\nPhoBERT / BERTopic",
        "#009E73",
        fontsize=8.5,
        fontweight="bold",
    )
    _box(
        ax,
        5.35,
        0.55,
        1.55,
        0.85,
        "Evaluate\nin-domain\n(YouTube, n=383)",
        COLOR_THIS_WORK,
        fontsize=8.5,
        edge=COLOR_THIS_WORK,
        fontweight="bold",
    )
    _box(
        ax,
        7.0,
        0.55,
        1.55,
        0.85,
        "Evaluate\ncross-domain\n(VSMEC, n=3,084)",
        COLOR_THIS_WORK,
        fontsize=8.5,
        edge=COLOR_THIS_WORK,
        fontweight="bold",
    )
    _box(
        ax,
        8.65,
        0.55,
        1.05,
        0.85,
        "ΔF1 gap\n≈ 0.53",
        COLOR_THIS_WORK,
        fontsize=8.5,
        edge=COLOR_THIS_WORK,
        fontweight="bold",
    )

    # ----- Inter-layer arrows -----
    # Layer 1 → Layer 2: "barriers motivate alternative"
    _arrow(ax, 1.75, 4.55, 1.75, 3.4, color=COLOR_CLINICAL, lw=1.2)
    _arrow(ax, 4.5, 4.55, 4.5, 3.4, color=COLOR_CLINICAL, lw=1.2)
    _arrow(ax, 7.5, 4.55, 7.5, 3.4, color=COLOR_CLINICAL, lw=1.2)
    ax.text(
        8.55,
        4.0,
        "barriers\n→ seek passive\nscreening",
        ha="left",
        va="center",
        fontsize=8,
        color=COLOR_MUTED,
        style="italic",
    )

    # Layer 2 → Layer 3: "operationalise as signal"
    _arrow(ax, 5.0, 2.55, 5.0, 1.4, color=COLOR_SIGNAL, lw=1.2)
    ax.text(
        5.15,
        1.95,
        "operationalise\nas training signal",
        ha="left",
        va="center",
        fontsize=8,
        color=COLOR_MUTED,
        style="italic",
    )

    # Intra-layer 3 arrows
    _arrow(ax, 1.95, 0.975, 2.05, 0.975, color="#009E73", lw=1.4)
    _arrow(ax, 3.6, 0.975, 3.7, 0.975, color="#009E73", lw=1.4)
    _arrow(ax, 5.25, 0.975, 5.35, 0.975, color="#009E73", lw=1.4)
    _arrow(ax, 6.9, 0.975, 7.0, 0.975, color="#009E73", lw=1.4)
    _arrow(ax, 8.55, 0.975, 8.65, 0.975, color="#009E73", lw=1.4)

    # ----- Legend (highlight) -----
    legend_patch = mpatches.Patch(
        facecolor=COLOR_THIS_WORK,
        alpha=0.25,
        edgecolor=COLOR_THIS_WORK,
        linewidth=1.0,
        label="Highlight: this work's empirical evaluation scope",
    )
    ax.legend(
        handles=[legend_patch],
        loc="lower left",
        bbox_to_anchor=(0.0, -0.02),
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
    stem = "fig-c1-01-conceptual-framework"
    render(
        FIGURES_DIR / f"{stem}.png",
        FIGURES_DIR / f"{stem}.svg",
    )