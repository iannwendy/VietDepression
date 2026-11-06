"""Generate Figure 2: Weak-labeler score distribution on the 125,329 cleaned YouTube comments.

Data source: docs/paper_report.html §3.5.5 Table 3.8.
Outputs: report_pdf/figures/fig-02-weak-label-distribution.png (300 DPI).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "report_pdf" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data extracted from paper_report.html §3.5.5 Table 3.8 (post-round-3 numbers).
# ---------------------------------------------------------------------------
LABELS = [
    {"name": "depression_auto", "count": 3223, "high": 779, "medium": 2444},
    {"name": "normal_auto",    "count": 23695, "high": 3449, "medium": 20246},
    {"name": "uncertain",       "count": 98410, "high": 0,   "medium": 0},
]
TOTAL = 125329

COLOR_HIGH = "#D55E00"   # vermillion (high confidence)
COLOR_MEDIUM = "#F0E442"  # yellow (medium confidence)
COLOR_REST = "#999999"    # grey (rest)


def render(out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.5), dpi=300,
                                    gridspec_kw={"width_ratios": [2.2, 1]})

    # --- LEFT: stacked horizontal bar ---
    names = [d["name"] for d in LABELS]
    counts = [d["count"] for d in LABELS]
    highs = [d["high"] for d in LABELS]
    mediums = [d["medium"] for d in LABELS]

    y = np.arange(len(names))
    bars_h = ax1.barh(y, highs, color=COLOR_HIGH, edgecolor="black",
                       linewidth=0.6, label="High confidence")
    bars_m = ax1.barh(y, mediums, left=highs, color=COLOR_MEDIUM, edgecolor="black",
                       linewidth=0.6, label="Medium confidence")
    bars_r = ax1.barh(
        y,
        [c - h - m for c, h, m in zip(counts, highs, mediums)],
        left=[h + m for h, m in zip(highs, mediums)],
        color=COLOR_REST, edgecolor="black", linewidth=0.6,
        label="(low / unclassified)",
    )

    # In-bar count labels
    for i, (h, m, c) in enumerate(zip(highs, mediums, counts)):
        if h > 0:
            ax1.text(h / 2, i, f"{h:,}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if m > 0:
            ax1.text(h + m / 2, i, f"{m:,}", ha="center", va="center", fontsize=9, color="black")
        if (c - h - m) > 0:
            ax1.text(h + m + (c - h - m) / 2, i, f"{c - h - m:,}", ha="center", va="center", fontsize=9, color="black")

    # Right-side total + percentage
    for i, c in enumerate(counts):
        ax1.text(c + max(counts) * 0.012, i, f"{c:,}\n({c/TOTAL*100:.1f}%)",
                 ha="left", va="center", fontsize=9, fontweight="bold")

    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=10.5)
    ax1.invert_yaxis()  # depression_auto on top
    ax1.set_xlabel("Number of comments", fontsize=10.5)
    ax1.set_xlim(0, max(counts) * 1.18)
    ax1.set_title("Distribution of weak labels on the 125,329-comment YouTube corpus",
                  fontsize=11.5, pad=12)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # --- RIGHT: pie chart of overall proportions ---
    pie_labels = [f"{n}\n({c:,}, {c/TOTAL*100:.1f}%)" for n, c in zip(names, counts)]
    pie_colors = [COLOR_HIGH if "depression" in n else (COLOR_MEDIUM if "normal" in n else COLOR_REST) for n in names]
    wedges, texts = ax2.pie(
        counts,
        labels=pie_labels,
        colors=pie_colors,
        startangle=90,
        wedgeprops={"edgecolor": "black", "linewidth": 0.6},
        textprops={"fontsize": 9.5},
    )
    ax2.set_title("Class proportions\n(total = 125,329 comments)", fontsize=11.5, pad=12)

    # Master title
    fig.suptitle(
        "Figure 2. Weak-label distribution on the cleaned YouTube corpus (lexicon-based auto-labeling)",
        fontsize=12.5, y=1.02, fontweight="bold",
    )

    # Footnote
    fig.text(
        0.5, -0.02,
        "Source: docs/paper_report.html §3.5.5 Table 3.8.  "
        "Threshold: score ≥ 5 → depression_auto (high ≥ 8); score ≤ -2 → normal_auto (high ≤ -4); else uncertain.  "
        "Note: 78.52% of comments fall into 'uncertain' — the keyword weak-labeler cannot classify them.  "
        "Need-review flag was raised on 120,334 (96.01%) of comments.",
        ha="center", fontsize=8, color="#555555", wrap=True,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    render(FIGURES_DIR / "fig-02-weak-label-distribution.png")