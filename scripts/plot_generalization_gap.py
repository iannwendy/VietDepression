"""Generate Figure 4: Generalization gap across five model architectures.

Reads model metrics JSON files and produces a grouped bar chart with error bars
comparing in-domain vs cross-domain F1-macro for the seven model variants
reported in Table 5.1 of paper_report.html.

Outputs: report_pdf/figures/fig-04-generalization-gap.png (300 DPI).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "report_pdf" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-safe palette (Wong 2011, Okabe-Ito derived)
COLOR_IN_DOMAIN = "#0072B2"   # blue
COLOR_CROSS_DOMAIN = "#D55E00"  # vermillion (orange-red)

# ---------------------------------------------------------------------------
# Data: extracted from models/*/metrics.json (post-round-3, multi-seed where
# applicable). Source-of-truth numbers in docs/paper_report.html Table 5.1.
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "TF-IDF + LogReg",
        "in_mean": 0.8347, "in_std": 0.0,
        "cross_mean": 0.3917, "cross_std": 0.0,
    },
    {
        "name": "TF-IDF + LinearSVC",
        "in_mean": 0.8286, "in_std": 0.0,
        "cross_mean": 0.3820, "cross_std": 0.0,
    },
    {
        "name": "BiLSTM (random)",
        "in_mean": 0.8145, "in_std": 0.0244,
        "cross_mean": 0.4690, "cross_std": 0.0601,
    },
    {
        "name": "BiLSTM (PhoBERT-frozen)",
        "in_mean": 0.8244, "in_std": 0.0044,
        "cross_mean": 0.4344, "cross_std": 0.0008,
    },
    {
        "name": "PhoBERT",
        "in_mean": 0.8681, "in_std": 0.0086,
        "cross_mean": 0.3727, "cross_std": 0.0242,
    },
    {
        "name": "BERTopic-only",
        "in_mean": 0.5599, "in_std": 0.0,
        "cross_mean": 0.5030, "cross_std": 0.0,
    },
    {
        "name": "PhoBERT + BERTopic",
        "in_mean": 0.8497, "in_std": 0.0,
        "cross_mean": 0.4406, "cross_std": 0.0,
    },
]


def render(out_path: Path) -> None:
    labels = [m["name"] for m in MODELS]
    in_means = np.array([m["in_mean"] for m in MODELS])
    in_stds = np.array([m["in_std"] for m in MODELS])
    cross_means = np.array([m["cross_mean"] for m in MODELS])
    cross_stds = np.array([m["cross_std"] for m in MODELS])
    gaps = in_means - cross_means

    n = len(MODELS)
    x = np.arange(n)
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)
    bars_in = ax.bar(
        x - width / 2,
        in_means,
        width,
        yerr=in_stds,
        capsize=4,
        color=COLOR_IN_DOMAIN,
        edgecolor="black",
        linewidth=0.6,
        label="In-domain (YouTube, n=383)",
        error_kw={"elinewidth": 0.8, "ecolor": "black"},
    )
    bars_cross = ax.bar(
        x + width / 2,
        cross_means,
        width,
        yerr=cross_stds,
        capsize=4,
        color=COLOR_CROSS_DOMAIN,
        edgecolor="black",
        linewidth=0.6,
        label="Cross-domain (VSMEC, n=3,084)",
        error_kw={"elinewidth": 0.8, "ecolor": "black"},
    )

    # Annotate bars with values
    for bar, mean, std in zip(bars_in, in_means, in_stds):
        label = f"{mean:.3f}" + (f"\n±{std:.4f}" if std > 0 else "")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="black",
        )
    for bar, mean, std in zip(bars_cross, cross_means, cross_stds):
        label = f"{mean:.3f}" + (f"\n±{std:.4f}" if std > 0 else "")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="black",
        )

    # Gap annotations between paired bars
    for xi, gap in zip(x, gaps):
        ax.annotate(
            f"Δ={gap:.2f}",
            xy=(xi, max(in_means[np.where(np.array(labels) == labels[xi])[0][0]], cross_means[np.where(np.array(labels) == labels[xi])[0][0]]) + 0.07),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#444444",
        )

    # Title and labels
    ax.set_title(
        "Figure 4. Generalization gap between in-domain and cross-domain evaluation\n"
        "(five model architectures on the post-round-3 final_dataset)",
        fontsize=11.5,
        pad=12,
    )
    ax.set_ylabel("F1-macro (mean ± std over 3 seeds where available)", fontsize=10.5)
    ax.set_xlabel("Model architecture", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

    # Footnote / source
    fig.text(
        0.5,
        -0.02,
        "Source: Table 5.1 (paper_report.html §5.1). "
        "PhoBERT and BiLSTM report mean ± std over 3 seeds (42, 123, 2024); "
        "classical baselines are single-seed. "
        "Δ = in-domain F1 − cross-domain F1.",
        ha="center",
        fontsize=8,
        color="#555555",
        wrap=True,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out = FIGURES_DIR / "fig-04-generalization-gap.png"
    render(out)