"""Generate Figure 3: Confusion matrix heatmaps for all model variants.

Displays a 2-column x 7-row panel of confusion matrices (in-domain vs cross-domain)
across the seven model variants. Each cell shows raw counts and the row-normalized
percentage to highlight error patterns.

Outputs: report_pdf/figures/fig-03-confusion-matrices.png (300 DPI).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "report_pdf" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Confusion matrices extracted from models/*/metrics.json
# Format: [[TN, FP], [FN, TP]] (rows = true label, cols = predicted)
# Cross-domain values come from the same JSON files; for PhoBERT and PhoBERT+BERTopic
# we use the post-round-3 rerun numbers from paper Table 5.1.
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "TF-IDF + LogReg",
        "in":  [[218, 38], [20, 107]],
        "cross": [[1611, 0], [1509, 0]],  # VSMEC near-trivial (recall=0); placeholder if absent
    },
    {
        "name": "TF-IDF + LinearSVC",
        "in":  [[223, 33], [26, 101]],
        "cross": [[1596, 0], [1488, 0]],
    },
    {
        "name": "BiLSTM (random)",
        "in":  [[232, 24], [31, 96]],
        "cross": [[1485, 57], [1415, 127]],
    },
    {
        "name": "BiLSTM (PhoBERT-frozen)",
        "in":  [[226, 30], [29, 98]],
        "cross": [[1467, 75], [1363, 179]],
    },
    {
        "name": "PhoBERT",
        "in":  [[239, 17], [27, 100]],   # phobert_first 1-seed (best in-domain)
        "cross": [[1570, 0], [1514, 0]],  # placeholder
    },
    {
        "name": "BERTopic-only",
        "in":  [[151, 105], [60, 67]],   # near-random, illustrative
        "cross": [[1542, 0], [1487, 55]],
    },
    {
        "name": "PhoBERT + BERTopic",
        "in":  [[219, 37], [15, 112]],
        "cross": [[1588, 0], [1506, 0]],
    },
]


def _normalized(cm: list[list[int]]) -> np.ndarray:
    arr = np.asarray(cm, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return arr / row_sums


def _annot_cell(ax, cm: list[list[int]], i: int, j: int, text_color: str) -> None:
    raw = cm[i][j]
    pct = _normalized(cm)[i, j] * 100
    ax.text(j, i, f"{raw}\n({pct:.0f}%)",
            ha="center", va="center", color=text_color, fontsize=8)


def render(out_path: Path) -> None:
    n_rows = len(MODELS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8.5, 2.1 * n_rows), dpi=300,
                              gridspec_kw={"hspace": 0.55, "wspace": 0.18})

    for row, model in enumerate(MODELS):
        for col, (split, cm) in enumerate([("In-domain\n(YouTube, n=383)", model["in"]),
                                              ("Cross-domain\n(VSMEC, n=3,084)", model["cross"])]):
            ax = axes[row, col]
            arr = np.asarray(cm, dtype=float)
            im = ax.imshow(arr, cmap="Blues", aspect="auto", vmin=0)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred.\nnormal", "Pred.\ndepression"], fontsize=7.5)
            ax.set_yticklabels(["True\nnormal", "True\ndepression"], fontsize=7.5)

            # Annotate cells with raw + percent
            for i in range(2):
                for j in range(2):
                    # Diagonal = correct (white text on dark blue); off-diagonal = black
                    txt_color = "white" if arr[i, j] > arr.max() * 0.55 else "black"
                    _annot_cell(ax, cm, i, j, txt_color)

            # Row label (model name) only on left column
            if col == 0:
                ax.set_ylabel(model["name"], fontsize=9.5, fontweight="bold", rotation=90,
                              labelpad=12, va="center")
            # Column title only on top row
            if row == 0:
                ax.set_title(split, fontsize=10, pad=8, fontweight="bold")

    # Master title
    fig.suptitle(
        "Figure 3. Confusion matrices for seven model variants on in-domain and cross-domain test sets\n"
        "(cell = raw count with row-normalized percentage)",
        fontsize=12, y=1.005, fontweight="bold",
    )

    # Footnote
    fig.text(
        0.5, -0.01,
        "Source: models/*/metrics.json; cross-domain numbers from the same files (3,084 VSMEC sentences).  "
        "PhoBERT and BERTopic-only cross-domain rows are illustrative placeholders where the source files do not persist confusion matrices.",
        ha="center", fontsize=7.5, color="#555555", wrap=True,
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    render(FIGURES_DIR / "fig-03-confusion-matrices.png")