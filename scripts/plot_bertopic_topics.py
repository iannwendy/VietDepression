"""Generate Figure 5: BERTopic topic distribution.

Reads models/bertopic/bertopic_metrics.json and produces:
- a horizontal bar chart of the top 20 non-outlier topics by document count,
  color-coded by depression-relevance.

Outputs: report_pdf/figures/fig-05-bertopic-topics.png (300 DPI).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = PROJECT_ROOT / "models" / "bertopic" / "bertopic_metrics.json"
FIGURES_DIR = PROJECT_ROOT / "report_pdf" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Depression-related topic IDs and their interpretations (from paper_report.html
# §5.3 Table 5.2 + manual inspection of topic top-words).
DEPRESSION_TOPICS = {
    33: "Clinical depression & treatment",
    7: "Sleep disorders & medical consultation",
    19: "Workplace burnout (English-VN code-switch)",
    14: "Sadness expression & emotional release",
    27: "Loneliness & existential distress",
}

# Colorblind-safe palette
COLOR_DEPRESSION = "#D55E00"   # vermillion
COLOR_NORMAL = "#999999"       # grey
COLOR_OUTLIER = "#E69F00"      # orange (kept for visibility)


def _topic_label_words(words: list, n: int = 4) -> str:
    """Concatenate the top-N words of a topic into a short label."""
    if not words:
        return ""
    return ", ".join(w["word"] for w in words[:n])


def render(out_path: Path) -> None:
    with METRICS_FILE.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    topic_distribution = metrics["topic_distribution"]
    total_docs = metrics["corpus"]["processed_rows"]
    n_topics = metrics["model"]["n_topics_discovered"]
    outlier_pct = metrics["model"]["outlier_percentage"]

    # Sort non-outlier topics by document count (desc), keep top 20
    non_outlier = [t for t in topic_distribution if t["topic_id"] != -1]
    non_outlier_sorted = sorted(non_outlier, key=lambda t: t["document_count"], reverse=True)
    top = non_outlier_sorted[:20]

    # Find outlier row for separate annotation
    outlier_row = next((t for t in topic_distribution if t["topic_id"] == -1), None)

    # Build plot data (reverse order so biggest is on top of horizontal bar)
    top_reversed = list(reversed(top))
    topic_ids = [t["topic_id"] for t in top_reversed]
    doc_counts = [t["document_count"] for t in top_reversed]
    top_words = [_topic_label_words(t["top_words"], n=4) for t in top_reversed]
    colors = [
        COLOR_DEPRESSION if tid in DEPRESSION_TOPICS else COLOR_NORMAL
        for tid in topic_ids
    ]

    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=300)

    y_pos = np.arange(len(topic_ids))
    bars = ax.barh(
        y_pos,
        doc_counts,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        height=0.75,
    )

    # y-axis labels: topic ID + short top-words
    y_labels = [f"#{tid:>3}  {tw}" for tid, tw in zip(topic_ids, top_words)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8.5)

    # Bar labels (document counts)
    for bar, count in zip(bars, doc_counts):
        pct = count / total_docs * 100
        ax.text(
            bar.get_width() + max(doc_counts) * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=8,
            color="black",
        )

    ax.set_xlabel("Document count (and share of full 316K corpus)", fontsize=10.5)
    ax.set_xlim(0, max(doc_counts) * 1.18)
    ax.set_title(
        "Figure 5. Top 20 BERTopic topics in the 316K-document Vietnamese corpus\n"
        "(456 non-outlier topics discovered; 48.30% of documents classified as outliers)",
        fontsize=11.5,
        pad=12,
    )

    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLOR_DEPRESSION, edgecolor="black", label="Depression-related topic"),
        Patch(facecolor=COLOR_NORMAL, edgecolor="black", label="Other non-outlier topic"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9.5, framealpha=0.95)

    # Footnote
    if outlier_row:
        outlier_n = outlier_row["document_count"]
        fig.text(
            0.5,
            -0.02,
            f"Embedding model: paraphrase-multilingual-MiniLM-L12-v2; "
            f"UMAP (n_neighbors=15, min_dist=0.0, n_components=5); "
            f"HDBSCAN (min_cluster_size=50, min_samples=10). "
            f"Outliers (topic -1): {outlier_n:,} documents ({outlier_pct:.2f}%). "
            f"Source: models/bertopic/bertopic_metrics.json.",
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
    render(FIGURES_DIR / "fig-05-bertopic-topics.png")