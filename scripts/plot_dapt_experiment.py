"""Generate Figure 6: Domain-Adaptive Pretraining (DAPT) counter-experiment scatter.

Plots per-seed F1-macro for the two PhoBERT variants (original vs DAPT) on both
in-domain and cross-domain test sets, with mean lines and seed-level connecting lines.

Outputs: report_pdf/figures/fig-06-dapt-counter-experiment.png (300 DPI).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = PROJECT_ROOT / "results" / "domain_adapted_eval_2026-06-26_181310" / "metrics.json"
FIGURES_DIR = PROJECT_ROOT / "report_pdf" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLOR_ORIGINAL = "#0072B2"   # blue
COLOR_DAPT = "#D55E00"        # vermillion
SEED_COLORS = {"42": "#1f77b4", "123": "#ff7f0e", "2024": "#2ca02c"}


def _load_runs() -> dict:
    with METRICS_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)["runs"]


def _per_seed(runs: list, model_tag: str, test_set: str) -> dict[int, float]:
    out = {}
    for r in runs:
        if r["model_tag"] == model_tag and r["test_set"] == test_set and r["status"] == "ok":
            out[r["seed"]] = r["f1_macro"]
    return out


def _per_seed_f1dep(runs: list, model_tag: str, test_set: str) -> dict[int, float]:
    out = {}
    for r in runs:
        if r["model_tag"] == model_tag and r["test_set"] == test_set and r["status"] == "ok":
            out[r["seed"]] = r["f1_depression"]
    return out


def render(out_path: Path) -> None:
    runs = _load_runs()

    orig_in = _per_seed(runs, "original", "final_test")
    dapt_in = _per_seed(runs, "domain_adapted", "final_test")
    orig_cross = _per_seed(runs, "original", "vsmec")
    dapt_cross = _per_seed(runs, "domain_adapted", "vsmec")
    orig_dep_cross = _per_seed_f1dep(runs, "original", "vsmec")
    dapt_dep_cross = _per_seed_f1dep(runs, "domain_adapted", "vsmec")

    seeds = sorted(orig_in.keys())

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 8.2), dpi=300,
        gridspec_kw={"height_ratios": [1, 1.1], "hspace": 0.32},
    )

    # --- TOP panel: in-domain F1-macro ---
    for seed in seeds:
        x0 = 0
        x1 = 1
        # Connecting line between original and DAPT for this seed
        ax_top.plot([x0, x1], [orig_in[seed], dapt_in[seed]],
                    color=SEED_COLORS[str(seed)], alpha=0.5, linewidth=1.2,
                    linestyle="--", zorder=1)
    ax_top.scatter([0] * len(seeds), [orig_in[s] for s in seeds],
                   color=COLOR_ORIGINAL, s=110, zorder=3, edgecolor="black", linewidth=0.8,
                   label="PhoBERT (original)")
    ax_top.scatter([1] * len(seeds), [dapt_in[s] for s in seeds],
                   color=COLOR_DAPT, s=110, zorder=3, edgecolor="black", linewidth=0.8,
                   marker="s", label="PhoBERT + DAPT")

    # Per-seed annotations
    for seed in seeds:
        ax_top.annotate(f"seed={seed}", xy=(0, orig_in[seed]), xytext=(-0.18, orig_in[seed]),
                        fontsize=7.5, color=SEED_COLORS[str(seed)], ha="right", va="center")
        ax_top.annotate(f"{orig_in[seed]:.4f}", xy=(0, orig_in[seed]),
                        xytext=(0, -12), textcoords="offset points",
                        fontsize=7, color=COLOR_ORIGINAL, ha="center", va="top")
        ax_top.annotate(f"{dapt_in[seed]:.4f}", xy=(1, dapt_in[seed]),
                        xytext=(0, -12), textcoords="offset points",
                        fontsize=7, color=COLOR_DAPT, ha="center", va="top")

    orig_in_mean = np.mean(list(orig_in.values()))
    dapt_in_mean = np.mean(list(dapt_in.values()))
    ax_top.axhline(orig_in_mean, color=COLOR_ORIGINAL, alpha=0.4, linestyle=":", linewidth=1)
    ax_top.axhline(dapt_in_mean, color=COLOR_DAPT, alpha=0.4, linestyle=":", linewidth=1)
    ax_top.text(1.5, orig_in_mean, f"mean={orig_in_mean:.4f}", color=COLOR_ORIGINAL,
                fontsize=8, va="center")
    ax_top.text(1.5, dapt_in_mean, f"mean={dapt_in_mean:.4f}", color=COLOR_DAPT,
                fontsize=8, va="center")

    ax_top.set_xticks([0, 1])
    ax_top.set_xticklabels(["PhoBERT (original)", "PhoBERT + DAPT"], fontsize=10)
    ax_top.set_xlim(-0.5, 2.0)
    ax_top.set_ylabel("F1-macro (in-domain, n=383)", fontsize=10.5)
    ax_top.set_title(
        "In-domain test set (YouTube comments)\n"
        f"Δ in-domain = {dapt_in_mean - orig_in_mean:+.4f} (not significant, p≈0.21)",
        fontsize=11,
    )
    ax_top.grid(axis="y", linestyle=":", alpha=0.5)
    ax_top.set_axisbelow(True)
    ax_top.legend(loc="lower right", fontsize=9)

    # --- BOTTOM panel: cross-domain F1-macro AND F1-depression ---
    width = 0.32
    x_orig = 0
    x_dapt = 1

    for seed in seeds:
        ax_bottom.plot([x_orig, x_dapt], [orig_cross[seed], dapt_cross[seed]],
                       color=SEED_COLORS[str(seed)], alpha=0.5, linewidth=1.2,
                       linestyle="--", zorder=1)

    ax_bottom.scatter([x_orig] * len(seeds), [orig_cross[s] for s in seeds],
                      color=COLOR_ORIGINAL, s=110, zorder=3, edgecolor="black", linewidth=0.8,
                      label="F1-macro (orig)")
    ax_bottom.scatter([x_dapt] * len(seeds), [dapt_cross[s] for s in seeds],
                      color=COLOR_DAPT, s=110, zorder=3, edgecolor="black", linewidth=0.8,
                      marker="s", label="F1-macro (DAPT)")
    # Depression F1 (open circles) — offset slightly to the right of each point
    ax_bottom.scatter([x_orig + 0.12] * len(seeds), [orig_dep_cross[s] for s in seeds],
                      color=COLOR_ORIGINAL, s=80, zorder=3, edgecolor="black",
                      marker="^", facecolor="none", linewidths=1.5,
                      label="F1-depression (orig)")
    ax_bottom.scatter([x_dapt + 0.12] * len(seeds), [dapt_dep_cross[s] for s in seeds],
                      color=COLOR_DAPT, s=80, zorder=3, edgecolor="black",
                      marker="^", facecolor="none", linewidths=1.5,
                      label="F1-depression (DAPT)")

    for seed in seeds:
        ax_bottom.annotate(f"{seed}", xy=(x_orig - 0.18, orig_cross[seed]),
                           fontsize=7.5, color=SEED_COLORS[str(seed)], ha="right", va="center")

    orig_cross_mean = np.mean(list(orig_cross.values()))
    dapt_cross_mean = np.mean(list(dapt_cross.values()))
    ax_bottom.axhline(orig_cross_mean, color=COLOR_ORIGINAL, alpha=0.4, linestyle=":", linewidth=1)
    ax_bottom.axhline(dapt_cross_mean, color=COLOR_DAPT, alpha=0.4, linestyle=":", linewidth=1)
    ax_bottom.text(1.5, orig_cross_mean, f"mean F1-macro={orig_cross_mean:.4f}",
                   color=COLOR_ORIGINAL, fontsize=8, va="center")
    ax_bottom.text(1.5, dapt_cross_mean, f"mean F1-macro={dapt_cross_mean:.4f}",
                   color=COLOR_DAPT, fontsize=8, va="center")

    ax_bottom.set_xticks([0, 1])
    ax_bottom.set_xticklabels(["PhoBERT (original)", "PhoBERT + DAPT"], fontsize=10)
    ax_bottom.set_xlim(-0.5, 2.0)
    ax_bottom.set_ylabel("F1 score (cross-domain VSMEC, n=3,084)", fontsize=10.5)
    ax_bottom.set_title(
        "Cross-domain test set (VSMEC)\n"
        f"Δ cross-domain F1-macro = {dapt_cross_mean - orig_cross_mean:+.4f} (not significant, p≈0.34)",
        fontsize=11,
    )
    ax_bottom.grid(axis="y", linestyle=":", alpha=0.5)
    ax_bottom.set_axisbelow(True)
    ax_bottom.legend(loc="upper right", fontsize=8.5, ncol=2)

    fig.suptitle(
        "Figure 6. DAPT counter-experiment: per-seed F1 scores across three seeds (42, 123, 2024)\n"
        "(filled markers = F1-macro; triangle markers = F1-depression; dashed lines connect same-seed pairs)",
        fontsize=12, y=1.005, fontweight="bold",
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    render(FIGURES_DIR / "fig-06-dapt-counter-experiment.png")