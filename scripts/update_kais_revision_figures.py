#!/usr/bin/env python3
"""
Render all six publication-grade 600-DPI figures (fig1_new.png .. fig6_new.png)
for the Springer KAIS manuscript (VietDepression) using exact audited metrics
from top_venue_baselines_results.json, visobert_and_ablations_results.json,
evaluation_results.json, bertopic_evaluation_results.json, and data/labeled/*.csv.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops
from sklearn.metrics import precision_recall_curve

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

MODEL_DIR = BASE_DIR / "models" / "round6_v2_retrained"
AUDIT_DIR = BASE_DIR / "results" / "scientific_audit"
HTML_DIR = AUDIT_DIR / "mcp_html_rev"
FIG_AUDIT_DIR = AUDIT_DIR / "figures"
OVERLEAF_DIR = BASE_DIR / "VietDepression_Springer_KAIS_Overleaf"

HTML_DIR.mkdir(parents=True, exist_ok=True)
FIG_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

COMMON_CSS = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #ffffff;
    font-family: "Times New Roman", "STIX Two Text", "Georgia", serif;
    color: #0f172a;
    -webkit-font-smoothing: antialiased;
    padding: 0;
    margin: 0;
  }
  .sans {
    font-family: "Helvetica Neue", "Inter", "Arial", sans-serif;
  }
  .mono {
    font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
  }
  #figure-root {
    background: #ffffff;
    display: inline-block;
    padding: 16px 20px;
  }
"""


def write_and_render(name: str, width: int, height: int, body_html: str, out_filename: str) -> None:
    html_path = HTML_DIR / f"{name}.html"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
{COMMON_CSS}
#figure-root {{
  width: {width}px;
}}
</style>
</head>
<body>
<div id="figure-root">
{body_html}
</div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    raw_png = HTML_DIR / f"{name}_raw.png"
    subprocess.run(
        [
            CHROME_BIN,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=2.5",
            f"--window-size={width + 160},{height + 400}",
            f"--screenshot={raw_png}",
            f"file://{html_path}",
        ],
        check=True,
        capture_output=True,
    )
    img = Image.open(raw_png).convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        pad = 24
        x0 = max(0, bbox[0] - pad)
        y0 = max(0, bbox[1] - pad)
        x1 = min(img.width, bbox[2] + pad)
        y1 = min(img.height, bbox[3] + pad)
        img = img.crop((x0, y0, x1, y1))

    out_overleaf = OVERLEAF_DIR / out_filename
    out_audit = FIG_AUDIT_DIR / out_filename
    img.save(out_overleaf, dpi=(600, 600))
    shutil.copy2(out_overleaf, out_audit)
    print(f"Saved {out_overleaf.name}: {img.width}x{img.height}px @ 600 DPI (synced to {out_audit})")


# ==============================================================================
# FIGURE 1: Six-Stage VietDepression Benchmark Construction & Inductive Protocol
# ==============================================================================
def build_fig1() -> None:
    body = """
    <div style="width: 100%; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 18px 20px; background: #ffffff;">
      <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 2px solid #0f172a; padding-bottom: 8px; margin-bottom: 14px;">
        <div style="font-size: 16px; font-weight: 700; letter-spacing: 0.2px; color: #0f172a;">
          VietDepression-YouTube Benchmark Construction &amp; Strictly Inductive Evaluation Protocol
        </div>
        <div class="sans" style="font-size: 11.5px; font-weight: 600; color: #334155; background: #f1f5f9; padding: 3px 10px; border-radius: 4px; border: 1px solid #cbd5e1;">
          Zero Transductive Leakage &bull; 100% Human-Verified Gold Splits (<i>n</i> = 9,134)
        </div>
      </div>

      <div style="display: flex; align-items: stretch; justify-content: space-between; gap: 0px; margin-bottom: 12px;">
        <div style="flex: 1; border: 1.5px solid #1e40af; border-radius: 6px; background: #f8fafc; overflow: hidden;">
          <div class="sans" style="background: #1e3a8a; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
            <span>STAGE 1: YouTube API Retrieval</span>
            <span style="background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 3px; font-size: 10px;">Phase I: Corpus</span>
          </div>
          <div style="padding: 10px 12px; font-size: 12.1px; line-height: 1.48; color: #1e293b;">
            <div style="margin-bottom: 4px;">&bull; <b>Query Design:</b> 264 Vietnamese distress, symptom, sleep &amp; help-seeking queries</div>
            <div style="margin-bottom: 4px;">&bull; <b>Video Coverage:</b> 4,413 crawled videos &rarr; <b>2,906</b> comment-yielding videos</div>
            <div style="margin-bottom: 4px;">&bull; <b>Raw Inventory:</b> <b>125,329</b> public top-level comments (2,949 channels)</div>
            <div class="mono" style="font-size: 10.4px; background: #e2e8f0; padding: 3px 6px; border-radius: 3px; color: #0f172a; margin-top: 5px;">
              Mean length: 23.8 &plusmn; 43.4w (median = 12w)
            </div>
          </div>
        </div>

        <div style="width: 54px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <div class="sans" style="font-size: 9.5px; font-weight: 700; color: #1e3a8a; margin-bottom: 2px;">125,329</div>
          <svg width="46" height="20" viewBox="0 0 46 20">
            <line x1="2" y1="10" x2="36" y2="10" stroke="#1e3a8a" stroke-width="2.2"/>
            <polygon points="34,4 44,10 34,16" fill="#1e3a8a"/>
          </svg>
          <div class="sans" style="font-size: 8.5px; color: #64748b; margin-top: 1px;">Raw UTF-8</div>
        </div>

        <div style="flex: 1; border: 1.5px solid #0284c7; border-radius: 6px; background: #f8fafc; overflow: hidden;">
          <div class="sans" style="background: #0369a1; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
            <span>STAGE 2: NFC &amp; Quality Filtering</span>
            <span style="background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 3px; font-size: 10px;">Normalization</span>
          </div>
          <div style="padding: 10px 12px; font-size: 12.1px; line-height: 1.48; color: #1e293b;">
            <div style="margin-bottom: 4px;">&bull; <b>Unicode NFC:</b> Canonicalize Vietnamese tone marks; preserve case &amp; emoji</div>
            <div style="margin-bottom: 4px;">&bull; <b>Noise Removal:</b> Strip URLs, spam, &lt;5 chars &amp; exact duplicates</div>
            <div style="margin-bottom: 4px;">&bull; <b>Discarded Noise:</b> <b>5,680</b> records removed (<b>4.53%</b> of raw corpus)</div>
            <div class="mono" style="font-size: 10.4px; background: #e0f2fe; padding: 3px 6px; border-radius: 3px; color: #0369a1; font-weight: 600; margin-top: 5px;">
              Filtered Pool: 119,649 (95.47% retained)
            </div>
          </div>
        </div>

        <div style="width: 54px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <div class="sans" style="font-size: 9.2px; font-weight: 700; color: #b45309; margin-bottom: 2px;">119,649</div>
          <svg width="46" height="20" viewBox="0 0 46 20">
            <line x1="2" y1="10" x2="36" y2="10" stroke="#b45309" stroke-width="2.2"/>
            <polygon points="34,4 44,10 34,16" fill="#b45309"/>
          </svg>
          <div class="sans" style="font-size: 8.2px; color: #64748b; margin-top: 1px; text-align:center;">(125,328 scored)</div>
        </div>

        <div style="flex: 1; border: 1.5px solid #d97706; border-radius: 6px; background: #fffbeb; overflow: hidden;">
          <div class="sans" style="background: #b45309; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
            <span>STAGE 3: Weighted Lexicon Triage</span>
            <span style="background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 3px; font-size: 10px;">N = 125,328</span>
          </div>
          <div style="padding: 10px 12px; font-size: 12.1px; line-height: 1.48; color: #1e293b;">
            <div style="margin-bottom: 4px;">&bull; <b>335-Entry Lexicon:</b> Strong (<i>w</i> = +5), Medium (+3), Normal/Idiom (&minus;2)</div>
            <div style="margin-bottom: 4px;">&bull; <b>Scored Raw Pool (125,328):</b> Pos (<i>s</i> &ge; 5): <b>3,223</b> (2.57%) | Neg (<i>s</i> &le; &minus;2): <b>23,695</b> (18.91%)</div>
            <div style="margin-bottom: 4px;">&bull; <b>Abstention Zone (&minus;1 &le; <i>s</i> &le; 4):</b> <b>98,410</b> uncertain records (<b>78.52%</b>)</div>
            <div class="mono" style="font-size: 10.4px; background: #fef3c7; padding: 3px 6px; border-radius: 3px; color: #92400e; font-weight: 600; margin-top: 5px;">
              22-Phrase Flag: 820 (0.65%) | Composite: 120,334
            </div>
          </div>
        </div>
      </div>

      <div style="display:flex; justify-content:flex-end; padding-right: 145px; margin: -4px 0 6px 0;">
        <div style="display:flex; align-items:center; gap: 8px;">
          <span class="sans" style="font-size: 10px; font-weight: 600; color: #0f766e; background: #f0fdfa; padding: 2px 8px; border: 1px solid #99f6e4; border-radius: 4px;">
            Candidate Triage &amp; Blind Label Studio Queue (Machine Scores Hidden)
          </span>
          <svg width="24" height="26" viewBox="0 0 24 26">
            <line x1="12" y1="0" x2="12" y2="18" stroke="#0f766e" stroke-width="2.4"/>
            <polygon points="6,16 12,25 18,16" fill="#0f766e"/>
          </svg>
        </div>
      </div>

      <div style="display: flex; align-items: stretch; justify-content: space-between; gap: 0px; margin-bottom: 12px;">
        <div style="flex: 1; border: 1.5px solid #334155; border-radius: 6px; background: #f8fafc; overflow: hidden;">
          <div class="sans" style="background: #1e293b; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
            <span>STAGE 6: Zero-Leakage Splits</span>
            <span style="background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 3px; font-size: 10px;">70 / 15 / 15</span>
          </div>
          <div style="padding: 10px 12px; font-size: 12.1px; line-height: 1.48; color: #1e293b;">
            <div style="margin-bottom: 4px;">&bull; <b>Train (70%):</b> <b>6,392</b> (1,262 Pos [19.74%] / 5,130 Neg [80.26%])</div>
            <div style="margin-bottom: 4px;">&bull; <b>Validation (15%):</b> <b>1,371</b> (271 Pos [19.77%] / 1,100 Neg [80.23%])</div>
            <div style="margin-bottom: 4px;">&bull; <b>In-Domain Test (15%):</b> <b>1,371</b> (271 Pos [19.77%] / 1,100 Neg [80.23%])</div>
            <div class="mono" style="font-size: 10.4px; background: #e2e8f0; padding: 3px 6px; border-radius: 3px; color: #0f172a; font-weight: 600; margin-top: 5px;">
              0% Text Overlap | Mean: 63.1 &plusmn; 73.1w (med = 38w)
            </div>
          </div>
        </div>

        <div style="width: 54px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <div class="sans" style="font-size: 9.5px; font-weight: 700; color: #1e293b; margin-bottom: 2px;">9,134 Gold</div>
          <svg width="46" height="20" viewBox="0 0 46 20">
            <line x1="10" y1="10" x2="44" y2="10" stroke="#1e293b" stroke-width="2.2"/>
            <polygon points="12,4 2,10 12,16" fill="#1e293b"/>
          </svg>
          <div class="sans" style="font-size: 8.5px; color: #64748b; margin-top: 1px;">Stratified</div>
        </div>

        <div style="flex: 1; border: 1.5px solid #0d9488; border-radius: 6px; background: #f0fdfa; overflow: hidden;">
          <div class="sans" style="background: #0f766e; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
            <span>STAGE 5: 6-Round Active Learning</span>
            <span style="background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 3px; font-size: 10px;">Phase III: AL</span>
          </div>
          <div style="padding: 10px 12px; font-size: 12.1px; line-height: 1.48; color: #1e293b;">
            <div style="margin-bottom: 4px;">&bull; <b>Uncertainty Policy:</b> <i>u</i>(<i>x</i>) = 1 &minus; max{<i>p</i>(<i>x</i>), 1&minus;<i>p</i>(<i>x</i>)} on <span class="mono" style="font-size:11px;">unlabeled_pool</span></div>
            <div style="margin-bottom: 4px;">&bull; <b>Per-Round Gold Ledger:</b> R1&ndash;R4: <b>2,072</b> &rarr; R5: <b>+1,360</b> &rarr; R6: <b>+5,702</b></div>
            <div style="margin-bottom: 4px;">&bull; <b>Matched-Budget Control:</b> Benchmarked vs. Random, BADGE &amp; CoMAL</div>
            <div class="mono" style="font-size: 10.4px; background: #ccfbf1; padding: 3px 6px; border-radius: 3px; color: #115e59; font-weight: 600; margin-top: 5px;">
              1,804 Positive (19.75%) | 7,330 Negative (80.25%)
            </div>
          </div>
        </div>

        <div style="width: 54px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <div class="sans" style="font-size: 9.5px; font-weight: 700; color: #0f766e; margin-bottom: 2px;">Adjudicated</div>
          <svg width="46" height="20" viewBox="0 0 46 20">
            <line x1="10" y1="10" x2="44" y2="10" stroke="#0f766e" stroke-width="2.2"/>
            <polygon points="12,4 2,10 12,16" fill="#0f766e"/>
          </svg>
          <div class="sans" style="font-size: 8.5px; color: #64748b; margin-top: 1px;">Iterative</div>
        </div>

        <div style="flex: 1; border: 1.5px solid #0f766e; border-radius: 6px; background: #f0fdfa; overflow: hidden;">
          <div class="sans" style="background: #115e59; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
            <span>STAGE 4: Blind Human Annotation</span>
            <span style="background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 3px; font-size: 10px;">Phase II: Gold</span>
          </div>
          <div style="padding: 10px 12px; font-size: 12.1px; line-height: 1.48; color: #1e293b;">
            <div style="margin-bottom: 4px;">&bull; <b>Annotator Pool:</b> 2 trained native VN NLP annotators + senior adjudicator</div>
            <div style="margin-bottom: 4px;">&bull; <b>4-Rule Rubric:</b> Speaker perspective, polysemy, persistence &amp; crisis cues</div>
            <div style="margin-bottom: 4px;">&bull; <b>Double-Annotated QC (<i>n</i> = 300):</b> Raw agreement = <b>93.33%</b> (280/300)</div>
            <div class="mono" style="font-size: 10.4px; background: #ccfbf1; padding: 3px 6px; border-radius: 3px; color: #115e59; font-weight: 600; margin-top: 5px;">
              Cohen's &kappa; = 0.8667 | Matrix: [[139, 8], [12, 141]]
            </div>
          </div>
        </div>
      </div>

      <div style="display:flex; justify-content:flex-start; padding-left: 145px; margin: -4px 0 6px 0;">
        <div style="display:flex; align-items:center; gap: 8px;">
          <svg width="24" height="26" viewBox="0 0 24 26">
            <line x1="12" y1="0" x2="12" y2="18" stroke="#991b1b" stroke-width="2.4"/>
            <polygon points="6,16 12,25 18,16" fill="#991b1b"/>
          </svg>
          <span class="sans" style="font-size: 10px; font-weight: 600; color: #991b1b; background: #fef2f2; padding: 2px 8px; border: 1px solid #fecaca; border-radius: 4px;">
            Strictly Inductive Fitting on final_train.csv (n = 6,392) Only
          </span>
        </div>
      </div>

      <div style="border: 1.5px solid #991b1b; border-radius: 6px; background: #fff1f2; overflow: hidden;">
        <div class="sans" style="background: #991b1b; color: #ffffff; padding: 6px 14px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
          <span>PHASE IV: Strictly Inductive Multi-Family Benchmark, Active-Learning Control &amp; Cross-Corpus Calibration</span>
          <span style="background: rgba(255,255,255,0.2); padding: 1px 8px; border-radius: 3px; font-size: 10px;">14 Benchmark Configs (3 Seeds + 2,000-Bootstrap CIs)</span>
        </div>
        <div style="display: flex; gap: 14px; padding: 10px 14px; background: #fffafa; font-size: 11.8px; line-height: 1.45; color: #1e293b;">
          <div style="flex: 1.18; border-right: 1px dashed #fca5a5; padding-right: 12px;">
            <div class="sans" style="font-weight: 700; color: #991b1b; font-size: 11.3px; margin-bottom: 3px;">1. Evaluated Models (Fitted Strictly on Train <i>n</i> = 6,392)</div>
            <div>&bull; <b>Lexical &amp; Recurrent:</b> All-Neg, 335-entry Lexicon, TF-IDF + <b>LinearSVC</b> / <b>LogReg</b>, Class-Weighted <b>BiLSTM</b></div>
            <div>&bull; <b>Multilingual, Formal &amp; Social Encoders:</b> <b>XLM-R-base</b>, <b>PhoBERT-base</b>, 74-d <b>BERTopic</b>, 806-d <b>Hybrid</b>, <b>ViSoBERT</b></div>
            <div>&bull; <b>Top-Venue Objectives &amp; LLM:</b> <b>LA-Focal</b> (TKDE '24), <b>SupCon</b> (KDD '26), <b>NCNL</b> (TKDE '23), 4-Shot <b>Qwen2.5-1.5B</b></div>
          </div>
          <div style="flex: 0.82;">
            <div class="sans" style="font-weight: 700; color: #991b1b; font-size: 11.3px; margin-bottom: 3px;">2. Statistical Rigor &amp; Transfer Calibration</div>
            <div>&bull; <b>In-Domain (<i>n</i> = 1,371):</b> 2,000-Bootstrap 95% CIs, Paired Bootstrap &Delta;F1 &amp; Exact McNemar tests; ROC-AUC &amp; PR-AUC</div>
            <div>&bull; <b>Active Learning:</b> Equal-budget Random vs. CoMAL, BADGE &amp; Least-Confidence (400 &rarr; 6,392)</div>
            <div>&bull; <b>UIT-VSMEC (<i>n</i> = 3,084 / 2,691):</b> Static Bayes (&tau; = 0.1974) &amp; Source-Rank Quantile (<i>Q</i><sub>80.3%</sub>) across 5 encoders</div>
          </div>
        </div>
      </div>
    </div>
    """
    write_and_render("fig1_rev", 1220, 560, body, "fig1_new.png")


# ==============================================================================
# FIGURE 2: Weak-Score Distribution (Boundary-Aligned Bins) & Length Distribution
# ==============================================================================
def build_fig2() -> None:
    from scripts.vietdepression_lexicon import label_comment_detailed

    dfs = [pd.read_csv(BASE_DIR / "data" / "labeled" / sp) for sp in ["final_train.csv", "final_val.csv", "final_test.csv"]]
    full_df = pd.concat(dfs, ignore_index=True)

    scores = []
    word_lens = []
    for txt in full_df["comment_text"].fillna("").astype(str):
        det = label_comment_detailed(txt)
        scores.append(det.depression_score)
        word_lens.append(len(txt.split()))
    full_df["weak_score"] = np.clip(scores, -7, 18)
    full_df["word_len"] = np.array(word_lens)

    pos_scores = full_df[full_df["label"] == 1]["weak_score"].values
    neg_scores = full_df[full_df["label"] == 0]["weak_score"].values

    pos_lens = full_df[full_df["label"] == 1]["word_len"].values
    neg_lens = full_df[full_df["label"] == 0]["word_len"].values

    # Panel (a): Boundary-aligned 2-point bins [-7,-5), [-5,-3), [-3,-1) | [-1,1), [1,3), [3,5) | [5,7)...[17,19)
    # Zero integer scores cross s <= -2 (x < -1.0) or s >= 5 (x >= 5.0)!
    bins_a = np.arange(-7, 20, 2)
    neg_hist_a, _ = np.histogram(neg_scores, bins=bins_a, density=False)
    pos_hist_a, _ = np.histogram(pos_scores, bins=bins_a, density=False)
    neg_frac_a = neg_hist_a / len(neg_scores)
    pos_frac_a = pos_hist_a / len(pos_scores)

    W_a, H_a = 565, 395
    pad_l, pad_r, pad_t, pad_b = 56, 18, 34, 52
    pw_a = W_a - pad_l - pad_r
    ph_a = H_a - pad_t - pad_b
    xmin_a, xmax_a = -7.3, 19.3
    max_y_a = 0.55

    def sx_a(val: float) -> float:
        return pad_l + (val - xmin_a) / (xmax_a - xmin_a) * pw_a

    def sy_a(val: float) -> float:
        return pad_t + ph_a - (val / max_y_a) * ph_a

    svg_a = []
    x_neg_cut = sx_a(-1.0)
    x_pos_cut = sx_a(5.0)
    svg_a.append(
        f'<rect x="{sx_a(xmin_a):.1f}" y="{pad_t}" width="{x_neg_cut - sx_a(xmin_a):.1f}" height="{ph_a}" fill="#eff6ff" opacity="0.78"/>'
        f'<rect x="{x_neg_cut:.1f}" y="{pad_t}" width="{x_pos_cut - x_neg_cut:.1f}" height="{ph_a}" fill="#fef9c3" opacity="0.65"/>'
        f'<rect x="{x_pos_cut:.1f}" y="{pad_t}" width="{sx_a(xmax_a) - x_pos_cut:.1f}" height="{ph_a}" fill="#fef2f2" opacity="0.78"/>'
    )

    svg_a.append(
        f'<text x="{(sx_a(xmin_a)+x_neg_cut)/2:.1f}" y="{pad_t+14}" text-anchor="middle" font-family="Helvetica Neue, sans-serif" font-size="8.6" font-weight="700" fill="#1e40af">s &le; &minus;2 (Raw: 18.9%)</text>'
        f'<text x="{(x_neg_cut+x_pos_cut)/2:.1f}" y="{pad_t+14}" text-anchor="middle" font-family="Helvetica Neue, sans-serif" font-size="8.6" font-weight="700" fill="#92400e">Abstain [&minus;1..4] (Raw: 78.5%)</text>'
        f'<text x="{(x_pos_cut+sx_a(xmax_a))/2:.1f}" y="{pad_t+14}" text-anchor="middle" font-family="Helvetica Neue, sans-serif" font-size="8.9" font-weight="700" fill="#991b1b">Positive Zone s(x) &ge; 5 (Raw: 2.57%)</text>'
    )

    for yv in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        yy = sy_a(yv)
        svg_a.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W_a-pad_r}" y2="{yy:.1f}" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="3,3"/>'
            f'<text x="{pad_l-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10.8" fill="#334155">{yv:.1f}</text>'
        )

    bar_w = (sx_a(2) - sx_a(0)) * 0.37
    for i in range(len(bins_a) - 1):
        center = 0.5 * (bins_a[i] + bins_a[i + 1])
        cx = sx_a(center)
        nf = neg_frac_a[i]
        ny = sy_a(nf)
        nh = sy_a(0.0) - ny
        if nh > 0.3:
            svg_a.append(
                f'<rect x="{cx - bar_w - 0.7:.1f}" y="{ny:.1f}" width="{bar_w:.1f}" height="{nh:.1f}" fill="#2563eb" fill-opacity="0.86" stroke="#1e3a8a" stroke-width="0.8"/>'
            )
        pf = pos_frac_a[i]
        py = sy_a(pf)
        ph = sy_a(0.0) - py
        if ph > 0.3:
            svg_a.append(
                f'<rect x="{cx + 0.7:.1f}" y="{py:.1f}" width="{bar_w:.1f}" height="{ph:.1f}" fill="#dc2626" fill-opacity="0.86" stroke="#7f1d1d" stroke-width="0.8"/>'
            )

    svg_a.append(
        f'<line x1="{x_neg_cut:.1f}" y1="{pad_t}" x2="{x_neg_cut:.1f}" y2="{pad_t+ph_a}" stroke="#1d4ed8" stroke-width="2" stroke-dasharray="5,3"/>'
        f'<line x1="{x_pos_cut:.1f}" y1="{pad_t}" x2="{x_pos_cut:.1f}" y2="{pad_t+ph_a}" stroke="#b91c1c" stroke-width="2" stroke-dasharray="5,3"/>'
    )

    for xv in range(-7, 20, 2):
        xx = sx_a(xv)
        svg_a.append(
            f'<line x1="{xx:.1f}" y1="{pad_t+ph_a}" x2="{xx:.1f}" y2="{pad_t+ph_a+5}" stroke="#0f172a" stroke-width="1"/>'
            f'<text x="{xx:.1f}" y="{pad_t+ph_a+17}" text-anchor="middle" font-size="10.4" fill="#0f172a">{xv}</text>'
        )

    svg_a.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph_a}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<line x1="{pad_l}" y1="{pad_t+ph_a}" x2="{W_a-pad_r}" y2="{pad_t+ph_a}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<text x="{pad_l + pw_a/2:.1f}" y="{H_a-10}" text-anchor="middle" font-size="12.0" font-weight="700" fill="#0f172a">Additive Lexical Weak Score s(x) [Boundary-Aligned 2-Pt Bins, Clipped to [&minus;7, 18]]</text>'
        f'<text x="15" y="{pad_t + ph_a/2:.1f}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#0f172a" transform="rotate(-90 15 {pad_t + ph_a/2:.1f})">Within-Class Empirical Probability Mass</text>'
    )

    svg_a.append(
        f'<rect x="{W_a-pad_r-222}" y="{pad_t+24}" width="214" height="74" rx="4" fill="#ffffff" fill-opacity="0.96" stroke="#94a3b8" stroke-width="1"/>'
        f'<rect x="{W_a-pad_r-212}" y="{pad_t+34}" width="14" height="10" fill="#2563eb" stroke="#1e3a8a" stroke-width="0.8"/>'
        f'<text x="{W_a-pad_r-192}" y="{pad_t+43}" font-size="10.8" fill="#0f172a">Gold Negative (n = 7,330; 75.0% abstain)</text>'
        f'<rect x="{W_a-pad_r-212}" y="{pad_t+52}" width="14" height="10" fill="#dc2626" stroke="#7f1d1d" stroke-width="0.8"/>'
        f'<text x="{W_a-pad_r-192}" y="{pad_t+61}" font-size="10.8" fill="#0f172a">Gold Positive (n = 1,804; 50.7% abstain)</text>'
        f'<text x="{W_a-pad_r-212}" y="{pad_t+82}" font-family="Helvetica Neue, sans-serif" font-size="9.6" font-weight="600" fill="#475569">Test Weak F1 = 0.6448 (70.39% abstain)</text>'
    )

    # Panel (b): Comment Word-Length Distribution & Marginal IQR Boxplots
    W_b, H_b = 565, 395
    bins_b = np.linspace(0, 220, 23)
    neg_hist_b, _ = np.histogram(np.clip(neg_lens, 0, 220), bins=bins_b, density=True)
    pos_hist_b, _ = np.histogram(np.clip(pos_lens, 0, 220), bins=bins_b, density=True)
    max_y_b = 0.022

    box_h = 42
    plot_top_b = pad_t + box_h
    ph_b = H_b - plot_top_b - pad_b
    pw_b = W_b - pad_l - pad_r

    def sx_b(val: float) -> float:
        return pad_l + (val / 220.0) * pw_b

    def sy_b(val: float) -> float:
        return plot_top_b + ph_b - (val / max_y_b) * ph_b

    svg_b = []
    svg_b.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw_b}" height="{box_h-4}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.8"/>'
    )
    neg_q25, neg_med, neg_q75 = np.percentile(neg_lens, [25, 50, 75])
    pos_q25, pos_med, pos_q75 = np.percentile(pos_lens, [25, 50, 75])
    neg_mean = float(np.mean(neg_lens))
    pos_mean = float(np.mean(pos_lens))

    by_neg = pad_t + 11
    svg_b.append(
        f'<line x1="{sx_b(5):.1f}" y1="{by_neg}" x2="{sx_b(105):.1f}" y2="{by_neg}" stroke="#1e40af" stroke-width="1.3"/>'
        f'<line x1="{sx_b(5):.1f}" y1="{by_neg-3.5}" x2="{sx_b(5):.1f}" y2="{by_neg+3.5}" stroke="#1e40af" stroke-width="1.3"/>'
        f'<line x1="{sx_b(105):.1f}" y1="{by_neg-3.5}" x2="{sx_b(105):.1f}" y2="{by_neg+3.5}" stroke="#1e40af" stroke-width="1.3"/>'
        f'<rect x="{sx_b(neg_q25):.1f}" y="{by_neg-5}" width="{sx_b(neg_q75)-sx_b(neg_q25):.1f}" height="10" fill="#93c5fd" stroke="#1e3a8a" stroke-width="1.2"/>'
        f'<line x1="{sx_b(neg_med):.1f}" y1="{by_neg-5}" x2="{sx_b(neg_med):.1f}" y2="{by_neg+5}" stroke="#1e3a8a" stroke-width="2.2"/>'
        f'<circle cx="{sx_b(neg_mean):.1f}" cy="{by_neg}" r="3.2" fill="#1e3a8a"/>'
    )
    by_pos = pad_t + 27
    svg_b.append(
        f'<line x1="{sx_b(6):.1f}" y1="{by_pos}" x2="{sx_b(115):.1f}" y2="{by_pos}" stroke="#991b1b" stroke-width="1.3"/>'
        f'<line x1="{sx_b(6):.1f}" y1="{by_pos-3.5}" x2="{sx_b(6):.1f}" y2="{by_pos+3.5}" stroke="#991b1b" stroke-width="1.3"/>'
        f'<line x1="{sx_b(115):.1f}" y1="{by_pos-3.5}" x2="{sx_b(115):.1f}" y2="{by_pos+3.5}" stroke="#991b1b" stroke-width="1.3"/>'
        f'<rect x="{sx_b(pos_q25):.1f}" y="{by_pos-5}" width="{sx_b(pos_q75)-sx_b(pos_q25):.1f}" height="10" fill="#fca5a5" stroke="#7f1d1d" stroke-width="1.2"/>'
        f'<line x1="{sx_b(pos_med):.1f}" y1="{by_pos-5}" x2="{sx_b(pos_med):.1f}" y2="{by_pos+5}" stroke="#7f1d1d" stroke-width="2.2"/>'
        f'<circle cx="{sx_b(pos_mean):.1f}" cy="{by_pos}" r="3.2" fill="#7f1d1d"/>'
    )
    svg_b.append(
        f'<text x="{W_b-pad_r-6}" y="{pad_t+14}" text-anchor="end" font-family="Helvetica Neue, sans-serif" font-size="9.2" font-weight="600" fill="#1e3a8a">Normal IQR [{neg_q25:.0f}, {neg_q75:.0f}]w (med={neg_med:.0f}, &mu;={neg_mean:.1f})</text>'
        f'<text x="{W_b-pad_r-6}" y="{pad_t+30}" text-anchor="end" font-family="Helvetica Neue, sans-serif" font-size="9.2" font-weight="600" fill="#991b1b">Depression IQR [{pos_q25:.0f}, {pos_q75:.0f}]w (med={pos_med:.0f}, &mu;={pos_mean:.1f})</text>'
    )

    for yv in [0.000, 0.005, 0.010, 0.015, 0.020]:
        yy = sy_b(yv)
        svg_b.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W_b-pad_r}" y2="{yy:.1f}" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="3,3"/>'
            f'<text x="{pad_l-6}" y="{yy+4:.1f}" text-anchor="end" font-size="10.5" fill="#334155">{yv:.3f}</text>'
        )

    def build_step_path(hist_vals: np.ndarray, bins_arr: np.ndarray) -> str:
        pts = [f"{sx_b(bins_arr[0]):.1f},{sy_b(0.0):.1f}"]
        for i, hval in enumerate(hist_vals):
            x0 = sx_b(bins_arr[i])
            x1 = sx_b(bins_arr[i + 1])
            yy = sy_b(hval)
            pts.append(f"{x0:.1f},{yy:.1f}")
            pts.append(f"{x1:.1f},{yy:.1f}")
        pts.append(f"{sx_b(bins_arr[-1]):.1f},{sy_b(0.0):.1f}")
        return " ".join(pts)

    svg_b.append(
        f'<polygon points="{build_step_path(neg_hist_b, bins_b)}" fill="#3b82f6" fill-opacity="0.30" stroke="#1d4ed8" stroke-width="1.8"/>'
        f'<polygon points="{build_step_path(pos_hist_b, bins_b)}" fill="#ef4444" fill-opacity="0.30" stroke="#b91c1c" stroke-width="1.8"/>'
    )

    # Exact audited Raw Inventory Mean = 23.81w (SD = 43.39w, median = 12w)
    x_raw_mean = sx_b(23.81)
    x_neg_mean = sx_b(neg_mean)
    x_pos_mean = sx_b(pos_mean)
    svg_b.append(
        f'<line x1="{x_raw_mean:.1f}" y1="{plot_top_b}" x2="{x_raw_mean:.1f}" y2="{plot_top_b+ph_b}" stroke="#475569" stroke-width="1.6" stroke-dasharray="2,2"/>'
        f'<line x1="{x_neg_mean:.1f}" y1="{plot_top_b}" x2="{x_neg_mean:.1f}" y2="{plot_top_b+ph_b}" stroke="#1e3a8a" stroke-width="2" stroke-dasharray="5,3"/>'
        f'<line x1="{x_pos_mean:.1f}" y1="{plot_top_b}" x2="{x_pos_mean:.1f}" y2="{plot_top_b+ph_b}" stroke="#991b1b" stroke-width="2" stroke-dasharray="5,3"/>'
    )

    svg_b.append(
        f'<rect x="{W_b-pad_r-254}" y="{plot_top_b+12}" width="246" height="94" rx="4" fill="#ffffff" fill-opacity="0.96" stroke="#94a3b8" stroke-width="1"/>'
        f'<rect x="{W_b-pad_r-244}" y="{plot_top_b+22}" width="14" height="10" fill="#3b82f6" fill-opacity="0.45" stroke="#1d4ed8" stroke-width="1.5"/>'
        f'<text x="{W_b-pad_r-224}" y="{plot_top_b+31}" font-size="10.6" fill="#0f172a">Gold Negative (&mu; = 60.0 &plusmn; 69.3w, med = 37w)</text>'
        f'<rect x="{W_b-pad_r-244}" y="{plot_top_b+42}" width="14" height="10" fill="#ef4444" fill-opacity="0.45" stroke="#b91c1c" stroke-width="1.5"/>'
        f'<text x="{W_b-pad_r-224}" y="{plot_top_b+51}" font-size="10.6" fill="#0f172a">Gold Positive (&mu; = 75.4 &plusmn; 85.8w, med = 44w)</text>'
        f'<line x1="{W_b-pad_r-244}" y1="{plot_top_b+66}" x2="{W_b-pad_r-230}" y2="{plot_top_b+66}" stroke="#475569" stroke-width="1.8" stroke-dasharray="2,2"/>'
        f'<text x="{W_b-pad_r-224}" y="{plot_top_b+70}" font-size="10.4" fill="#334155">Raw Corpus (125k: &mu; = 23.8 &plusmn; 43.4w, med = 12w)</text>'
        f'<text x="{W_b-pad_r-244}" y="{plot_top_b+91}" font-family="Helvetica Neue, sans-serif" font-size="9.5" font-weight="600" fill="#0f766e">Benchmark Total (n = 9,134): &mu; = 63.1 &plusmn; 73.1w (med = 38w)</text>'
    )

    for xv in range(0, 221, 25):
        xx = sx_b(xv)
        svg_b.append(
            f'<line x1="{xx:.1f}" y1="{plot_top_b+ph_b}" x2="{xx:.1f}" y2="{plot_top_b+ph_b+5}" stroke="#0f172a" stroke-width="1"/>'
            f'<text x="{xx:.1f}" y="{plot_top_b+ph_b+17}" text-anchor="middle" font-size="10.8" fill="#0f172a">{xv}</text>'
        )

    svg_b.append(
        f'<line x1="{pad_l}" y1="{plot_top_b}" x2="{pad_l}" y2="{plot_top_b+ph_b}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<line x1="{pad_l}" y1="{plot_top_b+ph_b}" x2="{W_b-pad_r}" y2="{plot_top_b+ph_b}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<text x="{pad_l + pw_b/2:.1f}" y="{H_b-10}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#0f172a">Comment Length in Whitespace-Delimited Words [Clipped at 220]</text>'
        f'<text x="15" y="{plot_top_b + ph_b/2:.1f}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#0f172a" transform="rotate(-90 15 {plot_top_b + ph_b/2:.1f})">Empirical Probability Density</text>'
    )

    body = f"""
    <div style="display: flex; gap: 18px; align-items: flex-start; background: #ffffff;">
      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 10px 6px 10px;">
        <div style="text-align: center; font-size: 13.8px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (a) Additive Lexical Weak-Score Distribution <i>s</i>(<i>x</i>) by Gold Class (<i>n</i> = 9,134)
        </div>
        <svg width="{W_a}" height="{H_a}" viewBox="0 0 {W_a} {H_a}">
          {''.join(svg_a)}
        </svg>
      </div>
      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 10px 6px 10px;">
        <div style="text-align: center; font-size: 13.8px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (b) Comment Word-Length Distribution &amp; Marginal IQR Boxplots (<i>n</i> = 9,134)
        </div>
        <svg width="{W_b}" height="{H_b}" viewBox="0 0 {W_b} {H_b}">
          {''.join(svg_b)}
        </svg>
      </div>
    </div>
    """
    write_and_render("fig2_rev", 1220, 455, body, "fig2_new.png")


# ==============================================================================
# FIGURE 3: Inductive 806-d Hybrid Architecture & Top BERTopic Clusters
# ==============================================================================
def build_fig3() -> None:
    topics_data = [
        ("Topic -1", "Outlier / Unassigned Cluster (Slot 0)", 2141, 21.0, "#d97706"),
        ("Topic 2", "Family Conflict & Parental Pressure", 347, 34.6, "#b91c1c"),
        ("Topic 0", "Depressive Symptoms, Therapy & Meds", 558, 29.6, "#b91c1c"),
        ("Topic 5", "Nighttime Crying, Isolation & Despair", 200, 19.5, "#2563eb"),
        ("Topic 7", "Music, Healing & Self-Comfort", 171, 18.1, "#2563eb"),
        ("Topic 3", "Peer Encouragement & Video Appreciation", 287, 13.9, "#0284c7"),
        ("Topic 1", "Sleep Routines, Fatigue & Physical Rest", 404, 12.9, "#0284c7"),
        ("Topic 6", "Academic Exams, Career & Tuition Stress", 188, 11.2, "#0284c7"),
        ("Topic 4", "Short Undiacritized Casual Reactions", 208, 5.8, "#0284c7"),
    ]

    W_b, H_b = 620, 415
    pad_l, pad_r, pad_t, pad_b = 258, 48, 24, 48
    pw_b = W_b - pad_l - pad_r
    ph_b = H_b - pad_t - pad_b
    max_x = 42.0

    def sx(val: float) -> float:
        return pad_l + (val / max_x) * pw_b

    svg_b = []
    for xv in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
        xx = sx(xv)
        svg_b.append(
            f'<line x1="{xx:.1f}" y1="{pad_t}" x2="{xx:.1f}" y2="{pad_t+ph_b}" stroke="#e2e8f0" stroke-width="1"/>'
            f'<text x="{xx:.1f}" y="{pad_t+ph_b+17}" text-anchor="middle" font-size="11" fill="#0f172a">{xv}%</text>'
        )

    row_h = ph_b / len(topics_data)
    for idx, (tid, tdesc, n_cnt, pos_pct, col) in enumerate(topics_data):
        cy = pad_t + (idx + 0.5) * row_h
        bh = row_h * 0.62
        bw = sx(pos_pct) - pad_l
        p = pos_pct / 100.0
        se_pct = 100.0 * math.sqrt(p * (1 - p) / n_cnt)
        ci_lo = max(0.0, pos_pct - 1.96 * se_pct)
        ci_hi = min(41.5, pos_pct + 1.96 * se_pct)

        svg_b.append(
            f'<text x="{pad_l-8}" y="{cy-2:.1f}" text-anchor="end" font-size="11.2" font-weight="700" fill="#0f172a">{tid}: {tdesc}</text>'
            f'<text x="{pad_l-8}" y="{cy+11:.1f}" text-anchor="end" font-family="Helvetica Neue, sans-serif" font-size="9.5" fill="#475569">Support: n = {n_cnt} ({n_cnt/6392*100:.1f}% of Train)</text>'
            f'<rect x="{pad_l}" y="{cy - bh/2:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" fill="{col}" fill-opacity="0.86" stroke="#0f172a" stroke-width="0.8"/>'
            f'<line x1="{sx(ci_lo):.1f}" y1="{cy:.1f}" x2="{sx(ci_hi):.1f}" y2="{cy:.1f}" stroke="#0f172a" stroke-width="1.4"/>'
            f'<line x1="{sx(ci_lo):.1f}" y1="{cy-4:.1f}" x2="{sx(ci_lo):.1f}" y2="{cy+4:.1f}" stroke="#0f172a" stroke-width="1.4"/>'
            f'<line x1="{sx(ci_hi):.1f}" y1="{cy-4:.1f}" x2="{sx(ci_hi):.1f}" y2="{cy+4:.1f}" stroke="#0f172a" stroke-width="1.4"/>'
        )
        lbl_x = sx(ci_hi) + 6
        if 15.5 <= ci_hi <= 19.5:
            lbl_x = sx(19.74) + 6
        svg_b.append(
            f'<text x="{lbl_x:.1f}" y="{cy+4:.1f}" font-family="Helvetica Neue, sans-serif" font-size="10.2" font-weight="700" fill="#0f172a">{pos_pct:.1f}%</text>'
        )

    x_base = sx(19.74)
    svg_b.append(
        f'<line x1="{x_base:.1f}" y1="{pad_t-4}" x2="{x_base:.1f}" y2="{pad_t+ph_b}" stroke="#0f172a" stroke-width="2" stroke-dasharray="5,3"/>'
        f'<rect x="{x_base+6:.1f}" y="{pad_t+ph_b-32}" width="175" height="22" rx="3" fill="#ffffff" stroke="#64748b" stroke-width="0.9"/>'
        f'<text x="{x_base+12:.1f}" y="{pad_t+ph_b-17}" font-family="Helvetica Neue, sans-serif" font-size="10" font-weight="700" fill="#0f172a">Train Base Rate: &pi; = 19.74%</text>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph_b}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<line x1="{pad_l}" y1="{pad_t+ph_b}" x2="{W_b-pad_r}" y2="{pad_t+ph_b}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<text x="{pad_l + pw_b/2 - 18:.1f}" y="{H_b-8}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#0f172a">Empirical Gold Positive Rate (% Depression in Train, x-axis starts at 0%)</text>'
    )

    body = f"""
    <div style="display: flex; gap: 18px; align-items: stretch; background: #ffffff;">
      <div style="width: 535px; border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px 14px; display:flex; flex-direction:column; justify-content:space-between;">
        <div style="text-align: center; font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 8px;">
          (a) Strictly Inductive PhoBERT&ndash;BERTopic Hybrid Architecture (<i>z</i>(<i>x</i>) &isin; &#8477;<sup>806</sup>)
        </div>

        <div style="border: 1.5px solid #334155; border-radius: 6px; background: #f8fafc; padding: 8px 12px; text-align:center;">
          <div class="sans" style="font-size: 11.4px; font-weight: 700; color: #0f172a;">
            Input Comment <i>x</i> (Unicode NFC-Normalized Raw Vietnamese Text)
          </div>
          <div style="font-size: 10.9px; color: #475569; margin-top: 2px;">
            Strictly Inductive Protocol: 0% Exposure to <span class="mono">final_val</span>, <span class="mono">final_test</span>, or UIT-VSMEC During Topic Fitting
          </div>
        </div>

        <div style="display:flex; justify-content:space-around; align-items:center; height: 24px;">
          <svg width="160" height="24" viewBox="0 0 160 24">
            <line x1="80" y1="0" x2="80" y2="16" stroke="#1e3a8a" stroke-width="2"/>
            <polygon points="75,14 80,23 85,14" fill="#1e3a8a"/>
          </svg>
          <svg width="160" height="24" viewBox="0 0 160 24">
            <line x1="80" y1="0" x2="80" y2="16" stroke="#b45309" stroke-width="2"/>
            <polygon points="75,14 80,23 85,14" fill="#b45309"/>
          </svg>
        </div>

        <div style="display:flex; gap: 12px;">
          <div style="flex: 1; border: 1.5px solid #1e40af; border-radius: 6px; background: #eff6ff; padding: 8px 10px;">
            <div class="sans" style="font-size: 11.4px; font-weight: 700; color: #1e3a8a; border-bottom: 1px solid #bfdbfe; padding-bottom: 4px; margin-bottom: 5px;">
              Branch 1: Contextual Encoder
            </div>
            <div style="font-size: 11.1px; line-height: 1.42; color: #1e293b;">
              <div>&bull; <b>Model:</b> Fine-tuned <span class="mono">vinai/phobert-base</span></div>
              <div>&bull; <b>Input:</b> Raw NFC text (no external word seg.)</div>
              <div>&bull; <b>State:</b> Frozen per seed (<i>s</i> &isin; {{42, 123, 2024}})</div>
              <div class="mono" style="margin-top: 5px; background: #dbeafe; padding: 3px 6px; border-radius: 3px; font-size: 10.4px; font-weight: 700; color: #1e3a8a; text-align:center;">
                [CLS] Pooling: h(x) &isin; &#8477;<sup>768</sup>
              </div>
            </div>
          </div>

          <div style="flex: 1.08; border: 1.5px solid #d97706; border-radius: 6px; background: #fffbeb; padding: 8px 10px;">
            <div class="sans" style="font-size: 11.4px; font-weight: 700; color: #92400e; border-bottom: 1px solid #fde68a; padding-bottom: 4px; margin-bottom: 5px;">
              Branch 2: Train-Only BERTopic
            </div>
            <div style="font-size: 11.1px; line-height: 1.42; color: #1e293b;">
              <div>&bull; <b>Fit Scope:</b> Strictly <span class="mono">final_train.csv</span> (<i>n</i> = 6,392)</div>
              <div>&bull; <b>Clusters:</b> <i>K</i> = 36 proper topics + 1 outlier</div>
              <div>&bull; <b>Outlier Rule:</b> Topic &minus;1 &rarr; <b>Slot 0</b>; Topics 0..35 &rarr; <b>Slots 1..36</b></div>
              <div class="mono" style="margin-top: 5px; background: #fef3c7; padding: 3px 6px; border-radius: 3px; font-size: 10.4px; font-weight: 700; color: #92400e; text-align:center;">
                OneHot(t(x)) &isin; {{0,1}}<sup>37</sup> + q(x) &isin; [0,1]
              </div>
            </div>
          </div>
        </div>

        <div style="display:flex; justify-content:space-around; align-items:center; height: 24px;">
          <svg width="160" height="24" viewBox="0 0 160 24">
            <line x1="80" y1="0" x2="115" y2="16" stroke="#0f766e" stroke-width="2"/>
            <polygon points="109,17 120,21 114,11" fill="#0f766e"/>
          </svg>
          <svg width="160" height="24" viewBox="0 0 160 24">
            <line x1="80" y1="0" x2="45" y2="16" stroke="#0f766e" stroke-width="2"/>
            <polygon points="46,11 40,21 51,17" fill="#0f766e"/>
          </svg>
        </div>

        <div style="border: 1.5px solid #0f766e; border-radius: 6px; background: #f0fdfa; padding: 8px 10px;">
          <div class="sans" style="font-size: 11.4px; font-weight: 700; color: #115e59; text-align:center; margin-bottom: 6px;">
            Concatenated Hybrid Representation: <i>z</i>(<i>x</i>) = [<i>h</i>(<i>x</i>); OneHot(<i>t</i>(<i>x</i>)); <i>q</i>(<i>x</i>)] &isin; &#8477;<sup>806</sup> &nbsp;(Eq. 7)
          </div>
          <div class="mono" style="display:flex; border: 1.5px solid #0f172a; border-radius: 4px; overflow:hidden; font-size: 10px; font-weight: 700; text-align:center;">
            <div style="flex: 4.5; background: #bfdbfe; color: #1e3a8a; padding: 5px 4px; border-right: 1.5px solid #0f172a;">
              h(x): [CLS] Embedding (768-d)
            </div>
            <div style="flex: 1.3; background: #fed7aa; color: #9a3412; padding: 5px 2px; border-right: 1.5px solid #0f172a;">
              Slot 0 (-1)
            </div>
            <div style="flex: 2.2; background: #fde68a; color: #92400e; padding: 5px 4px; border-right: 1.5px solid #0f172a;">
              Slots 1..36 (36-d)
            </div>
            <div style="flex: 1.1; background: #99f6e4; color: #115e59; padding: 5px 2px;">
              q(x) (1-d)
            </div>
          </div>
        </div>

        <div style="display:flex; justify-content:center; align-items:center; height: 22px;">
          <svg width="40" height="22" viewBox="0 0 40 22">
            <line x1="20" y1="0" x2="20" y2="14" stroke="#991b1b" stroke-width="2"/>
            <polygon points="15,12 20,21 25,12" fill="#991b1b"/>
          </svg>
        </div>

        <div style="border: 1.5px solid #991b1b; border-radius: 6px; background: #fff1f2; padding: 8px 12px; text-align:center;">
          <div class="sans" style="font-size: 11.3px; font-weight: 700; color: #991b1b;">
            StandardScaler &rarr; L2 Logistic Regression Head (<i>C</i> = 0.01, Selected on <span class="mono">final_val.csv</span>)
          </div>
          <div class="mono" style="font-size: 10.4px; color: #7f1d1d; font-weight: 600; margin-top: 3px;">
            3-Seed Majority Vote Macro-F1 = 0.7302 [0.697, 0.761] | Soft Vote = 0.7261
          </div>
        </div>
      </div>

      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px 12px 6px 12px;">
        <div style="text-align: center; font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (b) Positive-Class Rate (&plusmn;95% CI) Across Top Inductive BERTopic Clusters (<i>n</i> = 6,392)
        </div>
        <svg width="{W_b}" height="{H_b}" viewBox="0 0 {W_b} {H_b}">
          {''.join(svg_b)}
        </svg>
      </div>
    </div>
    """
    write_and_render("fig3_rev", 1215, 470, body, "fig3_new.png")


# ==============================================================================
# FIGURE 4: In-Domain PR Curves & Paired-Bootstrap Forest Plot vs Lexical Ceiling
# ==============================================================================
def build_fig4() -> None:
    top_data = json.loads((AUDIT_DIR / "top_venue_baselines_results.json").read_text())
    vis_data = json.loads((MODEL_DIR / "visobert_and_ablations_results.json").read_text())

    tp = top_data["test_probabilities"]
    y_true = np.array(tp["y_test"], dtype=int)
    supcon_prob = np.array(tp["visobert_supcon_soft_prob"], dtype=float)
    lafocal_prob = np.array(tp["visobert_lafocal_soft_prob"], dtype=float)
    viso_prob = np.array(tp["visobert_soft_prob"], dtype=float)
    pho_prob = np.array(tp["phobert_soft_prob"], dtype=float)
    xlmr_prob = np.array(tp["xlmr_soft_prob"], dtype=float)
    llm_fs_prob = np.array(tp["qwen25_rubric_prob"], dtype=float)
    lr_prob = np.array(vis_data["curves_in_domain"]["tfidf_logreg_prob"], dtype=float)

    W_a, H_a = 505, 470
    pad_l, pad_r, pad_t, pad_b = 52, 16, 20, 46
    pw_a = W_a - pad_l - pad_r
    ph_a = H_a - pad_t - pad_b

    def sx_a(r: float) -> float:
        return pad_l + r * pw_a

    def sy_a(p: float) -> float:
        return pad_t + ph_a - p * ph_a

    svg_a = []
    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        yy = sy_a(v)
        xx = sx_a(v)
        svg_a.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W_a-pad_r}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="0.9"/>'
            f'<text x="{pad_l-7}" y="{yy+4:.1f}" text-anchor="end" font-size="10.8" fill="#334155">{v:.1f}</text>'
            f'<line x1="{xx:.1f}" y1="{pad_t}" x2="{xx:.1f}" y2="{pad_t+ph_a}" stroke="#e2e8f0" stroke-width="0.9"/>'
            f'<text x="{xx:.1f}" y="{pad_t+ph_a+17}" text-anchor="middle" font-size="10.8" fill="#334155">{v:.1f}</text>'
        )

    for f1_iso in [0.3, 0.4, 0.5, 0.6, 0.7]:
        r_vals = np.linspace(f1_iso / (2 - f1_iso) + 0.01, 1.0, 60)
        pts = []
        for rv in r_vals:
            pv = (f1_iso * rv) / (2 * rv - f1_iso)
            if 0.0 <= pv <= 1.0:
                pts.append(f"{sx_a(rv):.1f},{sy_a(pv):.1f}")
        if pts:
            svg_a.append(
                f'<polyline points="{" ".join(pts)}" fill="none" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="2,3"/>'
            )

    y_prev = sy_a(0.1977)
    svg_a.append(
        f'<line x1="{pad_l}" y1="{y_prev:.1f}" x2="{W_a-pad_r}" y2="{y_prev:.1f}" stroke="#475569" stroke-width="1.4" stroke-dasharray="4,3"/>'
        f'<text x="{sx_a(0.62):.1f}" y="{y_prev-5:.1f}" font-family="Helvetica Neue, sans-serif" font-size="9.2" font-weight="600" fill="#475569">No-Skill (&pi; = 0.1977)</text>'
    )

    def pr_polyline(y_t: np.ndarray, scores: np.ndarray, color: str, dash: str = "", width_px: float = 2.2) -> str:
        prec, rec, _ = precision_recall_curve(y_t, scores)
        idx = np.linspace(0, len(prec) - 1, min(220, len(prec))).astype(int)
        pts = [f"{sx_a(rec[i]):.1f},{sy_a(prec[i]):.1f}" for i in idx]
        dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width_px}" {dash_attr}/>'

    svg_a.append(pr_polyline(y_true, llm_fs_prob, "#64748b", "3,3", 1.9))
    svg_a.append(pr_polyline(y_true, lr_prob, "#d97706", "5,2,1.5,2", 2.0))
    svg_a.append(pr_polyline(y_true, xlmr_prob, "#9333ea", "2,2", 2.0))
    svg_a.append(pr_polyline(y_true, pho_prob, "#dc2626", "6,2", 2.1))
    svg_a.append(pr_polyline(y_true, lafocal_prob, "#0f766e", "4,1.5", 2.2))
    svg_a.append(pr_polyline(y_true, viso_prob, "#2563eb", "", 2.3))
    svg_a.append(pr_polyline(y_true, supcon_prob, "#1e3a8a", "", 2.5))

    op_points = [
        (0.5978, 0.6045, "#0f766e"),  # ViSoBERT+LA-Focal Maj
        (0.6200, 0.5773, "#1e3a8a"),  # ViSoBERT+SupCon Maj
        (0.6273, 0.5743, "#2563eb"),  # ViSoBERT Maj
        (0.6531, 0.5145, "#dc2626"),  # PhoBERT-base Maj
        (0.6716, 0.4840, "#d97706"),  # TF-IDF + LogReg
        (0.7491, 0.4203, "#9333ea"),  # XLM-R-base Maj
        (0.5867, 0.2860, "#64748b"),  # Qwen2.5 Rubric 4-Shot
    ]
    for r_op, p_op, col in op_points:
        svg_a.append(
            f'<circle cx="{sx_a(r_op):.1f}" cy="{sy_a(p_op):.1f}" r="4.0" fill="{col}" stroke="#ffffff" stroke-width="1.2"/>'
        )

    lx, ly = sx_a(0.548), sy_a(0.995)
    legend_rows = [
        ("#1e3a8a", "", "ViSoBERT+SupCon (.661)", True),
        ("#2563eb", "", "ViSoBERT (.658)", True),
        ("#0f766e", "4,1.5", "ViSoBERT+LA-Focal (.655)", True),
        ("#dc2626", "6,2", "PhoBERT-base (.621)", False),
        ("#9333ea", "2,2", "XLM-R-base (.616)", False),
        ("#d97706", "5,2,1.5,2", "TF-IDF+LogReg (.595)", False),
        ("#64748b", "3,3", "Qwen2.5 4-Shot (.343)", False),
    ]
    svg_a.append(
        f'<rect x="{lx:.1f}" y="{ly:.1f}" width="196" height="111" rx="4" fill="#ffffff" fill-opacity="0.96" stroke="#94a3b8" stroke-width="1"/>'
        f'<text x="{lx+6:.1f}" y="{ly+11:.1f}" font-family="Helvetica Neue, sans-serif" font-size="8.0" font-weight="700" fill="#475569">Soft Curve (PR-AUC) | &bull; Maj. Op. Pt.</text>'
    )
    for idx_l, (col, dsh, lbl, bold) in enumerate(legend_rows):
        ry = ly + 23.0 + idx_l * 13.4
        da = f'stroke-dasharray="{dsh}"' if dsh else ""
        fw = "700" if bold else "400"
        svg_a.append(
            f'<line x1="{lx+6:.1f}" y1="{ry:.1f}" x2="{lx+22:.1f}" y2="{ry:.1f}" stroke="{col}" stroke-width="2.2" {da}/>'
            f'<circle cx="{lx+14:.1f}" cy="{ry:.1f}" r="2.8" fill="{col}"/>'
            f'<text x="{lx+26:.1f}" y="{ry+3.0:.1f}" font-size="8.5" font-weight="{fw}" fill="#0f172a">{lbl}</text>'
        )

    svg_a.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph_a}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<line x1="{pad_l}" y1="{pad_t+ph_a}" x2="{W_a-pad_r}" y2="{pad_t+ph_a}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<text x="{pad_l + pw_a/2:.1f}" y="{H_a-8}" text-anchor="middle" font-size="12.0" font-weight="700" fill="#0f172a">Positive-Class Recall (Depression, n = 271)</text>'
        f'<text x="15" y="{pad_t + ph_a/2:.1f}" text-anchor="middle" font-size="12.0" font-weight="700" fill="#0f172a" transform="rotate(-90 15 {pad_t + ph_a/2:.1f})">Positive-Class Precision (y-axis starts at 0.0)</text>'
    )

    # Panel (b): Forest Plot with metric-aligned 2,000-Iteration Paired Bootstrap ΔF1 p-values vs. TF-IDF + LogReg (Lexical Ceiling = 0.7138)
    forest_models = [
        ("ViSoBERT + LA-Focal (TKDE '24)", 0.7518, 0.7220, 0.7809, "0.8462", "p = 0.0045*", "#0f766e"),
        ("ViSoBERT (Social-Media Subword)", 0.7476, 0.7176, 0.7748, "0.8474", "p = 0.0090*", "#2563eb"),
        ("ViSoBERT + SupCon (KDD '26)", 0.7471, 0.7172, 0.7751, "0.8459", "p = 0.0080*", "#1e3a8a"),
        ("ViSoBERT + NCNL (TKDE '23)", 0.7465, 0.7150, 0.7749, "0.8411", "p = 0.0145*", "#0369a1"),
        ("PhoBERT + BERTopic (806-d)", 0.7302, 0.6974, 0.7606, "0.8184", "p = 0.2670", "#0d9488"),
        ("PhoBERT-base (Formal VN)", 0.7265, 0.6977, 0.7550, "0.8356", "p = 0.3440", "#dc2626"),
        ("TF-IDF + LogReg (L2, C = 1.0)", 0.7138, 0.6840, 0.7412, "0.8297", "Lexical Ref.", "#d97706"),
        ("TF-IDF + LinearSVC (C = 1.0)", 0.7025, 0.6721, 0.7327, "0.7927", "p = 0.3265", "#ea580c"),
        ("XLM-RoBERTa-base (Multilingual)", 0.6817, 0.6539, 0.7086, "0.8351", "p = 0.0130*", "#9333ea"),
        ("BiLSTM Class-Weighted", 0.6507, 0.6219, 0.6794, "0.7671", "p < 0.0001*", "#7c3aed"),
        ("Lexical Weak Labeler (All 1,371)", 0.6448, 0.6153, 0.6753, "0.6546", "p < 0.0001*", "#475569"),
        ("Qwen2.5-1.5B (4-Rule + 4-Shot)", 0.5594, 0.5310, 0.5859, "0.6739", "p < 0.0001*", "#64748b"),
        ("Inductive BERTopic-Only (74-d)", 0.5539, 0.5210, 0.5865, "0.6475", "p < 0.0001*", "#334155"),
    ]

    W_b, H_b = 715, 470
    fl_pad = 204
    fr_table = 266
    fp_t, fp_b = 32, 46
    fp_w = W_b - fl_pad - fr_table
    fp_h = H_b - fp_t - fp_b
    xmin_b, xmax_b = 0.43, 0.80

    def sx_b(v: float) -> float:
        return fl_pad + (v - xmin_b) / (xmax_b - xmin_b) * fp_w

    svg_b = []
    svg_b.append(
        f'<rect x="{sx_b(0.7220):.1f}" y="{fp_t}" width="{sx_b(0.7809)-sx_b(0.7220):.1f}" height="{fp_h}" fill="#ccfbf1" opacity="0.50"/>'
    )
    svg_b.append(
        f'<line x1="6" y1="{fp_t-4}" x2="{W_b-6}" y2="{fp_t-4}" stroke="#0f172a" stroke-width="1.5"/>'
        f'<text x="10" y="{fp_t-10}" font-family="Helvetica Neue, sans-serif" font-size="9.8" font-weight="700" fill="#0f172a">Model Family (Majority Vote)</text>'
        f'<text x="{fl_pad + fp_w/2:.1f}" y="{fp_t-10}" text-anchor="middle" font-family="Helvetica Neue, sans-serif" font-size="9.8" font-weight="700" fill="#0f172a">Macro-F1 &amp; 95% CI</text>'
        f'<text x="{W_b-fr_table+6}" y="{fp_t-10}" font-family="Helvetica Neue, sans-serif" font-size="9.8" font-weight="700" fill="#0f172a">Macro-F1 [95% CI]</text>'
        f'<text x="{W_b-118}" y="{fp_t-10}" font-family="Helvetica Neue, sans-serif" font-size="9.8" font-weight="700" fill="#0f172a">ROC</text>'
        f'<text x="{W_b-74}" y="{fp_t-10}" font-family="Helvetica Neue, sans-serif" font-size="9.6" font-weight="700" fill="#0f172a">&Delta;F1 p (vs LR)</text>'
    )

    for xv in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        xx = sx_b(xv)
        svg_b.append(
            f'<line x1="{xx:.1f}" y1="{fp_t}" x2="{xx:.1f}" y2="{fp_t+fp_h}" stroke="#e2e8f0" stroke-width="0.9"/>'
            f'<text x="{xx:.1f}" y="{fp_t+fp_h+17}" text-anchor="middle" font-size="10.4" fill="#334155">{xv:.2f}</text>'
        )

    x_triv = sx_b(0.4452)
    x_lexref = sx_b(0.7138)
    svg_b.append(
        f'<line x1="{x_triv:.1f}" y1="{fp_t}" x2="{x_triv:.1f}" y2="{fp_t+fp_h}" stroke="#64748b" stroke-width="1.4" stroke-dasharray="4,3"/>'
        f'<line x1="{x_lexref:.1f}" y1="{fp_t}" x2="{x_lexref:.1f}" y2="{fp_t+fp_h}" stroke="#d97706" stroke-width="1.2" stroke-dasharray="2,2"/>'
    )

    row_hb = fp_h / len(forest_models)
    for i, (mname, mean_v, lo_v, hi_v, roc_v, p_str, col) in enumerate(forest_models):
        cy = fp_t + (i + 0.5) * row_hb
        if i % 2 == 1:
            svg_b.append(
                f'<rect x="6" y="{cy - row_hb/2:.1f}" width="{W_b-12}" height="{row_hb:.1f}" fill="#f8fafc" opacity="0.75"/>'
            )
        fw = "700" if i < 4 or mname.startswith("TF-IDF + LogReg") or mname.startswith("XLM-RoBERTa") else "400"
        svg_b.append(
            f'<text x="{fl_pad-7}" y="{cy+3.5:.1f}" text-anchor="end" font-size="10.0" font-weight="{fw}" fill="#0f172a">{mname}</text>'
            f'<line x1="{sx_b(lo_v):.1f}" y1="{cy:.1f}" x2="{sx_b(hi_v):.1f}" y2="{cy:.1f}" stroke="{col}" stroke-width="2.2"/>'
            f'<line x1="{sx_b(lo_v):.1f}" y1="{cy-3.8:.1f}" x2="{sx_b(lo_v):.1f}" y2="{cy+3.8:.1f}" stroke="{col}" stroke-width="1.5"/>'
            f'<line x1="{sx_b(hi_v):.1f}" y1="{cy-3.8:.1f}" x2="{sx_b(hi_v):.1f}" y2="{cy+3.8:.1f}" stroke="{col}" stroke-width="1.5"/>'
            f'<circle cx="{sx_b(mean_v):.1f}" cy="{cy:.1f}" r="4.0" fill="{col}" stroke="#ffffff" stroke-width="1.1"/>'
        )
        p_col = "#0f766e" if (i < 4 and "*" in p_str) else ("#b45309" if p_str == "Lexical Ref." else ("#991b1b" if "*" in p_str else "#334155"))
        svg_b.append(
            f'<text x="{W_b-fr_table+6}" y="{cy+3.5:.1f}" font-family="SFMono-Regular, Menlo, monospace" font-size="9.2" font-weight="{fw}" fill="#0f172a">{mean_v:.4f} [{lo_v:.3f},{hi_v:.3f}]</text>'
            f'<text x="{W_b-118}" y="{cy+3.5:.1f}" font-family="SFMono-Regular, Menlo, monospace" font-size="9.2" fill="#334155">{roc_v}</text>'
            f'<text x="{W_b-74}" y="{cy+3.5:.1f}" font-family="Helvetica Neue, sans-serif" font-size="9.0" font-weight="600" fill="{p_col}">{p_str}</text>'
        )

    svg_b.append(
        f'<line x1="{fl_pad}" y1="{fp_t}" x2="{fl_pad}" y2="{fp_t+fp_h}" stroke="#0f172a" stroke-width="1.3"/>'
        f'<line x1="{fl_pad}" y1="{fp_t+fp_h}" x2="{W_b-fr_table}" y2="{fp_t+fp_h}" stroke="#0f172a" stroke-width="1.3"/>'
        f'<text x="{W_b/2:.1f}" y="{H_b-8}" text-anchor="middle" font-size="11.2" font-weight="700" fill="#0f172a">In-Domain Macro-F1 (2,000-Bootstrap 95% CIs; Paired Bootstrap &Delta;F1 p vs. TF-IDF+LogReg; Dashed = All-Neg 0.4452)</text>'
    )

    body = f"""
    <div style="display: flex; gap: 14px; align-items: stretch; background: #ffffff;">
      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 10px 6px 10px;">
        <div style="text-align: center; font-size: 13.5px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (a) In-Domain Precision&ndash;Recall Curves &amp; Iso-<i>F</i><sub>1</sub> Contours (<i>n</i> = 1,371)
        </div>
        <svg width="{W_a}" height="{H_a}" viewBox="0 0 {W_a} {H_a}">
          {''.join(svg_a)}
        </svg>
      </div>
      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 10px 6px 10px;">
        <div style="text-align: center; font-size: 13.5px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (b) Benchmark Macro-F1 Forest Plot &amp; Paired-Bootstrap Tests vs. Lexical Ceiling
        </div>
        <svg width="{W_b}" height="{H_b}" viewBox="0 0 {W_b} {H_b}">
          {''.join(svg_b)}
        </svg>
      </div>
    </div>
    """
    write_and_render("fig4_rev", 1290, 480, body, "fig4_new.png")


# ==============================================================================
# FIGURE 5: Reconciled Confusion Matrices (Flagship ViSoBERT + LA-Focal & PhoBERT)
# ==============================================================================
def build_fig5() -> None:
    matrices = [
        {
            "title": "(a) ViSoBERT + LA-Focal In-Domain (Majority Vote)",
            "subtitle": "In-Domain (n = 1,371): Macro-F1 = 0.7518 | Acc = 84.32% | F1-Pos = 0.6011 | Pred Pos = 268",
            "tn": 994, "fp": 106, "fn": 109, "tp": 162,
            "row0": "True Normal (0)", "row1": "True Depression (1)",
            "col0": "Pred Normal (0)", "col1": "Pred Depression (1)",
            "accent": "#115e59", "light_tp": "#ccfbf1", "dark_tn": "#0f766e",
        },
        {
            "title": "(b) PhoBERT-base In-Domain (Majority Vote)",
            "subtitle": "In-Domain (n = 1,371): Macro-F1 = 0.7265 | Acc = 80.96% | F1-Pos = 0.5756 | Pred Pos = 344",
            "tn": 933, "fp": 167, "fn": 94, "tp": 177,
            "row0": "True Normal (0)", "row1": "True Depression (1)",
            "col0": "Pred Normal (0)", "col1": "Pred Depression (1)",
            "accent": "#1e3a8a", "light_tp": "#dbeafe", "dark_tn": "#1e40af",
        },
        {
            "title": "(c) ViSoBERT + LA-Focal External (Uncalibrated \u03c4 = 0.50)",
            "subtitle": "UIT-VSMEC Sadness-Only (n = 2,691): Macro-F1 = 0.3826 | Acc = 57.93% | Pred Pos = 23",
            "tn": 1539, "fp": 3, "fn": 1129, "tp": 20,
            "row0": "True Enjoyment (0)", "row1": "True Sadness (1)",
            "col0": "Pred Enjoyment (0)", "col1": "Pred Sadness (1)",
            "accent": "#9a3412", "light_tp": "#ffedd5", "dark_tn": "#c2410c",
        },
        {
            "title": "(d) ViSoBERT + LA-Focal External (Static Bayes \u03c4 = 0.1974)",
            "subtitle": "UIT-VSMEC Sadness-Only (n = 2,691): Macro-F1 = 0.6433 | Acc = 68.93% | Pred Pos = 575",
            "tn": 1411, "fp": 131, "fn": 705, "tp": 444,
            "row0": "True Enjoyment (0)", "row1": "True Sadness (1)",
            "col0": "Pred Enjoyment (0)", "col1": "Pred Sadness (1)",
            "accent": "#7f1d1d", "light_tp": "#fee2e2", "dark_tn": "#b91c1c",
        },
    ]

    cards_html = []
    for m in matrices:
        tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
        n_tot = tn + fp + fn + tp
        r0_tot = tn + fp
        r1_tot = fn + tp
        c0_tot = tn + fn
        c1_tot = fp + tp

        rec0 = tn / r0_tot * 100
        fp_rate = fp / r0_tot * 100
        fn_rate = fn / r1_tot * 100
        rec1 = tp / r1_tot * 100

        prec0 = tn / c0_tot * 100 if c0_tot else 0
        prec1 = tp / c1_tot * 100 if c1_tot else 0
        acc = (tn + tp) / n_tot * 100

        card = f"""
        <div style="flex: 1; border: 1.5px solid #cbd5e1; border-radius: 6px; padding: 10px 12px; background: #ffffff;">
          <div style="font-size: 13.2px; font-weight: 700; color: #0f172a; text-align:center;">{m['title']}</div>
          <div class="mono" style="font-size: 9.8px; font-weight: 600; color: {m['accent']}; text-align:center; margin: 2px 0 8px 0;">
            {m['subtitle']}
          </div>

          <table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 11.5px;">
            <thead>
              <tr>
                <th style="width: 26%; padding: 4px; border: 1px solid #cbd5e1; background: #f8fafc; font-size: 10px; color: #475569;">Actual \\ Pred</th>
                <th class="sans" style="width: 27%; padding: 5px; border: 1px solid #cbd5e1; background: #f1f5f9; font-size: 10.5px; color: #0f172a;">{m['col0']}</th>
                <th class="sans" style="width: 27%; padding: 5px; border: 1px solid #cbd5e1; background: #f1f5f9; font-size: 10.5px; color: #0f172a;">{m['col1']}</th>
                <th class="sans" style="width: 20%; padding: 5px; border: 1px solid #cbd5e1; background: #e2e8f0; font-size: 10px; color: #0f172a;">Row Recall</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="sans" style="padding: 8px 6px; border: 1px solid #cbd5e1; background: #f1f5f9; font-weight: 700; font-size: 10.5px; color: #0f172a;">
                  {m['row0']}<br/><span class="mono" style="font-size:9.5px; font-weight:400; color:#475569;">(n = {r0_tot:,})</span>
                </td>
                <td style="padding: 10px 6px; border: 1.5px solid #0f172a; background: {m['dark_tn']}; color: #ffffff;">
                  <div class="sans" style="font-size: 9px; font-weight: 700; opacity: 0.9; letter-spacing:0.3px;">TRUE NEGATIVE (TN)</div>
                  <div class="mono" style="font-size: 15px; font-weight: 700; margin: 2px 0;">{tn:,}</div>
                  <div class="mono" style="font-size: 10px; opacity: 0.95;">({rec0:.1f}% of row)</div>
                </td>
                <td style="padding: 10px 6px; border: 1.5px solid #0f172a; background: #fff1f2; color: #881337;">
                  <div class="sans" style="font-size: 9px; font-weight: 700; color: #9f1239;">FALSE POSITIVE (FP)</div>
                  <div class="mono" style="font-size: 15px; font-weight: 700; margin: 2px 0;">{fp:,}</div>
                  <div class="mono" style="font-size: 10px;">({fp_rate:.1f}% of row)</div>
                </td>
                <td class="mono" style="padding: 8px 4px; border: 1px solid #cbd5e1; background: #f8fafc; font-size: 11px; font-weight: 700; color: #0f766e;">
                  {rec0:.1f}%<br/><span style="font-size:9px; color:#64748b; font-weight:400;">Spec.</span>
                </td>
              </tr>
              <tr>
                <td class="sans" style="padding: 8px 6px; border: 1px solid #cbd5e1; background: #f1f5f9; font-weight: 700; font-size: 10.5px; color: #0f172a;">
                  {m['row1']}<br/><span class="mono" style="font-size:9.5px; font-weight:400; color:#475569;">(n = {r1_tot:,})</span>
                </td>
                <td style="padding: 10px 6px; border: 1.5px solid #0f172a; background: #fffbeb; color: #78350f;">
                  <div class="sans" style="font-size: 9px; font-weight: 700; color: #92400e;">FALSE NEGATIVE (FN)</div>
                  <div class="mono" style="font-size: 15px; font-weight: 700; margin: 2px 0;">{fn:,}</div>
                  <div class="mono" style="font-size: 10px;">({fn_rate:.1f}% of row)</div>
                </td>
                <td style="padding: 10px 6px; border: 1.5px solid #0f172a; background: {m['light_tp']}; color: #0f172a;">
                  <div class="sans" style="font-size: 9px; font-weight: 700; color: {m['accent']};">TRUE POSITIVE (TP)</div>
                  <div class="mono" style="font-size: 15px; font-weight: 700; margin: 2px 0; color: {m['accent']};">{tp:,}</div>
                  <div class="mono" style="font-size: 10px; font-weight: 600;">({rec1:.1f}% of row)</div>
                </td>
                <td class="mono" style="padding: 8px 4px; border: 1px solid #cbd5e1; background: #f8fafc; font-size: 11px; font-weight: 700; color: {m['accent']};">
                  {rec1:.1f}%<br/><span style="font-size:9px; color:#64748b; font-weight:400;">Sens.</span>
                </td>
              </tr>
              <tr>
                <td class="sans" style="padding: 6px; border: 1px solid #cbd5e1; background: #e2e8f0; font-weight: 700; font-size: 10px; color: #0f172a;">
                  Col Precision
                </td>
                <td class="mono" style="padding: 6px; border: 1px solid #cbd5e1; background: #f8fafc; font-size: 10.6px; font-weight: 700; color: #0f172a;">
                  {prec0:.1f}% <span style="font-size:9.0px; font-weight:400; color:#475569;">(NPV, n={c0_tot:,})</span>
                </td>
                <td class="mono" style="padding: 6px; border: 1px solid #cbd5e1; background: #f8fafc; font-size: 10.6px; font-weight: 700; color: {m['accent']};">
                  {prec1:.1f}% <span style="font-size:9.0px; font-weight:400; color:#475569;">(PPV, n={c1_tot:,})</span>
                </td>
                <td class="mono" style="padding: 6px; border: 1.5px solid #0f172a; background: #0f172a; font-size: 10.5px; font-weight: 700; color: #ffffff;">
                  Acc: {acc:.2f}%
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        """
        cards_html.append(card)

    body = f"""
    <div style="display: flex; flex-direction: column; gap: 14px; background: #ffffff;">
      <div style="display: flex; gap: 14px;">
        {cards_html[0]}
        {cards_html[1]}
      </div>
      <div style="display: flex; gap: 14px;">
        {cards_html[2]}
        {cards_html[3]}
      </div>
    </div>
    """
    write_and_render("fig5_rev", 1220, 525, body, "fig5_new.png")


# ==============================================================================
# FIGURE 6: Equal-Budget Active Learning & 4-Encoder Label-Free Calibration
# ==============================================================================
def build_fig6() -> None:
    top_data = json.loads((AUDIT_DIR / "top_venue_baselines_results.json").read_text())
    al = top_data["active_learning"]
    cal = top_data["calibration"]

    # Panel (a): Equal-Budget Pool-Based Active Learning Curves
    W_a, H_a = 570, 405
    pad_l, pad_r, pad_t, pad_b = 54, 18, 26, 52
    pw_a = W_a - pad_l - pad_r
    ph_a = H_a - pad_t - pad_b
    ymin_a, ymax_a = 0.595, 0.71

    budgets = [400, 800, 1600, 3200, 4800, 6392]

    def sx_a(idx: int) -> float:
        return pad_l + (idx / (len(budgets) - 1)) * pw_a

    def sy_a(v: float) -> float:
        return pad_t + ph_a - ((v - ymin_a) / (ymax_a - ymin_a)) * ph_a

    svg_a = []
    for yv in [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]:
        yy = sy_a(yv)
        svg_a.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W_a-pad_r}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="0.9"/>'
            f'<text x="{pad_l-7}" y="{yy+4:.1f}" text-anchor="end" font-size="10.8" fill="#334155">{yv:.2f}</text>'
        )

    for idx_b, bval in enumerate(budgets):
        xx = sx_a(idx_b)
        pct = bval / 6392 * 100
        pct_lbl = "100%" if bval == 6392 else f"{pct:.1f}%"
        svg_a.append(
            f'<line x1="{xx:.1f}" y1="{pad_t}" x2="{xx:.1f}" y2="{pad_t+ph_a}" stroke="#f1f5f9" stroke-width="0.9"/>'
            f'<line x1="{xx:.1f}" y1="{pad_t+ph_a}" x2="{xx:.1f}" y2="{pad_t+ph_a+5}" stroke="#0f172a" stroke-width="1"/>'
            f'<text x="{xx:.1f}" y="{pad_t+ph_a+17}" text-anchor="middle" font-size="10.4" font-weight="700" fill="#0f172a">{bval:,}</text>'
            f'<text x="{xx:.1f}" y="{pad_t+ph_a+29}" text-anchor="middle" font-family="Helvetica Neue, sans-serif" font-size="8.7" fill="#64748b">({pct_lbl})</text>'
        )

    strat_cfg = [
        ("random", "Random Sampling (Matched-Budget Control)", "#64748b", "4,3"),
        ("contrastive_cal", "Contrastive AL / CoMAL (KDD '24)", "#d97706", "6,2"),
        ("badge", "BADGE Diversity-Uncertainty", "#2563eb", "3,1.5"),
        ("least_confidence", "Least-Confidence u(x) (Eq. 2, Ours)", "#dc2626", ""),
    ]

    for skey, slbl, scol, sdash in strat_cfg:
        means = [al["strategies"][skey][str(b)]["mean"] for b in budgets]
        stds = [al["strategies"][skey][str(b)]["std"] for b in budgets]
        upper = [f"{sx_a(i):.1f},{sy_a(min(ymax_a, m + s)):.1f}" for i, (m, s) in enumerate(zip(means, stds))]
        lower = [f"{sx_a(i):.1f},{sy_a(max(ymin_a, m - s)):.1f}" for i, (m, s) in reversed(list(enumerate(zip(means, stds))))]
        svg_a.append(
            f'<polygon points="{" ".join(upper + lower)}" fill="{scol}" fill-opacity="0.13"/>'
        )
        pts = [f"{sx_a(i):.1f},{sy_a(m):.1f}" for i, m in enumerate(means)]
        da = f'stroke-dasharray="{sdash}"' if sdash else ""
        svg_a.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{scol}" stroke-width="2.4" {da}/>'
        )
        for i, m in enumerate(means):
            svg_a.append(
                f'<circle cx="{sx_a(i):.1f}" cy="{sy_a(m):.1f}" r="4.0" fill="{scol}" stroke="#ffffff" stroke-width="1.2"/>'
            )

    x3200 = sx_a(3)
    y_lc_3200 = sy_a(0.6868)
    y_rd_3200 = sy_a(0.6333)
    svg_a.append(
        f'<line x1="{x3200:.1f}" y1="{y_lc_3200+5:.1f}" x2="{x3200:.1f}" y2="{y_rd_3200-5:.1f}" stroke="#0f172a" stroke-width="1.3" stroke-dasharray="2,2"/>'
        f'<rect x="{x3200+6:.1f}" y="{(y_lc_3200+y_rd_3200)/2 - 10:.1f}" width="114" height="20" rx="3" fill="#fef2f2" stroke="#dc2626" stroke-width="0.9"/>'
        f'<text x="{x3200+63:.1f}" y="{(y_lc_3200+y_rd_3200)/2 + 3.5:.1f}" text-anchor="middle" font-family="Helvetica Neue, sans-serif" font-size="9.2" font-weight="700" fill="#991b1b">+5.35 F1 vs. Random</text>'
    )

    lx_a, ly_a = W_a - pad_r - 246, pad_t + ph_a - 70
    svg_a.append(
        f'<rect x="{lx_a:.1f}" y="{ly_a:.1f}" width="242" height="67" rx="4" fill="#ffffff" fill-opacity="0.96" stroke="#94a3b8" stroke-width="0.9"/>'
    )
    for idx_s, (_, slbl, scol, sdash) in enumerate(reversed(strat_cfg)):
        ry = ly_a + 12 + idx_s * 15.0
        da = f'stroke-dasharray="{sdash}"' if sdash else ""
        fw = "700" if idx_s < 2 else "400"
        svg_a.append(
            f'<line x1="{lx_a+7:.1f}" y1="{ry:.1f}" x2="{lx_a+25:.1f}" y2="{ry:.1f}" stroke="{scol}" stroke-width="2.3" {da}/>'
            f'<circle cx="{lx_a+16:.1f}" cy="{ry:.1f}" r="3.1" fill="{scol}"/>'
            f'<text x="{lx_a+30:.1f}" y="{ry+3.3:.1f}" font-size="9.1" font-weight="{fw}" fill="#0f172a">{slbl}</text>'
        )

    svg_a.append(
        f'<rect x="{pad_l+7}" y="{pad_t+4}" width="222" height="19" rx="3" fill="#fffbeb" stroke="#d97706" stroke-width="0.9"/>'
        f'<text x="{pad_l+13}" y="{pad_t+16.5}" font-family="Helvetica Neue, sans-serif" font-size="8.8" font-weight="700" fill="#92400e">Note: Zoomed y-axis [0.60, 0.70] to resolve &plusmn;1 SD</text>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph_a}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<line x1="{pad_l-4}" y1="{pad_t+ph_a-5}" x2="{pad_l+4}" y2="{pad_t+ph_a-9}" stroke="#0f172a" stroke-width="1.3"/>'
        f'<line x1="{pad_l-4}" y1="{pad_t+ph_a-1}" x2="{pad_l+4}" y2="{pad_t+ph_a-5}" stroke="#0f172a" stroke-width="1.3"/>'
        f'<line x1="{pad_l}" y1="{pad_t+ph_a}" x2="{W_a-pad_r}" y2="{pad_t+ph_a}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<text x="{pad_l + pw_a/2:.1f}" y="{H_a-6}" text-anchor="middle" font-size="11.4" font-weight="700" fill="#0f172a">Ordinal Labeled Pool Budget on final_train.csv (% of N = 6,392; 3 Seeds Mean &plusmn; SD)</text>'
        f'<text x="15" y="{pad_t + ph_a/2:.1f}" text-anchor="middle" font-size="11.3" font-weight="700" fill="#0f172a" transform="rotate(-90 15 {pad_t + ph_a/2:.1f})">In-Domain Test Macro-F1 (Zoomed y-axis [0.60, 0.70])</text>'
    )

    # Panel (b): Complete 5 Pretrained Encoders x 3 Label-Free Calibration Regimes (Table 15 exact match)
    W_b, H_b = 675, 405
    pb_l, pb_r, pb_t, pb_b = 48, 14, 42, 52
    pw_b = W_b - pb_l - pb_r
    ph_b = H_b - pb_t - pb_b
    max_y_b = 0.75

    def sy_b(v: float) -> float:
        return pb_t + ph_b - (v / max_y_b) * ph_b

    svg_b = []
    for yv in [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]:
        yy = sy_b(yv)
        svg_b.append(
            f'<line x1="{pb_l}" y1="{yy:.1f}" x2="{W_b-pb_r}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="0.9"/>'
            f'<text x="{pb_l-7}" y="{yy+4:.1f}" text-anchor="end" font-size="10.8" fill="#334155">{yv:.2f}</text>'
        )

    svg_b.append(
        f'<rect x="{pb_l+2}" y="4" width="{pw_b-4}" height="28" rx="4" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.9"/>'
        f'<rect x="{pb_l+16}" y="12" width="12" height="12" fill="#94a3b8" stroke="#334155" stroke-width="0.8"/>'
        f'<text x="{pb_l+33}" y="21.5" font-size="10.2" fill="#0f172a">Uncalibrated (&tau; = 0.50)</text>'
        f'<rect x="{pb_l+188}" y="12" width="12" height="12" fill="#f59e0b" stroke="#92400e" stroke-width="0.8"/>'
        f'<text x="{pb_l+205}" y="21.5" font-size="10.2" font-weight="700" fill="#92400e">Static Bayes (&tau; = 0.1974)</text>'
        f'<rect x="{pb_l+378}" y="12" width="12" height="12" fill="#0f766e" stroke="#115e59" stroke-width="0.8"/>'
        f'<text x="{pb_l+395}" y="21.5" font-size="10.2" font-weight="700" fill="#115e59">Unsup. Source-Rank (Q<tspan dy="2" font-size="7.6">80.3%</tspan><tspan dy="-2">)</tspan></text>'
    )

    cal_models = [
        ("XLM-R-base", "xlmr_base"),
        ("PhoBERT-base", "phobert_base"),
        ("ViSoBERT", "visobert"),
        ("ViSoBERT+SupCon", "visobert_supcon"),
        ("ViSoBERT+LA-Focal", "visobert_lafocal"),
    ]
    grp_w = pw_b / len(cal_models)
    bw = 28.0
    for idx_m, (mlbl, mkey) in enumerate(cal_models):
        cm = cal[mkey]
        vals = [
            (cm["default_tau050"]["vsmec_sad"]["f1_macro"], "#94a3b8", "#334155", False),
            (cm["static_bayes_tau01974"]["vsmec_sad"]["f1_macro"], "#f59e0b", "#92400e", True),
            (cm["unsupervised_source_rank_quantile"]["vsmec_sad"]["f1_macro"], "#0f766e", "#115e59", True),
        ]
        gcx = pb_l + (idx_m + 0.5) * grp_w
        offsets = [-1.5 * bw - 2.5, -0.5 * bw, 0.5 * bw + 2.5]
        for off, (v_f1, bfill, bstrk, is_b) in zip(offsets, vals):
            bx = gcx + off
            by = sy_b(v_f1)
            bh = sy_b(0.0) - by
            fw = "700" if is_b else "400"
            svg_b.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{bfill}" stroke="{bstrk}" stroke-width="1.0"/>'
                f'<text x="{bx + bw/2:.1f}" y="{by - 5.0:.1f}" text-anchor="middle" font-family="Helvetica Neue, Arial, sans-serif" font-size="8.2" font-weight="{fw}" fill="#0f172a">{v_f1:.3f}</text>'
            )
        svg_b.append(
            f'<text x="{gcx:.1f}" y="{pb_t+ph_b+19}" text-anchor="middle" font-size="10.0" font-weight="700" fill="#0f172a">{mlbl}</text>'
        )

    svg_b.append(
        f'<line x1="{pb_l}" y1="{pb_t}" x2="{pb_l}" y2="{pb_t+ph_b}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<line x1="{pb_l}" y1="{pb_t+ph_b}" x2="{W_b-pb_r}" y2="{pb_t+ph_b}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<text x="{pb_l + pw_b/2:.1f}" y="{H_b-6}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0f172a">UIT-VSMEC Sadness-Only Transfer (n = 2,691; All 5 Encoders, Zero Target Labels)</text>'
        f'<text x="15" y="{pb_t + ph_b/2:.1f}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#0f172a" transform="rotate(-90 15 {pb_t + ph_b/2:.1f})">External Macro-F1 (y-axis starts at 0.0)</text>'
    )

    body = f"""
    <div style="display: flex; gap: 14px; align-items: stretch; background: #ffffff;">
      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 10px 6px 10px;">
        <div style="text-align: center; font-size: 13.0px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (a) Equal-Budget Active Learning [Zoomed y-axis: 0.60&ndash;0.70] (<i>n</i> = 400 &rarr; 6,392)
        </div>
        <svg width="{W_a}" height="{H_a}" viewBox="0 0 {W_a} {H_a}">
          {''.join(svg_a)}
        </svg>
      </div>
      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 10px 6px 10px;">
        <div style="text-align: center; font-size: 13.0px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (b) Label-Free Cross-Corpus Calibration Across All Five Encoders (<i>n</i> = 2,691)
        </div>
        <svg width="{W_b}" height="{H_b}" viewBox="0 0 {W_b} {H_b}">
          {''.join(svg_b)}
        </svg>
      </div>
    </div>
    """
    write_and_render("fig6_rev", 1335, 458, body, "fig6_new.png")


if __name__ == "__main__":
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    build_fig5()
    build_fig6()
