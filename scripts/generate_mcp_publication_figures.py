#!/usr/bin/env python3
"""
Generate IEEE Transactions / Nature / ACL Publication-Grade HTML+SVG Figure Templates
to be rendered via the Playwright MCP Server at High-DPI (Retina Device Scale).

Strictly uses ONLY the original model families:
- PhoBERT + BERTopic Hybrid (806-d, Strictly Inductive)
- PhoBERT-base (vinai/phobert-base)
- TF-IDF + Logistic Regression (L2, C=1.0, balanced)
- TF-IDF + LinearSVC (C=1.0, balanced)
- BiLSTM (Class-Weighted w1=4.0650 & Unweighted w1=1.0)
- Inductive BERTopic-Only (74-d, Train-Only K=36 + Slot 0 Outlier)
- Lexical Weak Labeler (335 entries) & All-Negative Baseline
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "round6_v2_retrained"
AUDIT_DIR = BASE_DIR / "results" / "scientific_audit"
HTML_DIR = AUDIT_DIR / "mcp_html"
FIG_DIR = AUDIT_DIR / "figures"
HTML_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


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
    padding: 20px 24px;
    zoom: 3.2;
  }
"""


def write_html(filename: str, width: int, height: int, body_html: str) -> Path:
    path = HTML_DIR / filename
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
{COMMON_CSS}
#figure-root {{
  width: {width}px;
  min-height: {height}px;
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
    path.write_text(html, encoding="utf-8")
    return path


# ==============================================================================
# FIGURE 1: Six-Stage Benchmark Construction & Evaluation Workflow
# ==============================================================================
def build_fig1():
    body = """
    <div style="width: 100%; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 18px 20px; background: #ffffff;">
      <!-- Top Title Banner -->
      <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 2px solid #0f172a; padding-bottom: 8px; margin-bottom: 14px;">
        <div style="font-size: 16px; font-weight: 700; letter-spacing: 0.2px; color: #0f172a;">
          VietDepression-YouTube Benchmark Construction &amp; Strictly Inductive Evaluation Protocol
        </div>
        <div class="sans" style="font-size: 11.5px; font-weight: 600; color: #334155; background: #f1f5f9; padding: 3px 10px; border-radius: 4px; border: 1px solid #cbd5e1;">
          Zero Transductive Leakage &bull; 100% Human-Verified Gold Splits (n = 9,134)
        </div>
      </div>

      <!-- PHASE I & II ROW (Stages 1 -> 2 -> 3) -->
      <div style="display: flex; align-items: stretch; justify-content: space-between; gap: 0px; margin-bottom: 12px;">
        
        <!-- STAGE 1 -->
        <div style="flex: 1; border: 1.5px solid #1e40af; border-radius: 6px; background: #f8fafc; overflow: hidden; box-shadow: 0 1px 3px rgba(15,23,42,0.06);">
          <div class="sans" style="background: #1e3a8a; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
            <span>STAGE 1: YouTube API Retrieval</span>
            <span style="background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 3px; font-size: 10px;">Phase I: Corpus</span>
          </div>
          <div style="padding: 10px 12px; font-size: 12.2px; line-height: 1.48; color: #1e293b;">
            <div style="margin-bottom: 4px;">&bull; <b>Query Design:</b> 264 Vietnamese distress, symptom, sleep &amp; help-seeking queries</div>
            <div style="margin-bottom: 4px;">&bull; <b>Video Coverage:</b> 4,413 crawled videos &rarr; <b>2,906</b> valid Vietnamese videos retained</div>
            <div style="margin-bottom: 4px;">&bull; <b>Raw Inventory:</b> <b>125,329</b> public top-level &amp; reply comments</div>
            <div class="mono" style="font-size: 10.5px; background: #e2e8f0; padding: 3px 6px; border-radius: 3px; color: #0f172a; margin-top: 5px;">
              Mean length: 19.63 &plusmn; 33.41w (median = 11w)
            </div>
          </div>
        </div>

        <!-- Arrow 1->2 -->
        <div style="width: 54px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <div class="sans" style="font-size: 9.5px; font-weight: 700; color: #1e3a8a; margin-bottom: 2px;">125,329</div>
          <svg width="48" height="20" viewBox="0 0 48 20">
            <line x1="2" y1="10" x2="38" y2="10" stroke="#1e3a8a" stroke-width="2.2"/>
            <polygon points="36,4 46,10 36,16" fill="#1e3a8a"/>
          </svg>
          <div class="sans" style="font-size: 8.5px; color: #64748b; margin-top: 1px;">Raw UTF-8</div>
        </div>

        <!-- STAGE 2 -->
        <div style="flex: 1; border: 1.5px solid #0284c7; border-radius: 6px; background: #f8fafc; overflow: hidden; box-shadow: 0 1px 3px rgba(15,23,42,0.06);">
          <div class="sans" style="background: #0369a1; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
            <span>STAGE 2: NFC &amp; Quality Filtering</span>
            <span style="background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 3px; font-size: 10px;">Normalization</span>
          </div>
          <div style="padding: 10px 12px; font-size: 12.2px; line-height: 1.48; color: #1e293b;">
            <div style="margin-bottom: 4px;">&bull; <b>Unicode NFC:</b> Canonicalize Vietnamese tone marks; preserve case &amp; emoji</div>
            <div style="margin-bottom: 4px;">&bull; <b>Noise Removal:</b> Strip URLs, spam, non-VN text, &lt;5 chars &amp; exact duplicates</div>
            <div style="margin-bottom: 4px;">&bull; <b>Discarded Noise:</b> <b>5,680</b> records removed (<b>4.53%</b> of raw corpus)</div>
            <div class="mono" style="font-size: 10.5px; background: #e0f2fe; padding: 3px 6px; border-radius: 3px; color: #0369a1; font-weight: 600; margin-top: 5px;">
              Retained Pool: 119,649 (95.47%) | Audit: 125,336
            </div>
          </div>
        </div>

        <!-- Arrow 2->3 -->
        <div style="width: 54px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <div class="sans" style="font-size: 9.5px; font-weight: 700; color: #b45309; margin-bottom: 2px;">119,649</div>
          <svg width="48" height="20" viewBox="0 0 48 20">
            <line x1="2" y1="10" x2="38" y2="10" stroke="#b45309" stroke-width="2.2"/>
            <polygon points="36,4 46,10 36,16" fill="#b45309"/>
          </svg>
          <div class="sans" style="font-size: 8.5px; color: #64748b; margin-top: 1px;">Normalized</div>
        </div>

        <!-- STAGE 3 -->
        <div style="flex: 1; border: 1.5px solid #d97706; border-radius: 6px; background: #fffbeb; overflow: hidden; box-shadow: 0 1px 3px rgba(15,23,42,0.06);">
          <div class="sans" style="background: #b45309; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
            <span>STAGE 3: Weighted Lexicon Triage</span>
            <span style="background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 3px; font-size: 10px;">Weak Scoring</span>
          </div>
          <div style="padding: 10px 12px; font-size: 12.2px; line-height: 1.48; color: #1e293b;">
            <div style="margin-bottom: 4px;">&bull; <b>335-Entry Lexicon:</b> Strong (<i>w</i> = +5), Medium (+3), Normal/Idiom (&minus;2)</div>
            <div style="margin-bottom: 4px;">&bull; <b>Triage Zones:</b> Pos (<i>s</i> &ge; 5): <b>3,223</b> (2.57%) | Neg (<i>s</i> &le; &minus;2): <b>23,695</b> (18.91%)</div>
            <div style="margin-bottom: 4px;">&bull; <b>Abstention Zone (&minus;1 &le; <i>s</i> &le; 4):</b> <b>98,468</b> uncertain records (<b>78.52%</b>)</div>
            <div class="mono" style="font-size: 10.5px; background: #fef3c7; padding: 3px 6px; border-radius: 3px; color: #92400e; font-weight: 600; margin-top: 5px;">
              22-Phrase Flag: 820 (0.65%) | Composite: 120,334
            </div>
          </div>
        </div>

      </div>

      <!-- Downward Turn Connector from Stage 3 to Stage 4 -->
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

      <!-- PHASE II & III ROW (Stage 6 <- Stage 5 <- Stage 4) -->
      <div style="display: flex; align-items: stretch; justify-content: space-between; gap: 0px; margin-bottom: 12px;">
        
        <!-- STAGE 6 (Left) -->
        <div style="flex: 1; border: 1.5px solid #334155; border-radius: 6px; background: #f8fafc; overflow: hidden; box-shadow: 0 1px 3px rgba(15,23,42,0.06);">
          <div class="sans" style="background: #1e293b; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
            <span>STAGE 6: Zero-Leakage Splits</span>
            <span style="background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 3px; font-size: 10px;">70 / 15 / 15</span>
          </div>
          <div style="padding: 10px 12px; font-size: 12.2px; line-height: 1.48; color: #1e293b;">
            <div style="margin-bottom: 4px;">&bull; <b>Train (70%):</b> <b>6,392</b> (1,262 Pos [19.74%] / 5,130 Neg [80.26%])</div>
            <div style="margin-bottom: 4px;">&bull; <b>Validation (15%):</b> <b>1,371</b> (271 Pos [19.77%] / 1,100 Neg [80.23%])</div>
            <div style="margin-bottom: 4px;">&bull; <b>In-Domain Test (15%):</b> <b>1,371</b> (271 Pos [19.77%] / 1,100 Neg [80.23%])</div>
            <div class="mono" style="font-size: 10.5px; background: #e2e8f0; padding: 3px 6px; border-radius: 3px; color: #0f172a; font-weight: 600; margin-top: 5px;">
              0% Exact-Text Overlap | Mean Length: 63.0 &plusmn; 69.5w
            </div>
          </div>
        </div>

        <!-- Arrow 5->6 (Pointing Left) -->
        <div style="width: 54px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <div class="sans" style="font-size: 9.5px; font-weight: 700; color: #1e293b; margin-bottom: 2px;">9,134 Gold</div>
          <svg width="48" height="20" viewBox="0 0 48 20">
            <line x1="10" y1="10" x2="46" y2="10" stroke="#1e293b" stroke-width="2.2"/>
            <polygon points="12,4 2,10 12,16" fill="#1e293b"/>
          </svg>
          <div class="sans" style="font-size: 8.5px; color: #64748b; margin-top: 1px;">Stratified</div>
        </div>

        <!-- STAGE 5 (Center) -->
        <div style="flex: 1; border: 1.5px solid #0d9488; border-radius: 6px; background: #f0fdfa; overflow: hidden; box-shadow: 0 1px 3px rgba(15,23,42,0.06);">
          <div class="sans" style="background: #0f766e; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
            <span>STAGE 5: 6-Round Active Learning</span>
            <span style="background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 3px; font-size: 10px;">Phase III: AL</span>
          </div>
          <div style="padding: 10px 12px; font-size: 12.2px; line-height: 1.48; color: #1e293b;">
            <div style="margin-bottom: 4px;">&bull; <b>Uncertainty Policy:</b> <i>u</i>(<i>x</i>) = 1 &minus; |2<i>p</i><sub>&theta;</sub>(<i>y</i>=1|<i>x</i>) &minus; 1| on <span class="mono" style="font-size:11px;">unlabeled_pool</span></div>
            <div style="margin-bottom: 4px;">&bull; <b>Per-Round Gold Ledger:</b> R1–R4: <b>2,072</b> &rarr; R5: <b>+1,360</b> &rarr; R6: <b>+5,702</b></div>
            <div style="margin-bottom: 4px;">&bull; <b>100% Human Benchmark:</b> <b>9,134</b> gold comments (0 pseudo-labels)</div>
            <div class="mono" style="font-size: 10.5px; background: #ccfbf1; padding: 3px 6px; border-radius: 3px; color: #115e59; font-weight: 600; margin-top: 5px;">
              1,804 Positive (19.75%) | 7,330 Negative (80.25%)
            </div>
          </div>
        </div>

        <!-- Arrow 4->5 (Pointing Left) -->
        <div style="width: 54px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <div class="sans" style="font-size: 9.5px; font-weight: 700; color: #0f766e; margin-bottom: 2px;">Adjudicated</div>
          <svg width="48" height="20" viewBox="0 0 48 20">
            <line x1="10" y1="10" x2="46" y2="10" stroke="#0f766e" stroke-width="2.2"/>
            <polygon points="12,4 2,10 12,16" fill="#0f766e"/>
          </svg>
          <div class="sans" style="font-size: 8.5px; color: #64748b; margin-top: 1px;">Iterative</div>
        </div>

        <!-- STAGE 4 (Right) -->
        <div style="flex: 1; border: 1.5px solid #0f766e; border-radius: 6px; background: #f0fdfa; overflow: hidden; box-shadow: 0 1px 3px rgba(15,23,42,0.06);">
          <div class="sans" style="background: #115e59; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
            <span>STAGE 4: Blind Human Annotation</span>
            <span style="background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 3px; font-size: 10px;">Phase II: Gold</span>
          </div>
          <div style="padding: 10px 12px; font-size: 12.2px; line-height: 1.48; color: #1e293b;">
            <div style="margin-bottom: 4px;">&bull; <b>Annotator Pool:</b> 2 trained native VN NLP annotators + 1 clinical psych adjudicator</div>
            <div style="margin-bottom: 4px;">&bull; <b>4-Rule Rubric:</b> First-person perspective, polysemy filter, implicit despair</div>
            <div style="margin-bottom: 4px;">&bull; <b>Double-Annotated QC (<i>n</i> = 300):</b> Raw agreement = <b>93.33%</b> (280/300)</div>
            <div class="mono" style="font-size: 10.5px; background: #ccfbf1; padding: 3px 6px; border-radius: 3px; color: #115e59; font-weight: 600; margin-top: 5px;">
              Cohen's &kappa; = 0.8667 | Matrix: [[139, 8], [12, 141]]
            </div>
          </div>
        </div>

      </div>

      <!-- Downward Turn Connector from Stage 6 to Evaluation Bar -->
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

      <!-- PHASE IV: BOTTOM FULL-WIDTH PROTOCOL PANEL -->
      <div style="border: 1.5px solid #991b1b; border-radius: 6px; background: #fff1f2; overflow: hidden;">
        <div class="sans" style="background: #991b1b; color: #ffffff; padding: 6px 14px; font-size: 12px; font-weight: 700; display:flex; justify-content:space-between; align-items:center;">
          <span>PHASE IV: Strictly Inductive Model Training, Multi-Seed Aggregation &amp; Statistical Evaluation Protocol</span>
          <span style="background: rgba(255,255,255,0.2); padding: 1px 8px; border-radius: 3px; font-size: 10px;">6 Original Model Families + 2 Baselines</span>
        </div>
        <div style="display: flex; gap: 14px; padding: 10px 14px; background: #fffafa; font-size: 12px; line-height: 1.45; color: #1e293b;">
          <div style="flex: 1.15; border-right: 1px dashed #fca5a5; padding-right: 12px;">
            <div class="sans" style="font-weight: 700; color: #991b1b; font-size: 11.5px; margin-bottom: 3px;">1. Evaluated Architectures (Fitted Strictly on Train n = 6,392)</div>
            <div>&bull; <b>Sparse Lexical:</b> Word (1–2g) + Char (3–5g) TF-IDF (160k features) + <b>LinearSVC</b> &amp; <b>LogReg</b> (<i>L</i><sub>2</sub>, <i>C</i>=1.0)</div>
            <div>&bull; <b>Recurrent Neural:</b> 2-Layer <b>BiLSTM</b> (Unweighted <i>w</i><sub>1</sub>=1.0 &amp; Class-Weighted <i>w</i><sub>1</sub>=4.0650; 3 seeds)</div>
            <div>&bull; <b>Contextual &amp; Topic Hybrid:</b> <b>PhoBERT-base</b>, Train-Only <b>BERTopic-Only</b> (<i>K</i>=36 + Slot 0), and 806-d <b>PhoBERT + BERTopic Hybrid</b> (Seeds: 42, 123, 2024; Mean&plusmn;SD, Majority Vote, Soft Vote)</div>
          </div>
          <div style="flex: 0.85;">
            <div class="sans" style="font-weight: 700; color: #991b1b; font-size: 11.5px; margin-bottom: 3px;">2. Dual-Domain &amp; Statistical Hypothesis Testing</div>
            <div>&bull; <b>In-Domain Test (<i>n</i> = 1,371):</b> 2,000-Iteration Bootstrap 95% CIs, Exact McNemar &amp; Paired Bootstrap tests; ROC-AUC &amp; PR-AUC</div>
            <div>&bull; <b>External Proxy (UIT-VSMEC):</b> Full Sadness+Fear (<i>n</i> = 3,084) &amp; Cleaner <b>Sadness-Only (<i>n</i> = 2,691)</b></div>
            <div>&bull; <b>Prior-Shift Calibration:</b> Bayes threshold &tau;<sub>Bayes</sub> = &pi;<sub>train</sub> = 0.1974 &amp; High-Sensitivity (&tau; = 0.05)</div>
          </div>
        </div>
      </div>

    </div>
    """
    return write_html("fig1.html", 1220, 560, body)


# ==============================================================================
# FIGURE 2: Weak-Score Distribution & Comment Length Distribution
# ==============================================================================
def build_fig2():
    # Load exact benchmark data (n = 9,134)
    dfs = []
    for sp in ["final_train.csv", "final_val.csv", "final_test.csv"]:
        dfs.append(pd.read_csv(BASE_DIR / "data" / "labeled" / sp))
    full_df = pd.concat(dfs, ignore_index=True)

    import sys
    sys.path.insert(0, str(BASE_DIR))
    from scripts.vietdepression_lexicon import label_comment_detailed

    scores = []
    word_lens = []
    for txt in full_df["comment_text"].fillna("").astype(str):
        det = label_comment_detailed(txt)
        scores.append(det.depression_score)
        word_lens.append(len(txt.split()))
    full_df["weak_score"] = np.clip(scores, -6, 18)
    full_df["word_len"] = np.array(word_lens)

    pos_scores = full_df[full_df["label"] == 1]["weak_score"].values
    neg_scores = full_df[full_df["label"] == 0]["weak_score"].values

    pos_lens = full_df[full_df["label"] == 1]["word_len"].values
    neg_lens = full_df[full_df["label"] == 0]["word_len"].values

    # Panel (a): Dodged histogram bins from -6 to 18 (step 2)
    bins_a = np.arange(-6, 20, 2)
    neg_hist_a, _ = np.histogram(neg_scores, bins=bins_a, density=False)
    pos_hist_a, _ = np.histogram(pos_scores, bins=bins_a, density=False)
    neg_frac_a = neg_hist_a / len(neg_scores)
    pos_frac_a = pos_hist_a / len(pos_scores)

    # Build SVG for Panel (a)
    W_a, H_a = 550, 390
    pad_l, pad_r, pad_t, pad_b = 58, 20, 34, 52
    pw_a = W_a - pad_l - pad_r
    ph_a = H_a - pad_t - pad_b
    max_y_a = 0.65

    def sx_a(val):
        return pad_l + (val - (-6.5)) / (18.5 - (-6.5)) * pw_a

    def sy_a(val):
        return pad_t + ph_a - (val / max_y_a) * ph_a

    svg_a_elements = []
    # Shaded background zones
    x_neg_zone = sx_a(-2.0)
    x_pos_zone = sx_a(4.5)
    # Negative zone (-6.5 to -2.0)
    svg_a_elements.append(
        f'<rect x="{sx_a(-6.5):.1f}" y="{pad_t}" width="{x_neg_zone - sx_a(-6.5):.1f}" height="{ph_a}" fill="#eff6ff" opacity="0.75"/>'
    )
    # Uncertain zone (-2.0 to 4.5)
    svg_a_elements.append(
        f'<rect x="{x_neg_zone:.1f}" y="{pad_t}" width="{x_pos_zone - x_neg_zone:.1f}" height="{ph_a}" fill="#fef9c3" opacity="0.65"/>'
    )
    # Positive zone (4.5 to 18.5)
    svg_a_elements.append(
        f'<rect x="{x_pos_zone:.1f}" y="{pad_t}" width="{sx_a(18.5) - x_pos_zone:.1f}" height="{ph_a}" fill="#fef2f2" opacity="0.75"/>'
    )

    # Zone labels at top (well separated)
    svg_a_elements.append(
        f'<text x="{(sx_a(-6.5)+x_neg_zone)/2 - 2:.1f}" y="{pad_t+15}" text-anchor="middle" font-family="Helvetica Neue, sans-serif" font-size="8.8" font-weight="700" fill="#1e40af">s &le; &minus;2 (18.9%)</text>'
    )
    svg_a_elements.append(
        f'<text x="{(x_neg_zone+x_pos_zone)/2 + 2:.1f}" y="{pad_t+15}" text-anchor="middle" font-family="Helvetica Neue, sans-serif" font-size="8.8" font-weight="700" fill="#92400e">Abstention [&minus;1..4] (78.5%)</text>'
    )
    svg_a_elements.append(
        f'<text x="{(x_pos_zone+sx_a(18.5))/2:.1f}" y="{pad_t+15}" text-anchor="middle" font-family="Helvetica Neue, sans-serif" font-size="9.2" font-weight="700" fill="#991b1b">Positive Rule Zone s(x) &ge; 5 (2.57%)</text>'
    )

    # Horizontal grid lines (y starts strictly at 0.0)
    for yv in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        yy = sy_a(yv)
        svg_a_elements.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W_a-pad_r}" y2="{yy:.1f}" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="3,3"/>'
        )
        svg_a_elements.append(
            f'<text x="{pad_l-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#334155">{yv:.1f}</text>'
        )

    # Bars (dodged side-by-side)
    bar_w = (sx_a(2) - sx_a(0)) * 0.38
    for i in range(len(bins_a) - 1):
        center = 0.5 * (bins_a[i] + bins_a[i + 1])
        cx = sx_a(center)
        # Neg bar
        nf = neg_frac_a[i]
        ny = sy_a(nf)
        nh = sy_a(0.0) - ny
        if nh > 0.3:
            svg_a_elements.append(
                f'<rect x="{cx - bar_w - 0.8:.1f}" y="{ny:.1f}" width="{bar_w:.1f}" height="{nh:.1f}" fill="#2563eb" fill-opacity="0.85" stroke="#1e3a8a" stroke-width="0.8"/>'
            )
        # Pos bar
        pf = pos_frac_a[i]
        py = sy_a(pf)
        ph = sy_a(0.0) - py
        if ph > 0.3:
            svg_a_elements.append(
                f'<rect x="{cx + 0.8:.1f}" y="{py:.1f}" width="{bar_w:.1f}" height="{ph:.1f}" fill="#dc2626" fill-opacity="0.85" stroke="#7f1d1d" stroke-width="0.8"/>'
            )

    # Vertical threshold lines
    svg_a_elements.append(
        f'<line x1="{x_neg_zone:.1f}" y1="{pad_t}" x2="{x_neg_zone:.1f}" y2="{pad_t+ph_a}" stroke="#1d4ed8" stroke-width="2" stroke-dasharray="5,3"/>'
    )
    svg_a_elements.append(
        f'<line x1="{x_pos_zone:.1f}" y1="{pad_t}" x2="{x_pos_zone:.1f}" y2="{pad_t+ph_a}" stroke="#b91c1c" stroke-width="2" stroke-dasharray="5,3"/>'
    )

    # X-axis ticks
    for xv in range(-6, 19, 2):
        xx = sx_a(xv)
        svg_a_elements.append(
            f'<line x1="{xx:.1f}" y1="{pad_t+ph_a}" x2="{xx:.1f}" y2="{pad_t+ph_a+5}" stroke="#0f172a" stroke-width="1"/>'
        )
        svg_a_elements.append(
            f'<text x="{xx:.1f}" y="{pad_t+ph_a+18}" text-anchor="middle" font-size="11" fill="#0f172a">{xv}</text>'
        )

    # Axes border
    svg_a_elements.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph_a}" stroke="#0f172a" stroke-width="1.4"/>'
    )
    svg_a_elements.append(
        f'<line x1="{pad_l}" y1="{pad_t+ph_a}" x2="{W_a-pad_r}" y2="{pad_t+ph_a}" stroke="#0f172a" stroke-width="1.4"/>'
    )

    # Axis titles
    svg_a_elements.append(
        f'<text x="{pad_l + pw_a/2:.1f}" y="{H_a-10}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">Additive Lexical Weak Score s(x) [2-Point Bins, Clipped to [&minus;6, 18]]</text>'
    )
    svg_a_elements.append(
        f'<text x="16" y="{pad_t + ph_a/2:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a" transform="rotate(-90 16 {pad_t + ph_a/2:.1f})">Within-Class Empirical Probability Mass</text>'
    )

    # Legend box inside Panel (a) top-right
    svg_a_elements.append(
        f'<rect x="{W_a-pad_r-218}" y="{pad_t+26}" width="210" height="72" rx="4" fill="#ffffff" fill-opacity="0.95" stroke="#94a3b8" stroke-width="1"/>'
        f'<rect x="{W_a-pad_r-208}" y="{pad_t+36}" width="14" height="10" fill="#2563eb" stroke="#1e3a8a" stroke-width="0.8"/>'
        f'<text x="{W_a-pad_r-188}" y="{pad_t+45}" font-size="11" fill="#0f172a">Gold Negative (n = 7,330)</text>'
        f'<rect x="{W_a-pad_r-208}" y="{pad_t+54}" width="14" height="10" fill="#dc2626" stroke="#7f1d1d" stroke-width="0.8"/>'
        f'<text x="{W_a-pad_r-188}" y="{pad_t+63}" font-size="11" fill="#0f172a">Gold Positive (n = 1,804)</text>'
        f'<text x="{W_a-pad_r-208}" y="{pad_t+84}" font-family="Helvetica Neue, sans-serif" font-size="9.8" font-weight="600" fill="#475569">Full-Test Weak F1 = 0.6448 (70.4% abstain)</text>'
    )

    # Panel (b): Comment Word-Length Distribution (step histogram + smooth KDE + top boxplot strip)
    W_b, H_b = 550, 390
    bins_b = np.linspace(0, 220, 23)  # 10-word bins
    neg_hist_b, _ = np.histogram(np.clip(neg_lens, 0, 220), bins=bins_b, density=True)
    pos_hist_b, _ = np.histogram(np.clip(pos_lens, 0, 220), bins=bins_b, density=True)
    max_y_b = 0.022

    # Reserve top 46px of plot area for horizontal IQR boxplots
    box_h = 42
    plot_top_b = pad_t + box_h
    ph_b = H_b - plot_top_b - pad_b
    pw_b = W_b - pad_l - pad_r

    def sx_b(val):
        return pad_l + (val / 220.0) * pw_b

    def sy_b(val):
        return plot_top_b + ph_b - (val / max_y_b) * ph_b

    svg_b_elements = []

    # Top Boxplot Strip background
    svg_b_elements.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw_b}" height="{box_h-4}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.8"/>'
    )
    # Compute quantiles
    neg_q25, neg_med, neg_q75 = np.percentile(neg_lens, [25, 50, 75])
    pos_q25, pos_med, pos_q75 = np.percentile(pos_lens, [25, 50, 75])
    neg_mean = float(np.mean(neg_lens))
    pos_mean = float(np.mean(pos_lens))

    # Draw Negative Boxplot in strip (y = pad_t + 11)
    by_neg = pad_t + 11
    svg_b_elements.append(
        f'<line x1="{sx_b(5):.1f}" y1="{by_neg}" x2="{sx_b(105):.1f}" y2="{by_neg}" stroke="#1e40af" stroke-width="1.3"/>'
        f'<line x1="{sx_b(5):.1f}" y1="{by_neg-3.5}" x2="{sx_b(5):.1f}" y2="{by_neg+3.5}" stroke="#1e40af" stroke-width="1.3"/>'
        f'<line x1="{sx_b(105):.1f}" y1="{by_neg-3.5}" x2="{sx_b(105):.1f}" y2="{by_neg+3.5}" stroke="#1e40af" stroke-width="1.3"/>'
        f'<rect x="{sx_b(neg_q25):.1f}" y="{by_neg-5}" width="{sx_b(neg_q75)-sx_b(neg_q25):.1f}" height="10" fill="#93c5fd" stroke="#1e3a8a" stroke-width="1.2"/>'
        f'<line x1="{sx_b(neg_med):.1f}" y1="{by_neg-5}" x2="{sx_b(neg_med):.1f}" y2="{by_neg+5}" stroke="#1e3a8a" stroke-width="2.2"/>'
        f'<circle cx="{sx_b(neg_mean):.1f}" cy="{by_neg}" r="3.2" fill="#1e3a8a"/>'
    )
    # Draw Positive Boxplot in strip (y = pad_t + 27)
    by_pos = pad_t + 27
    svg_b_elements.append(
        f'<line x1="{sx_b(6):.1f}" y1="{by_pos}" x2="{sx_b(115):.1f}" y2="{by_pos}" stroke="#991b1b" stroke-width="1.3"/>'
        f'<line x1="{sx_b(6):.1f}" y1="{by_pos-3.5}" x2="{sx_b(6):.1f}" y2="{by_pos+3.5}" stroke="#991b1b" stroke-width="1.3"/>'
        f'<line x1="{sx_b(115):.1f}" y1="{by_pos-3.5}" x2="{sx_b(115):.1f}" y2="{by_pos+3.5}" stroke="#991b1b" stroke-width="1.3"/>'
        f'<rect x="{sx_b(pos_q25):.1f}" y="{by_pos-5}" width="{sx_b(pos_q75)-sx_b(pos_q25):.1f}" height="10" fill="#fca5a5" stroke="#7f1d1d" stroke-width="1.2"/>'
        f'<line x1="{sx_b(pos_med):.1f}" y1="{by_pos-5}" x2="{sx_b(pos_med):.1f}" y2="{by_pos+5}" stroke="#7f1d1d" stroke-width="2.2"/>'
        f'<circle cx="{sx_b(pos_mean):.1f}" cy="{by_pos}" r="3.2" fill="#7f1d1d"/>'
    )
    svg_b_elements.append(
        f'<text x="{W_b-pad_r-6}" y="{pad_t+14}" text-anchor="end" font-family="Helvetica Neue, sans-serif" font-size="9.2" font-weight="600" fill="#1e3a8a">Normal IQR [{neg_q25:.0f}, {neg_q75:.0f}]w (med={neg_med:.0f}, &mu;={neg_mean:.1f})</text>'
        f'<text x="{W_b-pad_r-6}" y="{pad_t+30}" text-anchor="end" font-family="Helvetica Neue, sans-serif" font-size="9.2" font-weight="600" fill="#991b1b">Depression IQR [{pos_q25:.0f}, {pos_q75:.0f}]w (med={pos_med:.0f}, &mu;={pos_mean:.1f})</text>'
    )

    # Grid lines for main density plot
    for yv in [0.000, 0.005, 0.010, 0.015, 0.020]:
        yy = sy_b(yv)
        svg_b_elements.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W_b-pad_r}" y2="{yy:.1f}" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="3,3"/>'
        )
        svg_b_elements.append(
            f'<text x="{pad_l-6}" y="{yy+4:.1f}" text-anchor="end" font-size="10.5" fill="#334155">{yv:.3f}</text>'
        )

    # Step-area polygons for Neg and Pos
    def build_step_path(hist_vals, bins_arr):
        pts = [f"{sx_b(bins_arr[0]):.1f},{sy_b(0.0):.1f}"]
        for i, hval in enumerate(hist_vals):
            x0 = sx_b(bins_arr[i])
            x1 = sx_b(bins_arr[i + 1])
            yy = sy_b(hval)
            pts.append(f"{x0:.1f},{yy:.1f}")
            pts.append(f"{x1:.1f},{yy:.1f}")
        pts.append(f"{sx_b(bins_arr[-1]):.1f},{sy_b(0.0):.1f}")
        return " ".join(pts)

    neg_poly = build_step_path(neg_hist_b, bins_b)
    pos_poly = build_step_path(pos_hist_b, bins_b)
    svg_b_elements.append(
        f'<polygon points="{neg_poly}" fill="#3b82f6" fill-opacity="0.30" stroke="#1d4ed8" stroke-width="1.8"/>'
    )
    svg_b_elements.append(
        f'<polygon points="{pos_poly}" fill="#ef4444" fill-opacity="0.30" stroke="#b91c1c" stroke-width="1.8"/>'
    )

    # Vertical reference lines for Raw Inventory Mean (19.63w), Normal Mean (60.0w), Depression Mean (75.4w)
    x_raw_mean = sx_b(19.63)
    x_neg_mean = sx_b(neg_mean)
    x_pos_mean = sx_b(pos_mean)
    svg_b_elements.append(
        f'<line x1="{x_raw_mean:.1f}" y1="{plot_top_b}" x2="{x_raw_mean:.1f}" y2="{plot_top_b+ph_b}" stroke="#475569" stroke-width="1.6" stroke-dasharray="2,2"/>'
        f'<line x1="{x_neg_mean:.1f}" y1="{plot_top_b}" x2="{x_neg_mean:.1f}" y2="{plot_top_b+ph_b}" stroke="#1e3a8a" stroke-width="2" stroke-dasharray="5,3"/>'
        f'<line x1="{x_pos_mean:.1f}" y1="{plot_top_b}" x2="{x_pos_mean:.1f}" y2="{plot_top_b+ph_b}" stroke="#991b1b" stroke-width="2" stroke-dasharray="5,3"/>'
    )

    # Legend box in Panel (b)
    svg_b_elements.append(
        f'<rect x="{W_b-pad_r-246}" y="{plot_top_b+14}" width="238" height="92" rx="4" fill="#ffffff" fill-opacity="0.95" stroke="#94a3b8" stroke-width="1"/>'
        f'<rect x="{W_b-pad_r-236}" y="{plot_top_b+24}" width="14" height="10" fill="#3b82f6" fill-opacity="0.45" stroke="#1d4ed8" stroke-width="1.5"/>'
        f'<text x="{W_b-pad_r-216}" y="{plot_top_b+33}" font-size="10.8" fill="#0f172a">Gold Negative (&mu; = 60.0w, med = 37w)</text>'
        f'<rect x="{W_b-pad_r-236}" y="{plot_top_b+44}" width="14" height="10" fill="#ef4444" fill-opacity="0.45" stroke="#b91c1c" stroke-width="1.5"/>'
        f'<text x="{W_b-pad_r-216}" y="{plot_top_b+53}" font-size="10.8" fill="#0f172a">Gold Positive (&mu; = 75.4w, med = 44w)</text>'
        f'<line x1="{W_b-pad_r-236}" y1="{plot_top_b+68}" x2="{W_b-pad_r-222}" y2="{plot_top_b+68}" stroke="#475569" stroke-width="1.8" stroke-dasharray="2,2"/>'
        f'<text x="{W_b-pad_r-216}" y="{plot_top_b+72}" font-size="10.5" fill="#334155">Raw Corpus (125k: &mu; = 19.6w, med = 11w)</text>'
        f'<text x="{W_b-pad_r-236}" y="{plot_top_b+93}" font-family="Helvetica Neue, sans-serif" font-size="9.6" font-weight="600" fill="#0f766e">Benchmark Total (n = 9,134): &mu; = 63.0 &plusmn; 69.5w</text>'
    )

    # X-axis ticks
    for xv in range(0, 221, 25):
        xx = sx_b(xv)
        svg_b_elements.append(
            f'<line x1="{xx:.1f}" y1="{plot_top_b+ph_b}" x2="{xx:.1f}" y2="{plot_top_b+ph_b+5}" stroke="#0f172a" stroke-width="1"/>'
        )
        svg_b_elements.append(
            f'<text x="{xx:.1f}" y="{plot_top_b+ph_b+18}" text-anchor="middle" font-size="11" fill="#0f172a">{xv}</text>'
        )

    # Axes border
    svg_b_elements.append(
        f'<line x1="{pad_l}" y1="{plot_top_b}" x2="{pad_l}" y2="{plot_top_b+ph_b}" stroke="#0f172a" stroke-width="1.4"/>'
    )
    svg_b_elements.append(
        f'<line x1="{pad_l}" y1="{plot_top_b+ph_b}" x2="{W_b-pad_r}" y2="{plot_top_b+ph_b}" stroke="#0f172a" stroke-width="1.4"/>'
    )

    svg_b_elements.append(
        f'<text x="{pad_l + pw_b/2:.1f}" y="{H_b-10}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">Comment Length in Whitespace-Delimited Words [Clipped at 220]</text>'
    )
    svg_b_elements.append(
        f'<text x="15" y="{plot_top_b + ph_b/2:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a" transform="rotate(-90 15 {plot_top_b + ph_b/2:.1f})">Empirical Probability Density</text>'
    )

    body = f"""
    <div style="display: flex; gap: 22px; align-items: flex-start; background: #ffffff;">
      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 10px 6px 10px;">
        <div style="text-align: center; font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (a) Additive Lexical Weak-Score Distribution <i>s</i>(<i>x</i>) by Gold Class
        </div>
        <svg width="{W_a}" height="{H_a}" viewBox="0 0 {W_a} {H_a}">
          {''.join(svg_a_elements)}
        </svg>
      </div>
      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 10px 6px 10px;">
        <div style="text-align: center; font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (b) Comment Word-Length Distribution &amp; Marginal IQR Boxplots (<i>n</i> = 9,134)
        </div>
        <svg width="{W_b}" height="{H_b}" viewBox="0 0 {W_b} {H_b}">
          {''.join(svg_b_elements)}
        </svg>
      </div>
    </div>
    """
    return write_html("fig2.html", 1180, 445, body)


# ==============================================================================
# FIGURE 3: Inductive 806-d Hybrid Architecture & Top BERTopic Clusters
# ==============================================================================
def build_fig3():
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

    # Build Panel (b) SVG horizontal bar chart with Wilson 95% error bars
    W_b, H_b = 620, 415
    pad_l, pad_r, pad_t, pad_b = 258, 48, 24, 48
    pw_b = W_b - pad_l - pad_r
    ph_b = H_b - pad_t - pad_b
    max_x = 42.0

    def sx(val):
        return pad_l + (val / max_x) * pw_b

    svg_b = []
    # Vertical grid lines
    for xv in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
        xx = sx(xv)
        svg_b.append(
            f'<line x1="{xx:.1f}" y1="{pad_t}" x2="{xx:.1f}" y2="{pad_t+ph_b}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        svg_b.append(
            f'<text x="{xx:.1f}" y="{pad_t+ph_b+17}" text-anchor="middle" font-size="11" fill="#0f172a">{xv}%</text>'
        )

    row_h = ph_b / len(topics_data)
    for idx, (tid, tdesc, n_cnt, pos_pct, col) in enumerate(topics_data):
        cy = pad_t + (idx + 0.5) * row_h
        bh = row_h * 0.62
        bw = sx(pos_pct) - pad_l
        # Binomial SE in pct
        p = pos_pct / 100.0
        se_pct = 100.0 * math.sqrt(p * (1 - p) / n_cnt)
        ci_lo = max(0.0, pos_pct - 1.96 * se_pct)
        ci_hi = min(41.5, pos_pct + 1.96 * se_pct)

        # Left labels
        svg_b.append(
            f'<text x="{pad_l-8}" y="{cy-2:.1f}" text-anchor="end" font-size="11.2" font-weight="700" fill="#0f172a">{tid}: {tdesc}</text>'
            f'<text x="{pad_l-8}" y="{cy+11:.1f}" text-anchor="end" font-family="Helvetica Neue, sans-serif" font-size="9.5" fill="#475569">Support: n = {n_cnt} ({n_cnt/6392*100:.1f}% of Train)</text>'
        )
        # Bar starting strictly at 0.0%
        svg_b.append(
            f'<rect x="{pad_l}" y="{cy - bh/2:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" fill="{col}" fill-opacity="0.86" stroke="#0f172a" stroke-width="0.8"/>'
        )
        # 95% Binomial CI whisker
        svg_b.append(
            f'<line x1="{sx(ci_lo):.1f}" y1="{cy:.1f}" x2="{sx(ci_hi):.1f}" y2="{cy:.1f}" stroke="#0f172a" stroke-width="1.4"/>'
            f'<line x1="{sx(ci_lo):.1f}" y1="{cy-4:.1f}" x2="{sx(ci_lo):.1f}" y2="{cy+4:.1f}" stroke="#0f172a" stroke-width="1.4"/>'
            f'<line x1="{sx(ci_hi):.1f}" y1="{cy-4:.1f}" x2="{sx(ci_hi):.1f}" y2="{cy+4:.1f}" stroke="#0f172a" stroke-width="1.4"/>'
        )
        # Value label (placed cleanly inside bar in white, or right of whisker if clear of 19.74% line)
        lbl_x = sx(ci_hi) + 6
        if 15.5 <= ci_hi <= 19.5:
            lbl_x = sx(19.74) + 6
        svg_b.append(
            f'<text x="{lbl_x:.1f}" y="{cy+4:.1f}" font-family="Helvetica Neue, sans-serif" font-size="10.2" font-weight="700" fill="#0f172a">{pos_pct:.1f}%</text>'
        )

    # Train base rate line (19.74%)
    x_base = sx(19.74)
    svg_b.append(
        f'<line x1="{x_base:.1f}" y1="{pad_t-4}" x2="{x_base:.1f}" y2="{pad_t+ph_b}" stroke="#0f172a" stroke-width="2" stroke-dasharray="5,3"/>'
        f'<rect x="{x_base+6:.1f}" y="{pad_t+ph_b-32}" width="175" height="22" rx="3" fill="#ffffff" stroke="#64748b" stroke-width="0.9"/>'
        f'<text x="{x_base+12:.1f}" y="{pad_t+ph_b-17}" font-family="Helvetica Neue, sans-serif" font-size="10" font-weight="700" fill="#0f172a">Train Base Rate: &pi; = 19.74%</text>'
    )

    # Axes
    svg_b.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph_b}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<line x1="{pad_l}" y1="{pad_t+ph_b}" x2="{W_b-pad_r}" y2="{pad_t+ph_b}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<text x="{pad_l + pw_b/2 - 18:.1f}" y="{H_b-8}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#0f172a">Empirical Gold Positive Rate (% Depression in Train, x-axis starts at 0%)</text>'
    )

    body = f"""
    <div style="display: flex; gap: 18px; align-items: stretch; background: #ffffff;">
      <!-- Panel (a): Crisp Neural Architecture Schematic -->
      <div style="width: 530px; border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px 14px; display:flex; flex-direction:column; justify-content:space-between;">
        <div style="text-align: center; font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 8px;">
          (a) Strictly Inductive PhoBERT–BERTopic Hybrid Architecture (<i>z</i>(<i>x</i>) &isin; &#8477;<sup>806</sup>)
        </div>

        <!-- Step 1: Input -->
        <div style="border: 1.5px solid #334155; border-radius: 6px; background: #f8fafc; padding: 8px 12px; text-align:center;">
          <div class="sans" style="font-size: 11.5px; font-weight: 700; color: #0f172a;">
            Input Comment <i>x</i> (NFC Normalized &amp; <span class="mono">underthesea</span> Word-Segmented)
          </div>
          <div style="font-size: 11px; color: #475569; margin-top: 2px;">
            Strictly Inductive Protocol: 0% Exposure to <span class="mono">final_val</span>, <span class="mono">final_test</span>, or UIT-VSMEC During Topic Fitting
          </div>
        </div>

        <!-- Split Arrows -->
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

        <!-- Step 2: Parallel Dual Branches -->
        <div style="display:flex; gap: 12px;">
          <!-- Left Branch: PhoBERT -->
          <div style="flex: 1; border: 1.5px solid #1e40af; border-radius: 6px; background: #eff6ff; padding: 8px 10px;">
            <div class="sans" style="font-size: 11.5px; font-weight: 700; color: #1e3a8a; border-bottom: 1px solid #bfdbfe; padding-bottom: 4px; margin-bottom: 5px;">
              Branch 1: Contextual Encoder
            </div>
            <div style="font-size: 11.2px; line-height: 1.42; color: #1e293b;">
              <div>&bull; <b>Model:</b> Fine-tuned <span class="mono">vinai/phobert-base</span></div>
              <div>&bull; <b>State:</b> Encoder weights frozen per seed (<i>s</i> &isin; {{42, 123, 2024}})</div>
              <div>&bull; <b>Pooling:</b> Top-layer <span class="mono">[CLS]</span> token</div>
              <div class="mono" style="margin-top: 5px; background: #dbeafe; padding: 3px 6px; border-radius: 3px; font-size: 10.5px; font-weight: 700; color: #1e3a8a; text-align:center;">
                h(x) &isin; &#8477;<sup>768</sup>
              </div>
            </div>
          </div>

          <!-- Right Branch: Inductive BERTopic -->
          <div style="flex: 1.08; border: 1.5px solid #d97706; border-radius: 6px; background: #fffbeb; padding: 8px 10px;">
            <div class="sans" style="font-size: 11.5px; font-weight: 700; color: #92400e; border-bottom: 1px solid #fde68a; padding-bottom: 4px; margin-bottom: 5px;">
              Branch 2: Train-Only BERTopic
            </div>
            <div style="font-size: 11.2px; line-height: 1.42; color: #1e293b;">
              <div>&bull; <b>Fit Scope:</b> Strictly <span class="mono">final_train.csv</span> (<i>n</i> = 6,392)</div>
              <div>&bull; <b>Clusters:</b> <i>K</i> = 36 proper topics + 1 outlier</div>
              <div>&bull; <b>Outlier Rule:</b> Topic &minus;1 &rarr; <b>Slot 0</b>; Topics 0..35 &rarr; <b>Slots 1..36</b></div>
              <div class="mono" style="margin-top: 5px; background: #fef3c7; padding: 3px 6px; border-radius: 3px; font-size: 10.5px; font-weight: 700; color: #92400e; text-align:center;">
                OneHot(t(x)) &isin; {{0,1}}<sup>37</sup> + q(x) &isin; [0,1]
              </div>
            </div>
          </div>
        </div>

        <!-- Merge Arrows -->
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

        <!-- Step 3: Explicit 806-d Tensor Block Diagram -->
        <div style="border: 1.5px solid #0f766e; border-radius: 6px; background: #f0fdfa; padding: 8px 10px;">
          <div class="sans" style="font-size: 11.5px; font-weight: 700; color: #115e59; text-align:center; margin-bottom: 6px;">
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

        <!-- Down Arrow -->
        <div style="display:flex; justify-content:center; align-items:center; height: 22px;">
          <svg width="40" height="22" viewBox="0 0 40 22">
            <line x1="20" y1="0" x2="20" y2="14" stroke="#991b1b" stroke-width="2"/>
            <polygon points="15,12 20,21 25,12" fill="#991b1b"/>
          </svg>
        </div>

        <!-- Step 4: Classification Head -->
        <div style="border: 1.5px solid #991b1b; border-radius: 6px; background: #fff1f2; padding: 8px 12px; text-align:center;">
          <div class="sans" style="font-size: 11.5px; font-weight: 700; color: #991b1b;">
            StandardScaler &rarr; Regularized Classifier Head (<i>w</i><sub>1</sub> = 4.0650, Selected on <span class="mono">final_val.csv</span>)
          </div>
          <div class="mono" style="font-size: 10.5px; color: #7f1d1d; font-weight: 600; margin-top: 3px;">
            3-Seed Majority Vote Macro-F1 = 0.7302 [0.695, 0.758] | Soft Vote = 0.7261
          </div>
        </div>

      </div>

      <!-- Panel (b): Empirical Positive Rates across Inductive BERTopic Clusters -->
      <div style=" border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px 12px 6px 12px;">
        <div style="text-align: center; font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (b) Positive-Class Rate (&plusmn;95% CI) Across Top Inductive BERTopic Clusters (<i>n</i> = 6,392)
        </div>
        <svg width="{W_b}" height="{H_b}" viewBox="0 0 {W_b} {H_b}">
          {''.join(svg_b)}
        </svg>
      </div>
    </div>
    """
    return write_html("fig3.html", 1210, 470, body)


# ==============================================================================
# FIGURE 4: In-Domain PR Curves & 2,000-Bootstrap 95% CI Forest Plot
# ==============================================================================
def build_fig4():
    vis_data = json.loads((MODEL_DIR / "visobert_and_ablations_results.json").read_text())
    curves = vis_data["curves_in_domain"]
    y_true = np.array(curves["y_test"], dtype=int)
    pho_probs = np.array(curves["phobert_soft_prob"], dtype=float)
    lr_probs = np.array(curves["tfidf_logreg_prob"], dtype=float)
    lex_scores = np.array(curves["lexicon_prob"], dtype=float)

    hyb_df = pd.read_csv(BASE_DIR / "results" / "phobert_bertopic_predictions_in_domain.csv")
    hyb_probs = hyb_df["prob_depression"].values

    # Panel (a): Precision-Recall Curves SVG
    W_a, H_a = 515, 420
    pad_l, pad_r, pad_t, pad_b = 54, 18, 22, 48
    pw_a = W_a - pad_l - pad_r
    ph_a = H_a - pad_t - pad_b

    def sx_a(r):
        return pad_l + r * pw_a

    def sy_a(p):
        return pad_t + ph_a - p * ph_a

    svg_a = []
    # Grid lines
    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        yy = sy_a(v)
        xx = sx_a(v)
        svg_a.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W_a-pad_r}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="0.9"/>'
            f'<text x="{pad_l-7}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#334155">{v:.1f}</text>'
            f'<line x1="{xx:.1f}" y1="{pad_t}" x2="{xx:.1f}" y2="{pad_t+ph_a}" stroke="#e2e8f0" stroke-width="0.9"/>'
            f'<text x="{xx:.1f}" y="{pad_t+ph_a+17}" text-anchor="middle" font-size="11" fill="#334155">{v:.1f}</text>'
        )

    # Iso-F1 contours (F1 = 0.3, 0.4, 0.5, 0.6, 0.7)
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
            r_end = 0.96
            p_end = (f1_iso * r_end) / (2 * r_end - f1_iso)
            if p_end <= 0.95:
                svg_a.append(
                    f'<text x="{sx_a(r_end)-2:.1f}" y="{sy_a(p_end)-3:.1f}" text-anchor="end" font-family="Helvetica Neue, sans-serif" font-size="8.5" fill="#94a3b8">F1={f1_iso:.1f}</text>'
                )

    # No-skill baseline line (prevalence = 0.1977)
    y_prev = sy_a(0.1977)
    svg_a.append(
        f'<line x1="{pad_l}" y1="{y_prev:.1f}" x2="{W_a-pad_r}" y2="{y_prev:.1f}" stroke="#475569" stroke-width="1.5" stroke-dasharray="4,3"/>'
        f'<text x="{sx_a(0.56):.1f}" y="{y_prev-5:.1f}" font-family="Helvetica Neue, sans-serif" font-size="9.5" font-weight="600" fill="#475569">No-Skill Baseline (&pi; = 0.1977)</text>'
    )

    def pr_polyline(y_t, scores, color, dash=""):
        prec, rec, _ = precision_recall_curve(y_t, scores)
        # Subsample to ~180 points for crisp SVG
        idx = np.linspace(0, len(prec) - 1, min(220, len(prec))).astype(int)
        pts = [f"{sx_a(rec[i]):.1f},{sy_a(prec[i]):.1f}" for i in idx]
        dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.3" {dash_attr}/>'

    svg_a.append(pr_polyline(y_true, lex_scores, "#64748b", "3,3"))
    svg_a.append(pr_polyline(y_true, lr_probs, "#d97706", "5,2,1.5,2"))
    svg_a.append(pr_polyline(y_true, hyb_probs, "#0d9488", "7,2.5"))
    svg_a.append(pr_polyline(y_true, pho_probs, "#dc2626", ""))

    # Operating points at tau=0.50
    op_points = [
        (0.6531, 0.5145, "#dc2626"),  # PhoBERT Maj Vote
        (0.4982, 0.6400, "#0d9488"),  # Hybrid Maj Vote
        (0.6716, 0.4840, "#d97706"),  # TF-IDF + LogReg
        (0.5351, 0.4090, "#7c3aed"),  # BiLSTM Weighted
        (0.4502, 0.4207, "#64748b"),  # Weak Labeler
    ]
    for r_op, p_op, col in op_points:
        svg_a.append(
            f'<circle cx="{sx_a(r_op):.1f}" cy="{sy_a(p_op):.1f}" r="4.2" fill="{col}" stroke="#ffffff" stroke-width="1.3"/>'
        )

    # Legend in top-right clear space (Recall 0.38..0.99, Precision 0.71..0.99) - zero overlap with any curve or dot!
    lx, ly = sx_a(0.39), sy_a(0.985)
    svg_a.append(
        f'<rect x="{lx:.1f}" y="{ly:.1f}" width="268" height="104" rx="4" fill="#ffffff" fill-opacity="0.96" stroke="#94a3b8" stroke-width="1"/>'
        f'<line x1="{lx+8:.1f}" y1="{ly+15:.1f}" x2="{lx+30:.1f}" y2="{ly+15:.1f}" stroke="#dc2626" stroke-width="2.5"/>'
        f'<circle cx="{lx+19:.1f}" cy="{ly+15:.1f}" r="3.5" fill="#dc2626"/>'
        f'<text x="{lx+36:.1f}" y="{ly+19:.1f}" font-size="10.2" font-weight="700" fill="#0f172a">PhoBERT-base (PR-AUC = 0.6208)</text>'
        f'<line x1="{lx+8:.1f}" y1="{ly+34:.1f}" x2="{lx+30:.1f}" y2="{ly+34:.1f}" stroke="#0d9488" stroke-width="2.5" stroke-dasharray="6,2"/>'
        f'<circle cx="{lx+19:.1f}" cy="{ly+34:.1f}" r="3.5" fill="#0d9488"/>'
        f'<text x="{lx+36:.1f}" y="{ly+38:.1f}" font-size="10.2" font-weight="700" fill="#0f172a">PhoBERT + BERTopic (PR-AUC = 0.6111)</text>'
        f'<line x1="{lx+8:.1f}" y1="{ly+53:.1f}" x2="{lx+30:.1f}" y2="{ly+53:.1f}" stroke="#d97706" stroke-width="2.5" stroke-dasharray="5,2,1.5,2"/>'
        f'<circle cx="{lx+19:.1f}" cy="{ly+53:.1f}" r="3.5" fill="#d97706"/>'
        f'<text x="{lx+36:.1f}" y="{ly+57:.1f}" font-size="10.2" fill="#0f172a">TF-IDF + LogReg (PR-AUC = 0.5954)</text>'
        f'<circle cx="{lx+19:.1f}" cy="{ly+72:.1f}" r="4.0" fill="#7c3aed"/>'
        f'<text x="{lx+36:.1f}" y="{ly+76:.1f}" font-size="10.2" fill="#0f172a">BiLSTM Class-Weighted Op. Pt. (PR-AUC = 0.5074)</text>'
        f'<line x1="{lx+8:.1f}" y1="{ly+91:.1f}" x2="{lx+30:.1f}" y2="{ly+91:.1f}" stroke="#64748b" stroke-width="2.2" stroke-dasharray="3,3"/>'
        f'<circle cx="{lx+19:.1f}" cy="{ly+91:.1f}" r="3.5" fill="#64748b"/>'
        f'<text x="{lx+36:.1f}" y="{ly+95:.1f}" font-size="10.2" fill="#0f172a">Lexicon Weak-Labeler (PR-AUC = 0.3321)</text>'
    )

    # Axes border & labels
    svg_a.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph_a}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<line x1="{pad_l}" y1="{pad_t+ph_a}" x2="{W_a-pad_r}" y2="{pad_t+ph_a}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<text x="{pad_l + pw_a/2:.1f}" y="{H_a-8}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">Positive-Class Recall (Depression, n = 271)</text>'
        f'<text x="15" y="{pad_t + ph_a/2:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a" transform="rotate(-90 15 {pad_t + ph_a/2:.1f})">Positive-Class Precision (y-axis starts at 0.0)</text>'
    )

    # Panel (b): Forest Plot with Aligned Statistical Table Columns
    forest_models = [
        ("PhoBERT + BERTopic (Maj. Vote)", 0.7302, 0.6950, 0.7578, "0.5602", "0.8184", "p = 0.674", "#0d9488"),
        ("PhoBERT-base (Maj. Vote)", 0.7265, 0.6977, 0.7550, "0.5756", "0.8356", "Reference", "#dc2626"),
        ("PhoBERT-base (Soft Vote)", 0.7263, 0.6985, 0.7543, "0.5761", "0.8356", "p = 0.942", "#dc2626"),
        ("PhoBERT + BERTopic (Soft Vote)", 0.7261, 0.6912, 0.7541, "0.5527", "0.8184", "p = 0.968", "#0d9488"),
        ("TF-IDF + LogReg (L2, C = 1.0)", 0.7138, 0.6840, 0.7412, "0.5626", "0.8297", "p = 0.306", "#d97706"),
        ("TF-IDF + LinearSVC (C = 1.0)", 0.7025, 0.6721, 0.7327, "0.5289", "0.7927", "p = 0.068", "#ea580c"),
        ("BiLSTM Class-Weighted (Maj. Vote)", 0.6507, 0.6219, 0.6794, "0.4639", "0.7671", "p < 0.0001*", "#7c3aed"),
        ("Lexical Weak Labeler (All 1,371)", 0.6448, 0.6153, 0.6753, "0.4349", "0.6546", "p < 0.0001*", "#475569"),
        ("BiLSTM Unweighted (Maj. Vote)", 0.6421, 0.6097, 0.6736, "0.3868", "0.7787", "p < 0.0001*", "#8b5cf6"),
        ("Inductive BERTopic-Only (74-d)", 0.5539, 0.5210, 0.5865, "0.3730", "0.6475", "p < 0.0001*", "#334155"),
    ]

    W_b, H_b = 695, 420
    fl_pad = 198
    fr_table = 258
    fp_t, fp_b = 34, 48
    fp_w = W_b - fl_pad - fr_table
    fp_h = H_b - fp_t - fp_b
    xmin_b, xmax_b = 0.42, 0.80

    def sx_b(v):
        return fl_pad + (v - xmin_b) / (xmax_b - xmin_b) * fp_w

    svg_b = []
    # Shaded top-tier CI band (PhoBERT 95% CI: 0.6977 to 0.7550)
    svg_b.append(
        f'<rect x="{sx_b(0.6977):.1f}" y="{fp_t}" width="{sx_b(0.7550)-sx_b(0.6977):.1f}" height="{fp_h}" fill="#fee2e2" opacity="0.45"/>'
    )

    # Column Headers above plot & table
    svg_b.append(
        f'<line x1="6" y1="{fp_t-4}" x2="{W_b-6}" y2="{fp_t-4}" stroke="#0f172a" stroke-width="1.5"/>'
        f'<text x="10" y="{fp_t-10}" font-family="Helvetica Neue, sans-serif" font-size="10" font-weight="700" fill="#0f172a">Evaluated Configuration</text>'
        f'<text x="{fl_pad + fp_w/2:.1f}" y="{fp_t-10}" text-anchor="middle" font-family="Helvetica Neue, sans-serif" font-size="10" font-weight="700" fill="#0f172a">Macro-F1 &amp; 95% Bootstrap CI</text>'
        f'<text x="{W_b-fr_table+6}" y="{fp_t-10}" font-family="Helvetica Neue, sans-serif" font-size="10" font-weight="700" fill="#0f172a">Macro-F1 [95% CI]</text>'
        f'<text x="{W_b-112}" y="{fp_t-10}" font-family="Helvetica Neue, sans-serif" font-size="10" font-weight="700" fill="#0f172a">ROC</text>'
        f'<text x="{W_b-68}" y="{fp_t-10}" font-family="Helvetica Neue, sans-serif" font-size="10" font-weight="700" fill="#0f172a">vs PhoBERT</text>'
    )

    # Grid lines
    for xv in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        xx = sx_b(xv)
        svg_b.append(
            f'<line x1="{xx:.1f}" y1="{fp_t}" x2="{xx:.1f}" y2="{fp_t+fp_h}" stroke="#e2e8f0" stroke-width="0.9"/>'
            f'<text x="{xx:.1f}" y="{fp_t+fp_h+17}" text-anchor="middle" font-size="10.8" fill="#334155">{xv:.2f}</text>'
        )

    # Trivial all-negative baseline at 0.4452
    x_triv = sx_b(0.4452)
    svg_b.append(
        f'<line x1="{x_triv:.1f}" y1="{fp_t}" x2="{x_triv:.1f}" y2="{fp_t+fp_h}" stroke="#64748b" stroke-width="1.6" stroke-dasharray="4,3"/>'
        f'<text x="{x_triv+4:.1f}" y="{fp_t+fp_h-8}" font-family="Helvetica Neue, sans-serif" font-size="8.8" font-weight="600" fill="#475569">All-Neg (0.4452)</text>'
    )

    row_hb = fp_h / len(forest_models)
    for i, (mname, mean_v, lo_v, hi_v, f1_dep, roc_v, p_str, col) in enumerate(forest_models):
        cy = fp_t + (i + 0.5) * row_hb
        if i % 2 == 1:
            svg_b.append(
                f'<rect x="6" y="{cy - row_hb/2:.1f}" width="{W_b-12}" height="{row_hb:.1f}" fill="#f8fafc" opacity="0.7"/>'
            )
        # Left model name
        fw = "700" if i < 2 else "400"
        svg_b.append(
            f'<text x="{fl_pad-8}" y="{cy+4:.1f}" text-anchor="end" font-size="10.8" font-weight="{fw}" fill="#0f172a">{mname}</text>'
        )
        # CI line
        svg_b.append(
            f'<line x1="{sx_b(lo_v):.1f}" y1="{cy:.1f}" x2="{sx_b(hi_v):.1f}" y2="{cy:.1f}" stroke="{col}" stroke-width="2.4"/>'
            f'<line x1="{sx_b(lo_v):.1f}" y1="{cy-4.5:.1f}" x2="{sx_b(lo_v):.1f}" y2="{cy+4.5:.1f}" stroke="{col}" stroke-width="1.8"/>'
            f'<line x1="{sx_b(hi_v):.1f}" y1="{cy-4.5:.1f}" x2="{sx_b(hi_v):.1f}" y2="{cy+4.5:.1f}" stroke="{col}" stroke-width="1.8"/>'
            f'<circle cx="{sx_b(mean_v):.1f}" cy="{cy:.1f}" r="4.5" fill="{col}" stroke="#ffffff" stroke-width="1.2"/>'
        )
        # Right aligned tabular numbers
        p_col = "#991b1b" if "*" in p_str else "#334155"
        svg_b.append(
            f'<text x="{W_b-fr_table+6}" y="{cy+4:.1f}" font-family="SFMono-Regular, Menlo, monospace" font-size="9.6" font-weight="{fw}" fill="#0f172a">{mean_v:.4f} [{lo_v:.3f}, {hi_v:.3f}]</text>'
            f'<text x="{W_b-112}" y="{cy+4:.1f}" font-family="SFMono-Regular, Menlo, monospace" font-size="9.6" fill="#334155">{roc_v}</text>'
            f'<text x="{W_b-68}" y="{cy+4:.1f}" font-family="Helvetica Neue, sans-serif" font-size="9.3" font-weight="600" fill="{p_col}">{p_str}</text>'
        )

    svg_b.append(
        f'<line x1="{fl_pad}" y1="{fp_t}" x2="{fl_pad}" y2="{fp_t+fp_h}" stroke="#0f172a" stroke-width="1.3"/>'
        f'<line x1="{fl_pad}" y1="{fp_t+fp_h}" x2="{W_b-fr_table}" y2="{fp_t+fp_h}" stroke="#0f172a" stroke-width="1.3"/>'
        f'<text x="{W_b/2:.1f}" y="{H_b-8}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#0f172a">In-Domain Macro-F1 (2,000-Iteration Bootstrap 95% CIs; *Paired Bootstrap p &lt; 0.0001)</text>'
    )

    body = f"""
    <div style="display: flex; gap: 16px; align-items: stretch; background: #ffffff;">
      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 10px 6px 10px;">
        <div style="text-align: center; font-size: 13.8px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (a) In-Domain Precision–Recall Curves &amp; Iso-<i>F</i><sub>1</sub> Contours (<i>n</i> = 1,371)
        </div>
        <svg width="{W_a}" height="{H_a}" viewBox="0 0 {W_a} {H_a}">
          {''.join(svg_a)}
        </svg>
      </div>
      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 10px 6px 10px;">
        <div style="text-align: center; font-size: 13.8px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (b) In-Domain Macro-F1 Forest Plot with 95% Bootstrap CIs &amp; Significance Tests
        </div>
        <svg width="{W_b}" height="{H_b}" viewBox="0 0 {W_b} {H_b}">
          {''.join(svg_b)}
        </svg>
      </div>
    </div>
    """
    return write_html("fig4.html", 1285, 475, body)


# ==============================================================================
# FIGURE 5: Reconciled Confusion Matrices with Marginal Precision/Recall Strips
# ==============================================================================
def build_fig5():
    matrices = [
        {
            "title": "(a) PhoBERT-base In-Domain (Majority Vote)",
            "subtitle": "Macro-F1 = 0.7265 | Acc = 80.96% | F1-Dep = 0.5756 | Pred Pos = 344",
            "tn": 933, "fp": 167, "fn": 94, "tp": 177,
            "row0": "True Normal (0)", "row1": "True Depression (1)",
            "col0": "Pred Normal (0)", "col1": "Pred Depression (1)",
            "accent": "#1e3a8a", "light_tp": "#dbeafe", "dark_tn": "#1e40af",
        },
        {
            "title": "(b) TF-IDF + Logistic Regression In-Domain",
            "subtitle": "Macro-F1 = 0.7138 | Acc = 79.36% | F1-Dep = 0.5626 | Pred Pos = 376",
            "tn": 906, "fp": 194, "fn": 89, "tp": 182,
            "row0": "True Normal (0)", "row1": "True Depression (1)",
            "col0": "Pred Normal (0)", "col1": "Pred Depression (1)",
            "accent": "#9a3412", "light_tp": "#ffedd5", "dark_tn": "#c2410c",
        },
        {
            "title": "(c) PhoBERT-base External (Uncalibrated \u03c4 = 0.50)",
            "subtitle": "Full UIT-VSMEC (n = 3,084): Macro-F1 = 0.3620 | ROC-AUC = 0.6993 | Pred Pos = 53",
            "tn": 1532, "fp": 10, "fn": 1499, "tp": 43,
            "row0": "True Enjoyment (0)", "row1": "True Sad/Fear (1)",
            "col0": "Pred Enjoyment (0)", "col1": "Pred Sad/Fear (1)",
            "accent": "#115e59", "light_tp": "#ccfbf1", "dark_tn": "#0f766e",
        },
        {
            "title": "(d) PhoBERT-base External (Calibrated \u03c4 = 0.05)",
            "subtitle": "Sadness-Only (n = 2,691): Macro-F1 = 0.6161 | ROC-AUC = 0.7078 | Rec-Pos = 75.8%",
            "tn": 788, "fp": 754, "fn": 278, "tp": 871,
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
          <div class="mono" style="font-size: 10px; font-weight: 600; color: {m['accent']}; text-align:center; margin: 2px 0 8px 0;">
            {m['subtitle']}
          </div>

          <!-- 3x3 Diagnostic Confusion Matrix Table (with Row Recall & Col Precision Marginals) -->
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
              <!-- Row 0: True Negative Class -->
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
              <!-- Row 1: True Positive Class -->
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
              <!-- Bottom Marginal Row: Column Precision -->
              <tr>
                <td class="sans" style="padding: 6px; border: 1px solid #cbd5e1; background: #e2e8f0; font-weight: 700; font-size: 10px; color: #0f172a;">
                  Col Precision
                </td>
                <td class="mono" style="padding: 6px; border: 1px solid #cbd5e1; background: #f8fafc; font-size: 10.8px; font-weight: 700; color: #0f172a;">
                  {prec0:.1f}% <span style="font-size:9.2px; font-weight:400; color:#475569;">(NPV, n={c0_tot:,})</span>
                </td>
                <td class="mono" style="padding: 6px; border: 1px solid #cbd5e1; background: #f8fafc; font-size: 10.8px; font-weight: 700; color: {m['accent']};">
                  {prec1:.1f}% <span style="font-size:9.2px; font-weight:400; color:#475569;">(PPV, n={c1_tot:,})</span>
                </td>
                <td class="mono" style="padding: 6px; border: 1.5px solid #0f172a; background: #0f172a; font-size: 10.8px; font-weight: 700; color: #ffffff;">
                  Acc: {acc:.1f}%
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
    return write_html("fig5.html", 1180, 515, body)


# ==============================================================================
# FIGURE 6: Class-Weighting Ablation & External Prior-Shift Calibration
# ==============================================================================
def build_fig6():
    eval_data = json.loads((MODEL_DIR / "evaluation_results.json").read_text())
    sweep = eval_data["threshold_calibration_analysis"]["phobert_soft_vote"]["threshold_sweep"]

    # Panel (a): Grouped Bar Chart of BOTH Macro-F1 and Positive-Class F1 (y-axis starts at 0.0)
    W_a, H_a = 545, 405
    pad_l, pad_r, pad_t, pad_b = 52, 16, 42, 54
    pw_a = W_a - pad_l - pad_r
    ph_a = H_a - pad_t - pad_b
    max_y_a = 0.88

    def sy_a(v):
        return pad_t + ph_a - (v / max_y_a) * ph_a

    ablation_groups = [
        ("PhoBERT-base (Maj. Vote)", 0.7177, 0.7265, "+0.0088", 0.5430, 0.5756, "+0.0326"),
        ("TF-IDF + LogReg (L2)", 0.6579, 0.7138, "+0.0559", 0.4086, 0.5626, "+0.1540"),
        ("BiLSTM (Maj. Vote)", 0.6421, 0.6507, "+0.0086", 0.3868, 0.4639, "+0.0771"),
    ]

    svg_a = []
    # Grid lines starting strictly at 0.0
    for yv in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        yy = sy_a(yv)
        svg_a.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W_a-pad_r}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="0.9"/>'
            f'<text x="{pad_l-7}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#334155">{yv:.1f}</text>'
        )

    # Top Legend bar
    svg_a.append(
        f'<rect x="{pad_l+4}" y="4" width="{pw_a-8}" height="28" rx="4" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.9"/>'
        f'<rect x="{pad_l+12}" y="12" width="12" height="11" fill="#94a3b8" stroke="#334155" stroke-width="0.9"/>'
        f'<text x="{pad_l+28}" y="21" font-size="10.2" fill="#0f172a">Macro-F1 (Unw.)</text>'
        f'<rect x="{pad_l+118}" y="12" width="12" height="11" fill="#1d4ed8" stroke="#1e3a8a" stroke-width="0.9"/>'
        f'<text x="{pad_l+134}" y="21" font-size="10.2" font-weight="700" fill="#1e3a8a">Macro-F1 (Weighted)</text>'
        f'<rect x="{pad_l+252}" y="12" width="12" height="11" fill="#fca5a5" stroke="#991b1b" stroke-width="0.9"/>'
        f'<text x="{pad_l+268}" y="21" font-size="10.2" fill="#0f172a">F1-Dep (Unw.)</text>'
        f'<rect x="{pad_l+350}" y="12" width="12" height="11" fill="#dc2626" stroke="#7f1d1d" stroke-width="0.9"/>'
        f'<text x="{pad_l+366}" y="21" font-size="10.2" font-weight="700" fill="#991b1b">F1-Dep (Weighted)</text>'
    )

    grp_w = pw_a / len(ablation_groups)
    bw = 30.0
    for idx, (gname, m_unw, m_w, m_diff, d_unw, d_w, d_diff) in enumerate(ablation_groups):
        gcx = pad_l + (idx + 0.5) * grp_w
        x_m_unw = gcx - 2 * bw - 3
        x_m_w = gcx - bw - 3
        x_d_unw = gcx + 3
        x_d_w = gcx + bw + 3

        bars = [
            (x_m_unw, m_unw, "#94a3b8", "#334155", f"{m_unw:.4f}", False),
            (x_m_w, m_w, "#1d4ed8", "#1e3a8a", f"{m_w:.4f}", True),
            (x_d_unw, d_unw, "#fca5a5", "#991b1b", f"{d_unw:.4f}", False),
            (x_d_w, d_w, "#dc2626", "#7f1d1d", f"{d_w:.4f}", True),
        ]
        for bx, bval, bfill, bstroke, blbl, is_w in bars:
            by = sy_a(bval)
            bh = sy_a(0.0) - by
            svg_a.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{bfill}" stroke="{bstroke}" stroke-width="1"/>'
            )
            fw = "700" if is_w else "400"
            dx = 2.0 if is_w else -2.0
            svg_a.append(
                f'<text x="{bx + bw/2 + dx:.1f}" y="{by-5:.1f}" text-anchor="middle" font-family="SFMono-Regular, Menlo, monospace" font-size="8.4" font-weight="{fw}" fill="#0f172a">{blbl}</text>'
            )

        # Delta pills above each pair
        y_m_top = min(sy_a(m_unw), sy_a(m_w)) - 22
        y_d_top = min(sy_a(d_unw), sy_a(d_w)) - 22
        svg_a.append(
            f'<rect x="{x_m_unw+4:.1f}" y="{y_m_top-11:.1f}" width="{2*bw-8:.1f}" height="14" rx="3" fill="#dbeafe" stroke="#1e40af" stroke-width="0.8"/>'
            f'<text x="{x_m_unw+bw:.1f}" y="{y_m_top-1:.1f}" text-anchor="middle" font-family="Helvetica Neue, sans-serif" font-size="9" font-weight="700" fill="#1e3a8a">{m_diff}</text>'
            f'<rect x="{x_d_unw+4:.1f}" y="{y_d_top-11:.1f}" width="{2*bw-8:.1f}" height="14" rx="3" fill="#fee2e2" stroke="#991b1b" stroke-width="0.8"/>'
            f'<text x="{x_d_unw+bw:.1f}" y="{y_d_top-1:.1f}" text-anchor="middle" font-family="Helvetica Neue, sans-serif" font-size="9" font-weight="700" fill="#991b1b">{d_diff}</text>'
        )

        # X-axis group label
        svg_a.append(
            f'<text x="{gcx:.1f}" y="{pad_t+ph_a+20}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0f172a">{gname}</text>'
        )

    svg_a.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph_a}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<line x1="{pad_l}" y1="{pad_t+ph_a}" x2="{W_a-pad_r}" y2="{pad_t+ph_a}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<text x="{pad_l + pw_a/2:.1f}" y="{H_a-8}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#0f172a">Model Family on In-Domain Test (n = 1,371; w₁ = 4.0650 vs. w₁ = 1.0)</text>'
        f'<text x="15" y="{pad_t + ph_a/2:.1f}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#0f172a" transform="rotate(-90 15 {pad_t + ph_a/2:.1f})">Score on In-Domain Test (y-axis starts at 0.0)</text>'
    )

    # Panel (b): PhoBERT Prior-Shift Threshold Calibration Across Domains (y-axis starts at 0.0)
    W_b, H_b = 595, 405
    pb_l, pb_r, pb_t, pb_b = 52, 20, 26, 54
    pw_b = W_b - pb_l - pb_r
    ph_b = H_b - pb_t - pb_b
    max_y_b = 0.85

    def sx_b(t):
        return pb_l + (t - 0.03) / (0.57 - 0.03) * pw_b

    def sy_b(v):
        return pb_t + ph_b - (v / max_y_b) * ph_b

    thrs = []
    in_f1, full_f1, sad_f1 = [], [], []
    for k, v in sorted(sweep.items(), key=lambda item: item[1]["threshold"]):
        tv = float(v["threshold"])
        if tv <= 0.56:
            thrs.append(tv)
            in_f1.append(v["in_domain_test"]["f1_macro"])
            full_f1.append(v["vsmec_full"]["f1_macro"])
            sad_f1.append(v["vsmec_sadness_only"]["f1_macro"])

    svg_b = []
    for yv in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        yy = sy_b(yv)
        svg_b.append(
            f'<line x1="{pb_l}" y1="{yy:.1f}" x2="{W_b-pb_r}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="0.9"/>'
            f'<text x="{pb_l-7}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#334155">{yv:.1f}</text>'
        )
    for xv in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        xx = sx_b(xv)
        svg_b.append(
            f'<line x1="{xx:.1f}" y1="{pb_t}" x2="{xx:.1f}" y2="{pb_t+ph_b}" stroke="#e2e8f0" stroke-width="0.9"/>'
            f'<text x="{xx:.1f}" y="{pb_t+ph_b+18}" text-anchor="middle" font-size="10.8" fill="#334155">{xv:.2f}</text>'
        )

    # Vertical lines for Bayes Prior Threshold (0.1974) and Default (0.50)
    xb_bayes = sx_b(0.1974)
    xb_def = sx_b(0.50)
    svg_b.append(
        f'<line x1="{xb_bayes:.1f}" y1="{pb_t}" x2="{xb_bayes:.1f}" y2="{pb_t+ph_b}" stroke="#0f766e" stroke-width="2" stroke-dasharray="4,3"/>'
        f'<line x1="{xb_def:.1f}" y1="{pb_t}" x2="{xb_def:.1f}" y2="{pb_t+ph_b}" stroke="#475569" stroke-width="2" stroke-dasharray="5,3"/>'
    )

    def curve_pts(xs, ys):
        return " ".join(f"{sx_b(x):.1f},{sy_b(y):.1f}" for x, y in zip(xs, ys))

    svg_b.append(
        f'<polyline points="{curve_pts(thrs, in_f1)}" fill="none" stroke="#1d4ed8" stroke-width="2.5"/>'
        f'<polyline points="{curve_pts(thrs, sad_f1)}" fill="none" stroke="#0f766e" stroke-width="2.5"/>'
        f'<polyline points="{curve_pts(thrs, full_f1)}" fill="none" stroke="#dc2626" stroke-width="2.3" stroke-dasharray="6,3"/>'
    )
    for t, y_in, y_sad, y_fl in zip(thrs, in_f1, sad_f1, full_f1):
        svg_b.append(
            f'<circle cx="{sx_b(t):.1f}" cy="{sy_b(y_in):.1f}" r="3.8" fill="#1d4ed8" stroke="#ffffff" stroke-width="1"/>'
            f'<rect x="{sx_b(t)-3.5:.1f}" y="{sy_b(y_sad)-3.5:.1f}" width="7" height="7" fill="#0f766e" stroke="#ffffff" stroke-width="1"/>'
            f'<polygon points="{sx_b(t):.1f},{sy_b(y_fl)-4.5:.1f} {sx_b(t)-4.2:.1f},{sy_b(y_fl)+3.8:.1f} {sx_b(t)+4.2:.1f},{sy_b(y_fl)+3.8:.1f}" fill="#dc2626" stroke="#ffffff" stroke-width="0.9"/>'
        )

    # Callout badges on Panel (b)
    # 1. High-sensitivity tau=0.05
    svg_b.append(
        f'<rect x="{sx_b(0.05)+8:.1f}" y="{sy_b(0.6250)-24:.1f}" width="172" height="20" rx="3" fill="#fef2f2" stroke="#dc2626" stroke-width="1"/>'
        f'<text x="{sx_b(0.05)+13:.1f}" y="{sy_b(0.6250)-11:.1f}" font-family="Helvetica Neue, sans-serif" font-size="9.5" font-weight="700" fill="#991b1b">&tau;=0.05: F1=0.6250 (Full) / 0.6161 (Sad)</text>'
    )
    # 2. Bayes threshold tau=0.1974
    svg_b.append(
        f'<rect x="{xb_bayes+6:.1f}" y="{sy_b(0.4677)-26:.1f}" width="196" height="20" rx="3" fill="#f0fdfa" stroke="#0f766e" stroke-width="1"/>'
        f'<text x="{xb_bayes+11:.1f}" y="{sy_b(0.4677)-13:.1f}" font-family="Helvetica Neue, sans-serif" font-size="9.5" font-weight="700" fill="#115e59">&tau;_Bayes=0.1974: F1=0.4677 (Sad) / 0.4346</text>'
    )
    # 3. Default tau=0.50
    svg_b.append(
        f'<rect x="{xb_def-178:.1f}" y="{sy_b(0.3620)+10:.1f}" width="172" height="20" rx="3" fill="#f8fafc" stroke="#475569" stroke-width="1"/>'
        f'<text x="{xb_def-173:.1f}" y="{sy_b(0.3620)+23:.1f}" font-family="Helvetica Neue, sans-serif" font-size="9.5" font-weight="700" fill="#334155">Default &tau;=0.50: F1=0.3978 / 0.3620</text>'
    )

    # Legend in bottom-left clear region (y = 0.03..0.28)
    lx_b, ly_b = pb_l + 44, sy_b(0.27)
    svg_b.append(
        f'<rect x="{lx_b}" y="{ly_b}" width="315" height="72" rx="4" fill="#ffffff" fill-opacity="0.96" stroke="#94a3b8" stroke-width="1"/>'
        f'<line x1="{lx_b+8}" y1="{ly_b+15}" x2="{lx_b+30}" y2="{ly_b+15}" stroke="#1d4ed8" stroke-width="2.5"/>'
        f'<circle cx="{lx_b+19}" cy="{ly_b+15}" r="3.5" fill="#1d4ed8"/>'
        f'<text x="{lx_b+36}" y="{ly_b+19}" font-size="10.3" font-weight="700" fill="#0f172a">In-Domain YouTube Test (n = 1,371, &pi; = 19.8%, ROC = 0.8356)</text>'
        f'<line x1="{lx_b+8}" y1="{ly_b+36}" x2="{lx_b+30}" y2="{ly_b+36}" stroke="#0f766e" stroke-width="2.5"/>'
        f'<rect x="{lx_b+15.5}" y="{ly_b+32.5}" width="7" height="7" fill="#0f766e"/>'
        f'<text x="{lx_b+36}" y="{ly_b+40}" font-size="10.3" font-weight="700" fill="#0f172a">UIT-VSMEC Sadness-Only (n = 2,691, &pi; = 42.7%, ROC = 0.7078)</text>'
        f'<line x1="{lx_b+8}" y1="{ly_b+57}" x2="{lx_b+30}" y2="{ly_b+57}" stroke="#dc2626" stroke-width="2.3" stroke-dasharray="5,2.5"/>'
        f'<polygon points="{lx_b+19},{ly_b+52.5} {lx_b+15},{ly_b+60.5} {lx_b+23},{ly_b+60.5}" fill="#dc2626"/>'
        f'<text x="{lx_b+36}" y="{ly_b+61}" font-size="10.3" fill="#0f172a">UIT-VSMEC Full Sadness+Fear (n = 3,084, &pi; = 50.0%, ROC = 0.6993)</text>'
    )

    svg_b.append(
        f'<line x1="{pb_l}" y1="{pb_t}" x2="{pb_l}" y2="{pb_t+ph_b}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<line x1="{pb_l}" y1="{pb_t+ph_b}" x2="{W_b-pb_r}" y2="{pb_t+ph_b}" stroke="#0f172a" stroke-width="1.4"/>'
        f'<text x="{pb_l + pw_b/2:.1f}" y="{H_b-8}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#0f172a">Classification Decision Threshold &tau; on P(y = 1 | x)</text>'
        f'<text x="15" y="{pb_t + ph_b/2:.1f}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#0f172a" transform="rotate(-90 15 {pb_t + ph_b/2:.1f})">Macro-F1 Score (y-axis starts at 0.0)</text>'
    )

    body = f"""
    <div style="display: flex; gap: 18px; align-items: stretch; background: #ffffff;">
      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 10px 6px 10px;">
        <div style="text-align: center; font-size: 13.8px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (a) Class-Weighting Ablation Across Model Families (<i>n</i> = 1,371)
        </div>
        <svg width="{W_a}" height="{H_a}" viewBox="0 0 {W_a} {H_a}">
          {''.join(svg_a)}
        </svg>
      </div>
      <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 10px 6px 10px;">
        <div style="text-align: center; font-size: 13.8px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
          (b) PhoBERT-base Prior-Shift Threshold Calibration Across Domains
        </div>
        <svg width="{W_b}" height="{H_b}" viewBox="0 0 {W_b} {H_b}">
          {''.join(svg_b)}
        </svg>
      </div>
    </div>
    """
    return write_html("fig6.html", 1215, 460, body)


def main():
    p1 = build_fig1()
    p2 = build_fig2()
    p3 = build_fig3()
    p4 = build_fig4()
    p5 = build_fig5()
    p6 = build_fig6()
    for p in [p1, p2, p3, p4, p5, p6]:
        print("Generated HTML template:", p)


if __name__ == "__main__":
    main()
