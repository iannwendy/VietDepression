#!/usr/bin/env python3
"""
Deterministic, Zero-Raw-Text Quantitative Error Taxonomy Audit on final_test.csv (n = 1,371).

Computes mutually exclusive False Positive (FP) and False Negative (FN) category counts
and percentages aligned with the Four-Rule Annotation Rubric across representative
benchmark classifiers (ViSoBERT + LA-Focal, ViSoBERT, PhoBERT-base, XLM-RoBERTa-base,
and TF-IDF + Logistic Regression), saving only aggregate statistics to
results/scientific_audit/error_taxonomy_audit.json (100% DUA-compliant).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.vietdepression_lexicon import NORMAL_KEYWORDS, label_comment_detailed  # noqa: E402

SECOND_THIRD_RE = re.compile(
    r"\b(bạn|cậu|bác|anh|chị|mọi người|mn|ai đó|người ta|chúc|mong|cố lên|đừng buồn|vững tin|yêu thương|video|bài hát|nhạc|ad|admin|kênh)\b",
    flags=re.IGNORECASE,
)
FIRST_PERSON_RE = re.compile(
    r"\b(tôi|mình|em|con|tớ|tao|tui|bản thân)\b",
    flags=re.IGNORECASE,
)
POLYSEMY_IDIOM_RE = re.compile(
    r"(buồn ngủ|buon ngu|buồn cười|buon cuoi|chết cười|chet cuoi|cười chết|cuoi chet|mệt cười|đau bụng cười|buồn tè|mệt xỉu|xỉu up|thi cử|bài tập|deadline)",
    flags=re.IGNORECASE,
)


def main() -> None:
    test_path = BASE_DIR / "data" / "labeled" / "final_test.csv"
    df_te = pd.read_csv(test_path)
    y_true = df_te["label"].values.astype(int)
    texts = df_te["comment_text"].fillna("").astype(str).tolist()

    cache_npz = np.load(BASE_DIR / "results" / "scientific_audit" / "stage_arrays_cache.npz", allow_pickle=True)
    vis_json = json.loads((BASE_DIR / "models" / "round6_v2_retrained" / "visobert_and_ablations_results.json").read_text())
    lr_pred = (np.array(vis_json["curves_in_domain"]["tfidf_logreg_prob"]) >= 0.5).astype(int)

    models = [
        ("visobert_lafocal", "ViSoBERT + LA-Focal", cache_npz["lafocal_maj_pred_in"].astype(int)),
        ("visobert", "ViSoBERT", cache_npz["visobert_maj_pred_in"].astype(int)),
        ("ncnl_tkde23", "ViSoBERT + NCNL", cache_npz["ncnl_maj_pred_in"].astype(int)),
        ("phobert_base", "PhoBERT-base", cache_npz["phobert_maj_pred_in"].astype(int)),
        ("xlmr_base", "XLM-RoBERTa-base", cache_npz["xlmr_maj_pred_in"].astype(int)),
        ("tfidf_logreg", "TF-IDF + Logistic Regression", lr_pred),
    ]

    summary = {
        "n_test": int(len(df_te)),
        "gold_positives": int((y_true == 1).sum()),
        "gold_negatives": int((y_true == 0).sum()),
        "taxonomy_definitions": {
            "fp_rule1_supportive_or_non_first_person": "Supportive replies, anti-crisis encouragement, or second/third-person/media commentary (Rule 1 & Rule 4)",
            "fp_rule2_polysemy_idiom_or_mixed_cue": "Polysemous compounds ('buồn ngủ', 'chết cười'), hyperbolic idioms, or co-occurring appreciative cues (Rule 2)",
            "fp_rule3_transient_or_routine_affect": "Transient complaints, situational fatigue, or brief negative slang lacking persistent distress (Rule 3)",
            "fn_implicit_zero_or_negative_lexicon": "Implicit or figurative distress lacking net positive lexicon score (s(x) <= 0)",
            "fn_brief_or_pronoun_dropped": "Brief (<=25 words) or pronoun-dropped distress disclosures with s(x) > 0",
            "fn_long_narrative_dilution": "Extended multi-clause personal narratives (>25 words with first-person pronoun and s(x) > 0)",
        },
        "models": {},
    }

    for key, display_name, pred in models:
        fp_idx = np.where((y_true == 0) & (pred == 1))[0]
        fn_idx = np.where((y_true == 1) & (pred == 0))[0]

        fp_r1, fp_r2, fp_r3 = 0, 0, 0
        for i in fp_idx:
            t = texts[i]
            tl = t.lower()
            det = label_comment_detailed(t)
            has_norm = any(nk in tl for nk in NORMAL_KEYWORDS)
            if POLYSEMY_IDIOM_RE.search(t) or has_norm or det.has_mixed_signal:
                fp_r2 += 1
            elif det.has_review_context_phrase or SECOND_THIRD_RE.search(t):
                fp_r1 += 1
            else:
                fp_r3 += 1

        fn_c1, fn_c2, fn_c3 = 0, 0, 0
        for i in fn_idx:
            t = texts[i]
            det = label_comment_detailed(t)
            wlen = len(t.split())
            if det.depression_score <= 0:
                fn_c1 += 1
            elif (not FIRST_PERSON_RE.search(t)) or wlen <= 25:
                fn_c2 += 1
            else:
                fn_c3 += 1

        n_fp = int(len(fp_idx))
        n_fn = int(len(fn_idx))
        assert fp_r1 + fp_r2 + fp_r3 == n_fp
        assert fn_c1 + fn_c2 + fn_c3 == n_fn

        summary["models"][key] = {
            "display_name": display_name,
            "total_errors": n_fp + n_fn,
            "error_rate_pct": round((n_fp + n_fn) / len(df_te) * 100, 2),
            "fp_total": n_fp,
            "fp_categories": {
                "rule1_supportive_or_non_first_person": {"count": fp_r1, "pct": round(fp_r1 / n_fp * 100, 1)},
                "rule2_polysemy_idiom_or_mixed_cue": {"count": fp_r2, "pct": round(fp_r2 / n_fp * 100, 1)},
                "rule3_transient_or_routine_affect": {"count": fp_r3, "pct": round(fp_r3 / n_fp * 100, 1)},
            },
            "fn_total": n_fn,
            "fn_categories": {
                "implicit_zero_or_negative_lexicon": {"count": fn_c1, "pct": round(fn_c1 / n_fn * 100, 1)},
                "brief_or_pronoun_dropped": {"count": fn_c2, "pct": round(fn_c2 / n_fn * 100, 1)},
                "long_narrative_dilution": {"count": fn_c3, "pct": round(fn_c3 / n_fn * 100, 1)},
            },
        }

    out_path = BASE_DIR / "results" / "scientific_audit" / "error_taxonomy_audit.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved DUA-compliant error taxonomy audit to {out_path}")


if __name__ == "__main__":
    main()
