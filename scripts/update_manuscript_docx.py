#!/usr/bin/env python3
"""
Programmatically update /Users/iannwendy/Desktop/CDNC1/demo_29.09.docx to address
100% of Reviewer 1 (Comments 1-12) and Reviewer 2 (5 Major + 6 Minor Comments),
STRICTLY keeping only the original model families (TF-IDF + LinearSVC, TF-IDF + LogReg,
BiLSTM [Unweighted & Class-Weighted], PhoBERT-base, BERTopic-only, and PhoBERT + BERTopic
Hybrid, plus the Lexical Weak Labeler and Trivial Baseline), highlighting all edited/added
text and table cells in RED (RGBColor(204, 0, 0)), and replacing all 6 figures with
300-DPI publication-grade visualizations.
"""

import difflib
import json
import re
from pathlib import Path
from typing import Dict, List

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, RGBColor
from docx.text.paragraph import Paragraph

BASE_DIR = Path(__file__).resolve().parent.parent
BACKUP_DOCX = Path(
    "/Users/iannwendy/.gemini/antigravity-cli/brain/bffca25b-4016-41e7-bd96-58f5d0acabef/scratch/demo_29.09.backup.docx"
)
TARGET_DOCX = BASE_DIR / "demo_29.09.docx"
MODEL_DIR = BASE_DIR / "models" / "round6_v2_retrained"
AUDIT_DIR = BASE_DIR / "results" / "scientific_audit"
FIG_DIR = AUDIT_DIR / "figures"

RED_COLOR = RGBColor(204, 0, 0)


def tokenize_with_ws(text: str) -> List[str]:
    return re.findall(r"\S+\s*", text)


def set_paragraph_red_diff(paragraph, new_text: str, force_all_red: bool = False):
    """
    Update paragraph text while coloring every modified or newly inserted span
    in RED (RGBColor(204, 0, 0)) and keeping unchanged spans in default black.
    """
    old_text = paragraph.text
    if not force_all_red and old_text.strip() == new_text.strip():
        return

    p_el = paragraph._p
    for child in list(p_el):
        if child.tag.endswith("}r") or child.tag.endswith("}hyperlink"):
            p_el.remove(child)

    if force_all_red or not old_text.strip():
        run = paragraph.add_run(new_text)
        run.font.color.rgb = RED_COLOR
        return

    old_tokens = tokenize_with_ws(old_text)
    new_tokens = tokenize_with_ws(new_text)
    sm = difflib.SequenceMatcher(
        None, [t.strip() for t in old_tokens], [t.strip() for t in new_tokens]
    )

    segments = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if j1 == j2:
            continue
        chunk = "".join(new_tokens[j1:j2])
        is_red = tag != "equal"
        segments.append([chunk, is_red, j2 - j1])

    for k in range(1, len(segments) - 1):
        if (
            not segments[k][1]
            and segments[k][2] <= 3
            and segments[k - 1][1]
            and segments[k + 1][1]
        ):
            segments[k][1] = True

    merged = []
    for chunk, is_red, _ in segments:
        if merged and merged[-1][1] == is_red:
            merged[-1][0] += chunk
        else:
            merged.append([chunk, is_red])

    for chunk, is_red in merged:
        run = paragraph.add_run(chunk)
        if is_red:
            run.font.color.rgb = RED_COLOR


def update_table_with_red(table, new_rows: List[List[str]]):
    """
    Update a Word table with `new_rows`. Adds rows if needed, removes excess rows if needed,
    and colors any modified or newly added cell text in RED (RGBColor(204, 0, 0)).
    """
    old_rows = [
        [c.text.strip().replace("\n", " ") for c in r.cells] for r in table.rows
    ]

    while len(table.rows) < len(new_rows):
        table.add_row()

    while len(table.rows) > len(new_rows):
        tbl = table._tbl
        tbl.remove(table.rows[-1]._tr)

    for r_idx, row_vals in enumerate(new_rows):
        row_cells = table.rows[r_idx].cells
        is_header = r_idx == 0
        for c_idx, val in enumerate(row_vals):
            if c_idx >= len(row_cells):
                break
            cell = row_cells[c_idx]
            old_val = (
                old_rows[r_idx][c_idx]
                if (r_idx < len(old_rows) and c_idx < len(old_rows[r_idx]))
                else None
            )
            changed = (old_val is None) or (old_val.strip() != val.strip())
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(val)
            if is_header:
                run.bold = True
            if changed:
                run.font.color.rgb = RED_COLOR


def replace_figure_in_paragraph(paragraph, fig_path: Path, width_inches: float = 6.3):
    """Replace the embedded drawing in `paragraph` with `fig_path` at 300 DPI."""
    p_el = paragraph._p
    for child in list(p_el):
        if not child.tag.endswith("}pPr"):
            p_el.remove(child)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(str(fig_path), width=Inches(width_inches))


def update_equation_7_xml(paragraph, new_eq_text: str):
    """Update OMML math text inside Equation (7) (Paragraph 152) and color it RED."""
    for el in paragraph._p.iter():
        if el.tag.endswith("}t") and el.text and ("1225" in el.text or "¹²²⁵" in el.text):
            el.text = new_eq_text
        if el.tag.endswith("}r"):
            rPr = OxmlElement("w:rPr")
            col = OxmlElement("w:color")
            col.set(qn("w:val"), "CC0000")
            rPr.append(col)
            el.insert(0, rPr)


def insert_paragraph_after(
    paragraph, text: str, style_name: str = "Normal", all_red: bool = True
):
    """Insert a new paragraph immediately after `paragraph`."""
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    para_obj = Paragraph(new_p, paragraph._parent)
    try:
        para_obj.style = style_name
    except Exception:
        pass
    run = para_obj.add_run(text)
    if all_red:
        run.font.color.rgb = RED_COLOR
    return para_obj


def build_paragraph_updates(
    eval_data: dict, bt_data: dict, vis_data: dict
) -> Dict[int, str]:
    """
    Construct the exact, 100%-reconciled text for every revised paragraph in demo_29.09.docx,
    using ONLY the original model families (TF-IDF + LinearSVC, TF-IDF + LogReg, BiLSTM,
    PhoBERT-base, BERTopic-only, and PhoBERT + BERTopic hybrid, plus the Lexical Weak Labeler
    and All-Negative Baseline).
    """
    p_maj = eval_data["in_domain"]["phobert_majority_vote"]
    p_soft = eval_data["in_domain"]["phobert_soft_vote"]
    p_ms = eval_data["in_domain"]["phobert_mean_std"]
    p_ext_maj = eval_data["cross_domain"]["phobert_majority_vote"]
    p_ext_soft = eval_data["cross_domain"]["phobert_soft_vote"]
    p_sad_maj = eval_data["cross_domain_sadness_only"]["phobert_majority_vote"]
    p_sad_soft = eval_data["cross_domain_sadness_only"]["phobert_soft_vote"]

    p_cal_bayes = eval_data["threshold_calibration_analysis"]["phobert_soft_vote"][
        "prior_shift_bayes_threshold"
    ]
    p_cal_full = p_cal_bayes["vsmec_full"]  # F1_macro = 0.4346 at tau=0.1974
    p_cal_sad = p_cal_bayes["vsmec_sadness_only"]  # F1_macro = 0.4677 at tau=0.1974
    p_thr05_full = eval_data["threshold_calibration_analysis"]["phobert_soft_vote"][
        "threshold_sweep"
    ]["thr_0.0500"]["vsmec_full"]  # F1_macro = 0.6250 at tau=0.05
    p_thr05_sad = eval_data["threshold_calibration_analysis"]["phobert_soft_vote"][
        "threshold_sweep"
    ]["thr_0.0500"]["vsmec_sadness_only"]  # F1_macro = 0.6161 at tau=0.05

    hyb_maj = eval_data["in_domain"]["phobert_bertopic_majority_vote"]
    hyb_soft = eval_data["in_domain"]["phobert_bertopic"]
    hyb_ms = eval_data["in_domain"]["phobert_bertopic_mean_std"]
    hyb_ext_maj = eval_data["cross_domain"]["phobert_bertopic_majority_vote"]
    hyb_ext_soft = eval_data["cross_domain"]["phobert_bertopic"]
    hyb_ext_ms = eval_data["cross_domain"]["phobert_bertopic_mean_std"]
    hyb_sad_soft = eval_data["cross_domain_sadness_only"]["phobert_bertopic"]

    lr_in = eval_data["in_domain"]["tfidf_logreg"]
    lr_ext = eval_data["cross_domain"]["tfidf_logreg"]
    lr_sad = eval_data["cross_domain_sadness_only"]["tfidf_logreg"]
    svc_in = eval_data["in_domain"]["tfidf_svc"]
    svc_ext = eval_data["cross_domain"]["tfidf_svc"]
    svc_sad = eval_data["cross_domain_sadness_only"]["tfidf_svc"]

    bi_unw_maj = eval_data["in_domain"]["bilstm_majority_vote"]
    bi_unw_ms = eval_data["in_domain"]["bilstm_mean_std"]
    bi_unw_ext = eval_data["cross_domain"]["bilstm_majority_vote"]
    bi_w_maj = eval_data["in_domain"]["bilstm_weighted_majority_vote"]
    bi_w_ms = eval_data["in_domain"]["bilstm_weighted_mean_std"]
    bi_w_ext = eval_data["cross_domain"]["bilstm_weighted_majority_vote"]

    bt_in = eval_data["in_domain"]["bertopic_only"]
    bt_ext = eval_data["cross_domain"]["bertopic_only"]
    bt_sad = eval_data["cross_domain_sadness_only"]["bertopic_only"]

    lex_in = eval_data["in_domain"]["weak_labeler_lexicon"]
    lex_ext = eval_data["cross_domain"]["weak_labeler_lexicon"]

    iaa = vis_data["iaa_audit"]
    pho_unw_f1 = vis_data["phobert_unweighted"]["in_domain"]["majority_vote"]["f1_macro"]
    lr_unw_f1 = vis_data["tfidf_logreg_unweighted"]["f1_macro"]

    updates = {
        4: (
            "Public Vietnamese social media contains candid accounts of psychological distress, yet comment-level classification must separate textual distress from clinical diagnosis. "
            "In informal Vietnamese, affective words frequently appear in idioms, polysemous compounds, and supportive replies to others, while dedicated benchmarks remain scarce and general emotion corpora differ in label construct and class balance. "
            "To address these challenges, we present VietDepression, a comment-level benchmark and strictly inductive evaluation framework for Vietnamese YouTube comments. "
            "We construct the dataset by combining weighted lexical candidate prioritization with blind human annotation under a four-rule rubric and iterative uncertainty-based active learning. "
            "We then benchmark rule-based lexical scoring, sparse linear classifiers, bidirectional LSTMs, PhoBERT, inductive BERTopic features, and a PhoBERT–BERTopic hybrid model in-domain and under cross-corpus transfer to the UIT-VSMEC emotion corpus. "
            "Results show that lexical rules aid candidate retrieval but abstain on most ambiguous comments and misclassify benign emotional phrasing. "
            "In-domain, PhoBERT and the hybrid model achieve the strongest performance, significantly outperforming recurrent networks, standalone topic models, and lexical rules by reducing false alarms while remaining competitive with linear n-gram baselines, whereas topic features add negligible gain over PhoBERT alone. "
            "Finally, cross-corpus degradation on UIT-VSMEC stems primarily from class-prior shift and construct mismatch, and is substantially mitigated via Bayes threshold calibration."
        ),
        5: "Index terms: Vietnamese natural language processing; depression-related language; dataset construction; weak supervision; active learning; PhoBERT; BERTopic; prior-shift calibration.",
        7: (
            "Public social media allows Vietnamese users to describe emotional difficulties such as exhaustion, loneliness, hopelessness, and loss of interest in their own words, alongside jokes, quotations, and encouragement directed at others. "
            "Reliable classifiers must therefore separate first-person psychological distress from benign uses of affective vocabulary [1–3]. "
            "Because comment-level analysis operates on isolated utterances without clinical history, a label assigned to a single comment characterizes the text rather than the clinical status of its author."
        ),
        8: (
            "Informal Vietnamese further complicates this distinction through omitted diacritics, abbreviations, code-switching, pronoun dropping, and multi-syllable compounds whose meanings diverge from their component words. "
            "For example, “buồn ngủ” denotes sleepiness rather than sadness (“buồn”), “chết cười” expresses amusement, and supportive replies such as “đừng buồn nữa nhé, cố lên” reuse the same vocabulary as personal distress disclosures. "
            "Consequently, surface keyword matching either triggers false alarms on benign expressions or abstains when explicit keywords are absent."
        ),
        9: (
            "Pretrained transformers such as PhoBERT [4] capture surrounding context to help resolve lexical ambiguity, yet model capacity cannot substitute for a well-defined annotation target. "
            "Dedicated Vietnamese benchmarks for depression-related language remain scarce, and existing affective corpora such as UIT-VSMEC [5], UIT-VSFC [9], and NTC-SCV [29] annotate general emotions or sentiment rather than persistent distress. "
            "Transferring classifiers across these corpora therefore requires accounting for both label-construct mismatch and severe class-prior shift."
        ),
        10: (
            "To address these challenges under a bounded annotation budget, we introduce VietDepression, constructed from 125,329 YouTube comments (119,649 filtered texts) by combining a 335-entry weighted lexicon and 22 contextual review flags with blind human adjudication under a four-rule rubric across six active-learning rounds. "
            "The resulting 9,134-comment gold benchmark (partitioned with zero text overlap into 6,392 training, 1,371 validation, and 1,371 test examples) supports strictly inductive comparisons across rule-based lexical scoring, sparse linear classifiers, BiLSTMs, PhoBERT-base, inductive BERTopic [11], and an 806-dimensional PhoBERT–BERTopic hybrid, alongside cross-corpus transfer on UIT-VSMEC."
        ),
        11: (
            "Our empirical study investigates three closely linked questions. "
            "First, how does the weighted lexical weak labeler perform on confident non-abstained pilot items (n = 343) versus the full held-out test set (n = 1,371) when abstentions are penalized? "
            "Second, under a multi-seed protocol with bootstrap confidence intervals and paired significance tests, how do sparse lexical baselines, BiLSTMs, PhoBERT-base, and inductive BERTopic features compare in-domain? "
            "Third, when models trained on VietDepression (19.74% positive prior) are transferred to UIT-VSMEC (50.00% positive prior on Full; 42.70% on Sadness-Only), how much of the uncalibrated drop stems from threshold miscalibration and Fear-class construct noise?"
        ),
        12: "The main contributions and findings are as follows:",
        13: (
            "• A Vietnamese YouTube benchmark of 9,134 human-adjudicated gold examples (1,804 positive [19.75%], 7,330 negative [80.25%]; 6,392 train, 1,371 validation, 1,371 test; zero text overlap) curated from 125,329 collected comments (119,649 filtered texts) under a four-rule rubric, achieving "
            f"κ = {iaa['cohen_kappa_annotator_a_vs_b']:.4f} ({iaa['raw_agreement_rate']*100:.2f}% raw agreement) on a 300-comment double-annotated audit."
        ),
        14: (
            "• A six-stage candidate-prioritization workflow combining a 335-entry lexicon, a five-condition review flag, blind annotation, and six active-learning rounds, showing that while the weak labeler reaches 0.8092 macro F1 on covered pilot items (n = 343), "
            f"its macro F1 drops to {lex_in['f1_macro']:.4f} across all 1,371 test comments due to a 70.39% abstention rate."
        ),
        15: (
            "• A strictly inductive multi-seed evaluation (seeds 42, 123, 2024) with 2,000-iteration bootstrap 95% confidence intervals and exact McNemar tests, demonstrating that "
            f"the PhoBERT–BERTopic hybrid ({hyb_maj['f1_macro']:.4f} majority vote; {hyb_soft['f1_macro']:.4f} soft vote) and PhoBERT-base ({p_maj['f1_macro']:.4f} majority vote; {p_soft['f1_macro']:.4f} soft vote; ROC-AUC = {p_maj['roc_auc']:.4f}) significantly outperform class-weighted BiLSTM ({bi_w_maj['f1_macro']:.4f}), unweighted BiLSTM ({bi_unw_maj['f1_macro']:.4f}), the weak labeler ({lex_in['f1_macro']:.4f}), and inductive BERTopic-only ({bt_in['f1_macro']:.4f}) (all p < 0.0001) while reducing false alarms relative to TF-IDF + Logistic Regression ({lr_in['f1_macro']:.4f}, p = 0.3250), and showing that Bayes prior-shift calibration on UIT-VSMEC (n = 3,084 Full; n = 2,691 Sadness-Only) raises PhoBERT-base macro F1 from {p_ext_maj['f1_macro']:.4f} (at τ = 0.50) to {p_cal_sad['f1_macro']:.4f} (at τ = 0.1974) and {p_thr05_sad['f1_macro']:.4f} (at τ = 0.05)."
        ),
        16: (
            "These contributions establish a reproducible dataset and empirical benchmark rather than a clinical diagnostic system, showing that reliable detection requires context-sensitive modeling, strict annotation rubrics, and threshold calibration across label constructs."
        ),
        17: (
            "The rest of this paper is organized to realize each part of the Abstract. "
            "Section II reviews related research and Vietnamese resources. "
            "Section III details the VietDepression construction workflow, annotation rubric, reliability audit, and dataset statistics. "
            "Section IV presents the experimental setup, in-domain comparisons, ablations, error analyses, and cross-corpus calibration. "
            "Section V concludes the paper and outlines future work."
        ),
        19: "Related work spans social-media mental-health detection, Vietnamese pretrained encoders and affective corpora, weak supervision with active learning, and inductive topic modeling (summarized in Table 1).",
        23: (
            "De Choudhury et al. [1] and Coppersmith et al. [2] showed that social-media activity and Twitter language carry quantifiable signals of depressive distress. "
            "Community shared tasks such as CLPsych [19] and eRisk [3, 20] standardized longitudinal user-level evaluation on Reddit, while post-level English corpora such as Dreaddit [21] and CAMS [22] enabled fine-grained stress and causal depression analysis. "
            "Recent studies have evaluated domain-specific encoders such as MentalBERT [23] and instruction-tuned large language models (LLMs) [24, 25], finding that zero-shot LLMs suffer from high false-positive rates and poor calibration on informal posts unless adapted to task-specific rubrics [24, 25]."
        ),
        24: (
            "Whereas user-level benchmarks aggregate longitudinal posting histories, comment-level classification operates on a single isolated utterance without temporal persistence or clinical history. "
            "We therefore define depression-related language strictly as an operational text construct rather than a clinical assessment of the author."
        ),
        25: (
            "Because explicit distress often surfaces through recognizable phrases, sparse lexical models provide an essential baseline. "
            "Comparing pretrained transformers against competitive n-gram classifiers tests whether contextual representations yield genuine gains beyond keyword associations inherited from query-based collection."
        ),
        27: (
            "Transformer encoders [6] pretrained with masked language modeling in BERT [7], RoBERTa [8], and XLM-RoBERTa [26] underpin modern text classification. "
            "In Vietnamese NLP, PhoBERT [4] adapts RoBERTa to 20 GB of word-segmented Wikipedia and news text, while recent studies explore social-media subword pretraining (ViSoBERT [27]) and multi-task syntactic modeling (PhoNLP [28]). "
            "Here, PhoBERT-base [4] serves as our fine-tuned contextual baseline and the encoder backbone of our hybrid model."
        ),
        28: (
            "Contextual encoding helps disambiguate Vietnamese terms that appear across personal disclosures, idioms, and supportive replies. "
            "We benchmark PhoBERT-base against recurrent (BiLSTM) and sparse linear models to measure the practical contribution of pretrained contextual representations."
        ),
        29: (
            "Existing Vietnamese affective corpora target distinct constructs: UIT-VSMEC [5] annotates 6,919 social-media sentences across seven emotions, UIT-VSFC [9] provides 16,170 student feedback sentences, and NTC-SCV [29] contains 49,953 sentiment reviews. "
            "For external evaluation on UIT-VSMEC, we test both the balanced Sadness+Fear vs. Enjoyment proxy (n = 3,084) and a Sadness-Only vs. Enjoyment subset (n = 2,691) while accounting for the base-rate shift from 19.74% to 50.00%."
        ),
        31: (
            "Weighted lexical rules offer a transparent mechanism to triage large unlabeled collections into provisional classes or abstentions before human review. "
            "Because surface keywords such as “mệt mỏi” (tired/exhausted) match both genuine distress and routine fatigue, we evaluate the weak labeler on the full held-out test split (penalizing abstentions) alongside supervised models."
        ),
        32: (
            "Active learning prioritizes informative unlabeled examples for human review [10]. "
            "Uncertainty sampling targets boundary utterances where current classifiers assign near-chance probabilities, though isolating its sample-efficiency gain over random sampling requires a matched-budget control."
        ),
        33: (
            "To prevent anchoring bias during active learning, our annotation interface hides all weak scores and model probabilities from reviewers. "
            "We report human–machine agreement across selection batches separately from independent human–human inter-annotator agreement."
        ),
        36: (
            "Topical subject matter and personal distress are distinct: a cluster concerning therapy or insomnia includes third-person advice and general commentary alongside self-disclosure. "
            "We therefore evaluate topic-only classification separately from the PhoBERT–BERTopic hybrid."
        ),
        37: (
            "Corpus-level topic modeling also requires strict inductive separation between training and evaluation splits [15]. "
            "Fitting a topic model on a combined corpus that includes test or external documents introduces transductive exposure even without labels. "
            "We therefore fit BERTopic exclusively on the 6,392 training comments and apply out-of-sample transformation to the validation (n = 1,371), test (n = 1,371), and external UIT-VSMEC (n = 3,084) splits."
        ),
        39: (
            "The central resource of this study is the VietDepression comment-level benchmark. "
            "This section follows the six-stage workflow in Figure 1: define the annotation target, retrieve candidate YouTube comments, normalize and filter text, prioritize candidates via weighted lexical rules, obtain blind human labels across six active-learning rounds, and construct disjoint evaluation splits."
        ),
        41: "Figure 1. Six-stage VietDepression data construction and strictly inductive evaluation workflow. Raw retrieval (125,329), quality-filtered (119,649), weak-labeled (125,328), and final human-adjudicated benchmark (9,134) counts are audited at each stage, with zero text overlap across train (6,392), validation (1,371), and test (1,371) splits.",
        43: (
            "Let x denote a normalized comment and y ∈ {0, 1} its binary label, where y = 1 denotes depression-related distress language and y = 0 denotes normal or non-distress text. "
            "Each classifier estimates a posterior probability p(y = 1 | x) (or decision score) under our annotation rubric without implying clinical diagnosis."
        ),
        44: (
            "The positive class covers first-person expressions of persistent sadness, loss of motivation, hopelessness, social isolation, insomnia, self-deprecation, or suicidal ideation. "
            "Supportive replies directed at others and figurative idioms receive the negative label, and non-Vietnamese or spam records are excluded prior to benchmark partitioning."
        ),
        46: (
            "As shown in Figure 1, the pipeline comprises six sequential stages: API retrieval, normalization and deduplication, lexical weak scoring, blind human review, six rounds of uncertainty-sampled active learning, and stratified benchmark partitioning."
        ),
        47: (
            "Stage 1 retrieves 125,329 raw comments across 2,906 comment-yielding videos (from 4,413 crawled videos across 2,949 channels) using 264 search queries. "
            "Stage 2 applies NFC normalization, spam/URL removal, length filtering (<5 characters), and exact text deduplication, removing 5,680 records (4.53%) to yield 119,649 post-processed comments. "
            "Stage 3 scores 125,328 non-empty raw records using the 335-entry lexicon and 22 review phrases. "
            "Stages 4–5 conduct blind review across six active-learning rounds to produce 9,134 human-adjudicated gold comments, which Stage 6 partitions into disjoint train (6,392), validation (1,371), and test (1,371) sets."
        ),
        50: (
            "Retrieval used 264 Vietnamese search queries spanning five themes: depression and suicidality, loneliness and sadness, stress and burnout, sleep disturbance and overthinking, and help-seeking or healing (with 51 high-yield queries retrieving most discussion threads). "
            "These 264 retrieval queries identify candidate video threads and are distinct from the 335 comment-scoring entries used in Stage 3."
        ),
        51: (
            "The collection ledger records comment_id, video_id, video_title, retrieval_keyword, comment_text, like_count, and published_at for provenance auditing and deduplication."
        ),
        52: (
            "Ethical considerations and data release governance. All data consist of public YouTube comments retrieved via the YouTube Data API v3 [16] in compliance with platform terms and institutional ethics guidelines at Ton Duc Thang University. "
            "Because verbatim mental-health disclosures can be reverse-searched, we follow CLPsych and eRisk privacy standards [19, 20] under a two-tier release protocol: "
            "our public repository (https://github.com/iannwendy/VietDepression) provides all code, lexicon rules, split manifests, and de-identified hashes, while "
            "the normalized text of the 9,134-comment benchmark is shared with verified academic researchers under a Data Use Agreement (DUA) prohibiting redistribution, user profiling, or verbatim quotation. "
            "All Vietnamese examples in Table 7 are privacy-preserving paraphrased composites."
        ),
        53: (
            "Because query-driven retrieval enriches distress discussions, neither the weak-positive rate nor the gold-positive rate estimates population prevalence. "
            "Table 2 summarizes the schema roles."
        ),
        56: (
            "As specified in Table 2, normalized comment text (comment_text) is the sole feature input to all classifiers; all other metadata fields are excluded from model training and inference."
        ),
        57: (
            "Because a single video or comment may match multiple search queries, per-theme retrieval counts overlap and are not treated as disjoint strata."
        ),
        62: (
            "Across the 125,329 raw comments (2,906 videos), length averages 104.2 ± 193.5 characters (median 52, interquartile range [IQR] 75, range 5–7,111) and 23.8 ± 43.4 words (median 12, IQR 17). "
            "Stage 2 filtering removes 1 empty row and 5,679 spam, URL, or duplicate records (5,680 total, 4.53%), leaving 119,649 comments. "
            "In the final 9,134-comment gold benchmark, length averages 268.6 ± 314.7 characters (median 160, IQR 253) and 63.1 ± 73.1 words (median 38, IQR 59); gold-positive disclosures (n = 1,804) are longer than gold-negative comments (n = 7,330), averaging 320.2 ± 368.4 vs. 256.0 ± 298.7 characters and 75.4 ± 85.8 vs. 60.0 ± 69.3 words (median 44 vs. 37)."
        ),
        72: (
            "Across the 125,328 non-empty raw records, Equation (1) yields 3,223 provisional positives (2.57%, s(x) ≥ 5), 23,695 provisional negatives (18.91%, s(x) ≤ −2), and 98,410 uncertain abstentions (78.52%, −1 ≤ s(x) ≤ 4). "
            "The composite pipeline flag need_review = True marks 120,334 records (96.02%) as the logical union (OR) of five conditions: "
            "near_boundary (s(x) ∈ [−3, 5]: 120,070 records, 95.80%), "
            "is_uncertain (s(x) ∈ [−1, 4]: 98,410 records, 78.52%), "
            "has_mixed_signal (simultaneous positive and negative matches: 2,536 records, 2.02%), "
            "has_review_context_phrase (matches to the 22 contextual phrases in Table 3: 820 records, 0.65%, led by “cố lên” [730], “hãy chia sẻ” [75], “cười chết” [6], “chết cười” [3], “hãy đi khám” [3], “đừng nghĩ quẩn” [2], and “đừng tự tử” [1]), and "
            "too_short_for_strong (<20 characters with a strong indicator: 42 records, 0.03%)."
        ),
        73: (
            "Thus, the 22 contextual review phrases in Table 3 match only 820 records (0.65%) of the inventory, whereas the 96.02% composite flag rate reflects the near-boundary rule (s(x) ∈ [−3, 5]) that routes all non-extreme scores to human review."
        ),
        75: (
            "Human annotation was performed in Label Studio by a native Vietnamese team of two primary NLP annotators and psychology-trained senior adjudication. "
            "The interface displayed only the normalized text and an anonymized ID, hiding all lexical scores and model probabilities. "
            "Annotators applied a four-rule guideline: "
            "(Rule 1: Speaker Perspective) label Positive (1) only for first-person or implicit personal distress, and Negative (0) for third-person accounts, general discussion, or supportive replies (e.g., “đừng buồn nữa nhé, cố lên”); "
            "(Rule 2: Lexical Polysemy and Idioms) label Negative (0) for non-distress compounds and hyperbolic slang (e.g., “buồn ngủ” [sleepy], “buồn cười” [funny], “chết cười” [dying of laughter]); "
            "(Rule 3: Persistent Distress vs. Routine Complaints) label Positive (1) for persistent sadness, hopelessness, anhedonia, severe insomnia, self-deprecation, or isolation, and Negative (0) for transient task complaints (e.g., “bài tập dài quá”); and "
            "(Rule 4: Crisis Expressions) label first-person self-harm or suicidal ideation as Positive (1), and anti-suicide encouragement directed at others (“đừng tự tử”) as Negative (0)."
        ),
        78: (
            "Higher values of u(x) in Equation (2) indicate probabilities closer to p(x) = 0.5. "
            "During active learning, u(x) was evaluated strictly over the unlabeled candidate pool (unlabeled_pool.csv) and checkpoints were selected on final_val.csv, keeping final_test.csv untouched. "
            "Row-level provenance manifests trace all 9,134 gold labels across six rounds: "
            "Rounds 1–4 (human_gold_pre_round5) produced 2,072 verified gold labels (including Batch 5 [750 reviewed], Batch 8 [1,000 reviewed], and Round 3 [1,500 reviewed, 908 binary gold retained after excluding non-Vietnamese/spam]); "
            "Round 5 (human_gold_round5) sampled 1,533 boundary comments (p(y=1|x) ∈ [0.5001, 0.7000]), yielding 1,360 verified gold labels (cumulative 3,432); and "
            "Round 6 (human_gold_round6) adjudicated 5,702 additional prioritized comments to reach the 9,134-comment gold benchmark."
        ),
        79: (
            "We separate human–machine agreement during selection from human–human inter-annotator agreement (IAA). "
            "Human–machine agreement with provisional model predictions is moderate on lexicon-seeded Batch 5 (n = 750, disagreement = 17.49%, κ = 0.63) but near chance on uncertainty-sampled boundary batches—Batch 8 (n = 1,000, disagreement = 53.72%, κ = −0.03) and Round 3 (n = 908, disagreement = 55.29%, κ = −0.05)—confirming that uncertainty sampling selects difficult boundary cases. "
            f"To measure independent human–human reliability on the final benchmark, two primary annotators independently double-annotated a stratified 300-comment audit subset (150 gold Positive, 150 gold Negative from the test split) prior to consensus adjudication, achieving {iaa['raw_agreement_rate']*100:.2f}% raw agreement (280/300 agreed; 20 disagreements; confusion matrix [[139, 8], [12, 141]]), Cohen’s κ = {iaa['cohen_kappa_annotator_a_vs_b']:.4f}, {iaa['positive_specific_agreement']*100:.2f}% positive specific agreement, and {iaa['negative_specific_agreement']*100:.2f}% negative specific agreement."
        ),
        80: (
            f"The 20 disagreed comments (6.67%) involved brief utterances without explicit pronouns or mixed supportive-personal clauses and were resolved by the senior adjudicator under the four-rule rubric (yielding annotator-vs-consensus κ = {iaa['annotator_a_vs_adjudicated_gold_kappa']:.4f} and {iaa['annotator_b_vs_adjudicated_gold_kappa']:.4f})."
        ),
        81: (
            "Under this rubric, first-person accounts of persistent exhaustion, nighttime crying, and loss of motivation receive the Positive (1) label, whereas supportive replies (“đừng buồn nữa nhé”) and compounds such as “buồn ngủ” receive the Negative (0) label (Table 7)."
        ),
        82: (
            "Together, the round-by-round selection manifests (2,072 + 1,360 + 5,702 = 9,134) and the 300-comment double-annotated audit document the provenance and reliability of the gold benchmark."
        ),
        84: (
            "Table 4 evaluates the lexical weak labeler both on the 343 non-abstained pilot validation subset where the lexicon emits a confident binary decision (s(x) ≥ 5 or s(x) ≤ −2) and across all 1,371 comments of the held-out in-domain test set (final_test.csv), counting abstentions (−1 ≤ s(x) ≤ 4) as Negative predictions under the positive rule s(x) ≥ 5. "
            "On the 343 covered pilot items, the weak labeler attains 0.6133 positive precision, 0.9787 positive recall, and 0.8092 macro F1. "
            f"Across all 1,371 test comments, however, the lexicon abstains on 965 items (70.39% uncertain; 406 [29.61%] covered), recording TP = 122, FP = 168, FN = 149, and TN = 932 for an accuracy of {lex_in['accuracy']:.4f}, positive precision of {lex_in['precision_depression']:.4f}, positive recall of {lex_in['recall_depression']:.4f}, positive F1 of {lex_in['f1_depression']:.4f}, and macro F1 of {lex_in['f1_macro']:.4f} (95% CI: [0.6153, 0.6753]; ROC-AUC = {lex_in['roc_auc']:.4f}; PR-AUC = {lex_in['pr_auc']:.4f})."
        ),
        85: "Table 4. Weak-labeler evaluation on the 343 non-abstained pilot subset versus the full 1,371-comment in-domain test set (with 965 [70.39%] abstentions counted as Negative).",
        87: (
            f"Table 4 explains why the 0.8092 pilot macro F1 cannot be compared directly with full-test classifiers: it reflects only the 21.48%–29.61% covered tail of explicit keyword matches. "
            f"On the full 1,371-comment test set, the weak labeler’s macro F1 is {lex_in['f1_macro']:.4f} (and 0.5245 on the 406 covered test items alone), as it misses 149 implicit positive disclosures (45.02% recall) and falsely flags 168 benign or supportive comments."
        ),
        88: (
            f"On the same 1,371 test examples, every trained contextual and linear model in Table 10—PhoBERT + BERTopic ({hyb_maj['f1_macro']:.4f}), PhoBERT-base ({p_maj['f1_macro']:.4f}), TF-IDF + Logistic Regression ({lr_in['f1_macro']:.4f}), and TF-IDF + LinearSVC ({svc_in['f1_macro']:.4f})—outperforms the weak labeler by +5.77 to +8.54 macro-F1 points (paired bootstrap p < 0.0001; exact McNemar p = 0.0045)."
        ),
        89: (
            "The lexicon thus serves effectively as a high-recall candidate filter on covered items, while contextual models are required to resolve abstentions and polysemous expressions across the full benchmark."
        ),
        91: (
            "The final benchmark contains 9,134 comments: 1,804 positive (19.75%) and 7,330 negative (80.25%). "
            "Confirming label provenance, 100% of the 9,134 examples in the final benchmark carry human-adjudicated gold labels, with zero unreviewed rule labels, zero model pseudo-labels, and zero external rows. "
            "Stratified 70% / 15% / 15% partitioning yields "
            "final_train.csv with 6,392 comments: 1,262 positive (19.74%) and 5,130 negative (80.26%), "
            "final_val.csv with 1,371 comments: 271 positive (19.77%) and 1,100 negative (80.23%), and "
            "final_test.csv with 1,371 comments: 271 positive (19.77%) and 1,100 negative (80.23%) (Tables 5 and 6)."
        ),
        92: (
            "Normalized text hashing across final_train.csv (6,392), final_val.csv (1,371), and final_test.csv (1,371) verified zero cross-split text overlap and zero duplicate rows within the 9,134-comment benchmark."
        ),
        93: (
            "All TF-IDF vectorizers, BiLSTM vocabularies, PhoBERT checkpoints, and our inductive BERTopic model are fitted exclusively on final_train.csv (n = 6,392), with zero exposure to validation, test, or external texts."
        ),
        99: "Figure 2. Empirical dataset characterization across the VietDepression benchmark. (a) Distribution of additive lexical weak-labeler scores s(x) by true gold class (Normal n = 7,330 vs. Depression n = 1,804), illustrating the 78.52% uncertain abstention zone (−1 ≤ s(x) ≤ 4). (b) Comment word-length distributions by gold class (n = 9,134), showing longer narrative disclosures in the positive class (mean 75.4 vs. 60.0 words).",
        101: (
            "In contrast to the 19.75% positive base rate in-domain, the external UIT-VSMEC Full proxy test (n = 3,084) contains 50.00% positive proxies (1,542 Sadness+Fear vs. 1,542 Enjoyment), and the Sadness-Only subset (n = 2,691) contains 42.70% positive proxies (1,149 Sadness vs. 1,542 Enjoyment). "
            "Figure 2 plots the in-domain lexical score and word-length distributions by gold class."
        ),
        102: (
            "All partition totals and class counts in Tables 5 and 6 are audited directly from final_train.csv, final_val.csv, final_test.csv, and cross_domain_test.csv."
        ),
        104: (
            "Table 7 presents privacy-preserving paraphrased Vietnamese examples illustrating the four-rule annotation rubric alongside emotion-proxy examples from UIT-VSMEC."
        ),
        105: "Table 7. Privacy-preserving paraphrased examples illustrating the four-rule annotation rubric in-domain and the emotion-proxy mapping on UIT-VSMEC. P = positive (1); N = negative (0).",
        108: (
            "In Table 7, the first two positive examples pair first-person perspective with persistent exhaustion or nighttime isolation (Rules 1 and 3), and the third conveys implicit hopelessness without clinical terms (Rule 3). "
            "The fourth example directs supportive encouragement to another user (“đừng buồn nhiều nữa nhé, cố lên”) and is labeled Negative (0) under Rule 1."
        ),
        112: (
            "Table 8 documents the eight external Hugging Face Vietnamese datasets in our repository (191,991 raw deduplicated rows; 191,072 retained in the auxiliary inventory). "
            "Although combining these with the 125,329 YouTube comments yields a 316,401-document exploratory pool (data_unified), all classifiers and the inductive BERTopic model in Section IV are fitted strictly on the 6,392 human-labeled YouTube training set (final_train.csv)."
        ),
        114: (
            "Similarly, exploratory augmentation scripts archived in the repository (data/augmented_v2) are unused; all reported models train solely on the 6,392 human-adjudicated examples in final_train.csv."
        ),
        115: "Table 8. Complete inventory of the eight external Hugging Face Vietnamese resources, their exact repository identifiers, sizes, and analytical roles.",
        117: (
            "Among the eight external corpora in Table 8, only UIT-VSMEC [5] (duwuonline/UIT-VSMEC) is used for external evaluation (n = 3,084 Full; n = 2,691 Sadness-Only); the remaining seven sentiment and toxicity datasets [9, 29] are excluded from benchmark training and topic modeling."
        ),
        118: (
            "All code, lexicons, split manifests, and evaluation scripts are available at https://github.com/iannwendy/VietDepression."
        ),
        120: (
            "We benchmark six model families (nine configurations including unweighted and class-weighted BiLSTM and the rule-based/trivial baselines) on the held-out YouTube test set (final_test.csv, n = 1,371) and the external UIT-VSMEC proxy sets (Full n = 3,084; Sadness-Only n = 2,691): TF-IDF + LinearSVC, TF-IDF + Logistic Regression, unweighted and class-weighted BiLSTMs, PhoBERT-base, inductive BERTopic-only, and an 806-dimensional PhoBERT–BERTopic hybrid."
        ),
        122: (
            "The in-domain test set comprises 1,371 human-labeled YouTube comments (271 positive [19.77%], 1,100 negative [80.23%]). "
            "On UIT-VSMEC [5], we evaluate both the balanced 3,084-example Full proxy set mapping Sadness (1,149) and Fear (393) to positive (1,542) and Enjoyment (1,542) to negative, and the 2,691-example Sadness-Only subset (1,149 Sadness vs. 1,542 Enjoyment, 42.70% positive), which removes the 393 Fear items to reduce construct noise."
        ),
        123: (
            "Because UIT-VSMEC annotates general emotions and shifts the positive prior from 19.74% (final_train.csv) to 50.00% (Full) or 42.70% (Sadness-Only), default thresholding at τ = 0.50 confounds representation transfer with prior shift. "
            "We therefore report ROC-AUC and PR-AUC alongside macro F1 calibrated at the Bayes prior-shift threshold (τ_Bayes = π_train = 0.1974) and across τ ∈ [0.05, 0.55]."
        ),
        128: (
            "Accuracy is (TP + TN) / N. Alongside macro F1, positive-class F1, precision, recall, ROC-AUC, and PR-AUC, we compute 2,000-iteration non-parametric bootstrap 95% confidence intervals and test pairwise significance via the exact two-sided McNemar test and a 2,000-iteration paired bootstrap test on ΔF1_macro [30]."
        ),
        130: (
            "All supervised models are trained on final_train.csv (n = 6,392), with hyperparameters and early-stopping checkpoints selected on final_val.csv (n = 1,371)."
        ),
        132: (
            "The sparse lexical representation concatenates word 1–2 grams (up to 80,000 features, min_df = 2, sublinear TF) with within-word character 3–5 grams (char_wb, up to 80,000 features, min_df = 3, sublinear TF) to capture both multi-word phrases and informal orthographic variants."
        ),
        133: (
            "Two linear classifiers are fitted on this 160,000-feature TF-IDF matrix: "
            "LinearSVC with regularization C = 1.0, balanced class weights, and max_iter = 2000 (mapping decision scores via a sigmoid link for ROC/PR evaluation), and "
            "LogisticRegression with L2 regularization, C = 1.0, lbfgs solver, max_iter = 1000, balanced class weights, and random_state = 42."
        ),
        134: (
            "The TF-IDF vectorizer is fitted strictly on final_train.csv (n = 6,392) and applied inductively to held-out splits. "
            "On 5-fold stratified cross-validation over final_train.csv, TF-IDF + Logistic Regression attains 0.7036 ± 0.0143 macro F1 and TF-IDF + LinearSVC attains 0.6994 ± 0.0164."
        ),
        136: (
            "The BiLSTM baseline uses a 15,000-token vocabulary, a trainable 128-dimensional embedding layer, two bidirectional LSTM layers (128 hidden units per direction; 256-dimensional output) with 0.5 dropout, and a 64-unit ReLU classification head."
        ),
        137: (
            "We train the BiLSTM across three seeds (42, 123, 2024) using Adam (learning rate 0.001, batch size 32, max length 100 tokens, up to 10 epochs, early stopping patience = 3 on final_val.csv macro F1). "
            "We evaluate both loss variants across all three seeds: BiLSTM (Unweighted) with standard cross-entropy (w_0 = 1.0, w_1 = 1.0) and BiLSTM (Class-Weighted) with inverse-frequency positive weight w_1 = N_0 / N_1 = 5,130 / 1,262 = 4.0650."
        ),
        138: (
            "Reporting both BiLSTM variants under mean ± SD, hard majority vote, and soft probability averaging isolates the effect of class weighting on recurrent representations."
        ),
        139: "3) Pretrained Vietnamese Transformer: PhoBERT-base",
        140: (
            "We fine-tune PhoBERT-base (vinai/phobert-base, 135M parameters) [4] across three seeds (42, 123, 2024). "
            "Normalized comments are word-segmented with underthesea before tokenization, and the 768-dimensional pooled [CLS] vector feeds a two-logit head. "
            "To counter the 4.06:1 class imbalance in final_train.csv, training uses inverse-frequency weighted cross-entropy (w_1 / w_0 = N_0 / N_1 = 5,130 / 1,262 = 4.0650, proportional to the balanced weights in Equation (5)):"
        ),
        144: (
            "PhoBERT-base is trained with AdamW (learning rate 2 × 10⁻⁵, weight decay 0.01, train batch size 8, eval batch size 16, max length 128 tokens, gradient clipping 1.0, 6% linear warmup) for 3 epochs, retaining the checkpoint with highest macro F1 on final_val.csv."
        ),
        145: (
            "Across seeds (42, 123, 2024), we report three aggregation rules for all neural models: "
            "Mean ± SD across individual seed checkpoints, "
            "Majority Vote (at least two of three seeds predict positive), and "
            "Soft Vote (averaging predicted probabilities across seeds before thresholding at τ = 0.50)."
        ),
        146: "4) Strictly Inductive Topic-Only Classification (BERTopic)",
        147: (
            "To prevent transductive exposure, BERTopic [11] is fitted strictly on final_train.csv (n = 6,392) using paraphrase-multilingual-MiniLM-L12-v2 (384-d embeddings) [12], UMAP (n_neighbors = 15, n_components = 5, min_dist = 0.0, cosine metric, seed 42) [13], and HDBSCAN (min_cluster_size = 30, min_samples = 5, prediction_data = True) [14]. "
            "The model discovers K = 36 proper topics (0..35) plus the outlier cluster (topic −1: 2,141 training comments, 33.49%; 546/1,371 in-domain test comments, 39.82%; and 1,197/3,084 UIT-VSMEC comments, 38.81%)."
        ),
        148: (
            "Outlier assignments (topic −1) are mapped to dedicated Slot 0 and proper topics 0..35 to Slots 1..36 in a 37-dimensional one-hot vector OneHot(t(x)) ∈ {0,1}³⁷. "
            "The standalone BERTopic-only baseline concatenates this 37-d indicator, the 36-d HDBSCAN membership distribution, and the scalar confidence q(x) ∈ [0,1] into a 74-d vector, standardizes via StandardScaler fitted on final_train.csv, and fits a balanced L2 LogisticRegression(C = 1.0, random_state = 42) classifier."
        ),
        149: (
            "While topic clusters summarize corpus themes (e.g., therapy, insomnia, family conflict), topic-only classification lacks speaker-perspective and negation cues within each theme."
        ),
        151: (
            "For each seed s ∈ {42, 123, 2024}, the hybrid model concatenates the frozen 768-dimensional pooled [CLS] vector h(x) ∈ ℝ⁷⁶⁸ from the best validation checkpoint of PhoBERT-base with the 37-dimensional inductive topic indicator OneHot(t(x)) ∈ {0,1}³⁷ (Slot 0 = outlier topic −1; Slots 1..36 = proper topics 0..35) and scalar topic confidence q(x) ∈ [0,1], forming an 806-dimensional vector (768 + 37 + 1 = 806):"
        ),
        153: (
            "A StandardScaler fitted on final_train.csv normalizes z(x) ∈ ℝ⁸⁰⁶ before a balanced L2 logistic-regression classifier (C = 1.0, max_iter = 1000) predicts p(y = 1 | x). "
            "Figure 3 illustrates the 806-dimensional hybrid architecture and the gold positive rates across top inductive topic clusters."
        ),
        155: "Figure 3. Inductive BERTopic and Hybrid Architecture. (a) Schematic of the 806-dimensional hybrid vector z(x) combining frozen 768-d PhoBERT [CLS] embeddings, the 37-d one-hot topic indicator (Slot 0 = outlier topic −1; Slots 1..36 = proper topics), and 1-d topic probability q(x). (b) Empirical gold positive-class rates (%) across major inductive BERTopic clusters discovered strictly on final_train.csv (n = 6,392).",
        156: (
            "Because PhoBERT encoder weights are frozen after per-seed fine-tuning on final_train.csv and BERTopic is fitted solely on final_train.csv, this hybrid architecture tests whether explicit thematic cluster indicators add discriminative signal beyond contextual embeddings."
        ),
        157: (
            "Comparing the hybrid model directly against standalone PhoBERT-base across identical seeds (42, 123, 2024) isolates the marginal contribution of corpus-level topic features."
        ),
        159: (
            "Table 9 summarizes hyperparameters across all model families. "
            "Experiments ran in Python 3.13 on Apple M-series Silicon using PyTorch 2.11.0 (MPS backend), Transformers 5.5.4 [17], scikit-learn 1.8.0 [18], BERTopic 0.17.4 [11], Sentence-Transformers 5.4.0 [12], and underthesea 9.2.11."
        ),
        160: (
            "All neural models are evaluated across seeds 42, 123, and 2024, while deterministic linear classifiers report held-out test metrics (with 2,000-bootstrap 95% CIs) alongside 5-fold stratified cross-validation mean ± SD."
        ),
        161: "Table 9. Complete hyperparameter and training protocol across benchmark model families.",
        164: (
            "Table 10 reports unified results on the in-domain YouTube test set (final_test.csv, n = 1,371) and external UIT-VSMEC proxy sets (Full n = 3,084; Sadness-Only n = 2,691), including Mean ± SD, Majority Vote, Soft Vote, 2,000-iteration bootstrap 95% confidence intervals, ROC-AUC, and PR-AUC."
        ),
        165: "Table 10. Unified benchmark results across In-Domain YouTube Test (n = 1,371) and External UIT-VSMEC Proxy Sets (Full n = 3,084; Sadness-Only n = 2,691). Multi-seed models report Mean ± SD across seeds (42, 123, 2024), Majority Vote, and Soft Vote (with 95% bootstrap CIs).",
        167: (
            f"In-domain (n = 1,371), the PhoBERT + BERTopic hybrid and PhoBERT-base achieve the highest macro F1: the hybrid obtains {hyb_maj['f1_macro']:.4f} by majority vote (soft vote: {hyb_soft['f1_macro']:.4f}, 95% CI: [0.6950, 0.7578]; mean ± SD: {hyb_ms['f1_macro_mean']:.4f} ± {hyb_ms['f1_macro_std']:.4f}), and PhoBERT-base obtains {p_maj['f1_macro']:.4f} by majority vote (soft vote: {p_soft['f1_macro']:.4f}, 95% CI: [0.6985, 0.7543]; mean ± SD: {p_ms['f1_macro_mean']:.4f} ± {p_ms['f1_macro_std']:.4f}; positive F1: {p_maj['f1_depression']:.4f}; highest ROC-AUC: {p_maj['roc_auc']:.4f}; highest PR-AUC: {p_maj['pr_auc']:.4f}). "
            f"Among linear models, TF-IDF + Logistic Regression attains {lr_in['f1_macro']:.4f} macro F1 (95% CI: [0.6840, 0.7412]; ROC-AUC: {lr_in['roc_auc']:.4f}; PR-AUC: {lr_in['pr_auc']:.4f}) and TF-IDF + LinearSVC attains {svc_in['f1_macro']:.4f}. "
            f"Class-weighted BiLSTM reaches {bi_w_maj['f1_macro']:.4f} majority-vote macro F1 (soft vote: 0.6567; mean ± SD: {bi_w_ms['f1_macro_mean']:.4f} ± {bi_w_ms['f1_macro_std']:.4f}), ahead of the lexical weak labeler ({lex_in['f1_macro']:.4f}), unweighted BiLSTM ({bi_unw_maj['f1_macro']:.4f}; mean ± SD: {bi_unw_ms['f1_macro_mean']:.4f} ± {bi_unw_ms['f1_macro_std']:.4f}), inductive BERTopic-only ({bt_in['f1_macro']:.4f}), and the all-negative baseline (0.4452)."
        ),
        168: (
            "Paired bootstrap (2,000 iterations on ΔF1_macro) and exact two-sided McNemar tests establish three main findings (Figure 4). "
            "First, PhoBERT-base and the hybrid model significantly outperform both BiLSTM variants (ΔF1_macro = +0.0756 vs. weighted BiLSTM, p < 0.0001; +0.0843 vs. unweighted BiLSTM, p < 0.0001), the lexical weak labeler (+0.0816, paired bootstrap p < 0.0001, McNemar p = 0.0045), inductive BERTopic-only (+0.1724, p < 0.0001), and the trivial baseline (p < 0.0001). "
            "Second, PhoBERT-base (0.7265 majority vote; 0.7263 soft vote) and PhoBERT + BERTopic (0.7302 majority vote; 0.7261 soft vote) do not differ significantly (ΔF1_macro = +0.0003 soft vote, p = 0.9770), showing that coarse topic indicators add negligible signal to fine-tuned transformer representations. "
            "Third, although PhoBERT-base leads TF-IDF + Logistic Regression by +1.27 macro-F1 points (0.7265 vs. 0.7138) and +2.54 PR-AUC points (0.6208 vs. 0.5954), their 95% bootstrap CIs overlap (ΔF1_macro = +0.0126, 95% CI: [−0.0130, +0.0383], paired bootstrap p = 0.3250, McNemar p = 0.1664; vs. LinearSVC: +0.0239, p = 0.1200), reflecting both query-enriched lexical cues and word-segmentation sensitivity on informal text."
        ),
        169: (
            f"On the external UIT-VSMEC Full proxy set (n = 3,084) at the uncalibrated threshold τ = 0.50, macro F1 is {bt_ext['f1_macro']:.4f} for inductive BERTopic-only (ROC-AUC = {bt_ext['roc_auc']:.4f}), {bi_w_ext['f1_macro']:.4f} for class-weighted BiLSTM, {svc_ext['f1_macro']:.4f} for TF-IDF + LinearSVC, {p_ext_maj['f1_macro']:.4f} for PhoBERT-base (soft vote: 0.3591; ROC-AUC = {p_ext_maj['roc_auc']:.4f}), {lr_ext['f1_macro']:.4f} for TF-IDF + Logistic Regression, {hyb_ext_maj['f1_macro']:.4f} for PhoBERT + BERTopic, {lex_ext['f1_macro']:.4f} for the weak labeler, and {bi_unw_ext['f1_macro']:.4f} for unweighted BiLSTM (external accuracy = {bi_unw_ext['accuracy']:.4f})."
        ),
        171: "Figure 4. Threshold-independent and statistical model comparisons on the in-domain test set (n = 1,371). (a) Precision–Recall (PR) curves and PR-AUC scores across model families. (b) Forest plot of in-domain Macro-F1 with 2,000-iteration non-parametric bootstrap 95% confidence intervals.",
        172: (
            f"Threshold-independent ranking and prior-shift calibration explain why uncalibrated external macro F1 scores cluster near the all-negative baseline (0.3333) at τ = 0.50. "
            f"While PhoBERT-base records {p_ext_maj['f1_macro']:.4f} macro F1 at τ = 0.50 on UIT-VSMEC Full, its ROC-AUC is {p_ext_maj['roc_auc']:.4f} (PR-AUC = {p_ext_maj['pr_auc']:.4f}) on Full and {p_sad_maj['roc_auc']:.4f} (PR-AUC = {p_sad_maj['pr_auc']:.4f}) on the 2,691-example Sadness-Only subset. "
            f"Because training uses a 19.74% positive prior and labels transient sadness as Negative under Rule 3, predicted probabilities on brief UIT-VSMEC emotion sentences fall below τ = 0.50. "
            f"Applying Bayes prior-shift calibration (τ_Bayes = π_train = 0.1974) without refitting model weights raises PhoBERT-base external macro F1 from 0.3591 to {p_cal_full['f1_macro']:.4f} on Full and from 0.3940 to {p_cal_sad['f1_macro']:.4f} on Sadness-Only; at τ = 0.05 (Figure 6b), macro F1 reaches {p_thr05_full['f1_macro']:.4f} on Full and {p_thr05_sad['f1_macro']:.4f} on Sadness-Only (75.81% positive recall)."
        ),
        173: (
            "Comparing UIT-VSMEC Sadness-Only (n = 2,691) against Full (n = 3,084) further shows that removing the 393 Fear items consistently improves external macro F1 and ROC-AUC across all models (e.g., PhoBERT-base macro F1 rises from 0.3620 to 0.3978 and ROC-AUC from 0.6993 to 0.7078; weighted BiLSTM rises from 0.3824 to 0.4220; and BERTopic-only rises from 0.5025 to 0.5235), confirming that Fear expressions introduce construct noise."
        ),
        174: (
            f"Fitted strictly on final_train.csv (n = 6,392), inductive BERTopic-only achieves {bt_in['f1_macro']:.4f} in-domain macro F1 (+10.87 points above the 0.4452 trivial baseline) and {bt_ext['f1_macro']:.4f} on UIT-VSMEC Full ({bt_sad['f1_macro']:.4f} on Sadness-Only; +0.0514 transfer gap), performing above chance in-domain but remaining below lexical and contextual classifiers that capture speaker perspective."
        ),
        176: (
            "Figure 5 and Table 11 report the exact diagnostic confusion matrices and derived metrics across PhoBERT-base and TF-IDF + Logistic Regression. "
            "In-domain PhoBERT-base Majority Vote (n = 1,371, Figure 5a) records TN = 933, FP = 167, FN = 94, TP = 177 (accuracy = 0.8096, positive precision = 0.5145, positive recall = 0.6531, positive F1 = 0.5756, negative F1 = 0.8773, macro F1 = 0.7265; Soft Vote: TN = 931, FP = 169, FN = 93, TP = 178, macro F1 = 0.7263), "
            "while in-domain TF-IDF + Logistic Regression (Figure 5b) records TN = 906, FP = 194, FN = 89, TP = 182 (accuracy = 0.7936, positive precision = 0.4840, positive recall = 0.6716, positive F1 = 0.5626, negative F1 = 0.8649, macro F1 = 0.7138). "
            "Externally on UIT-VSMEC, PhoBERT-base shifts from near-complete negative prediction at uncalibrated τ = 0.50 (Figure 5c: TN = 1,532, FP = 10, FN = 1,499, TP = 43 on Full n = 3,084; macro F1 = 0.3620) to balanced retrieval on Sadness-Only (n = 2,691) at τ_Bayes = 0.1974 (TN = 1,510, FP = 32, FN = 1,022, TP = 127; macro F1 = 0.4677) and τ = 0.05 (Figure 5d: TN = 788, FP = 754, FN = 278, TP = 871; macro F1 = 0.6161)."
        ),
        178: "Figure 5. Reconciled diagnostic confusion matrices with row-recall and column-precision marginals. (a) PhoBERT-base Majority-Vote in-domain confusion matrix (n = 1,371, F1_macro = 0.7265, Acc = 80.96%). (b) TF-IDF + Logistic Regression in-domain confusion matrix (n = 1,371, F1_macro = 0.7138, Acc = 79.36%). (c) PhoBERT-base external confusion matrix on UIT-VSMEC Full (n = 3,084) at uncalibrated default threshold τ = 0.50 (F1_macro = 0.3620, Acc = 51.07%). (d) PhoBERT-base external confusion matrix on UIT-VSMEC Sadness-Only (n = 2,691) at calibrated threshold τ = 0.05 (F1_macro = 0.6161, Acc = 61.65%).",
        179: (
            "Table 11 summarizes the class-specific precision, recall, F1, and predicted-positive counts computed directly from the confusion matrices in Figure 5."
        ),
        180: "Table 11. Exact class-specific metrics and confusion counts for PhoBERT-base and TF-IDF + Logistic Regression in-domain (n = 1,371) and PhoBERT-base externally across decision thresholds.",
        182: (
            "Comparing PhoBERT-base Majority Vote (FP = 167, TP = 177, macro F1 = 0.7265) against TF-IDF + Logistic Regression (FP = 194, TP = 182, macro F1 = 0.7138) shows that PhoBERT-base improves precision and macro F1 primarily by rejecting 27 false alarms on benign comments that contain surface emotional terms."
        ),
        183: (
            "Because the in-domain test set is 80.23% negative, the trivial all-negative baseline attains 0.8023 accuracy (1,100 / 1,371) but only 0.4452 macro F1. "
            "PhoBERT-base (0.8096 accuracy, 0.7265 macro F1) and PhoBERT + BERTopic (0.8373 accuracy, 0.7302 macro F1) exceed the trivial baseline on both metrics, whereas TF-IDF + Logistic Regression (0.7936 accuracy, 0.7138 macro F1) trades 0.87 percentage points of accuracy for a +0.2686 gain in macro F1 (recovering 67.16% of positive cases)."
        ),
        185: (
            f"Table 12 reports the class-weighting ablation on final_test.csv (n = 1,371) across PhoBERT-base, TF-IDF + Logistic Regression, and BiLSTM, comparing unweighted loss (w_1 = 1.0) against inverse-frequency weighted loss (w_1 = 4.0650). "
            f"Class weighting increases macro F1 for TF-IDF + Logistic Regression from {lr_unw_f1:.4f} to {lr_in['f1_macro']:.4f} (+{(lr_in['f1_macro'] - lr_unw_f1)*100:.2f} points; positive F1 from {vis_data['tfidf_logreg_unweighted']['f1_depression']:.4f} to {lr_in['f1_depression']:.4f}), "
            f"for PhoBERT-base (majority vote) from {pho_unw_f1:.4f} to {p_maj['f1_macro']:.4f} (+{(p_maj['f1_macro'] - pho_unw_f1)*100:.2f} points; positive F1 from {vis_data['phobert_unweighted']['in_domain']['majority_vote']['f1_depression']:.4f} to {p_maj['f1_depression']:.4f}), and "
            f"for BiLSTM (majority vote) from {bi_unw_maj['f1_macro']:.4f} to {bi_w_maj['f1_macro']:.4f} (+{(bi_w_maj['f1_macro'] - bi_unw_maj['f1_macro'])*100:.2f} points; positive F1 from {bi_unw_maj['f1_depression']:.4f} to {bi_w_maj['f1_depression']:.4f})."
        ),
        186: "Table 12. Audited class-weighting ablation on the in-domain test set (n = 1,371), comparing unweighted loss (w_1 = 1.0) versus inverse-frequency class-weighted loss (w_1 = 4.0650).",
        188: (
            "As Table 12 shows, unweighted objectives under 4.06:1 imbalance over-predict the majority Negative class, whereas inverse-frequency weighting improves positive-class F1 by +3.26 to +15.41 percentage points."
        ),
        189: (
            f"Across the six cumulative active-learning rounds—expanding the gold pool from Round 1 (n = 750) and Round 2 (n = 1,750) through Rounds 3–4 (n = 2,072), Round 5 (n = 3,432), and Round 6 (n = 9,134; 6,392 train)—in-domain PhoBERT macro F1 rose from 0.5821 (Round 1) to 0.6314 (Round 2), 0.6658 (Round 3), 0.6892 (Round 4), 0.7015 (Round 5), and {p_maj['f1_macro']:.4f} (Round 6 Majority Vote; validation macro F1 = 0.7318)."
        ),
        190: (
            "In every round, uncertainty sampling queried only unlabeled_pool.csv and checkpoint selection used final_val.csv, keeping final_test.csv strictly held out. "
            "Because the training pool grows across rounds without an equal-budget random-sampling control, this trajectory reflects cumulative dataset expansion rather than isolated active-learning sample efficiency."
        ),
        192: "Figure 6. Class-weighting ablation and cross-domain threshold calibration (both vertical axes starting at 0.0). (a) Unweighted vs. class-weighted in-domain Macro-F1 and Positive-Class F1 across PhoBERT-base, TF-IDF + Logistic Regression, and BiLSTM. (b) PhoBERT-base Macro-F1 as a function of classification threshold τ on In-Domain Test (n = 1,371), UIT-VSMEC Sadness-Only (n = 2,691), and UIT-VSMEC Full (n = 3,084), highlighting the Bayes prior-calibrated threshold τ_Bayes = 0.1974 and high-sensitivity regime τ = 0.05.",
        193: (
            "Figure 6(a) summarizes the class-weighting gains on a zero-anchored axis, and Figure 6(b) illustrates how threshold calibration (τ_Bayes = 0.1974 and τ = 0.05) mitigates prior shift on UIT-VSMEC."
        ),
        195: (
            "Qualitative error analysis highlights three recurring ambiguities: supportive expressions directed at others (“đừng buồn nữa nhé”), polysemous compounds (“buồn ngủ” [sleepy]), and brief implicit statements of helplessness (“mình chẳng biết phải làm sao nữa”) that lack explicit subject pronouns."
        ),
        196: (
            "Across PhoBERT-base (FP = 167, FN = 94) and TF-IDF + Logistic Regression (FP = 194, FN = 89), short pronoun-dropped utterances remain the primary source of residual false negatives."
        ),
        197: (
            "As shown in Figure 3(b), fitting inductive BERTopic on final_train.csv (n = 6,392) yields 36 proper clusters plus the outlier group (Topic −1: n = 2,141, 33.49%, 21.0% positive). "
            "The largest clusters capture depression and therapy (Topic 0: n = 558, 29.6% positive), chronic insomnia (Topic 1: n = 404, 12.9% positive), family conflict (Topic 2: n = 347, 34.6% positive), peer support (Topic 3: n = 287, 13.9% positive), undiacritized commentary (Topic 4: n = 208, 5.8% positive), and nighttime loneliness (Topic 5: n = 200, 19.5% positive)."
        ),
        198: (
            f"Figure 3(b) also explains why inductive BERTopic-only achieves {bt_in['f1_macro']:.4f} macro F1: even in the highest-risk clusters—family conflict (34.6% positive) and therapy discussion (29.6% positive)—65% to 70% of comments are gold Negative (third-person discussion or supportive advice)."
        ),
        200: (
            f"Four factors govern our empirical findings. "
            f"First, because both VietDepression and UIT-VSMEC consist of informal social-media text, low uncalibrated external F1 at τ = 0.50 reflects severe prior shift (from 19.74% to 50.00% positive) rather than a formal-versus-informal register gap; Bayes prior calibration (τ_Bayes = 0.1974) raises PhoBERT-base macro F1 from 0.3591 to {p_cal_full['f1_macro']:.4f} (Full) and {p_cal_sad['f1_macro']:.4f} (Sadness-Only), and τ = 0.05 reaches {p_thr05_full['f1_macro']:.4f} and {p_thr05_sad['f1_macro']:.4f}."
        ),
        201: (
            f"Second, construct divergence between transient emotions and persistent distress remains substantial: excluding the 393 Fear items in Sadness-Only (n = 2,691) improves PhoBERT-base ROC-AUC from {p_ext_maj['roc_auc']:.4f} to {p_sad_maj['roc_auc']:.4f} and uncalibrated macro F1 from {p_ext_maj['f1_macro']:.4f} to {p_sad_maj['f1_macro']:.4f}, but general sadness in UIT-VSMEC still differs from our four-rule distress rubric."
        ),
        202: (
            f"Third, bootstrap confidence intervals and paired tests establish that PhoBERT-base ({p_maj['f1_macro']:.4f}) and the hybrid model ({hyb_maj['f1_macro']:.4f}) significantly outperform both BiLSTM variants ({bi_w_maj['f1_macro']:.4f} weighted; {bi_unw_maj['f1_macro']:.4f} unweighted), the weak labeler ({lex_in['f1_macro']:.4f}), and BERTopic-only ({bt_in['f1_macro']:.4f}) (p < 0.0001), while their lead over TF-IDF + Logistic Regression ({lr_in['f1_macro']:.4f}) lies within overlapping 95% CIs (p = 0.3250)."
        ),
        203: (
            f"Fourth, while our 300-comment double-annotated audit confirms high reliability (κ = {iaa['cohen_kappa_annotator_a_vs_b']:.4f}, {iaa['raw_agreement_rate']*100:.2f}% agreement), query-based retrieval over-samples explicit distress terms relative to random streams, and round-by-round gains reflect cumulative dataset growth rather than a matched-budget comparison against random sampling."
        ),
        204: (
            "Fitting BERTopic and all feature scalers strictly on final_train.csv (n = 6,392) eliminates transductive leakage across all reported experiments."
        ),
        205: (
            "Finally, all labels characterize individual comments rather than clinical diagnosis or user-level risk."
        ),
        207: (
            f"Addressing the need to reliably detect comment-level psychological distress in informal Vietnamese social media without conflating textual expression with clinical diagnosis, we presented VietDepression, a benchmark of 9,134 human-adjudicated YouTube comments (curated from 125,329 retrieved texts via a six-stage workflow; κ = {iaa['cohen_kappa_annotator_a_vs_b']:.4f}). "
            f"Evaluating the 335-entry lexicon on all 1,371 test comments with abstentions counted ({lex_in['f1_macro']:.4f} macro F1 vs. 0.8092 on the 343 covered pilot subset) showed that lexical rules aid candidate triage but require human review and contextual modeling for broad coverage. "
            f"Across three seeds, PhoBERT + BERTopic ({hyb_maj['f1_macro']:.4f} majority vote; {hyb_soft['f1_macro']:.4f} soft vote) and PhoBERT-base ({p_maj['f1_macro']:.4f} majority vote; {p_soft['f1_macro']:.4f} soft vote; ROC-AUC = {p_maj['roc_auc']:.4f}) achieved the highest in-domain macro F1, significantly outperforming class-weighted BiLSTM ({bi_w_maj['f1_macro']:.4f}), unweighted BiLSTM ({bi_unw_maj['f1_macro']:.4f}), the weak labeler ({lex_in['f1_macro']:.4f}), and inductive BERTopic-only ({bt_in['f1_macro']:.4f}) (all p < 0.0001), while performing comparably to TF-IDF + Logistic Regression ({lr_in['f1_macro']:.4f}, p = 0.3250) and confirming that coarse topic indicators add negligible signal to fine-tuned transformer embeddings."
        ),
        208: (
            f"On the external UIT-VSMEC emotion benchmark, threshold-independent ranking (PhoBERT-base ROC-AUC = {p_sad_maj['roc_auc']:.4f} on Sadness-Only) and prior-shift calibration (raising macro F1 from 0.3620 at τ = 0.50 to {p_cal_sad['f1_macro']:.4f} at τ_Bayes = 0.1974 and {p_thr05_sad['f1_macro']:.4f} at τ = 0.05) confirmed that cross-corpus degradation stems primarily from severe base-rate shift and emotion-proxy label mismatch rather than representation failure."
        ),
        209: (
            "Future work will compare uncertainty sampling against equal-budget random sampling, expand multi-platform annotation (TikTok, Facebook, Threads), and evaluate social-media-pretrained encoders (ViSoBERT [27], XLM-RoBERTa [26]) and parameter-efficient fine-tuning of LLMs [24, 25]."
        ),
        210: (
            "Any downstream deployment in mental-health support requires human clinical oversight and domain-calibrated decision thresholds."
        ),
        211: (
            "AI-Use and Reproducibility Disclosure: In accordance with IEEE policy, the authors disclose that generative AI was used for English language editing and refactoring of evaluation and plotting scripts. All annotations, experiments, statistical tests, and conclusions were verified by the authors against the code and data artifacts at https://github.com/iannwendy/VietDepression."
        ),
        # Fix References [5], [8], [11], [13], [16], [17] per Reviewer 1 Comment 12 & user access date Oct. 15, 2026
        217: '[5] V. A. Ho, D. H.-C. Nguyen, D. H. Nguyen, L. T.-V. Pham, D.-V. Nguyen, K. V. Nguyen, and N. L.-T. Nguyen, “Emotion recognition for Vietnamese social media text,” in Proceedings of the 2019 International Conference of the Pacific Association for Computational Linguistics (PACLING 2019), pp. 319–333, 2019, doi: 10.1007/978-981-15-6168-9_27. [Online]. Available: https://arxiv.org/abs/1911.09339 (Accessed: Oct. 15, 2026).',
        220: '[8] Y. Liu et al., “RoBERTa: A robustly optimized BERT pretraining approach,” arXiv:1907.11692, 2019. https://arxiv.org/abs/1907.11692 (Accessed: Oct. 15, 2026).',
        223: '[11] M. Grootendorst, “BERTopic: Neural topic modeling with a class-based TF-IDF procedure,” arXiv:2203.05794, 2022. https://arxiv.org/abs/2203.05794 (Accessed: Oct. 15, 2026).',
        225: '[13] L. McInnes, J. Healy, N. Saul, and L. Großberger, “UMAP: Uniform manifold approximation and projection for dimension reduction,” Journal of Open Source Software, vol. 3, no. 29, p. 861, 2018, doi: 10.21105/joss.00861. [Online]. Available: https://arxiv.org/abs/1802.03426 (Accessed: Oct. 15, 2026).',
        228: '[16] Google, “YouTube Data API v3,” Google for Developers. https://developers.google.com/youtube/v3 (Accessed: Oct. 15, 2026).',
        229: '[17] T. Wolf et al., “Transformers: State-of-the-art natural language processing,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45, 2020. doi: 10.18653/v1/2020.emnlp-demos.6.',
    }
    return updates


def build_new_references() -> List[str]:
    """New references [19]-[30] addressing Reviewer 1 Comments 7, 8, 9, 12 and Reviewer 2."""
    return [
        '[19] G. Coppersmith, M. Dredze, C. Harman, K. Hollingshead, and M. Mitchell, “CLPsych 2015 shared task: Depression and PTSD on Twitter,” in Proceedings of the 2nd Workshop on Computational Linguistics and Clinical Psychology, pp. 31–39, 2015. doi: 10.3115/v1/W15-1204.',
        '[20] D. E. Losada, F. Crestani, and J. Parapar, “Overview of eRisk: Early risk prediction on the Internet,” in Experimental IR Meets Multilinguality, Multimodality, and Interaction (CLEF 2018), LNCS 11018, pp. 343–361, 2018. doi: 10.1007/978-3-319-98932-7_30.',
        '[21] E. Turcan and K. McKeown, “Dreaddit: A Reddit dataset for stress analysis in social media,” in Proceedings of the 10th International Workshop on Health Text Mining and Information Analysis (LOUHI), pp. 97–107, 2019. doi: 10.18653/v1/D19-6213.',
        '[22] M. Garg, C. Saxena, S. Saha, V. Krishnan, R. Joshi, and V. Mago, “CAMS: An annotated corpus for causal analysis of mental health issues in social media posts,” in Proceedings of the 13th Language Resources and Evaluation Conference (LREC), pp. 6387–6396, 2022.',
        '[23] S. Ji, T. Zhang, L. Ansari, J. Fu, P. Tiwari, and E. Cambria, “MentalBERT: Publicly available pretrained language models for mental healthcare,” in Proceedings of the 13th Language Resources and Evaluation Conference (LREC), pp. 7184–7190, 2022.',
        '[24] K. Yang, T. Zhang, Z. Kuang, Q. Xie, J. Huang, and S. Ananiadou, “MentaLLaMA: Interpretable mental health analysis on social media with large language models,” in Proceedings of the ACM Web Conference 2024 (WWW ’24), pp. 4489–4500, 2024. doi: 10.1145/3589334.3648137.',
        '[25] X. Xu, B. Yao, Y. Dong, S. Gabriel, H. Yu, J. Hendler, M. Ghassemi, A. K. Dey, and D. Wang, “Mental-LLM: Leveraging large language models for mental health prediction via online text data,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 8, no. 1, Art. 31, pp. 1–32, 2024. doi: 10.1145/3643540.',
        '[26] A. Conneau et al., “Unsupervised cross-lingual representation learning at scale,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 8440–8451, 2020. doi: 10.18653/v1/2020.acl-main.747.',
        '[27] N. Nguyen, T. Phan, D.-V. Nguyen, and K. Nguyen, “ViSoBERT: A pre-trained language model for Vietnamese social media text processing,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 5191–5207, 2023. doi: 10.18653/v1/2023.emnlp-main.315.',
        '[28] L. T. Nguyen and D. Q. Nguyen, “PhoNLP: A joint multi-task learning model for Vietnamese part-of-speech tagging, named entity recognition and dependency parsing,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations, pp. 1–7, 2021. doi: 10.18653/v1/2021.naacl-demos.1.',
        '[29] “NTC-SCV,” GitHub repository. [Online]. Available: https://github.com/congnghia0609/ntc-scv (Accessed: Oct. 15, 2026).',
        '[30] T. G. Dietterich, “Approximate statistical tests for comparing supervised classification learning algorithms,” Neural Computation, vol. 10, no. 7, pp. 1895–1923, 1998. doi: 10.1162/089976698300017197.',
    ]


def build_table_updates(
    eval_data: dict, bt_data: dict, vis_data: dict
) -> Dict[int, List[List[str]]]:
    """
    Build the exact, 100%-consistent rows for all 12 tables (doc.tables[0] to doc.tables[11]),
    STRICTLY using only the original model families.
    """
    p_maj = eval_data["in_domain"]["phobert_majority_vote"]
    p_soft = eval_data["in_domain"]["phobert_soft_vote"]
    p_ms = eval_data["in_domain"]["phobert_mean_std"]
    p_ext_maj = eval_data["cross_domain"]["phobert_majority_vote"]
    p_ext_soft = eval_data["cross_domain"]["phobert_soft_vote"]
    p_sad_maj = eval_data["cross_domain_sadness_only"]["phobert_majority_vote"]
    p_sad_soft = eval_data["cross_domain_sadness_only"]["phobert_soft_vote"]
    p_cal_sad = eval_data["threshold_calibration_analysis"]["phobert_soft_vote"][
        "prior_shift_bayes_threshold"
    ]["vsmec_sadness_only"]

    hyb_maj = eval_data["in_domain"]["phobert_bertopic_majority_vote"]
    hyb_soft = eval_data["in_domain"]["phobert_bertopic"]
    hyb_ms = eval_data["in_domain"]["phobert_bertopic_mean_std"]
    hyb_ext_maj = eval_data["cross_domain"]["phobert_bertopic_majority_vote"]
    hyb_ext_soft = eval_data["cross_domain"]["phobert_bertopic"]
    hyb_ext_ms = eval_data["cross_domain"]["phobert_bertopic_mean_std"]
    hyb_sad_soft = eval_data["cross_domain_sadness_only"]["phobert_bertopic"]

    lr_in = eval_data["in_domain"]["tfidf_logreg"]
    lr_ext = eval_data["cross_domain"]["tfidf_logreg"]
    lr_sad = eval_data["cross_domain_sadness_only"]["tfidf_logreg"]
    svc_in = eval_data["in_domain"]["tfidf_svc"]
    svc_ext = eval_data["cross_domain"]["tfidf_svc"]
    svc_sad = eval_data["cross_domain_sadness_only"]["tfidf_svc"]

    bi_unw_maj = eval_data["in_domain"]["bilstm_majority_vote"]
    bi_unw_ms = eval_data["in_domain"]["bilstm_mean_std"]
    bi_unw_ext = eval_data["cross_domain"]["bilstm_majority_vote"]
    bi_unw_sad = eval_data["cross_domain_sadness_only"]["bilstm_majority_vote"]

    bi_w_maj = eval_data["in_domain"]["bilstm_weighted_majority_vote"]
    bi_w_ms = eval_data["in_domain"]["bilstm_weighted_mean_std"]
    bi_w_ext = eval_data["cross_domain"]["bilstm_weighted_majority_vote"]
    bi_w_sad = eval_data["cross_domain_sadness_only"]["bilstm_weighted_majority_vote"]

    bt_in = eval_data["in_domain"]["bertopic_only"]
    bt_ext = eval_data["cross_domain"]["bertopic_only"]
    bt_sad = eval_data["cross_domain_sadness_only"]["bertopic_only"]

    lex_in = eval_data["in_domain"]["weak_labeler_lexicon"]
    lex_ext = eval_data["cross_domain"]["weak_labeler_lexicon"]
    lex_sad = eval_data["cross_domain_sadness_only"]["weak_labeler_lexicon"]

    pho_unw_f1 = vis_data["phobert_unweighted"]["in_domain"]["majority_vote"]["f1_macro"]
    lr_unw_f1 = vis_data["tfidf_logreg_unweighted"]["f1_macro"]

    tables = {}

    # TABLE 0 -> Table 1 (3 cols)
    tables[0] = [
        ["Resource or Study", "Target, Language, and Scale", "Relationship to VietDepression"],
        [
            "De Choudhury et al. [1]; Coppersmith et al. [2]",
            "English Twitter user-level depression & PTSD signals",
            "Foundational social-media mental-health studies; user-level rather than Vietnamese comment-level",
        ],
        [
            "CLPsych [19] & eRisk [3, 20]",
            "English Reddit/forum longitudinal early risk benchmarks",
            "Establish privacy/DUA ethics & user-history evaluation; differ from single-comment Vietnamese task",
        ],
        [
            "Dreaddit [21] & CAMS [22]",
            "English Reddit post-level stress (3,553) & causal depression (5,051)",
            "Post-level mental-health annotation rubrics; motivate our four-rule comment guideline",
        ],
        [
            "MentalBERT [23]; MentaLLaMA / Mental-LLM [24, 25]",
            "English domain encoders & instruction-tuned / zero-shot LLMs",
            "Highlight value of domain pretraining and calibration challenges on informal social media",
        ],
        [
            "UIT-VSMEC [5]; UIT-VSFC [9]; NTC-SCV [29]",
            "Vietnamese emotion (6,919) & sentiment (16,170; 49,953) corpora",
            "UIT-VSMEC supplies external emotion-proxy sets (n=3,084 / 2,691); no clinical distress labels",
        ],
        [
            "PhoBERT-base [4] (and Vietnamese NLP [27, 28])",
            "Vietnamese contextual RoBERTa encoder (135M parameters)",
            "Fine-tuned across 3 seeds as our core contextual encoder and hybrid backbone",
        ],
        [
            "VietDepression (Ours)",
            "125,329 raw / 9,134 gold Vietnamese YouTube comments (κ = 0.8667)",
            "Audited comment-level distress benchmark, multi-seed statistical comparison & prior calibration",
        ],
    ]

    # TABLE 1 -> Table 2 (3 cols) — Explicitly fixes Reviewer 2 Minor Comment 2
    tables[1] = [
        ["Collected Schema Field", "Role in Dataset Construction and Provenance Audit", "Use in Classification"],
        ["comment_id", "Detect repeated API records; trace upstream collection ledger", "Excluded from model input"],
        ["video_id", "Track 2,906 comment-yielding source videos (from 4,413 crawled)", "Excluded from model input"],
        ["video_title", "Document source video context during collection", "Excluded from model input"],
        ["retrieval_keyword", "Audit 264 sampling queries across 5 thematic domains", "Excluded from model input"],
        ["comment_text", "NFC-normalized, deduplicated Vietnamese comment text (0% split overlap)", "Included (sole model text input)"],
        ["like_count", "Record engagement metadata in upstream collection ledger", "Excluded from model input"],
        ["published_at", "Record collection timestamp; reserved for future temporal analysis", "Excluded from model input"],
    ]

    # TABLE 2 -> Table 3 (4 cols)
    tables[2] = [
        ["Group", "Entries", "Weight / Corpus Match Rate (N = 125,328)", "Illustrative Phrases"],
        ["Strong positive", "53", "+5 (High-priority distress cues)", "muốn chết; tự tử; trầm cảm"],
        ["Medium positive", "116", "+3 (Context-dependent distress cues)", "cô đơn; mệt mỏi; mất ngủ"],
        ["Normal context", "166", "−2 (Benign / appreciative cues)", "video hay; cảm ơn; tuyệt vời"],
        [
            "Review context (Table 3 phrases)",
            "22",
            "Flag only: 820 exact matches (0.65%); composite need_review = 120,334 (96.02%)",
            "đừng tự tử (1); cố lên (730); hãy chia sẻ (75); chết cười (3)",
        ],
    ]

    # TABLE 3 -> Table 4 (3 cols)
    tables[3] = [
        [
            "Statistic / Metric",
            "Pilot Covered Subset (n = 343, Abstentions Excluded)",
            "Full In-Domain Test Set (n = 1,371, 965 [70.39%] Abstentions = N)",
        ],
        [
            "Evaluated / Abstained",
            "343 evaluated / 0 abstained (confident s(x) ≥ 5 or ≤ −2)",
            "1,371 evaluated / 965 abstained (70.39% uncertain; 406 [29.61%] covered)",
        ],
        [
            "TP / FP / FN / TN",
            "92 / 58 / 2 / 191",
            "122 / 168 / 149 / 932",
        ],
        [
            "Positive precision (Class 1)",
            "0.6133 (92 / 150)",
            f"{lex_in['precision_depression']:.4f} (122 / 290)",
        ],
        [
            "Positive recall (Class 1)",
            "0.9787 (92 / 94)",
            f"{lex_in['recall_depression']:.4f} (122 / 271)",
        ],
        [
            "Positive F1 (Class 1)",
            "0.7541",
            f"{lex_in['f1_depression']:.4f}",
        ],
        [
            "Negative F1 (Class 0)",
            "0.8643",
            f"{lex_in['f1_normal']:.4f}",
        ],
        [
            "Macro F1 [95% Bootstrap CI]",
            "0.8092 (on 21.48% covered tail only)",
            f"{lex_in['f1_macro']:.4f} [0.6153, 0.6753] (0.5245 on 406 covered test items)",
        ],
        [
            "Accuracy / ROC-AUC / PR-AUC",
            "0.8251 / — / —",
            f"{lex_in['accuracy']:.4f} / {lex_in['roc_auc']:.4f} / {lex_in['pr_auc']:.4f}",
        ],
    ]

    # TABLE 4 -> Table 5 (3 cols)
    tables[4] = [
        ["Resource or Partition", "Audited Size (Class Support)", "Analytical Role and Leakage Policy"],
        ["Collected YouTube comments", "125,329 (2,906 videos / 2,949 channels)", "Raw API retrieval inventory (125,328 non-empty scored)"],
        ["Filtered YouTube comments", "119,649 (−5,680 [4.53%] filtered)", "Normalized, deduplicated post-processing inventory"],
        ["External text inventory", "191,072 (191,991 raw across 8 HF sets)", "Auxiliary Vietnamese sentiment/emotion/toxicity corpora"],
        ["Unified exploratory pool", "316,401", "Archived exploratory pool; excluded from inductive training"],
        ["Final human-labeled benchmark", "9,134 (1,804 P [19.75%]; 7,330 N [80.25%])", "100% human-adjudicated gold benchmark (0 duplicate texts)"],
        ["Training split (final_train.csv)", "6,392 (1,262 P [19.74%]; 5,130 N [80.26%])", "Sole split used to fit all classifiers, TF-IDF, and BERTopic"],
        ["Validation split (final_val.csv)", "1,371 (271 P [19.77%]; 1,100 N [80.23%])", "Hyperparameter tuning and early-stopping checkpoint selection"],
        ["In-domain test (final_test.csv)", "1,371 (271 P [19.77%]; 1,100 N [80.23%])", "Held-out 100% human-labeled test evaluation (0% overlap)"],
        ["UIT-VSMEC Full proxy test", "3,084 (1,542 Sad+Fear [50%]; 1,542 Enjoy [50%])", "Balanced external emotion-proxy evaluation"],
        ["UIT-VSMEC Sadness-Only subset", "2,691 (1,149 Sadness [42.7%]; 1,542 Enjoy [57.3%])", "Cleaner external proxy test excluding 393 Fear items"],
    ]

    # TABLE 5 -> Table 6 (5 cols)
    tables[5] = [
        ["Stage or Partition", "Positive (1)", "Negative (0)", "Uncertain / Excluded", "Audited Total"],
        ["Weak labeling (scored raw)", "3,223 (2.57%)", "23,695 (18.91%)", "98,410 (78.52%)", "125,328"],
        ["Final labeled benchmark", "1,804 (19.75%)", "7,330 (80.25%)", "0 (100% human binary gold)", "9,134"],
        ["Training split (final_train.csv)", "1,262 (19.74%)", "5,130 (80.26%)", "0 (0% overlap)", "6,392"],
        ["Validation split (final_val.csv)", "271 (19.77%)", "1,100 (80.23%)", "0 (0% overlap)", "1,371"],
        ["In-domain test (final_test.csv)", "271 (19.77%)", "1,100 (80.23%)", "0 (0% overlap)", "1,371"],
        ["UIT-VSMEC Full proxy test", "1,542 (50.00%)", "1,542 (50.00%)", "3,835 other emotions", "3,084"],
        ["UIT-VSMEC Sadness-Only test", "1,149 (42.70%)", "1,542 (57.30%)", "393 Fear excluded", "2,691"],
    ]

    # TABLE 6 -> Table 7 (5 cols)
    tables[6] = [
        ["Source & Rule", "Paraphrased Vietnamese Composite", "English Translation", "Label", "Rubric Interpretation"],
        [
            "YouTube (Rule 1 & 3)",
            "Dạo này mình kiệt sức thật sự, chẳng còn động lực để tiếp tục bất cứ việc gì.",
            "Lately I am truly exhausted and have no motivation left to continue anything.",
            "P (1)",
            "First-person persistent exhaustion and anhedonia",
        ],
        [
            "YouTube (Rule 1 & 3)",
            "Đêm nào mình cũng nằm khóc một mình trong phòng, cảm giác lạc lõng không ai hiểu.",
            "Every night I lie crying alone in my room, feeling isolated with no one understanding.",
            "P (1)",
            "First-person recurring nighttime crying and social isolation",
        ],
        [
            "YouTube (Rule 3)",
            "Thêm một ngày trôi qua vô nghĩa, mình thấy bản thân cứ mãi dậm chân tại chỗ trong bóng tối.",
            "Another meaningless day has passed; I feel stuck in the dark without direction.",
            "P (1)",
            "Implicit hopelessness and self-deprecation without naming depression",
        ],
        [
            "YouTube (Rule 1)",
            "Bạn đừng buồn nhiều nữa nhé, cố lên rồi mọi khó khăn cũng sẽ qua thôi.",
            "Please do not be so sad anymore, stay strong and all difficulties will pass.",
            "N (0)",
            "Second-person supportive encouragement (flagged by 'cố lên')",
        ],
        [
            "YouTube (Rule 2)",
            "Ngồi nghe giảng mà buồn ngủ ríu cả mắt, ngày mai lại kiểm tra sớm rồi.",
            "Sitting in lecture feeling so sleepy my eyes are closing, and tomorrow is an early exam.",
            "N (0)",
            "Lexical polysemy: compound 'buồn ngủ' (sleepy), not sadness",
        ],
        [
            "YouTube (Rule 3)",
            "Tuần này bài tập nhóm với hạn nộp dồn dập làm mình mệt bở hơi tai luôn.",
            "This week group assignments and deadlines piling up are making me completely worn out.",
            "N (0)",
            "Transient academic/situational fatigue without emotional decompensation",
        ],
        [
            "UIT-VSMEC [5]",
            "Nhìn kết quả mà lòng buồn rười rượi, chẳng muốn bước ra ngoài gặp ai.",
            "Looking at the result makes me feel deeply sad, not wanting to go out to meet anyone.",
            "P proxy",
            "Sadness emotion class mapped to positive proxy in UIT-VSMEC",
        ],
        [
            "UIT-VSMEC [5]",
            "Hôm nay vui quá trời vì được đi ăn uống cùng hội bạn thân lâu ngày gặp lại.",
            "Today is so joyful because I got to eat out with close friends after a long time.",
            "N proxy",
            "Enjoyment emotion class mapped to negative proxy in UIT-VSMEC",
        ],
    ]

    # TABLE 7 -> Table 8 (3 cols)
    tables[7] = [
        ["External Resource & Hugging Face Identifier", "Original Task & Audited Size", "Analytical Role in This Study"],
        [
            "1. UIT-VSMEC [5] (duwuonline/UIT-VSMEC)",
            "7-class social-media emotion (6,919 total; 5,536 deduplicated)",
            "Supplies 3,084-item Full & 2,691-item Sadness-Only external proxy tests",
        ],
        [
            "2. NTC-SCV [29] (ntc-scv)",
            "Binary sentiment reviews (49,953 deduplicated rows)",
            "Auxiliary pool only; strictly excluded from supervised train & inductive BERTopic",
        ],
        [
            "3. UIT-VSFC [9] (uitnlp/vietnamese_students_feedback)",
            "3-class student feedback sentiment (16,170 deduplicated rows)",
            "Auxiliary pool only; strictly excluded from supervised train & inductive BERTopic",
        ],
        [
            "4. UIT-VSFC-SA (visolex/uit-vsfc-sa)",
            "5-star / 3-class sentiment variant (15,824 deduplicated rows)",
            "Auxiliary pool only; strictly excluded from supervised train & inductive BERTopic",
        ],
        [
            "5. ODS Vietnamese Sentiment (ods/vietnamese_sentiment_analysis)",
            "Binary sentiment classification (26,645 deduplicated rows)",
            "Auxiliary pool only; strictly excluded from supervised train & inductive BERTopic",
        ],
        [
            "6. UIT-VSMEC Mirror (tridm/UIT-VSMEC)",
            "Emotion corpus mirror (6,903 deduplicated rows)",
            "Deduplicated against duwuonline/UIT-VSMEC; excluded from training",
        ],
        [
            "7. ViCTSD (tarudesu/ViCTSD)",
            "Vietnamese constructive & toxic speech (10,000 deduplicated rows)",
            "Auxiliary toxicity corpus only; strictly excluded from benchmark training",
        ],
        [
            "8. ViHSD (visolex/ViHSD)",
            "Vietnamese hate speech detection (60,960 deduplicated rows)",
            "Auxiliary toxicity corpus only; strictly excluded from benchmark training",
        ],
    ]

    # TABLE 8 -> Table 9 (4 cols): Original Models Only
    tables[8] = [
        ["Setting / Hyperparameter", "TF-IDF + LinearSVC / LogReg", "BiLSTM (Unweighted & Weighted)", "PhoBERT-base (+ 806-d BERTopic Hybrid)"],
        [
            "Representation & Tokenizer",
            "Word 1–2g (80k, min_df=2) + char_wb 3–5g (80k, min_df=3), sublinear TF",
            "15,000-token vocab; trainable 128-d embedding; underthesea tokens",
            "vinai/phobert-base (135M params, underthesea segmented, 768-d CLS representation)",
        ],
        [
            "Architecture & Head",
            "LinearSVC (C=1.0) & LogReg (penalty='l2', C=1.0, lbfgs); 160,000 features",
            "2-layer BiLSTM (128 hidden/dir = 256-d), dropout 0.5, 64-d ReLU head",
            "End-to-end 2-logit head; Hybrid concatenates frozen 768-d CLS + 37-d topic one-hot (Slot 0=outlier) + 1-d prob = 806-d",
        ],
        [
            "Optimizer & Regularization",
            "LIBLINEAR (SVC, max_iter=2000); L-BFGS (LogReg, L2, C=1.0, max_iter=1000)",
            "Adam (lr = 1e-3, weight_decay = 0.0)",
            "AdamW (lr = 2e-5, weight_decay = 0.01, grad clip = 1.0, linear warmup = 6%)",
        ],
        [
            "Batch Size & Max Length",
            "Full-batch convex optimization (N_train = 6,392)",
            "Batch size 32; max_len = 100 tokens",
            "Train batch size 8 (eval 16); max_len = 128 tokens",
        ],
        [
            "Epochs & Checkpoint Selection",
            "Converged on final_train.csv; 5-fold stratified CV also reported",
            "Up to 10 epochs; early stopping patience = 3 on final_val.csv Macro F1",
            "3 epochs; best checkpoint selected strictly on final_val.csv (n = 1,371) Macro F1",
        ],
        [
            "Class Imbalance Handling",
            "class_weight='balanced' (unweighted ablation also reported)",
            "Both Unweighted (w₁ = 1.0) and Weighted (w₁ = 4.0650) trained across 3 seeds",
            "Inverse-frequency weighted CE loss w₁ = 5,130/1,262 = 4.0650 (unweighted w₁ = 1.0 in Table 12)",
        ],
        [
            "Seeds & Aggregation Rules",
            "random_state = 42 + 5-fold CV (mean ± SD) + 2,000-bootstrap 95% CI",
            "3 seeds (42, 123, 2024): Mean ± SD, Majority Vote, and Soft Vote",
            "3 seeds (42, 123, 2024): Mean ± SD, Majority Vote, and Soft Vote + 2,000-bootstrap 95% CI",
        ],
        [
            "Hardware & Software Stack",
            "Apple Silicon MPS / CPU; Python 3.13, scikit-learn 1.8.0",
            "Apple M-series Silicon (PyTorch 2.11.0 MPS backend)",
            "Apple M-series Silicon (PyTorch 2.11.0 MPS, Transformers 5.5.4, BERTopic 0.17.4)",
        ],
    ]

    # TABLE 9 -> Table 10 (4 cols): Unified Benchmark Results (Original Models Only)
    tables[9] = [
        [
            "Model Configuration & Aggregation",
            "In-Domain Test (n = 1,371) Macro F1 [95% CI] | F1-Pos | ROC-AUC",
            "External UIT-VSMEC Full (n = 3,084) / Sad-Only (n = 2,691) Macro F1 (ROC-AUC)",
            "Transfer Gap ΔF1 (Full τ=0.50 / Bayes Calibrated τ=0.1974)",
        ],
        [
            "PhoBERT + BERTopic Hybrid (806-d, Maj. Vote)",
            f"{hyb_maj['f1_macro']:.4f} [0.6950, 0.7578] | {hyb_maj['f1_depression']:.4f} | {hyb_maj['roc_auc']:.4f}",
            f"{hyb_ext_maj['f1_macro']:.4f} ({hyb_ext_maj['roc_auc']:.4f}) / {hyb_sad_soft['f1_macro']:.4f} ({hyb_sad_soft['roc_auc']:.4f})",
            f"{hyb_maj['f1_macro'] - hyb_ext_maj['f1_macro']:.4f}",
        ],
        [
            "PhoBERT + BERTopic Hybrid (Soft Vote / Mean ± SD)",
            f"{hyb_soft['f1_macro']:.4f} ({hyb_ms['f1_macro_mean']:.4f} ± {hyb_ms['f1_macro_std']:.4f}) | {hyb_soft['f1_depression']:.4f} | {hyb_soft['roc_auc']:.4f}",
            f"{hyb_ext_soft['f1_macro']:.4f} ({hyb_ext_ms['f1_macro_mean']:.4f} ± {hyb_ext_ms['f1_macro_std']:.4f}) / {hyb_sad_soft['f1_macro']:.4f}",
            f"{hyb_soft['f1_macro'] - hyb_ext_soft['f1_macro']:.4f}",
        ],
        [
            "PhoBERT-base (Majority Vote, 3 seeds)",
            f"{p_maj['f1_macro']:.4f} [0.6977, 0.7550] | {p_maj['f1_depression']:.4f} | {p_maj['roc_auc']:.4f}",
            f"{p_ext_maj['f1_macro']:.4f} ({p_ext_maj['roc_auc']:.4f}) / {p_sad_maj['f1_macro']:.4f} ({p_sad_maj['roc_auc']:.4f})",
            f"{p_maj['f1_macro'] - p_ext_maj['f1_macro']:.4f} (Cal. Sad F1: {p_cal_sad['f1_macro']:.4f}; τ=0.05: 0.6161)",
        ],
        [
            "PhoBERT-base (Soft Vote / Mean ± SD)",
            f"{p_soft['f1_macro']:.4f} [0.6985, 0.7543] ({p_ms['f1_macro_mean']:.4f} ± {p_ms['f1_macro_std']:.4f}) | {p_soft['f1_depression']:.4f} | {p_soft['roc_auc']:.4f}",
            f"{p_ext_soft['f1_macro']:.4f} (0.3728 ± 0.0041) / {p_sad_soft['f1_macro']:.4f} (0.3980 ± 0.0055)",
            f"{p_soft['f1_macro'] - p_ext_soft['f1_macro']:.4f} (Cal. Full F1: 0.4346; τ=0.05: 0.6250)",
        ],
        [
            "TF-IDF + Logistic Regression (L2, C = 1.0)",
            f"{lr_in['f1_macro']:.4f} [0.6840, 0.7412] (5f-CV: 0.7036 ± 0.0143) | {lr_in['f1_depression']:.4f} | {lr_in['roc_auc']:.4f}",
            f"{lr_ext['f1_macro']:.4f} ({lr_ext['roc_auc']:.4f}) / {lr_sad['f1_macro']:.4f} ({lr_sad['roc_auc']:.4f})",
            f"{lr_in['f1_macro'] - lr_ext['f1_macro']:.4f}",
        ],
        [
            "TF-IDF + LinearSVC (C = 1.0, Balanced)",
            f"{svc_in['f1_macro']:.4f} [0.6721, 0.7327] (5f-CV: 0.6994 ± 0.0164) | {svc_in['f1_depression']:.4f} | {svc_in['roc_auc']:.4f}",
            f"{svc_ext['f1_macro']:.4f} ({svc_ext['roc_auc']:.4f}) / {svc_sad['f1_macro']:.4f} ({svc_sad['roc_auc']:.4f})",
            f"{svc_in['f1_macro'] - svc_ext['f1_macro']:.4f}",
        ],
        [
            "BiLSTM Class-Weighted (Maj. Vote / Mean ± SD)",
            f"{bi_w_maj['f1_macro']:.4f} [0.6219, 0.6794] ({bi_w_ms['f1_macro_mean']:.4f} ± {bi_w_ms['f1_macro_std']:.4f}) | {bi_w_maj['f1_depression']:.4f} | {bi_w_maj['roc_auc']:.4f}",
            f"{bi_w_ext['f1_macro']:.4f} ({bi_w_ext['roc_auc']:.4f}) / {bi_w_sad['f1_macro']:.4f} ({bi_w_sad['roc_auc']:.4f})",
            f"{bi_w_maj['f1_macro'] - bi_w_ext['f1_macro']:.4f}",
        ],
        [
            "BiLSTM Unweighted (Maj. Vote / Mean ± SD)",
            f"{bi_unw_maj['f1_macro']:.4f} [0.6097, 0.6736] ({bi_unw_ms['f1_macro_mean']:.4f} ± {bi_unw_ms['f1_macro_std']:.4f}) | {bi_unw_maj['f1_depression']:.4f} | {bi_unw_maj['roc_auc']:.4f}",
            f"{bi_unw_ext['f1_macro']:.4f} ({bi_unw_ext['roc_auc']:.4f}) / {bi_unw_sad['f1_macro']:.4f} ({bi_unw_sad['roc_auc']:.4f})",
            f"{bi_unw_maj['f1_macro'] - bi_unw_ext['f1_macro']:.4f} (Ext. Acc = 0.5003)",
        ],
        [
            "Lexical Weak Labeler (All 1,371, Abst. = N)",
            f"{lex_in['f1_macro']:.4f} [0.6153, 0.6753] | {lex_in['f1_depression']:.4f} | {lex_in['roc_auc']:.4f}",
            f"{lex_ext['f1_macro']:.4f} ({lex_ext['roc_auc']:.4f}) / {lex_sad['f1_macro']:.4f} ({lex_sad['roc_auc']:.4f})",
            f"{lex_in['f1_macro'] - lex_ext['f1_macro']:.4f}",
        ],
        [
            "Inductive BERTopic-Only (74-d, Train-Fitted)",
            f"{bt_in['f1_macro']:.4f} [0.5210, 0.5865] | {bt_in['f1_depression']:.4f} | {bt_in['roc_auc']:.4f}",
            f"{bt_ext['f1_macro']:.4f} ({bt_ext['roc_auc']:.4f}) / {bt_sad['f1_macro']:.4f} ({bt_sad['roc_auc']:.4f})",
            f"+{bt_in['f1_macro'] - bt_ext['f1_macro']:.4f} (+0.0304 on Sad-Only)",
        ],
        [
            "All-Negative Trivial Baseline",
            "0.4452 | 0.0000 | 0.5000 (Acc = 0.8023)",
            "0.3333 (Full, Acc = 0.5000) / 0.3643 (Sad-Only, Acc = 0.5730)",
            "0.1119",
        ],
    ]

    # TABLE 10 -> Table 11 (3 cols): Confusion-matrix derived metrics (PhoBERT-base & TF-IDF+LogReg Only)
    tables[10] = [
        [
            "Metric / Confusion Count",
            "In-Domain (n = 1,371): PhoBERT-base (Maj. Vote | Soft Vote) | TF-IDF + LogReg",
            "External UIT-VSMEC PhoBERT: Default (τ=0.50, Full) | Bayes (τ=0.1974, Sad) | High-Sens (τ=0.05, Sad)",
        ],
        [
            "Confusion Matrix (TN / FP / FN / TP)",
            "933 / 167 / 94 / 177 | 931 / 169 / 93 / 178 | 906 / 194 / 89 / 182",
            "1532 / 10 / 1499 / 43 | 1510 / 32 / 1022 / 127 | 788 / 754 / 278 / 871",
        ],
        [
            "Accuracy",
            "0.8096 | 0.8089 | 0.7936",
            "0.5107 | 0.6083 | 0.6165",
        ],
        [
            "Positive precision (Class 1)",
            "0.5145 | 0.5130 | 0.4840",
            "0.8113 | 0.7987 | 0.5360",
        ],
        [
            "Positive recall (Class 1)",
            "0.6531 | 0.6568 | 0.6716",
            "0.0279 | 0.1105 | 0.7581",
        ],
        [
            "Positive F1 (Class 1)",
            "0.5756 | 0.5761 | 0.5626",
            "0.0539 | 0.1942 | 0.6280",
        ],
        [
            "Negative F1 (Class 0)",
            "0.8773 | 0.8766 | 0.8649",
            "0.6700 | 0.7413 | 0.6043",
        ],
        [
            "Macro F1",
            "0.7265 | 0.7263 | 0.7138",
            "0.3620 | 0.4677 | 0.6161 (Full τ=0.05: 0.6250)",
        ],
        [
            "Predicted Positive / Total",
            "344 / 1,371 (25.09%) | 347 / 1,371 (25.31%) | 376 / 1,371 (27.43%)",
            "53 / 3,084 (1.72%) | 159 / 2,691 (5.91%) | 1,625 / 2,691 (60.39%)",
        ],
    ]

    # TABLE 11 -> Table 12 (4 cols): Class-Weighting Ablation on In-Domain Test (Original 3 Models Only)
    tables[11] = [
        [
            "Model Family (In-Domain Test, n = 1,371)",
            "Unweighted Loss (w₁ = 1.0) Macro F1 (F1-Pos)",
            "Class-Weighted Loss (w₁ = 4.0650) Macro F1 (F1-Pos)",
            "Absolute Gain ΔF1_macro (ΔF1-Pos)",
        ],
        [
            "PhoBERT-base (3-Seed Majority Vote)",
            f"{pho_unw_f1:.4f} ({vis_data['phobert_unweighted']['in_domain']['majority_vote']['f1_depression']:.4f})",
            f"{p_maj['f1_macro']:.4f} ({p_maj['f1_depression']:.4f})",
            f"+{p_maj['f1_macro'] - pho_unw_f1:.4f} (+{p_maj['f1_depression'] - vis_data['phobert_unweighted']['in_domain']['majority_vote']['f1_depression']:.4f})",
        ],
        [
            "TF-IDF + Logistic Regression (L2, C = 1.0)",
            f"{lr_unw_f1:.4f} ({vis_data['tfidf_logreg_unweighted']['f1_depression']:.4f})",
            f"{lr_in['f1_macro']:.4f} ({lr_in['f1_depression']:.4f})",
            f"+{lr_in['f1_macro'] - lr_unw_f1:.4f} (+{lr_in['f1_depression'] - vis_data['tfidf_logreg_unweighted']['f1_depression']:.4f})",
        ],
        [
            "BiLSTM (3-Seed Majority Vote)",
            f"{bi_unw_maj['f1_macro']:.4f} ({bi_unw_maj['f1_depression']:.4f})",
            f"{bi_w_maj['f1_macro']:.4f} ({bi_w_maj['f1_depression']:.4f})",
            f"+{bi_w_maj['f1_macro'] - bi_unw_maj['f1_macro']:.4f} (+{bi_w_maj['f1_depression'] - bi_unw_maj['f1_depression']:.4f})",
        ],
    ]

    return tables


def main():
    print(f"Loading backup docx from {BACKUP_DOCX} ...")
    doc = docx.Document(str(BACKUP_DOCX))

    with open(MODEL_DIR / "evaluation_results.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(MODEL_DIR / "bertopic_evaluation_results.json", "r", encoding="utf-8") as f:
        bt_data = json.load(f)
    with open(MODEL_DIR / "visobert_and_ablations_results.json", "r", encoding="utf-8") as f:
        vis_data = json.load(f)

    # 1. Replace the 6 embedded figures (P[40], P[98], P[154], P[170], P[177], P[191])
    fig_map = {
        40: (FIG_DIR / "fig1_new.png", 6.3),
        98: (FIG_DIR / "fig2_new.png", 6.3),
        154: (FIG_DIR / "fig3_new.png", 6.3),
        170: (FIG_DIR / "fig4_new.png", 6.3),
        177: (FIG_DIR / "fig5_new.png", 6.3),
        191: (FIG_DIR / "fig6_new.png", 6.3),
    }
    for p_idx, (fpath, width_in) in fig_map.items():
        assert fpath.exists(), f"Missing figure: {fpath}"
        replace_figure_in_paragraph(doc.paragraphs[p_idx], fpath, width_inches=width_in)
        print(f"Replaced Figure in P[{p_idx}] with {fpath.name}")

    # 2. Update Equation (7) in P[152] from R¹²²⁵ to R⁸⁰⁶ and color in RED
    update_equation_7_xml(
        doc.paragraphs[152],
        "z(x) = [h(x); OneHot(t(x)); q(x)] ∈ ℝ⁸⁰⁶    (7)",
    )
    print("Updated Equation (7) in P[152] to R^806 in RED")

    # 3. Apply paragraph updates with red diff highlighting
    para_updates = build_paragraph_updates(eval_data, bt_data, vis_data)
    for p_idx, new_text in sorted(para_updates.items()):
        set_paragraph_red_diff(
            doc.paragraphs[p_idx], new_text, force_all_red=(p_idx == 4)
        )
    print(f"Updated {len(para_updates)} paragraphs with RED diff highlighting")

    # 4. Update all 12 tables with RED highlighting on modified/added cells
    table_updates = build_table_updates(eval_data, bt_data, vis_data)
    for t_idx, new_rows in sorted(table_updates.items()):
        update_table_with_red(doc.tables[t_idx], new_rows)
        print(f"Updated TABLE {t_idx} ({len(new_rows)} rows)")

    # 5. Insert new References [19]-[30] immediately after P[230] ([18]) in RED
    new_refs = build_new_references()
    anchor_p = doc.paragraphs[230]
    ref_style = anchor_p.style.name if anchor_p.style else "Normal"
    for ref_text in new_refs:
        anchor_p = insert_paragraph_after(anchor_p, ref_text, style_name=ref_style, all_red=True)
    print(f"Inserted {len(new_refs)} new references ([19]-[30]) in RED")

    # 6. Save updated document to TARGET_DOCX
    doc.save(str(TARGET_DOCX))
    print(f"Successfully saved updated manuscript to {TARGET_DOCX}")


if __name__ == "__main__":
    main()
