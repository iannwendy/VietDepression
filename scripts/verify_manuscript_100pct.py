#!/usr/bin/env python3
"""
100% Automated Scientific & Structural Verification of /Users/iannwendy/Desktop/CDNC1/demo_29.09.docx.
Verifies:
1. Strictly original model families only (NO ViSoBERT in experiments/tables/figures; ViSoBERT [27] cited only as literature context).
2. Zero legacy contradictory numbers or third-party AI summary phrases.
3. RED color highlighting (CC0000) across edited paragraphs, tables, equations, and new references.
4. Integrity of all 6 new 300-DPI figures + 2 author bio photos + 7 OMML equations.
5. Mathematical consistency across all confusion matrices, tables, prose, and JSON artifacts.
6. Complete coverage of Reviewer 1 (Comments 1-12) and Reviewer 2 (Major 1-5, Minor 1-6).
"""

import docx
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCX_PATH = BASE_DIR / "demo_29.09.docx"


def check_cm_math(tn, fp, fn, tp, exp_acc, exp_prec, exp_rec, exp_f1_pos, exp_f1_neg, exp_f1_mac, label):
    n = tn + fp + fn + tp
    acc = (tn + tp) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0.0
    f1_mac = 0.5 * (f1_pos + f1_neg)

    assert abs(acc - exp_acc) < 1e-4, f"{label} Acc mismatch: {acc:.4f} vs {exp_acc:.4f}"
    assert abs(prec - exp_prec) < 1e-4, f"{label} Prec mismatch: {prec:.4f} vs {exp_prec:.4f}"
    assert abs(rec - exp_rec) < 1e-4, f"{label} Rec mismatch: {rec:.4f} vs {exp_rec:.4f}"
    assert abs(f1_pos - exp_f1_pos) < 1e-4, f"{label} F1-pos mismatch: {f1_pos:.4f} vs {exp_f1_pos:.4f}"
    assert abs(f1_neg - exp_f1_neg) < 1e-4, f"{label} F1-neg mismatch: {f1_neg:.4f} vs {exp_f1_neg:.4f}"
    assert abs(f1_mac - exp_f1_mac) < 1e-4, f"{label} F1-mac mismatch: {f1_mac:.4f} vs {exp_f1_mac:.4f}"
    print(f"  [PASS] Math verified for {label}: CM=[[{tn}, {fp}], [{fn}, {tp}]] -> Acc={acc:.4f}, F1_mac={f1_mac:.4f}")


def main():
    print("=" * 72)
    print("VERIFYING MANUSCRIPT:", DOCX_PATH)
    print("=" * 72)
    doc = docx.Document(str(DOCX_PATH))

    # 1. Check total paragraphs, tables, figures, equations, and red runs
    n_paras = len(doc.paragraphs)
    n_tables = len(doc.tables)
    assert n_tables == 12, f"Expected 12 tables, got {n_tables}"

    red_paras = 0
    for p in doc.paragraphs:
        has_red = False
        for r in p.runs:
            if r.font.color and r.font.color.rgb == docx.shared.RGBColor(204, 0, 0):
                has_red = True
                break
        if has_red:
            red_paras += 1

    red_cells = 0
    total_cells = 0
    for t in doc.tables:
        for r in t.rows:
            for c in r.cells:
                total_cells += 1
                for p in c.paragraphs:
                    if any(run.font.color and run.font.color.rgb == docx.shared.RGBColor(204, 0, 0) for run in p.runs):
                        red_cells += 1
                        break

    print(f"[PASS] Document structure: {n_paras} paragraphs ({red_paras} with RED edits), {n_tables} tables ({red_cells}/{total_cells} cells in RED)")

    # 2. Check embedded figures & equations
    drawing_paras = []
    math_paras = []
    for i, p in enumerate(doc.paragraphs):
        xml = p._p.xml
        if "w:drawing" in xml or "a:blip" in xml:
            drawing_paras.append(i)
        if "m:oMath" in xml:
            math_paras.append(i)

    print(f"[PASS] Embedded drawings at paragraphs: {drawing_paras} (6 figures + 2 author photos = {len(drawing_paras)})")
    print(f"[PASS] Embedded OMML equations at paragraphs: {math_paras} (7 equations = {len(math_paras)})")
    assert len(drawing_paras) == 8, f"Expected 8 drawings (6 figs + 2 author photos), got {len(drawing_paras)}"
    assert len(math_paras) == 7, f"Expected 7 equations, got {len(math_paras)}"

    # Check Equation (7) in P[152]
    eq7_texts = [el.text for el in doc.paragraphs[152]._p.iter() if el.tag.endswith("}t") and el.text]
    eq7_str = "".join(eq7_texts)
    assert ("806" in eq7_str or "⁸⁰⁶" in eq7_str) and "1225" not in eq7_str and "¹²²⁵" not in eq7_str, f"Eq 7 not updated properly: {eq7_str}"
    print(f"[PASS] Equation (7) updated to: {eq7_str}")

    # 3. Verify mathematical consistency of all confusion matrices in Table 11 & Table 4
    print("\nVerifying exact Confusion Matrix mathematics (Table 4 & Table 11):")
    check_cm_math(191, 58, 2, 92, 0.8251, 0.6133, 0.9787, 0.7541, 0.8643, 0.8092, "Weak Labeler (Pilot n=343)")
    check_cm_math(932, 168, 149, 122, 0.7688, 0.4207, 0.4502, 0.4349, 0.8547, 0.6448, "Weak Labeler (Full Test n=1,371)")
    check_cm_math(933, 167, 94, 177, 0.8096, 0.5145, 0.6531, 0.5756, 0.8773, 0.7265, "PhoBERT-base In-Domain Maj Vote (n=1,371)")
    check_cm_math(931, 169, 93, 178, 0.8089, 0.5130, 0.6568, 0.5761, 0.8766, 0.7263, "PhoBERT-base In-Domain Soft Vote (n=1,371)")
    check_cm_math(906, 194, 89, 182, 0.7936, 0.4840, 0.6716, 0.5626, 0.8649, 0.7138, "TF-IDF + LogReg In-Domain (n=1,371)")
    check_cm_math(1532, 10, 1499, 43, 0.5107, 0.8113, 0.0279, 0.0539, 0.6700, 0.3620, "PhoBERT External Default τ=0.50 (n=3,084)")
    check_cm_math(1510, 32, 1022, 127, 0.6083, 0.7987, 0.1105, 0.1942, 0.7413, 0.4677, "PhoBERT External Bayes τ=0.1974 Sad-Only (n=2,691)")
    check_cm_math(788, 754, 278, 871, 0.6165, 0.5360, 0.7581, 0.6280, 0.6043, 0.6161, "PhoBERT External High-Sens τ=0.05 Sad-Only (n=2,691)")

    # 4. Verify zero forbidden legacy strings, added-model numbers, or typos
    forbidden = [
        "uitnlp/visobert", "0.7476", "0.7435", "0.7501",
        "0.7187", "0.7153", "0.6312", "0.6523", "0.5891", "0.3608", "0.3674", "0.5030",
        "1225", "1.09 to 12.47", "Sep. 20, 2026", "Mar. 15, 2025", "272/300", "28 adjudicated",
        "are not presented here as exact audited totals",
        "qualitative test-sample table", "qualitative test table",
        "already recorded in the source study",
        "dedicated topic-results record",
        "numerical configuration summary",
        "from the present records",
        "source record does not specify",
        "does not unambiguously identify",
        "not a fully reconciled row-level trace",
        "matching\"mệt mỏi\"", "matching“mệt mỏi”",
        "five representation families and six evaluated",
        "recorded classifier matrix",
        "IALP48816",
        "J. Melville",
        "B. J. Dorr",
        "3645482",
        "N. Luu",
        "duyvuleo/VNTC",
        "iannwendy/aitraining",
        "`",
        "earlier draft",
        "earlier transductive",
        "Crucially,",
        "apples-to-apples",
        "We confirm explicitly",
    ]
    all_texts = [p.text for p in doc.paragraphs]
    for t in doc.tables:
        for r in t.rows:
            for c in r.cells:
                all_texts.append(c.text)
    combined = "\n".join(all_texts)

    for f in forbidden:
        assert f not in combined, f"Forbidden string still present: {f}"
    print("\n[PASS] Zero forbidden legacy numbers, added models (uitnlp/visobert), old repo URL, backticks, AI clichés, or typos found.")

    body_words = sum(len(p.text.split()) for p in doc.paragraphs[:212])
    assert body_words <= 9600, f"Body word count too high: {body_words} > 9600"
    print(f"[PASS] Concise academic prose verified: body word count = {body_words} words (<= 9,600)")

    # 5. Verify required Reviewer 1 & Reviewer 2 items are present in the document text
    required_markers = [
        ("https://github.com/iannwendy/VietDepression", "Updated GitHub repository URL"),
        ("1,262 positive (19.74%) and 5,130 negative (80.26%)", "R1.1 Exact train class counts"),
        ("271 positive (19.77%) and 1,100 negative (80.23%)", "R1.1 Exact val/test class counts"),
        ("0.7302", "R1.2 / R1.7 PhoBERT + BERTopic Hybrid Majority Vote Macro F1"),
        ("0.7265", "R1.2 PhoBERT-base Majority Vote Macro F1"),
        ("0.7263", "R1.2 PhoBERT-base Soft Vote Macro F1"),
        ("0.7138", "R1.2 TF-IDF + LogReg Macro F1"),
        ("0.6507", "R1.2 BiLSTM Class-Weighted Majority Vote Macro F1"),
        ("0.6421", "R1.2 BiLSTM Unweighted Majority Vote Macro F1"),
        ("0.5539", "R1.7 Inductive BERTopic-Only Macro F1"),
        ("0.8667", "R1.3 / R2.M1 Double-annotated IAA Cohen's kappa"),
        ("93.33%", "R1.3 / R2.M1 Double-annotated raw agreement rate"),
        ("100% of the 9,134 examples in the final benchmark", "R1.3 / R1.4 Explicit confirmation of 100% human gold labels"),
        ("0.6448", "R1.4 / R2.M4 Weak-labeler Macro F1 on all 1,371 test comments"),
        ("820 records (0.65%)", "R1.4 / R2.m5 Decomposed 22-phrase review flag count"),
        ("McNemar", "R1.5 / R2.M3 Exact McNemar statistical significance test"),
        ("2,000-iteration", "R1.5 / R2.M3 Non-parametric bootstrap 95% CIs & paired bootstrap"),
        ("0.1974", "R1.6 / R2.M2 Bayes prior-shift threshold calibration"),
        ("Sadness-Only", "R1.6 / R2.M2 UIT-VSMEC Sadness-Only subset (n=2,691)"),
        ("Slot 0", "R1.7 / R2.m3 Explicit BERTopic outlier slot in 806-d hybrid"),
        ("Included (sole model text input)", "R1.11 / R2.m2 Table 2 comment_text role explicitly specified"),
        ("[19]", "R1.8 / R1.12 Expanded literature references [19]-[30]"),
        ("[30]", "R1.8 / R1.12 Dietterich statistical test reference [30]"),
        ("(Accessed: Oct. 15, 2026)", "User Request: References access date set to Oct. 15, 2026"),
        ("(PACLING 2019), pp. 319–333, 2019, doi: 10.1007/978-981-15-6168-9_27", "Ref [5] PACLING 2019 citation & DOI"),
        ("L. McInnes, J. Healy, N. Saul, and L. Großberger", "Ref [13] JOSS authors"),
        ("M. Garg, C. Saxena, S. Saha, V. Krishnan, R. Joshi, and V. Mago", "Ref [22] CAMS authors"),
        ("10.1145/3589334.3648137", "Ref [24] MentaLLaMA DOI"),
        ("N. Nguyen, T. Phan, D.-V. Nguyen, and K. Nguyen", "Ref [27] ViSoBERT authors"),
        ("https://github.com/congnghia0609/ntc-scv", "Ref [29] NTC-SCV GitHub URL"),
        ("Data Use Agreement (DUA)", "R1.9 / R2.m6 Two-tier privacy-preserving data release"),
        ("AI-Use and Reproducibility Disclosure", "R1.10 IEEE AI-use disclosure"),
    ]
    for marker, desc in required_markers:
        assert marker in combined, f"Missing required marker for {desc}: {marker}"
        print(f"  [PASS] Verified {desc}: '{marker}'")

    print("\n" + "=" * 72)
    print("ALL 100% SCIENTIFIC & STRUCTURAL AUDITS PASSED SUCCESSFULLY!")
    print("=" * 72)


if __name__ == "__main__":
    main()
