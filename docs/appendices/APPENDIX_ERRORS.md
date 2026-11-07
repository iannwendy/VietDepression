# Appendix C: Privacy-Preserving Qualitative Error Analysis

To protect social-media user privacy against reverse-search re-identification under the **VietDepression Data Use Agreement (`DATA_USE_AGREEMENT.md`)** and CLPsych/eRisk ethics standards, all illustrative Vietnamese examples below are **privacy-preserving paraphrased composites** capturing the exact linguistic phenomena observed on the held-out in-domain test set (`final_test.csv`, $n = 1{,}371$: `271` Positive, `1,100` Negative).

---

## C.1 Reconciled Error Counts & Quantitative Error Taxonomy ($n = 1{,}371$)

| Model Configuration | True Neg (TN) | False Pos (FP) | False Neg (FN) | True Pos (TP) | Macro F1 | Accuracy |
|---------------------|---------------|----------------|----------------|---------------|----------|----------|
| **ViSoBERT + LA-Focal (Majority Vote)** | **994** | **106** | 109 | 162 | **0.7518** | **84.32%** |
| **ViSoBERT (Majority Vote)** | 974 | 126 | 101 | 170 | 0.7476 | 83.44% |
| **ViSoBERT + SupCon (Majority Vote)** | 977 | 123 | 103 | 168 | 0.7471 | 83.52% |
| **ViSoBERT + NCNL (Majority Vote)** | 980 | 120 | 105 | 166 | 0.7465 | 83.59% |
| **PhoBERT-base (Majority Vote)** | 933 | 167 | 94 | 177 | 0.7265 | 80.96% |
| **TF-IDF + Logistic Regression** | 906 | 194 | 89 | 182 | 0.7138 | 79.36% |
| **XLM-RoBERTa-base (Majority Vote)** | 820 | 280 | **68** | **203** | 0.6817 | 74.62% |
| **Lexical Weak Labeler (Abst. $\to$ Neg.)** | 932 | 168 | 149 | 122 | 0.6448 | 76.88% |
| **Qwen2.5-1.5B (4-Rule + 4-Shot)** | 703 | 397 | 112 | 159 | 0.5594 | 62.87% |

### Mutually Exclusive Error-Category Breakdown (Section 4.7, Table 16)

| Error Category (Count, `%` of FP/FN) | ViSoBERT + LA-Focal | PhoBERT-base | TF-IDF + LogReg | XLM-RoBERTa-base |
|--------------------------------------|--------------------:|-------------:|----------------:|-----------------:|
| **Total Errors (`%` of $n = 1{,}371$)** | **215 (15.68%)** | 261 (19.04%) | 283 (20.64%) | 348 (25.38%) |
| **False Positives (FP Total)** | **106 (100%)** | 167 (100%) | 194 (100%) | 280 (100%) |
| ↳ Rule 1: Supportive / Non-1st-person | 26 (24.5%) | 49 (29.3%) | 66 (34.0%) | 84 (30.0%) |
| ↳ Rule 2: Polysemy / Idiom / Mixed cue | 15 (14.2%) | 16 (9.6%) | 26 (13.4%) | 24 (8.6%) |
| ↳ Rule 3: Transient / Routine affect | 65 (61.3%) | 102 (61.1%) | 102 (52.6%) | 172 (61.4%) |
| **False Negatives (FN Total)** | 109 (100%) | 94 (100%) | 89 (100%) | **68 (100%)** |
| ↳ Implicit / Figurative ($s(x) \le 0$) | 43 (39.4%) | 41 (43.6%) | 46 (51.7%) | 34 (50.0%) |
| ↳ Brief ($\le 25$w) / Pronoun-dropped | 20 (18.3%) | 17 (18.1%) | 15 (16.9%) | 11 (16.2%) |
| ↳ Extended narrative dilution ($> 25$w) | 46 (42.2%) | 36 (38.3%) | 28 (31.5%) | 23 (33.8%) |

Key empirical shift: Social-media subword pretraining with logit-adjusted focal modulation (**ViSoBERT + LA-Focal**) cuts overall test misclassifications to **15.68%** (`215` errors), eliminating **61 false positives** relative to formal-register **PhoBERT-base** (`106` vs. `167`), **88** relative to **TF-IDF + Logistic Regression** (`106` vs. `194`), and **174** relative to multilingual **XLM-RoBERTa-base** (`106` vs. `280`), while cutting Rule 1 supportive/non-first-person false alarms by **69.0%** (`84 -> 26` vs. `XLM-R-base`) and **60.6%** (`66 -> 26` vs. `LogReg`).

---

## C.2 Characteristic False Positive (FP) Categories

False positives arise when a classifier predicts **`Positive (1)`** on comments whose gold rubric label is **`Negative (0)`**:

### 1. Empathetic Second/Third-Person Support (Rule 1 Ambiguity)
- **Paraphrased Vietnamese Composite:** *"Đọc tâm sự của bạn mà thương quá, đừng buồn nhiều nữa nhé, cố lên rồi bão giông nào cũng sẽ qua."*
- **English Translation:** *"Reading your story makes me feel so sorry for you; please don't be so sad anymore, stay strong and every storm will pass."*
- **Gold Label:** `Negative (0)` (Rule 1 — directed at another commenter)
- **Diagnostic Behavior:** Sparse TF-IDF (`LogReg`) and prompted `Qwen2.5-1.5B` frequently trigger false alarms due to dense affective lexemes (`"thương quá"`, `"buồn"`), whereas `ViSoBERT + LA-Focal` resolves second-person pronouns (`"của bạn"`) and imperative encouragement (`"đừng ... nữa nhé"`).

### 2. Undiacritized Slang & Lexical Polysemy (Rule 2 Ambiguity)
- **Paraphrased Vietnamese Composite:** *"Nghe giọng đọc ấm áp quá làm mình buồn ngủ díp cả mắt luôn, cảm ơn kênh nhiều."*
- **English Translation:** *"Listening to such a warm voice makes me so sleepy my eyes are closing, thank you so much to the channel."*
- **Gold Label:** `Negative (0)` (Rule 2 — lexical compound `"buồn ngủ"` denotes sleepiness, often written without diacritics as `"buon ngu"`)
- **Diagnostic Behavior:** Word-segmented/formal encoders occasionally split or misread undiacritized informal compounds, whereas native SentencePiece social-media subwords in `ViSoBERT` preserve phrase-level semantics.

### 3. Situational Academic / Work Fatigue Without Distress (Rule 3 Boundary)
- **Paraphrased Vietnamese Composite:** *"Cuối kỳ đồ án dồn dập thức trắng hai hôm nay làm mình mệt rã rời, chỉ mong thi xong để ngủ bù."*
- **English Translation:** *"End-of-term projects piling up for two sleepless nights have left me exhausted; I just hope exams finish so I can catch up on sleep."*
- **Gold Label:** `Negative (0)` (Rule 3 — transient situational deadline fatigue with concrete recovery expectation)
- **Diagnostic Behavior:** High-recall models flag `"thức trắng"` and `"mệt rã rời"`, representing the hardest boundary between routine exhaustion and persistent emotional burnout.

---

## C.3 Characteristic False Negative (FN) Categories

False negatives arise when a classifier predicts **`Negative (0)`** on comments whose gold rubric label is **`Positive (1)`**:

### 1. Pronoun-Dropped Implicit Helplessness (Rule 1 + Rule 3)
- **Paraphrased Vietnamese Composite:** *"Chẳng còn biết phải bước tiếp thế nào nữa, nhìn đâu cũng chỉ thấy một màu xám xịt."*
- **English Translation:** *"No longer knowing how to take the next step anymore; everywhere I look is just bleak gray."*
- **Gold Label:** `Positive (1)` (Implicit first-person hopelessness without explicit `"mình"`/`"tôi"` or clinical keywords)
- **Diagnostic Behavior:** The 335-entry lexicon abstains (`s(x) = 0`), and conservative classifiers miss short metaphor-driven utterances lacking explicit symptom nouns.

### 2. Understated Emotional Numbness & Anhedonia (Rule 3)
- **Paraphrased Vietnamese Composite:** *"Lâu lắm rồi mình không còn thấy vui hay buồn với bất cứ chuyện gì xung quanh nữa, cứ trôi qua ngày vậy thôi."*
- **English Translation:** *"For a very long time I haven't felt happy or sad about anything around me anymore, just drifting through each day."*
- **Gold Label:** `Positive (1)` (Rule 3 — classic emotional blunting/anhedonia)
- **Diagnostic Behavior:** Negation of affect terms (`"không còn thấy vui hay buồn"`) neutralizes surface sentiment cues, remaining the primary source of residual false negatives across all transformer families.
