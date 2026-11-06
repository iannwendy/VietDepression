# VietDepression Annotation Guidelines & Adjudication Protocol

This document provides the complete English specification of the **Four-Rule Operational Annotation Rubric**, blind Label Studio setup, edge-case decision criteria, and senior adjudication workflow used to construct the `9,134`-comment **VietDepression** gold benchmark. (The original Vietnamese annotator handbook for Label Studio is archived alongside this file at `docs/appendices/manual_review/HUONG_DAN_REVIEW_LABEL_STUDIO.md`.)

---

## 1. Construct Definition and Non-Clinical Scope

- **Unit of Analysis:** A single isolated, Unicode NFC-normalized Vietnamese YouTube comment $x$, without access to author posting history, clinical records, or video metadata during labeling.
- **Operational Target ($y \in \{0, 1\}$):**
  - **`Positive (1)` — Depression-Related Distress Language:** First-person (explicit or contextually implicit self-referential) expressions of persistent sadness, emotional exhaustion, anhedonia (loss of interest/motivation), hopelessness, social isolation, psychogenic sleep disturbance, severe self-deprecation/worthlessness, or self-harm/suicidal ideation.
  - **`Negative (0)` — Normal / Non-Distress Language:** Appreciative or neutral commentary, third-person discussions about mental health, supportive replies encouraging another user, figurative idioms/slang containing affective words, and transient situational complaints (e.g., exam or deadline fatigue without psychological decompensation).
  - **`Exclude` (Pre-Partition Filtering):** Non-Vietnamese text, commercial spam/URLs, or unintelligible symbol-only strings; excluded records do not enter the `9,134` binary gold benchmark.

---

## 2. Blind Annotation Protocol (Anti-Anchoring Design)

During all human review and active-learning rounds (Stages 4–5):
1. **Zero Machine Cues:** The Label Studio interface displays **only** the anonymized `row_id` and normalized `comment_text`. All lexical weak scores $s(x)$, matched keywords, review flags, and model probabilities $p(y = 1 \mid x)$ are strictly hidden from annotators to eliminate confirmation and anchoring bias.
2. **Annotator Pool & Adjudication:** Primary labeling is performed by trained native Vietnamese NLP annotators fluent in youth internet sociolects, omitted-diacritic writing, and Vietnamese pronoun pragmatics, with ambiguous or disagreed boundary items adjudicated by a psychology-trained senior adjudicator under Rules 1–4.

---

## 3. The Four-Rule Annotation Rubric

### Rule 1 — Speaker Perspective & Target Attribution
- **Assign `Positive (1)`** **only** when psychological distress is experienced by the **speaker/author** (`mình`, `tôi`, `em`, `con`, `bản thân`, or Vietnamese zero-pronoun first-person clauses where self-reference is unambiguous from context).
- **Assign `Negative (0)`** when distress vocabulary describes **another person**, a fictional character/song lyric, general educational facts about depression, or **second/third-person supportive replies** directed at fellow commenters.
  - *Paraphrased Composite (`Negative 0`):* *"Bạn đừng buồn nhiều nữa nhé, cố lên rồi mọi khó khăn cũng sẽ qua thôi."* ("Please don't be so sad anymore, stay strong and all difficulties will pass.") — Contains `"buồn"` and `"cố lên"`, but expresses empathy toward another user rather than personal distress.

### Rule 2 — Lexical Polysemy, Compound Words & Hyperbolic Idioms
- **Assign `Negative (0)`** when affective morphemes appear inside lexicalized Vietnamese compounds or colloquial hyperbole whose compositional meaning is non-depressive:
  - `"buồn ngủ"` (sleepy), `"buồn cười"` (amusing/funny), `"buồn miệng"` (wanting a snack) $\neq$ `"buồn"` (sadness).
  - `"chết cười"` / `"cười chết mất"` (dying of laughter), `"mệt xỉu vì dễ thương"` (slang admiration) $\neq$ self-harm or distress.
  - *Paraphrased Composite (`Negative 0`):* *"Ngồi nghe giảng mà buồn ngủ ríu cả mắt, ngày mai lại kiểm tra sớm rồi."* ("Sitting in lecture feeling so sleepy my eyes are closing, and tomorrow is an early exam.")

### Rule 3 — Persistent Psychological Distress vs. Transient Situational Complaints
- **Assign `Positive (1)`** when the utterance conveys **pervasive, recurring, or internalized emotional distress**, including:
  - Chronic emotional exhaustion, emptiness, or inability to find meaning/motivation (*anhedonia*).
  - Recurring nighttime crying, persistent insomnia tied to rumination, or feeling completely isolated/unheard.
  - Deep self-blame, feeling worthless/burdensome, or feeling trapped without a way forward—**even when clinical terms like `"trầm cảm"` are absent**.
  - *Paraphrased Composite (`Positive 1`):* *"Thêm một ngày trôi qua vô nghĩa, mình thấy bản thân cứ mãi dậm chân tại chỗ trong bóng tối."* ("Another meaningless day has passed; I feel stuck in the dark without direction.")
- **Assign `Negative (0)`** for **ordinary situational complaints**, temporary physical tiredness after work/exercise, traffic frustration, or short-term exam/assignment stress without emotional decompensation.
  - *Paraphrased Composite (`Negative 0`):* *"Tuần này bài tập nhóm với hạn nộp dồn dập làm mình mệt bở hơi tai luôn."* ("This week group assignments and deadlines piling up are making me completely worn out.")

### Rule 4 — Crisis Expressions vs. Protective Counter-Speech
- **Assign `Positive (1)`** to any first-person statement of self-harm, wishing to disappear/never wake up, or suicidal ideation (e.g., first-person `"không muốn tồn tại nữa"`).
- **Assign `Negative (0)`** to **protective counter-speech, helpline sharing, and anti-suicide encouragement** directed at others (e.g., `"đừng tự tử nhé, hãy chia sẻ với gia đình hoặc đi khám tâm lý"`), even though such replies match high-weight crisis lexemes.

---

## 4. Boundary Resolution & Senior Adjudication Checklist

When an utterance contains mixed signals or omitted pronouns:
1. **Mixed Supportive + Self-Disclosure Clauses:** If a commenter both encourages others *and* discloses their own ongoing depressive suffering (e.g., *"Cố lên bạn nhé, mình cũng đang rơi vào bế tắc suốt mấy tháng nay không thoát ra được"*), **Rule 1 + Rule 3** classify the comment as **`Positive (1)`** because genuine first-person persistent distress is present.
2. **Recovered Past vs. Active Distress:** Purely retrospective statements of completed recovery without current distress (e.g., *"Ba năm trước mình từng khủng hoảng nhưng giờ mình đã ổn và yêu đời lại rồi"*) are labeled **`Negative (0)`**, whereas ongoing relapse or unresolved suffering is labeled **`Positive (1)`**.
3. **Pronoun-Dropped Fragments:** Short Vietnamese clauses lacking explicit pronouns (e.g., *"Chẳng biết phải làm sao tiếp theo nữa, bế tắc thật sự"*) are evaluated for implicit first-person experiential framing; if unambiguously self-experiential and expressing helplessness/despair (Rule 3), assign **`Positive (1)`**; if purely ambiguous reaction commentary without distress context, refer to senior adjudication.
