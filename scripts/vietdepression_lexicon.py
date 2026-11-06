"""Standalone Vietnamese Depression Lexicon, Text Cleaner, and Weak-Labeler.

Extracted from the original pipeline so that all training, evaluation, and
scientific audit scripts run 100% independently after removing `yt_depression_crawler`
and `web_demo`.

Also provides `label_comment_detailed()` to decompose the composite `need_review`
flag into its five constituent conditions (addressing Reviewer 1 Comment 4).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ── Text Cleaning Constants ──────────────────────────────────────────────────
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{14,}")
SPAM_PHRASES = {
    "sub4sub",
    "subscribe",
    "đăng ký kênh",
    "dang ky kenh",
    "kiếm tiền online",
    "kiem tien online",
    "nhận quà miễn phí",
    "nhan qua mien phi",
}

# ── Keyword Scoring Weights & Thresholds ─────────────────────────────────────
DEPRESSION_STRONG_WEIGHT = 5
DEPRESSION_MEDIUM_WEIGHT = 3
NORMAL_WEIGHT = -2
DEPRESSION_AUTO_THRESHOLD = 5
NORMAL_AUTO_THRESHOLD = -2
HIGH_CONFIDENCE_DEPRESSION_THRESHOLD = 8
HIGH_CONFIDENCE_NORMAL_THRESHOLD = -4

# ── Three-Tier Weighted Keyword Lexicon + 22 Review-Context Phrases ──────────
DEPRESSION_STRONG_KEYWORDS: list[str] = [
    "muốn chết", "muon chet",
    "tự tử", "tu tu",
    "không muốn sống", "khong muon song",
    "không thiết sống", "khong thiet song",
    "muốn biến mất", "muon bien mat",
    "muốn tan biến", "muon tan bien",
    "muốn ngủ mãi", "muon ngu mai",
    "kết thúc cuộc đời", "ket thuc cuoc doi",
    "chấm dứt cuộc đời", "cham dut cuoc doi",
    "không còn lý do sống", "khong con ly do song",
    "sống vô nghĩa", "song vo nghia",
    "cuộc sống vô nghĩa", "cuoc song vo nghia",
    "vô nghĩa", "vo nghia",
    "tuyệt vọng", "tuyet vong",
    "trầm cảm", "tram cam",
    "rối loạn trầm cảm", "roi loan tram cam",
    "đau khổ quá", "dau kho qua",
    "không chịu nổi nữa", "khong chiu noi nua",
    "không thể chịu nổi", "khong the chiu noi",
    "chỉ muốn buông xuôi", "chi muon buong xuoi",
    "buông xuôi tất cả", "buong xuoi tat ca",
    "không còn hy vọng", "khong con hy vong",
    "không còn tương lai", "khong con tuong lai",
    "chẳng còn gì để mất", "chang con gi de mat",
]

DEPRESSION_MEDIUM_KEYWORDS: list[str] = [
    "cô đơn", "co don",
    "mệt mỏi", "met moi",
    "áp lực", "ap luc",
    "stress",
    "mất ngủ", "mat ngu",
    "kiệt sức", "kiet suc",
    "burnout",
    "không còn động lực", "khong con dong luc",
    "mất động lực", "mat dong luc",
    "chán đời", "chan doi",
    "chán nản", "chan nan",
    "trống rỗng", "trong rong",
    "suy sụp", "suy sup",
    "bất lực", "bat luc",
    "lo âu", "lo au",
    "overthinking",
    "khóc một mình", "khoc mot minh",
    "không ai hiểu", "khong ai hieu",
    "áp lực gia đình", "ap luc gia dinh",
    "áp lực học tập", "ap luc hoc tap",
    "áp lực công việc", "ap luc cong viec",
    "mất phương hướng", "mat phuong huong",
    "tự ti", "tu ti",
    "ghét bản thân", "ghet ban than",
    "mất niềm tin", "mat niem tin",
    "không muốn nói chuyện", "khong muon noi chuyen",
    "cạn năng lượng", "can nang luong",
    "mệt quá", "met qua",
    "buồn quá", "buon qua",
    "đau lòng", "dau long",
    "tổn thương", "ton thuong",
]

NORMAL_KEYWORDS: list[str] = [
    "video hay",
    "hay quá", "hay qua",
    "cảm ơn", "cam on",
    "vui quá", "vui qua",
    "bổ ích", "bo ich",
    "ủng hộ", "ung ho",
    "hài quá", "hai qua",
    "thích quá", "thich qua",
    "tuyệt vời", "tuyet voi",
    "ý nghĩa quá", "y nghia qua",
    "cười xỉu", "cuoi xiu",
    "giải trí", "giai tri",
    "dễ thương", "de thuong",
    "đáng yêu", "dang yeu",
    "học được nhiều", "hoc duoc nhieu",
    "cảm động quá", "cam dong qua",
    "xem cuốn", "xem cuon",
    "quá đỉnh", "qua dinh",
    "đỉnh quá", "dinh qua",
    "rất hay", "rat hay",
    "thật ý nghĩa", "that y nghia",
    "chúc mừng", "chuc mung",
    "iu quá", "iu qua",
    "yêu quá", "yeu qua",
]

REVIEW_CONTEXT_KEYWORDS: list[str] = [
    "đừng tự tử", "dung tu tu",
    "không nên tự tử", "khong nen tu tu",
    "đừng nghĩ quẩn", "dung nghi quan",
    "hãy đi khám", "hay di kham",
    "tìm chuyên gia", "tim chuyen gia",
    "gọi đường dây nóng", "goi duong day nong",
    "hãy chia sẻ", "hay chia se",
    "cố lên", "co len",
    "mong bạn ổn", "mong ban on",
    "chết cười", "chet cuoi",
    "cười chết", "cuoi chet",
]


def normalize_text(text: object) -> str:
    """Collapse whitespace while preserving Vietnamese Unicode and emojis."""
    return WHITESPACE_RE.sub(" ", str(text or "").strip())


def normalize_comparison_key(value: object) -> str:
    """Return a Unicode-normalized (NFKC), casefolded comparison key."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return WHITESPACE_RE.sub(" ", text).strip().casefold()


def is_basic_spam(text: str) -> bool:
    """Minimal-intervention spam filter for YouTube comments."""
    normalized = normalize_text(text)
    lowered = normalized.lower()

    if URL_RE.search(normalized):
        return True
    if any(phrase in lowered for phrase in SPAM_PHRASES):
        return True
    if REPEATED_CHAR_RE.search(normalized):
        return True

    tokens = lowered.split()
    if len(tokens) >= 8 and len(set(tokens)) <= 2:
        return True

    return False


def strip_accents(text: str) -> str:
    """Strip Vietnamese diacritics so accented and unaccented text match."""
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return without_marks.replace("đ", "d").replace("Đ", "D")


def normalize_for_match(text: str) -> str:
    base = normalize_text(text).lower()
    no_accent = strip_accents(base)
    return f" {no_accent} "


def _prepare_keywords(keywords: list[str]) -> list[tuple[str, str]]:
    prepared: list[tuple[str, str]] = []
    seen: set[str] = set()
    for keyword in keywords:
        display = normalize_text(keyword).lower()
        match_key = strip_accents(display)
        if not display or match_key in seen:
            continue
        seen.add(match_key)
        prepared.append((display, match_key))
    return prepared


STRONG_KEYWORDS = _prepare_keywords(DEPRESSION_STRONG_KEYWORDS)
MEDIUM_KEYWORDS = _prepare_keywords(DEPRESSION_MEDIUM_KEYWORDS)
NORMAL_KEYWORDS_PREPARED = _prepare_keywords(NORMAL_KEYWORDS)
REVIEW_CONTEXTS = _prepare_keywords(REVIEW_CONTEXT_KEYWORDS)


def find_matches(
    text_for_match: str, prepared_keywords: list[tuple[str, str]]
) -> list[str]:
    matches: list[str] = []
    for display, match_key in prepared_keywords:
        if f" {match_key} " in text_for_match or match_key in text_for_match.strip():
            matches.append(display)
    return matches


@dataclass(frozen=True)
class LabelResult:
    weak_label: str
    confidence: str
    depression_score: int
    matched_keywords: str
    need_review: bool


@dataclass(frozen=True)
class DetailedLabelResult:
    weak_label: str
    confidence: str
    depression_score: int
    matched_keywords: str
    need_review: bool
    is_uncertain: bool
    near_boundary: bool
    has_mixed_signal: bool
    too_short_for_strong: bool
    has_review_context_phrase: bool
    matched_review_phrases: list[str]


def label_comment_detailed(comment_text: str) -> DetailedLabelResult:
    """Label a comment and return decomposed `need_review` sub-flags."""
    text = normalize_text(comment_text)
    text_for_match = normalize_for_match(text)
    strong_matches = find_matches(text_for_match, STRONG_KEYWORDS)
    medium_matches = find_matches(text_for_match, MEDIUM_KEYWORDS)
    normal_matches = find_matches(text_for_match, NORMAL_KEYWORDS_PREPARED)
    review_matches = find_matches(text_for_match, REVIEW_CONTEXTS)

    score = (
        len(strong_matches) * DEPRESSION_STRONG_WEIGHT
        + len(medium_matches) * DEPRESSION_MEDIUM_WEIGHT
        + len(normal_matches) * NORMAL_WEIGHT
    )

    if score >= DEPRESSION_AUTO_THRESHOLD:
        weak_label = "depression_auto"
        confidence = (
            "high" if score >= HIGH_CONFIDENCE_DEPRESSION_THRESHOLD else "medium"
        )
    elif score <= NORMAL_AUTO_THRESHOLD:
        weak_label = "normal_auto"
        confidence = (
            "high" if score <= HIGH_CONFIDENCE_NORMAL_THRESHOLD else "medium"
        )
    else:
        weak_label = "uncertain"
        confidence = "low"

    is_uncertain = weak_label == "uncertain"
    has_mixed_signal = bool((strong_matches or medium_matches) and normal_matches)
    near_boundary = score in {-3, -2, -1, 0, 1, 2, 3, 4, 5}
    too_short_for_strong = len(text) < 20 and bool(strong_matches)
    has_review_context_phrase = bool(review_matches)

    need_review = (
        is_uncertain
        or has_mixed_signal
        or has_review_context_phrase
        or near_boundary
        or too_short_for_strong
    )

    matched = strong_matches + medium_matches + normal_matches + review_matches
    return DetailedLabelResult(
        weak_label=weak_label,
        confidence=confidence,
        depression_score=score,
        matched_keywords="|".join(dict.fromkeys(matched)),
        need_review=need_review,
        is_uncertain=is_uncertain,
        near_boundary=near_boundary,
        has_mixed_signal=has_mixed_signal,
        too_short_for_strong=too_short_for_strong,
        has_review_context_phrase=has_review_context_phrase,
        matched_review_phrases=review_matches,
    )


def label_comment(comment_text: str) -> LabelResult:
    res = label_comment_detailed(comment_text)
    return LabelResult(
        weak_label=res.weak_label,
        confidence=res.confidence,
        depression_score=res.depression_score,
        matched_keywords=res.matched_keywords,
        need_review=res.need_review,
    )


RAW_COMMENT_COLUMNS = [
    "comment_id",
    "video_id",
    "video_title",
    "keyword",
    "comment_text",
    "like_count",
    "published_at",
]


def clean_comments(input_path: str | object, output_path: str | object) -> int:
    """Clean raw YouTube comments CSV into normalized, deduplicated CSV."""
    from pathlib import Path
    import pandas as pd

    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists() or inp.stat().st_size == 0:
        pd.DataFrame(columns=RAW_COMMENT_COLUMNS).to_csv(out, index=False, encoding="utf-8-sig")
        return 0

    try:
        df = pd.read_csv(inp, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        pd.DataFrame(columns=RAW_COMMENT_COLUMNS).to_csv(out, index=False, encoding="utf-8-sig")
        return 0

    if df.empty or "comment_text" not in df.columns:
        pd.DataFrame(columns=RAW_COMMENT_COLUMNS).to_csv(out, index=False, encoding="utf-8-sig")
        return 0

    df["comment_text"] = df["comment_text"].fillna("").astype(str).map(normalize_text)
    df = df[df["comment_text"].str.len() >= 5]
    df = df[~df["comment_text"].map(is_basic_spam)]
    df = df.drop_duplicates(subset=["comment_text"], keep="first").reset_index(drop=True)

    cols = [c for c in RAW_COMMENT_COLUMNS if c in df.columns]
    df[cols].to_csv(out, index=False, encoding="utf-8-sig")
    return len(df)

