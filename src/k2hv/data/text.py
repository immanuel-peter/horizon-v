"""Question/answer normalization and rejection rules shared by every source."""

import re
import unicodedata

# Strings that would be split as Miles media placeholders, rejected by the processor or confused with template structure.
RESERVED = ("<image>", "</image>", "<video>", "<audio>", "reserved_special_token", "<|ifm|", "<ifm|think>", "</ifm|think>", "<think>", "</think>")
CODE = re.compile(r"(^|\n)\s*(import |from \S+ import |def |class \w+[:(])|```|\bplt\.|\bmatplotlib\b|\bpd\.DataFrame\b|\bnp\.array\(|<\s*(html|div|svg|table)\b|\\begin\{(tikzpicture|tabular|document)\}|\bprint\(", re.I)
FOLLOW_UP = re.compile(r"\b(previous|preceding|prior|last|earlier) (question|answer|response|turn)s?\b|\b(you|we) (said|mentioned|discussed)\b|\bas (mentioned|discussed|stated) (above|before|earlier)\b|^(and|also|what about|how about)\b", re.I)
MC_ANSWER = re.compile(r"^(?:answer\s*:\s*)?\(?([A-J])\)?\.?$", re.I)
SENTENCE_END = re.compile(r"[.!?](\s|$)")


def clean(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def key(text):
    """Comparison key for deduplication: case/space/punctuation-insensitive."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w%$.\-+/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().rstrip(".")


def words(text):
    return len(text.split())


def reserved(text):
    return next((r for r in RESERVED if r in text), None)


def strip_period(answer, max_words=8):
    """Drop a single packaging-added trailing period from a short answer ("Sweden." -> "Sweden")."""
    if answer.endswith(".") and not answer.endswith("..") and words(answer) <= max_words:
        return answer[:-1].rstrip()
    return answer


def mc_letter(answer):
    m = MC_ANSWER.match(answer.strip())
    return m.group(1).upper() if m else None


def sft_reasons(question, answer, rules):
    """Return the first rejection reason for an SFT question/answer pair, or None."""
    if not question:
        return "question_empty"
    if not answer:
        return "answer_empty"
    for field, text in (("question", question), ("answer", answer)):
        if reserved(text):
            return f"{field}_reserved_marker"
        if CODE.search(text):
            return f"{field}_code"
    if len(question) > rules["max_question_chars"]:
        return "question_too_long"
    if FOLLOW_UP.search(question):
        return "question_follow_up"
    if "\n" in answer and rules.get("single_line_answer", True):
        return "answer_multiline"
    if words(answer) > rules["max_answer_words"]:
        return "answer_too_long"
    if len(SENTENCE_END.findall(answer.rstrip("."))) >= rules.get("max_answer_sentences", 2):
        return "answer_explanation"
    return None


NUMBER = re.compile(r"^[-+]?(\d{1,3}(,\d{3})+|\d+)?(\.\d+)?$")


def is_number(text):
    t = text.strip()
    return bool(t) and any(c.isdigit() for c in t) and bool(NUMBER.match(t))


def decimals(text):
    t = text.strip()
    return len(t.split(".", 1)[1]) if "." in t else 0
