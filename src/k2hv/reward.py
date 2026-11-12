"""Final-answer scorers for K2-Horizon-V RL rows (data admission and RL training).

Contract, for every selector:

1. The response is the generated text after the generation prompt, which ends with the opening think
   tag for the reasoning effort (``<ifm|think>`` high, ``<ifm|think_fast>`` medium,
   ``<ifm|think_faster>`` low). Only the text after the matching closing tag is scored. A missing,
   repeated or mismatched closing tag is a format failure: score 0, never credit for an answer
   written inside the reasoning.
2. The final answer is the value on the last ``Answer:`` line after the closing tag, or else the last
   ``\\boxed{...}`` there. A boxed value on the ``Answer:`` line is unboxed.
3. Selectors (``metadata.data_source``):
   ``math_v1``    exact equality after normalization, or exact symbolic equivalence (rationals,
                       roots, pi, single-letter variables) with sympy. ``1/2``, ``\\frac12``,
                       ``\\dfrac{1}{2}`` and ``0.5`` are equal; ``0.33`` and ``\\frac13`` are not.
   ``choice_v1``  the option letter A-J.
   ``decimal_v1`` a number that rounds (half up) to the target at the precision the question
                       states, given as ``metadata.answer_decimals``.
   The legacy spellings ``k2hv_math_v1``, ``k2hv_choice_v1`` and ``k2hv_decimal_v1`` from the
   first dataset build are accepted as aliases for the same three selectors.
4. Model-format failures score 0 with an ``outcome`` naming the cause. Infrastructure failures
   (aborted or failed rollouts, scorer exceptions) return ``score=None`` so group filters can drop
   the attempt instead of training on a zero. An unknown selector raises.

Expressions are parsed by a small whitelist parser that builds sympy objects directly; model output
is never evaluated as Python. Dependencies: sympy (installed with torch).
"""

import logging
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

logger = logging.getLogger(__name__)

MATH = "math_v1"
CHOICE = "choice_v1"
DECIMAL = "decimal_v1"
SELECTORS = (MATH, CHOICE, DECIMAL)
LEGACY_SELECTORS = {"k2hv_math_v1": MATH, "k2hv_choice_v1": CHOICE, "k2hv_decimal_v1": DECIMAL}


def canonical_selector(selector):
    if selector in SELECTORS:
        return selector
    return LEGACY_SELECTORS.get(selector)

THINK = {"high": "ifm|think", "medium": "ifm|think_fast", "low": "ifm|think_faster"}
CLOSE_TAG = re.compile(r"</ifm\|think(?:_fast|_faster)?>")
OPEN_TAG = re.compile(r"<ifm\|think(?:_fast|_faster)?>")
END_TOKENS = ("<|ifm|im_end|>", "<|ifm|endoftext|>")
ANSWER_LINE = re.compile(r"(?im)^[ \t>*_#]*(?:final\s+)?answer\s*[:：]\s*(.*?)\s*$")
CHOICE_LETTER = re.compile(r"^\(?\s*([A-J])\s*\)?(?:[.:)：,]|\s|$)")

UNITS = ("square units", "sq units", "sq. units", "cubic units", "units", "unit", "square centimeters", "square meters", "square feet", "square inches", "square yards",
         "square miles", "square kilometers", "square millimeters", "cubic centimeters", "cubic meters", "cubic inches", "cubic feet",
         "sq in", "sq ft", "sq. in", "sq. ft", "in^2", "in^3", "in²", "in³", "ft^2", "ft^3", "ft²", "ft³", "yd^2", "mm^2", "km^2", "ft", "yd",
         "centimeters", "centimetres", "meters", "metres", "kilometers", "millimeters", "inches", "inch", "feet", "foot", "yards", "miles", "cm^2", "cm^3",
         "m^2", "m^3", "cm²", "cm³", "m²", "m³", "cm", "mm", "km", "m", "kg", "g", "degrees", "degree", "deg", "dollars", "dollar", "cents", "hours", "hour",
         "minutes", "minute", "seconds", "second", "years", "year", "days", "day", "percent", "people", "students", "items", "points", "times",
         "m/s^2", "m/s²", "m/s", "km/h", "rad/s", "g/cm^3", "kg/m^3", "N/m", "kPa", "Pa", "Hz", "kJ", "mol", "mL", "N", "J", "W", "V", "L", "s")
UNIT_TAIL = re.compile(r"(?:\s|\\,|\\;|~)*(?:" + "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True)) + r")\.?$")
MAX_LEN = 120


class ScoreError(Exception):
    pass


def result(score, outcome, pred=None):
    return {"score": score, "acc": score == 1.0, "outcome": outcome, "pred": pred}


def final_region(response, effort=None):
    """Text after the closing think tag, or (None, failure outcome)."""
    text = response
    for token in END_TOKENS:
        text = text.replace(token, "")
    closes = CLOSE_TAG.findall(text)
    if not closes:
        return None, "format_unclosed_think"
    if len(closes) > 1:
        return None, "format_repeated_think_close"
    if effort is not None and closes[0] != f"</{THINK[effort]}>":
        return None, "format_wrong_think_tag"
    region = text.split(closes[0], 1)[1]
    if OPEN_TAG.search(region):
        return None, "format_think_after_close"
    return region, None


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0 or text[idx + 6 : i].strip():
        return None
    depth = 0
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1 : j].strip()
    return None


def extract_answer(region):
    lines = ANSWER_LINE.findall(region)
    if lines:
        value = lines[-1].strip()
        boxed = last_boxed(value)
        return boxed if boxed is not None else value
    return last_boxed(region)


def strip_delimiters(s):
    s = s.strip()
    for left, right in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if s.startswith(left) and s.endswith(right) and len(s) >= len(left) + len(right):
            return strip_delimiters(s[len(left) : len(s) - len(right)])
    return s.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "").replace("$", "").strip()


def normalize(s):
    """Notation-only normalization shared by targets and predictions."""
    s = strip_delimiters(s.replace("\\$", ""))
    boxed = last_boxed(s)
    if boxed is not None and s.startswith("\\boxed"):
        s = boxed
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|displaystyle|,|;|!|:|quad|qquad)", " ", s)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\degree", "").replace("°", "").replace("\\%", "").replace("%", "")
    s = s.replace("−", "-").replace("×", "*").replace("·", "*").replace("\\cdot", "*").replace("\\times", "*")
    s = s.strip().rstrip(".").strip()
    if "=" in s:
        s = s.rsplit("=", 1)[1].strip()
    while True:
        stripped = UNIT_TAIL.sub("", s).strip()
        if stripped == s or not stripped or stripped[-1] in "+-*/^(=" or re.search(r"[A-Za-z]", re.sub(r"\\[A-Za-z]+", "", stripped)):
            break
        s = stripped
    s = s.replace("{,}", ",")
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s



def _group(s, i):
    """Return (content, next index) of a {...} group or a single character at s[i]."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        raise ScoreError("unexpected end")
    if s[i] != "{":
        return s[i], i + 1
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1 : j], j + 1
    raise ScoreError("unbalanced braces")


def _delatex(s):
    out, i = [], 0
    while i < len(s):
        if s.startswith("\\frac", i):
            a, i = _group(s, i + 5)
            b, i = _group(s, i)
            out.append(f"(({_delatex(a)})/({_delatex(b)}))")
        elif s.startswith("\\sqrt", i):
            i += 5
            index = None
            if i < len(s) and s[i] == "[":
                j = s.index("]", i)
                index, i = s[i + 1 : j], j + 1
            a, i = _group(s, i)
            out.append(f"root(({_delatex(a)}),({_delatex(index)}))" if index else f"sqrt({_delatex(a)})")
        elif s.startswith("\\pi", i):
            out.append(" pi ")
            i += 3
        elif s[i] == "√":
            m = re.match(r"\s*(\d+(?:\.\d+)?|[a-z]|\{|\()", s[i + 1 :])
            if not m:
                raise ScoreError("bad root")
            if m.group(1) in "{(":
                close = {"{": "}", "(": ")"}[m.group(1)]
                start = i + 1 + m.start(1)
                depth = 0
                for j in range(start, len(s)):
                    depth += 1 if s[j] == m.group(1) else -1 if s[j] == close else 0
                    if depth == 0:
                        break
                out.append(f"sqrt({_delatex(s[start + 1 : j])})")
                i = j + 1
            else:
                out.append(f"sqrt({m.group(1)})")
                i += 1 + m.end(1)
        elif s[i] == "π":
            out.append(" pi ")
            i += 1
        elif s[i] == "\\":
            raise ScoreError("unsupported command")
        elif s[i] in "{}":
            out.append("(" if s[i] == "{" else ")")
            i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


TOKEN = re.compile(r"\s*(?:(\d+\.\d*|\.\d+|\d+)|(sqrt|root|pi|[a-z])|(\*\*|[-+*/^(),]))")


def _tokens(s):
    toks, i = [], 0
    s = s.strip()
    while i < len(s):
        m = TOKEN.match(s, i)
        if not m or m.end() == i:
            raise ScoreError(f"bad token at {s[i:i + 8]!r}")
        num, name, op = m.groups()
        toks.append(("num", num) if num else ("name", name) if name else ("op", "**" if op == "^" else op))
        i = m.end()
        while i < len(s) and s[i] == " ":
            i += 1
    return toks


class _Parser:
    def __init__(self, toks, sympy):
        self.toks, self.i, self.sp = toks, 0, sympy

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self, value=None):
        tok = self.peek()
        if tok[0] is None or (value is not None and tok[1] != value):
            raise ScoreError(f"expected {value}")
        self.i += 1
        return tok

    def parse(self):
        e = self.expr()
        if self.i != len(self.toks):
            raise ScoreError("trailing tokens")
        return e

    def expr(self):
        e = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            t = self.term()
            e = e + t if op == "+" else e - t
        return e

    def term(self):
        e = self.unary()
        while True:
            tok = self.peek()
            if tok in (("op", "*"), ("op", "/")):
                self.take()
                f = self.unary()
                if tok[1] == "/" and f == 0:
                    raise ScoreError("division by zero")
                e = e * f if tok[1] == "*" else e / f
            elif tok[0] in ("num", "name") or tok == ("op", "("):
                if tok[0] == "num" and self.toks[self.i - 1][0] == "num":
                    raise ScoreError("adjacent numbers")
                e = e * self.power()
            else:
                return e

    def unary(self):
        if self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            u = self.unary()
            return u if op == "+" else -u
        return self.power()

    def power(self):
        base = self.atom()
        if self.peek() == ("op", "**"):
            self.take()
            exp = self.unary()
            if not exp.is_number or abs(exp) > 64:
                raise ScoreError("exponent out of range")
            return base**exp
        return base

    def atom(self):
        kind, value = self.take()
        sp = self.sp
        if kind == "num":
            if len(value) > 30:
                raise ScoreError("number too long")
            return sp.Rational(value)
        if kind == "name":
            if value == "pi":
                return sp.pi
            if value in ("sqrt", "root"):
                self.take("(")
                a = self.expr()
                b = None
                if value == "root":
                    self.take(",")
                    b = self.expr()
                self.take(")")
                if b is not None and (not b.is_Integer or not 2 <= b <= 10):
                    raise ScoreError("bad root index")
                return sp.sqrt(a) if b is None else sp.root(a, b)
            return sp.Symbol(value)
        if value == "(":
            e = self.expr()
            self.take(")")
            return e
        raise ScoreError(f"unexpected {value}")


def to_sympy(s):
    """Parse a normalized answer into a sympy expression, or raise ScoreError."""
    import sympy

    if len(s) > MAX_LEN:
        raise ScoreError("too long")
    if re.search(r"\d\s*\\frac", s):
        raise ScoreError("mixed number is ambiguous")
    toks = _tokens(_delatex(s))
    if not toks or len(toks) > 80:
        raise ScoreError("bad length")
    return _Parser(toks, sympy).parse()


def math_equal(pred, target):
    p, t = normalize(pred), normalize(target)
    if not p:
        return False
    if p.replace(" ", "") == t.replace(" ", ""):
        return True
    try:
        a, b = to_sympy(p), to_sympy(t)
    except ScoreError:
        return False
    import sympy

    diff = a - b
    if diff == 0:
        return True
    if diff.free_symbols:
        return sympy.simplify(sympy.expand(diff)) == 0
    return sympy.simplify(diff) == 0


def choice_letter(answer):
    m = CHOICE_LETTER.match(strip_delimiters(answer).strip())
    return m.group(1) if m else None


def decimal_equal(pred, target, decimals):
    p = normalize(pred)
    if not re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)", p):
        return False
    try:
        q = Decimal(1).scaleb(-int(decimals))
        return Decimal(p).quantize(q, rounding=ROUND_HALF_UP) == Decimal(normalize(target)).quantize(q, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return False


def score(selector, response, label, metadata=None, effort=None):
    """Score one response. Returns dict(score, acc, outcome, pred)."""
    canonical = canonical_selector(selector)
    if canonical is None:
        raise ValueError(f"Unsupported data_source selector: {selector!r}")
    selector = canonical
    if not isinstance(response, str):
        return result(None, "infra_no_response")
    region, failure = final_region(response, effort)
    if failure:
        return result(0.0, failure)
    pred = extract_answer(region)
    if pred is None or not pred.strip():
        return result(0.0, "format_missing_answer")
    if selector == CHOICE:
        letter = choice_letter(pred)
        if letter is None:
            return result(0.0, "format_no_option_letter", pred)
        return result(float(letter == label), "correct" if letter == label else "wrong", letter)
    if selector == DECIMAL:
        ok = decimal_equal(pred, label, (metadata or {})["answer_decimals"])
        return result(float(ok), "correct" if ok else "wrong", pred)
    ok = math_equal(pred, label)
    return result(float(ok), "correct" if ok else "wrong", pred)


def _effort(args, sample):
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    if meta.get("reasoning_effort"):
        return meta["reasoning_effort"]
    kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    return kwargs.get("reasoning_effort", "high")


def score_sample(args, sample):
    status = getattr(getattr(sample, "status", None), "value", None)
    if status in ("aborted", "failed"):
        return result(None, f"infra_{status}")
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    selector = meta.get("data_source")
    if canonical_selector(selector) is None:
        raise ValueError(f"Unsupported data_source selector: {selector!r}")
    try:
        out = score(selector, sample.response, sample.label, meta, _effort(args, sample))
    except Exception as e:  # scorer bug or bad label: not the policy's fault
        logger.exception("scorer error")
        return result(None, f"infra_scorer_{type(e).__name__}")
    if status == "truncated" and out["outcome"] == "format_unclosed_think":
        out["outcome"] = "format_truncated"
    return out


async def async_rm(args, sample, **kwargs):
    """Miles ``--custom-rm-path k2hv.reward.async_rm``; use ``--reward-key score``. Accepts one sample or a list."""
    if isinstance(sample, list):
        return [score_sample(args, s) for s in sample]
    return score_sample(args, sample)
