"""RL target classification and admission against the k2hv final-answer scorers.

Targets are scored by ``k2hv.reward`` (selectors ``k2hv_math_v1``, ``k2hv_choice_v1``,
``k2hv_decimal_v1``); see that module for the response contract. A row is admitted only if the
selected scorer accepts the canonical correct response in every think-tag form and in equivalent
notations, rejects wrong and near-wrong values, and gives no credit to an answer inside unclosed
reasoning. Long decimals without a stated precision stay pending: no defensible tolerance exists.
"""

import re

from .. import reward
from . import text

MATH = reward.MATH
CHOICE = reward.CHOICE
DECIMAL = reward.DECIMAL
PENDING_DECIMAL = "pending_decimal_precision"

MATH_SUFFIX = 'Solve the problem. After your reasoning, end your response with a line of the form "Answer: X" (without quotes), where X is the final answer as a number or expression, without units.'
CHOICE_SUFFIX = 'Choose the correct option. After your reasoning, end your response with a line of the form "Answer: X" (without quotes), where X is the letter of the correct option.'

OPTION = re.compile(r"(?m)^\s*(?:\(([A-J])\)|([A-J])[.):：])\s*(\S.*)$")
INLINE_OPTION = re.compile(r"\(([A-J])\)\s*([^()\n]+?)(?=\s*\([A-J]\)|$)")
UNIT = re.compile(r"^(?P<num>[-+]?[\d,]*\.?\d+)\s*(?P<unit>%|°|degrees?|cm|mm|m|km|kg|g|units?|inches|in|feet|ft|meters?|cm\^?2|cm²|m²|m\^2|square units|sq\.? units|dollars?|cents?|hours?|minutes?|seconds?|s|years?|days?)?\.?$", re.I)
CURRENCY = re.compile(r"^[$€£¥]\s*(?P<num>[-+]?[\d,]*\.?\d+)$")
SCIENTIFIC = re.compile(r"^[-+]?\d+(\.\d+)?[eE][-+]?\d+$")
SYMBOLIC_HINT = re.compile(r"\\frac|\\sqrt|\\pi|π|√|\^|/")
YES_NO = {"yes", "no", "true", "false"}
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
PRECISION = [
    (re.compile(r"(?i)\b(\d|one|two|three|four|five|six)\s+decimal\s+places?\b"), None),
    (re.compile(r"(?i)\bnearest\s+(?:whole\s+number|integer)\b"), 0),
    (re.compile(r"(?i)\bnearest\s+tenth\b"), 1),
    (re.compile(r"(?i)\bnearest\s+hundredth\b"), 2),
    (re.compile(r"(?i)\bnearest\s+thousandth\b"), 3),
    (re.compile(r"(?i)\bnearest\s+0\.(0*)1\b"), "zeros"),
]
SIGNIFICANT = re.compile(r"(?i)significant\s+(figures|digits)")


def _outer_boxed(ref):
    if not ref.startswith("\\boxed"):
        return None
    i = ref.find("{")
    if i < 0 or ref[6:i].strip():
        return None
    depth = 0
    for j in range(i, len(ref)):
        depth += {"{": 1, "}": -1}.get(ref[j], 0)
        if depth == 0:
            return ref[i + 1 : j].strip() if j == len(ref) - 1 else None
    return None


def unbox(reference):
    """Reference without an outer \\boxed{}, math delimiters ($, \\( \\), \\[ \\]) or a trailing period."""
    ref = reference.strip()
    while True:
        prev = ref
        inner = _outer_boxed(ref)
        ref = reward.strip_delimiters(inner if inner is not None else ref)
        if ref == prev:
            break
    return ref.rstrip(".").strip() if not text.is_number(ref) else ref


def stated_decimals(problem):
    """Decimal places the question asks for, or None. Significant-figure requests are not a decimal rule."""
    if SIGNIFICANT.search(problem):
        return None
    found = set()
    for pattern, value in PRECISION:
        for m in pattern.finditer(problem):
            if value is None:
                g = m.group(1).lower()
                found.add(WORDS.get(g, int(g) if g.isdigit() else -1))
            elif value == "zeros":
                found.add(len(m.group(1)) + 1)
            else:
                found.add(value)
    return found.pop() if len(found) == 1 else None


def options(problem):
    found = {}
    for m in OPTION.finditer(problem):
        letter = m.group(1) or m.group(2)
        found.setdefault(letter, m.group(3).strip())
    if len(found) < 2:
        for m in INLINE_OPTION.finditer(problem):
            found.setdefault(m.group(1), m.group(2).strip())
    letters = sorted(found)
    if len(letters) < 2 or letters != list("ABCDEFGHIJ"[: len(letters)]):
        return {}
    return found


def canonical_number(core):
    core = core.replace(",", "").lstrip("+")
    if core.startswith("."):
        core = "0" + core
    if core.startswith("-."):
        core = "-0" + core[1:]
    if "." in core:
        core = core.rstrip("0").rstrip(".")
    return "0" if core in ("-0", "") else core


def classify(problem, reference, max_decimals=3):
    """Return dict(answer_type, target, selector, reason[, answer_decimals]) for an RL reference answer."""
    ref = unbox(reference)
    opts = options(problem)
    letter = text.mc_letter(ref)
    if opts:
        if letter and letter in opts:
            return {"answer_type": "choice", "target": letter, "selector": CHOICE, "reason": None}
        raw = (_outer_boxed(reference.strip()) or reference.strip()).rstrip(".")
        matches = [k for k, v in opts.items() if text.key(v) in (text.key(ref), text.key(raw))]
        if len(matches) == 1:
            return {"answer_type": "choice", "target": matches[0], "selector": CHOICE, "reason": None, "mapped_from": ref}
        return {"answer_type": "choice", "target": None, "selector": None, "reason": "choice_unmatched"}
    if not ref:
        return {"answer_type": "empty", "target": None, "selector": None, "reason": "answer_empty"}
    if letter and len(ref) <= 3:
        return {"answer_type": "choice", "target": None, "selector": None, "reason": "choice_options_unparsed"}
    if text.key(ref) in YES_NO:
        return {"answer_type": "yes_no", "target": None, "selector": None, "reason": "answer_yes_no"}
    if SCIENTIFIC.match(ref):
        return {"answer_type": "numeric_scientific", "target": None, "selector": None, "reason": "answer_scientific"}
    plain = reward.normalize(ref)
    m = UNIT.match(ref) or CURRENCY.match(ref) or re.fullmatch(r"(?P<num>[-+]?[\d,]*\.?\d+)", plain)
    if m:
        core = m.group("num")
        if not re.match(r"^[-+]?(\d{1,3}(,\d{3})+|\d*)(\.\d+)?$", core):
            return {"answer_type": "numeric", "target": None, "selector": None, "reason": "answer_number_format"}
        target = canonical_number(core)
        unit = m.groupdict().get("unit") or None
        stated = stated_decimals(problem)
        written = core.replace(",", "").lstrip("+")
        if stated is not None and text.decimals(written) == stated:
            # The reference is written at exactly the precision the question asks for.
            return {"answer_type": "numeric_stated_precision", "target": written, "selector": DECIMAL, "reason": None, "unit": unit, "answer_decimals": stated}
        if text.decimals(target) > max_decimals:
            return {"answer_type": "numeric_long_decimal", "target": target, "selector": PENDING_DECIMAL, "reason": None, "unit": unit,
                    "pending": "precision_mismatch" if stated is not None else "no_stated_precision"}
        return {"answer_type": "numeric", "target": target, "selector": MATH, "reason": None, "unit": unit}
    target = reward.normalize(ref)
    if SYMBOLIC_HINT.search(target) and any(c.isdigit() or c in "π√" for c in target.replace("\\pi", "π")):
        try:
            expr = reward.to_sympy(target)
        except reward.ScoreError:
            return {"answer_type": "symbolic", "target": None, "selector": None, "reason": "answer_unparseable_expression"}
        if len(expr.free_symbols) > 1:
            return {"answer_type": "symbolic", "target": None, "selector": None, "reason": "answer_multivariable_expression"}
        return {"answer_type": "symbolic", "target": target, "selector": MATH, "reason": None}
    return {"answer_type": "free_form", "target": None, "selector": None, "reason": "answer_free_form"}


def _number_variants(target):
    """Wrong and near-wrong values for a numeric target: +1, *1.5+1, and one unit in the last place each way."""
    from decimal import Decimal

    d = Decimal(target)
    ulp = Decimal(1).scaleb(d.as_tuple().exponent)
    out = []
    for v in (d + 1, d * Decimal("1.5") + 1, d * 2 + 3, d + ulp, d - ulp):
        if v != d and str(v) not in out:
            out.append(str(v))
    return out


def responses(target, info, opts=None):
    """(correct responses, wrong responses, unclosed-reasoning responses) used for admission."""
    tags = {"high": "</ifm|think>", "medium": "</ifm|think_fast>", "low": "</ifm|think_faster>"}
    forms = [target, f"\\boxed{{{target}}}"]
    selector = info["selector"]
    if selector == MATH and info["answer_type"] == "symbolic":
        import sympy

        expr = reward.to_sympy(target)
        forms += [str(expr).replace("**", "^"), sympy.latex(expr)]
    elif selector == MATH:
        from fractions import Fraction

        frac = Fraction(target)
        forms += [f"{target}0" if "." in target else f"{target}.0"]
        if frac.denominator != 1:
            forms += [f"{frac.numerator}/{frac.denominator}", f"\\frac{{{frac.numerator}}}{{{frac.denominator}}}"]
        if info.get("unit") in ("%", "°", "cm", "m", "units"):
            forms.append(f"{target}{info['unit']}")
    if selector == CHOICE:
        bad = [k for k in sorted(opts or "ABCD") if k != target][:2]
        forms += [f"({target})", f"{target}. {(opts or {}).get(target, '')}".strip()]
    elif selector == MATH and info["answer_type"] == "symbolic":
        bad = [f"{target}+1", f"2({target})"]
    else:
        bad = _number_variants(target)
    correct = [(f"Reasoning.{tags[e]}Answer: {f}", e) for e in tags for f in forms[:1]] + [(f"Reasoning.</ifm|think>Answer: {f}", "high") for f in forms[1:]]
    correct.append((f"Reasoning.</ifm|think>The result follows.\n\\boxed{{{target}}}", "high"))
    wrong = [(f"Reasoning.</ifm|think>Answer: {b}", "high") for b in bad]
    wrong.append((f"Reasoning. Answer: {target}</ifm|think>Answer: {bad[0]}", "high"))
    unclosed = [(f"Reasoning. Answer: {target}", "high"), (f"Reasoning.</ifm|think_fast>Answer: {target}", "high")]
    return correct, wrong, unclosed


def admit(problem, info):
    """Run the selected scorer on correct, equivalent, wrong, near-wrong and unclosed-reasoning responses; return a failure reason or None."""
    target, selector = info["target"], info["selector"]
    if selector == PENDING_DECIMAL:
        return None
    meta = {"answer_decimals": info.get("answer_decimals")}
    correct, wrong, unclosed = responses(target, info, options(problem))
    for response, effort in correct:
        if reward.score(selector, response, target, meta, effort)["score"] != 1.0:
            return "scorer_rejects_reference"
    for response, effort in wrong:
        if reward.score(selector, response, target, meta, effort)["score"] != 0.0:
            return "scorer_accepts_wrong"
    for response, effort in unclosed:
        if reward.score(selector, response, target, meta, effort)["outcome"] not in ("format_unclosed_think", "format_wrong_think_tag"):
            return "scorer_credits_unclosed_reasoning"
    return None


def admit_row(data):
    """Re-run admission on an exported RL row (problem, answer, metadata)."""
    meta = data["metadata"]
    info = {"target": data["answer"], "selector": meta["data_source"], "answer_type": meta["answer_type"], "answer_decimals": meta.get("answer_decimals")}
    return admit(data["problem"], info)


def strip_format(problem):
    """Remove a source's answer-format instruction (the whole sentence mentioning a box); the RL prompt appends the scorer's format."""
    problem = problem.replace("<image>/n", "<image>\n")
    while True:
        idx = max(problem.rfind("\\boxed"), problem.rfind("boxed{"), problem.lower().rfind("in a box"))
        tail = problem[idx:] if idx >= 0 else ""
        if idx < 0 or "\n\n" in tail.strip() or len(tail) > 120:
            return problem.rstrip()
        start = max(problem.rfind(sep, 0, idx) for sep in (". ", "? ", "! ", "\n", ": "))
        problem = problem[: start + 1 if start >= 0 else 0].rstrip()


def rl_problem(problem, info):
    suffix = CHOICE_SUFFIX if info["selector"] == CHOICE else MATH_SUFFIX
    return f"{strip_format(problem)}\n\n{suffix}"
