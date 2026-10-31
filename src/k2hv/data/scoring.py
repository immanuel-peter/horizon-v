"""RL target classification and admission against the pinned RL360 scorers.

``math_dapo`` (Minerva path) reads the last ``Answer: ...`` line and compares normalized
strings exactly; it has no numeric tolerance and ignores a bare ``\\boxed{}``.
``stem__supergpqa`` extracts one option letter A-J. A row is admitted only if the pinned
scorer accepts its canonical correct response and rejects a wrong one.
"""

import math
import re

from . import runtime, text

MATH = "math_dapo"
CHOICE = "stem__supergpqa"
PENDING_NUMERIC = "pending_numeric_tolerance"

MATH_SUFFIX = 'Solve the problem. The last line of your response must be of the form "Answer: X" (without quotes), where X is the final answer as a number or expression, without units.'
CHOICE_SUFFIX = 'Choose the correct option. The last line of your response must be of the form "Answer: X" (without quotes), where X is the letter of the correct option.'

OPTION = re.compile(r"(?m)^\s*(?:\(([A-J])\)|([A-J])[.):：])\s*(\S.*)$")
INLINE_OPTION = re.compile(r"\(([A-J])\)\s*([^()\n]+?)(?=\s*\([A-J]\)|$)")
BOXED = re.compile(r"^\\boxed\{(.*)\}$", re.S)
UNIT = re.compile(r"^(?P<num>[-+]?[\d,]*\.?\d+)\s*(?P<unit>%|°|degrees?|cm|mm|m|km|kg|g|units?|inches|in|feet|ft|meters?|cm\^?2|cm²|m²|m\^2|square units|sq\.? units|dollars?|cents?|hours?|minutes?|seconds?|s|years?|days?)?\.?$", re.I)
CURRENCY = re.compile(r"^[$€£¥]\s*(?P<num>[-+]?[\d,]*\.?\d+)$")
SCIENTIFIC = re.compile(r"^[-+]?\d+(\.\d+)?[eE][-+]?\d+$")
SYMBOLIC = re.compile(r"^[-+]?[0-9a-z\\{}^()+\-*/.,π√ ]{1,40}$")
SYMBOLIC_HINT = re.compile(r"\\frac|\\dfrac|\\sqrt|\\pi|π|√|\^|/")
YES_NO = {"yes", "no", "true", "false"}


def unbox(reference):
    reference = reference.strip()
    m = BOXED.match(reference)
    reference = m.group(1).strip() if m else reference
    if reference.startswith("$") and reference.endswith("$") and len(reference) > 1:
        reference = reference.strip("$").strip()
    return reference.rstrip(".").strip() if not text.is_number(reference) else reference


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
    """Return dict(answer_type, target, selector, reason) for an RL reference answer."""
    ref = unbox(reference)
    opts = options(problem)
    letter = text.mc_letter(ref)
    if opts:
        if letter and letter in opts:
            return {"answer_type": "choice", "target": letter, "selector": CHOICE, "reason": None}
        matches = [k for k, v in opts.items() if text.key(v) == text.key(ref)]
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
    m = UNIT.match(ref) or CURRENCY.match(ref)
    if m:
        core = m.group("num")
        if not re.match(r"^[-+]?(\d{1,3}(,\d{3})+|\d*)(\.\d+)?$", core):
            return {"answer_type": "numeric", "target": None, "selector": None, "reason": "answer_number_format"}
        target = canonical_number(core)
        unit = (m.groupdict().get("unit") or ("$" if ref.lstrip().startswith(("$", "€", "£", "¥")) else "")) or None
        if text.decimals(target) > max_decimals:
            return {"answer_type": "numeric_long_decimal", "target": target, "selector": PENDING_NUMERIC, "reason": None, "unit": unit}
        return {"answer_type": "numeric", "target": target, "selector": MATH, "reason": None, "unit": unit}
    if SYMBOLIC.match(ref) and SYMBOLIC_HINT.search(ref) and any(c.isdigit() or c in "π√" or "\\pi" in ref for c in ref):
        return {"answer_type": "symbolic", "target": ref.replace("\\dfrac", "\\frac"), "selector": MATH, "reason": None}
    return {"answer_type": "free_form", "target": None, "selector": None, "reason": "answer_free_form"}


def wrong(target, answer_type, opts=None):
    if answer_type == "choice":
        return next(k for k in sorted(opts or "ABCD") if k != target)
    if answer_type in ("numeric", "numeric_long_decimal"):
        value = float(target)
        return canonical_number(repr(value + 1 if value == int(value) else value * 1.5 + 1))
    return target + "+1"


def scorer_correct(selector, response, target):
    rl = runtime.rl360()
    if selector == MATH:
        return bool(rl.math_dapo.compute_score(response, target)["acc"])
    if selector == CHOICE:
        return bool(rl.supergpqa.compute_score(response, target)["acc"])
    if selector == PENDING_NUMERIC:
        return numeric_close(response, target)
    raise ValueError(f"Unsupported data_source selector: {selector!r}")


def admit(problem, info):
    """Check the canonical correct and a wrong response with the selected scorer."""
    target, selector = info["target"], info["selector"]
    if scorer_correct(selector, f"Answer: {target}", target) is not True:
        return "scorer_rejects_reference"
    if scorer_correct(selector, f"Answer: {wrong(target, info['answer_type'], options(problem))}", target) is not False:
        return "scorer_accepts_wrong"
    return None


ANSWER_LINE = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")
FLOAT = re.compile(r"[-+]?(\d+(,\d{3})*(\.\d*)?|\.\d+)([eE][-+]?\d+)?")


def numeric_close(response, reference, rel_tol=0.01):
    """Tolerance rule for long-decimal references: last ``Answer:`` value within 1% relative error.

    Not an RL360 scorer. Step 8 must register a selector implementing this rule before these rows train.
    """
    found = ANSWER_LINE.findall(response[-300:])
    if not found:
        return False
    m = FLOAT.search(found[-1].replace("\\boxed{", ""))
    if not m:
        return False
    pred, ref = float(m.group(0).replace(",", "")), float(reference)
    return math.isclose(pred, ref, rel_tol=rel_tol, abs_tol=1e-9)


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
