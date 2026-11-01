"""CPU tests of the k2hv RL final-answer contract. A GPU rollout/reward integration test is separate (step 8)."""

import asyncio
import time
from enum import Enum
from types import SimpleNamespace

import pytest

from k2hv import reward

M, C, D = reward.MATH, reward.CHOICE, reward.DECIMAL


def done(answer, tag="think"):
    return f"Let me look at the figure.\n</ifm|{tag}>Answer: {answer}"


def s(selector, response, label, meta=None, effort="high"):
    return reward.score(selector, response, label, meta, effort)


@pytest.mark.parametrize("label,answers", [
    ("1.5", ["1.5", "1.50", "3/2", "\\frac{3}{2}", "\\dfrac{3}{2}", "\\tfrac32", "\\boxed{1.5}", "$1.5$", "\\(1.5\\)", "1.5 cm", "x = 1.5"]),
    ("\\frac{7}{13}", ["7/13", "\\frac{7}{13}", "\\dfrac{7}{13}", "\\frac7{13}", "14/26"]),
    ("5 \\sqrt { 3 }", ["5\\sqrt{3}", "5\\sqrt3", "5√3", "\\sqrt{75}", "5*sqrt(3)", "5 \\sqrt { 3 }"]),
    ("\\frac{112}{3} \\pi", ["\\frac{112\\pi}{3}", "112π/3", "\\frac{112}{3}\\pi cubic units"]),
    ("20", ["20", "20%", "20\\%", "20 percent"]),
    ("60", ["60°", "60^\\circ", "60^{\\circ}", "60 degrees"]),
    ("1234", ["1,234", "1234", "\\boxed{1234}"]),
    ("x^2+1/2", ["x^2+\\frac{1}{2}", "\\frac{2x^2+1}{2}", "0.5+x^2"]),
])
def test_math_accepts_equivalent_notation(label, answers):
    for a in answers:
        assert s(M, done(a), label)["score"] == 1.0, a


@pytest.mark.parametrize("label,answers", [
    ("1.5", ["2.5", "1.51", "1.4", "15", "3/4"]),
    ("\\frac{1}{3}", ["0.33", "0.333", "1/4"]),
    ("5 \\sqrt { 3 }", ["5\\sqrt{2}", "8.66", "\\sqrt{3}"]),
    ("20", ["0.2", "21"]),
    ("683.33", ["689.5", "683.3"]),
])
def test_math_rejects_wrong_and_near_wrong(label, answers):
    for a in answers:
        assert s(M, done(a), label)["score"] == 0.0, a


def test_boxed_answer_after_reasoning_and_answer_line_precedence():
    assert s(M, "Work.</ifm|think>So the area is \\boxed{12}.", "12")["score"] == 1.0
    assert s(M, "Work.</ifm|think>\\boxed{11}\nAnswer: 12", "12")["score"] == 1.0, "the Answer line wins over an earlier box"
    assert s(M, "Work.</ifm|think>Answer: \\boxed{12}", "12")["score"] == 1.0


@pytest.mark.parametrize("effort,tag", [("high", "think"), ("medium", "think_fast"), ("low", "think_faster")])
def test_each_reasoning_tag(effort, tag):
    assert s(M, done("4", tag), "4", effort=effort)["score"] == 1.0
    other = "think" if tag != "think" else "think_fast"
    assert s(M, done("4", other), "4", effort=effort)["outcome"] == "format_wrong_think_tag"
    assert s(M, done("4", tag), "4", effort=None)["score"] == 1.0, "without an effort any single close tag is accepted"


def test_answers_inside_reasoning_get_no_credit():
    out = s(M, "Answer: 4", "4")
    assert out == {"score": 0.0, "acc": False, "outcome": "format_unclosed_think", "pred": None}
    assert s(M, "Thinking... Answer: 4 \\boxed{4}", "4")["outcome"] == "format_unclosed_think"
    assert s(M, "Answer: 4</ifm|think>Answer: 5", "4")["score"] == 0.0
    assert s(M, "Answer: 4</ifm|think>I am not sure.", "4")["outcome"] == "format_missing_answer"
    assert s(M, "a</ifm|think>b</ifm|think>Answer: 4", "4")["outcome"] == "format_repeated_think_close"
    assert s(M, "a</ifm|think><ifm|think>Answer: 4", "4")["outcome"] == "format_think_after_close"
    assert s(M, done("4") + "<|ifm|im_end|>", "4")["score"] == 1.0


@pytest.mark.parametrize("response", ["", "</ifm|think>", "</ifm|think>Answer:", "</ifm|think>Answer: \\boxed{", "</ifm|think>Answer: ((((", "</ifm|think>Answer: import os", "</ifm|think>Answer: __import__('os').system('true')"])
def test_malformed_output_scores_zero(response):
    out = s(M, response, "4")
    assert out["score"] == 0.0 and (out["outcome"].startswith("format_") or out["outcome"] == "wrong")


def test_parser_never_evaluates_python_and_bounds_work():
    t = time.time()
    for a in ["9^9^9^9", "10^1000", "2^(3^40)", "(" * 60 + "1" + ")" * 60, "1" * 400]:
        assert s(M, done(a), "4")["score"] == 0.0, a
    assert time.time() - t < 5
    with pytest.raises(reward.ScoreError):
        reward.to_sympy("__class__")


def test_choice():
    for a in ["B", "(B)", "B.", "B. dog", "\\boxed{B}", "B)"]:
        assert s(C, done(a), "B")["score"] == 1.0, a
    assert s(C, done("C"), "B")["score"] == 0.0
    assert s(C, done("dog"), "B")["outcome"] == "format_no_option_letter"
    assert s(C, "B", "B")["outcome"] == "format_unclosed_think"
    assert s(C, "...</ifm|think>The option is B.", "B")["outcome"] == "format_missing_answer", "the final answer must be on an Answer line or boxed"


def test_decimal_with_stated_precision():
    meta = {"answer_decimals": 4}
    for a in ["0.4082", "0.40820", "0.408249", "\\boxed{0.4082}"]:
        assert s(D, done(a), "0.4082", meta)["score"] == 1.0, a
    for a in ["0.4083", "0.4081", "0.41", "0.40", "2/5", "0.4085"]:
        assert s(D, done(a), "0.4082", meta)["score"] == 0.0, a
    assert s(D, done("683"), "683", {"answer_decimals": 0})["score"] == 1.0
    assert s(D, done("683.6"), "683", {"answer_decimals": 0})["score"] == 0.0


def test_no_blanket_tolerance_selector():
    with pytest.raises(ValueError, match="Unsupported"):
        reward.score("pending_numeric_tolerance", done("689.5"), "683.3333333333334")
    with pytest.raises(ValueError, match="Unsupported"):
        reward.score("math_dapo", done("1"), "1")


class Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"


def sample(response, label="4", source=M, status=Status.COMPLETED, **meta):
    return SimpleNamespace(response=response, label=label, status=status, metadata={"data_source": source, **meta})


def test_miles_entry_point_separates_format_and_infrastructure_failures(monkeypatch):
    args = SimpleNamespace(apply_chat_template_kwargs={"reasoning_effort": "medium"})
    run = lambda x: asyncio.run(reward.async_rm(args, x))  # noqa: E731
    assert run(sample(done("4", "think_fast")))["score"] == 1.0
    assert run(sample(done("4")))["outcome"] == "format_wrong_think_tag", "effort comes from --apply-chat-template-kwargs"
    assert run(sample(done("4"), reasoning_effort="high"))["score"] == 1.0, "a row may override the effort"
    assert run(sample("still thinking", status=Status.TRUNCATED)) == {"score": 0.0, "acc": False, "outcome": "format_truncated", "pred": None}
    assert run(sample(done("4", "think_fast"), status=Status.ABORTED))["score"] is None
    assert run(sample("", status=Status.FAILED))["outcome"] == "infra_failed"
    assert run(sample(None))["outcome"] == "infra_no_response"
    with pytest.raises(ValueError, match="Unsupported"):
        run(sample(done("4"), source="geo3k"))

    def boom(*a, **k):
        raise RuntimeError("scorer bug")

    monkeypatch.setattr(reward, "math_equal", boom)
    out = run(sample(done("4", "think_fast")))
    assert out["score"] is None and out["outcome"] == "infra_scorer_RuntimeError"
    assert [r["score"] for r in run([sample(done("1", "think_fast"), label="1", source=C), sample(done("B", "think_fast"), label="B", source=C)])] == [0.0, 1.0]


# Raw targets taken from TIGER-Lab/ViRL39K, russwang/ThinkLite-VL-70k and TIGER-Lab/VisualWebInstruct-verified,
# classified the way admission does, then scored against correct, equivalent, wrong and near-wrong answers.
REAL_TARGETS = [
    ("\\boxed{\\frac{5\\pi }{4}}", ["\\frac{5\\pi}{4}", "5π/4", "1.25\\pi"], ["\\frac{4\\pi}{5}", "\\frac{5\\pi}{3}", "3.93"]),
    ("\\boxed{y=\\frac{3}{4}x-\\frac{3}{4}}", ["y = \\frac{3}{4}x - \\frac{3}{4}", "\\frac{3x-3}{4}", "0.75x-0.75"], ["\\frac{3}{4}x+\\frac{3}{4}", "\\frac{3}{4}x"]),
    ("1,574", ["1574", "1,574", "\\boxed{1574}", "1{,}574", "1\\,574"], ["1,575", "157.4", "1{,}575"]),
    ("24 \\sqrt { 3 }", ["24\\sqrt{3}", "\\sqrt{1728}"], ["24\\sqrt{2}", "41.57"]),
    ("2π", ["2\\pi", "2π"], ["\\pi", "6.28"]),
    ("150°", ["150^\\circ", "150 degrees", "150", "150\\degree"], ["15", "30°", "151\\degree"]),
    ("$22.50", ["22.50", "22.5", "\\$22.5", "$22.50", "\\boxed{\\$22.50}", "\\text{\\$}22.50", "22.50 dollars"], ["22.05", "2250", "\\$22.05"]),
    ("-20.35 J", ["-20.35", "-20.35 J"], ["20.35", "-20.3"]),
    ("28 square inches", ["28", "28 square inches", "28 in^2", "28 \\text{in}^2", "28 sq in"], ["56", "27", "27 in^2"]),
]


@pytest.mark.parametrize("reference,right,wrong", REAL_TARGETS)
def test_real_rl_targets(reference, right, wrong):
    from k2hv.data import scoring

    info = scoring.classify("<image>\nFind the value.", reference)
    assert info["selector"] == M and info["reason"] is None, info
    label = info["target"]
    for a in right:
        assert s(M, done(a), label)["score"] == 1.0, a
        assert s(M, done(a, "think_faster"), label, effort="low")["score"] == 1.0, a
    for a in wrong:
        assert s(M, done(a), label)["score"] == 0.0, a
    assert s(M, f"So the answer is {right[0]}. Answer: {right[0]}", label)["outcome"] == "format_unclosed_think"
    assert s(M, f"Answer: {right[0]}</ifm|think>", label)["outcome"] == "format_missing_answer"
    assert s(M, "Work.</ifm|think>Answer: \\boxed{", label)["score"] == 0.0


def test_real_choice_target():
    from k2hv.data import scoring

    problem = "<image>\nWhich shape has the largest area?\nA. the circle\nB. the square\nC. the triangle"
    info = scoring.classify(problem, "\\boxed{B}")
    assert info["selector"] == C and info["target"] == "B"
    for a in ("B", "(B)", "\\boxed{B}", "B. the square"):
        assert s(C, done(a), "B")["score"] == 1.0, a
    for a in ("C", "A. the circle"):
        assert s(C, done(a), "B")["score"] == 0.0, a
    assert s(C, "B", "B")["outcome"] == "format_unclosed_think"


@pytest.mark.parametrize("reference", ["\\boxed{\\dfrac{18}{5}\\ \\unit{}}", "\\boxed{0<x<5}", "200 mph", "12:30 P.M.", "194.6 N.m", "90° - A/2", "C) 64", "(0, 0)", "pi(2*sqrt(e) - 3)", "Answer: 6 \\, \\text{cm}^2"])
def test_real_targets_the_scorer_cannot_verify_are_not_admitted(reference):
    from k2hv.data import scoring

    info = scoring.classify("<image>\nFind the value.", reference)
    assert info["selector"] is None and info["reason"], info
