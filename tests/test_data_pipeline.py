import copy
import io
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image, ImageDraw

from k2hv.data import build, images, overlap, pipeline, runtime, scoring, sources, text

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "configs/data/step05.toml"
HF = REPO / "hf/K2-Horizon-V-init"


def have_checkout(root, commit):
    import subprocess

    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return head == commit


needs_miles = pytest.mark.skipif(not (HF / "tokenizer.json").exists() or not have_checkout(REPO / "third_party/miles", runtime.MILES_COMMIT), reason="needs the pinned tokenizer.json and third_party/miles at the pinned commit")
needs_rl360 = pytest.mark.skipif(not have_checkout(REPO / "third_party/rl360", runtime.RL360_COMMIT), reason="needs third_party/rl360 at the pinned commit")


def png(img, fmt="PNG", **kw):
    out = io.BytesIO()
    img.save(out, format=fmt, **kw)
    return out.getvalue()


def chart(values, size=(480, 320), color=(40, 90, 200)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    for k, v in enumerate(values):
        x = 40 + k * 60
        d.rectangle([x, size[1] - 30 - v, x + 40, size[1] - 30], fill=color)
        d.text((x, size[1] - 25), f"{v}", fill="black")
    d.line([30, 10, 30, size[1] - 30], fill="black")
    d.text((150, 5), "Revenue by year", fill="black")
    return img


def photo(seed=0, size=(400, 300)):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize(size, Image.Resampling.BICUBIC)


# ------------------------------------------------------------------- text rules

RULES = tomllib.loads(RECIPE.read_text())["rules"]


@pytest.mark.parametrize("question,answer,reason", [
    ("What is shown?", "", "answer_empty"),
    ("", "cat", "question_empty"),
    ("What does the <image> show?", "cat", "question_reserved_marker"),
    ("What is the label?", "uses reserved_special_token_12", "answer_reserved_marker"),
    ("What does the code print?", "import numpy as np\nprint(1)", "answer_code"),
    ("Which library?", "plt.plot(x)", "answer_code"),
    ("What about the previous question?", "yes", "question_follow_up"),
    ("And what color is it?", "red", "question_follow_up"),
    ("Name it.", "first line\nsecond line", "answer_multiline"),
    ("Explain.", " ".join(["word"] * 40), "answer_too_long"),
    ("Why?", "It is red. The sky is blue. Grass is green.", "answer_explanation"),
])
def test_sft_rejections(question, answer, reason):
    assert text.sft_reasons(question, answer, RULES) == reason


@pytest.mark.parametrize("question", ["What is the last step in the diagram?", "What is shown above the sign?", "How many previous winners are listed?"])
def test_ordinary_questions_are_not_follow_ups(question):
    assert text.sft_reasons(question, "Step 4", RULES) is None


def test_finevision_answers_are_normalized_without_truncation():
    src = {"adapter": "finevision"}
    assert build.sft_turn(src, RULES, "Which country?\nWrite a very short answer.", "Sweden.")[1] == "Sweden"
    q, a, kind, reason = build.sft_turn(src, RULES, "Question: What is green?\nChoices:\nA. Gum.\nB. Grass.\nAnswer with the letter.", "Answer: B")
    assert (a, kind, reason) == ("B", "choice", None)
    long = "The woman is pregnant. The other woman feels the baby move. They are happy."
    assert build.sft_turn(src, RULES, "Why?", long)[1:] == (long, "text", "answer_explanation")
    assert build.sft_turn({"adapter": "chartverse_sft", "max_answer_decimals": 2}, RULES, "What share?", "0.21358378184590313")[3] == "answer_unrounded_computation"
    assert build.sft_turn({"adapter": "x", "question_exclude": "(?i)genre"}, RULES, "What is the genre of this book?", "Travel")[3] == "question_excluded_by_source_rule"


# ------------------------------------------------------------------- images


def test_bad_images_are_rejected_with_reasons():
    good = png(photo())
    with pytest.raises(images.ImageError, match="image_missing"):
        images.prepare(b"")
    with pytest.raises(images.ImageError, match="image_corrupt"):
        images.prepare(b"not an image at all")
    with pytest.raises(images.ImageError, match="image_corrupt"):
        images.prepare(good[: len(good) // 2])
    with pytest.raises(images.ImageError, match="image_too_small"):
        images.prepare(png(Image.new("RGB", (4, 40))))
    with pytest.raises(images.ImageError, match="image_aspect_ratio"):
        images.prepare(png(Image.new("RGB", (4020, 20))))
    with pytest.raises(images.ImageError, match="image_mode_I;16"):
        images.prepare(png(Image.new("I;16", (64, 64))))


def test_transparent_palette_images_are_composited_like_rgba(tmp_path):
    img = Image.new("P", (64, 64), 0)
    img.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    ImageDraw.Draw(img).rectangle([10, 10, 30, 30], fill=1)
    img.info["transparency"] = 0
    data = png(img, transparency=0)
    out = images.prepare(data, tmp_path)
    assert out["image_transform"] == {"alpha_composite_white": __import__("hashlib").sha256(data).hexdigest()}
    stored = Image.open(tmp_path / out["image_path"]).convert("RGB")
    assert stored.getpixel((0, 0)) == (255, 255, 255) and stored.getpixel((20, 20)) == (255, 0, 0)


def test_content_addressed_store_is_idempotent(tmp_path):
    data = png(photo(3))
    a, b = images.prepare(data, tmp_path), images.prepare(data, tmp_path)
    assert a == b and (tmp_path / a["image_path"]).read_bytes() == data
    assert Path(a["image_path"]).name.startswith(a["image_sha256"])


def test_qwen_resize_matches_pinned_qwen_vl_utils():
    runtime._import_qwen_vl_utils()
    from qwen_vl_utils.vision_process import smart_resize

    for w, h in [(300, 200), (31, 47), (1586, 2048), (359, 2048), (6000, 4000), (5000, 3), (100, 100)]:
        try:
            expected = smart_resize(h, w, factor=28)
        except ValueError:
            continue
        assert images.qwen_resize(w, h) == (expected[1], expected[0])


def test_image_token_estimate_and_unpatchify_match_the_processor():
    from transformers.models.glm5_next.image_processing_pil_glm5_next import Glm5NextImageProcessorPil

    config = json.loads((HF / "preprocessor_config.json").read_text())
    ip = Glm5NextImageProcessorPil(**config)
    cfg = {k: config[k] for k in ("patch_size", "merge_size", "temporal_patch_size", "min_image_tokens", "max_image_tokens")}
    for size in [(300, 200), (1586, 2048), (90, 60), (448, 448)]:
        img = photo(1, size)
        w, h = images.qwen_resize(*size)
        resized = img.resize((w, h))
        out = ip(images=[resized], return_tensors="np")
        grid = out["image_grid_thw"][0]
        tokens, (tw, th) = images.image_tokens(*size, cfg)
        assert tokens == int(np.prod(grid)) // 4 and (th, tw) == (grid[1] * 14, grid[2] * 14)
        restored = images.unpatchify(out["pixel_values"], grid, ip)
        assert restored.size == (tw, th)
        if (tw, th) == (w, h):
            assert np.abs(np.asarray(restored, np.int16) - np.asarray(resized, np.int16)).max() <= 1


# ------------------------------------------------------------------- overlap

CFG = tomllib.loads(RECIPE.read_text())["overlap"]


def test_resized_and_reencoded_copies_are_confirmed():
    original = chart([120, 80, 200, 150, 60])
    for copy_ in (original.resize((336, 224)), Image.open(io.BytesIO(png(original, "JPEG", quality=60))), Image.open(io.BytesIO(png(original.resize((720, 480)), "JPEG", quality=75)))):
        m = overlap.aligned_metrics(original, copy_)
        assert overlap.classify_pair(m, CFG) == "confirmed", m
    recolored = chart([120, 80, 200, 150, 60], color=(200, 60, 40))
    assert overlap.classify_pair(overlap.aligned_metrics(original, recolored), CFG) == "confirmed"


def test_same_template_with_different_content_is_not_confirmed():
    a, b = chart([120, 80, 200, 150, 60]), chart([60, 180, 90, 40, 210])
    m = overlap.aligned_metrics(a, b)
    assert overlap.classify_pair(m, CFG) != "confirmed", m
    assert overlap.classify_pair(overlap.aligned_metrics(a, a.resize((480, 200))), CFG) is None


def test_hamming_and_band_candidates():
    rng = np.random.default_rng(0)
    base = rng.integers(-(2**63), 2**63 - 1, 50, dtype=np.int64)
    near = base.copy()
    near[7] ^= np.int64(0b101)
    pairs = overlap.hamming_pairs(base[:10], near[5:10], 2)
    assert (7, 2, 2) in pairs and all(d <= 2 for _, _, d in pairs)
    thumbs = [bytes([k]) * 1024 for k in range(50)] + [bytes([7]) * 1024]
    found, oversized = overlap.near_duplicate_pairs(np.concatenate([base, near[7:8]]), thumbs, {**CFG, "prefilter_similarity": 0.6})
    assert (7, 50) in found and oversized == 0


def test_eval_candidates_and_pixel_matches(tmp_path):
    img = chart([10, 20, 30])
    f = images.features(img)
    reenc = images.features(Image.open(io.BytesIO(png(img, "JPEG", quality=70))))
    other = images.features(photo(5))
    cand = {"phash": np.array([reenc["phash"], other["phash"]]), "thumb": [reenc["thumb"], other["thumb"]]}
    ev = {"phash": np.array([f["phash"]]), "thumb": [f["thumb"]]}
    found = overlap.eval_candidates(cand, ev, CFG)
    assert 0 in found and 1 not in found


def test_dedup_key_needs_image_identity_and_split_follows_group():
    q = "What is the title of this chart?\nAnswer briefly."
    assert pipeline.dedup_key("sft", "g1", q, "Sales.") == pipeline.dedup_key("sft", "g1", "What is the title of this chart?\nGive a very brief answer.", "sales")
    assert pipeline.dedup_key("sft", "g1", q, "Sales") != pipeline.dedup_key("sft", "g2", q, "Sales")
    cfg = {"sft_validation_fraction": 0.3, "rl_heldout_fraction": 0.3}
    splits = {g: pipeline.assign_split(7, g, "sft", cfg) for g in (f"group{k}" for k in range(400))}
    assert all(pipeline.assign_split(7, g, "sft", cfg) == s for g, s in splits.items())
    assert 0.2 < sum(s == "validation" for s in splits.values()) / 400 < 0.4
    uf = overlap.UnionFind(4)
    uf.union(3, 1)
    uf.union(1, 2)
    assert {uf.find(k) for k in (1, 2, 3)} == {1} and uf.find(0) == 0


# ------------------------------------------------------------------- RL scoring


@pytest.mark.parametrize("problem,reference,answer_type,target,selector", [
    ("<image>\nFind x.", "\\boxed{2016}", "numeric", "2016", scoring.MATH),
    ("<image>\nPerimeter?", "\\boxed{20 cm}", "numeric", "20", scoring.MATH),
    ("<image>\nAngle?", "60°", "numeric", "60", scoring.MATH),
    ("<image>\nCost?", "$1,234", "numeric", "1234", scoring.MATH),
    ("<image>\nRate?", "20%", "numeric", "20", scoring.MATH),
    ("<image>\nShare?", "35.294117647058826", "numeric_long_decimal", "35.294117647058826", scoring.PENDING_DECIMAL),
    ("<image>\nRMS? Provide the answer rounded to four decimal places.", "0.4082", "numeric_stated_precision", "0.4082", scoring.DECIMAL),
    ("<image>\nShare? Round to two decimal places.", "35.294117647058826", "numeric_long_decimal", "35.294117647058826", scoring.PENDING_DECIMAL),
    ("<image>\nFind b.", "2 \\sqrt { 22 }", "symbolic", "2 \\sqrt { 22 }", scoring.MATH),
    ("<image>\nVolume?", "\\(\\frac{112}{3} \\pi \\) cubic units", "symbolic", "\\frac{112}{3} \\pi", scoring.MATH),
    ("<image>\nRatio?", "\\boxed{\\(\\frac{3}{5}\\)}", "symbolic", "\\frac{3}{5}", scoring.MATH),
    ("<image>\nSide?", "\\(5\\sqrt{3}\\)", "symbolic", "5\\sqrt{3}", scoring.MATH),
    ("<image>\nWhich?\nA. cat\nB. dog", "B", "choice", "B", scoring.CHOICE),
    ("<image>\nWhich?\n(A) cat\n(B) dog", "dog", "choice", "B", scoring.CHOICE),
])
def test_rl_targets_are_classified_and_admitted(problem, reference, answer_type, target, selector):
    info = scoring.classify(problem, reference)
    assert (info["answer_type"], info["target"], info["selector"]) == (answer_type, target, selector)
    assert scoring.admit(problem, info) is None
    assert not any(d in (info["target"] or "") for d in ("\\(", "\\)", "\\[", "$")), "math delimiters are removed from targets"


def test_long_decimal_precision_rules():
    assert scoring.stated_decimals("Round your answer to 4 decimal places.") == 4
    assert scoring.stated_decimals("to the nearest tenth") == 1 and scoring.stated_decimals("Round the answer to the nearest 0.0001 mass unit.") == 4
    assert scoring.stated_decimals("Give three significant figures, two decimal places.") is None
    assert scoring.stated_decimals("What is the average?") is None
    info = scoring.classify("<image>\nAverage?", "683.3333333333334")
    assert info["selector"] == scoring.PENDING_DECIMAL and info["pending"] == "no_stated_precision"


@pytest.mark.parametrize("reference,reason", [("Yes", "answer_yes_no"), ("5.48e+04", "answer_scientific"), ("Consecutive interior angles are supplementary.", "answer_free_form"), ("", "answer_empty"), ("\\frac{2x}{3y}", "answer_multivariable_expression")])
def test_unscorable_rl_targets_are_excluded(reference, reason):
    assert scoring.classify("<image>\nQ?", reference)["reason"] == reason
    assert scoring.classify("<image>\nWhich one?", "C")["reason"] == "choice_options_unparsed"
    assert scoring.classify("<image>\nWhich?\nA. cat\nB. dog", "fish")["reason"] == "choice_unmatched"


def test_admission_catches_a_scorer_that_misses_equivalent_forms(monkeypatch):
    from k2hv import reward

    info = scoring.classify("<image>\nFind b.", "\\frac{3}{2}")
    assert scoring.admit("<image>\nFind b.", info) is None
    monkeypatch.setattr(reward, "to_sympy", lambda s: (_ for _ in ()).throw(reward.ScoreError("off")))
    assert scoring.admit("<image>\nFind b.", {**info, "answer_type": "numeric", "target": "1.5"}) == "scorer_rejects_reference"


@needs_rl360
def test_pinned_math_dapo_behavior_is_characterized():
    """Why the k2hv selectors exist: the published RL360 math_dapo compares strings and reads any Answer line."""
    rl = runtime.rl360()
    ok = lambda r, t: bool(rl.math_dapo.compute_score(r, t)["acc"])  # noqa: E731
    assert ok("Answer: 1.5", "1.5") and ok("Answer: \\boxed{1.5}", "1.5") and ok("Answer: 1.5 cm", "1.5")
    assert not ok("\\boxed{1.5}", "1.5") and not ok("Answer: 1.50", "1.5") and not ok("Answer: 3/2", "1.5") and not ok("Answer: 20%", "20")
    assert not ok("Answer: \\frac{3}{5}", "\\(\\frac{3}{5}\\)"), "a delimited v1 target only matches the delimited string"
    assert ok("<ifm|think>Answer: 4", "4"), "math_dapo credits an Answer line inside unclosed reasoning"


def test_rl_prompt_replaces_source_format_instructions():
    p = "<image>\nFind k. Let's think step by step and output the final answer within \\boxed{}."
    assert scoring.strip_format(p) == "<image>\nFind k."
    assert scoring.strip_format("<image>/nBased on the image.\nFind b.") == "<image>\nBased on the image.\nFind b."
    out = scoring.rl_problem(p, {"selector": scoring.MATH})
    assert out.startswith("<image>\nFind k.\n\n") and '"Answer: X"' in out and "boxed" not in out
    p, target, info = build.rl_turn("Which?\nA. cat\nB. dog\nAnswer with the letter inside \\boxed{}.", "B")
    assert p.startswith("<image>\nWhich?") and p.count("<image>") == 1 and target == "B" and "letter of the correct option" in p
    assert build.rl_turn("<image><image> two images", "1")[2]["reason"] == "multi_image_marker"


# ------------------------------------------------------------------- SFT/RL row contracts


@pytest.fixture(scope="module")
def checker():
    from k2hv.data import contract

    limits = tomllib.loads(RECIPE.read_text())["limits"]
    return contract.Checker(str(HF), limits)


def sft_row(path, answer="red", question="What color is the square?"):
    return {"messages": [{"role": "user", "content": "<image>\n" + question, "think": None}, {"role": "assistant", "content": answer, "think": ""}], "images": [str(path)]}


@needs_miles
def test_sft_contract_through_pinned_miles_path(tmp_path, checker):
    from k2hv.data import contract

    stored = images.prepare(png(photo(2, (300, 200))), tmp_path)
    path = tmp_path / stored["image_path"]
    res = checker.sft(sft_row(path), stored["image_sha256"], keep_pixels=True)
    pixels, grid, ids, start = res.pop("pixels")
    tok = checker.processor.tokenizer
    assert res["image_tokens"] == images.image_tokens(300, 200, checker.image_config)[0] == int(np.prod(grid)) // 4
    assert tok.decode(ids[start:]) == "red<|ifm|im_end|>" and res["supervised_tokens"] == len(ids) - start
    assert ids[start - len(checker.closing) : start] == checker.closing
    prompt = ids[:start]
    s, e = prompt.index(checker.ids["start"]), prompt.index(checker.ids["end"])
    assert set(prompt[s + 1 : e]) == {checker.ids["pad"]} and e - s - 1 == res["image_tokens"]
    assert res["total_tokens"] == len(ids) and res["bos_tokens"] == 1 and ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1

    bad = []
    two = sft_row(path)
    two["messages"][0]["content"] = "<image><image>\nTwo?"
    two["images"] = [str(path), str(path)]
    bad.append((two, "row_image_marker_count"))
    thinking = sft_row(path)
    thinking["messages"][1]["think"] = "hmm"
    bad.append((thinking, "row_think_not_empty"))
    bad.append((sft_row(tmp_path / "missing.png"), "image_file_missing"))
    bad.append((sft_row(path, answer="see <image>"), "row_marker_in_answer"))
    three = sft_row(path)
    three["messages"].append({"role": "user", "content": "And?", "think": None})
    bad.append((three, "row_not_single_pair"))
    for row, reason in bad:
        with pytest.raises(contract.ContractError, match=reason):
            checker.sft(row, stored["image_sha256"])
    with pytest.raises(contract.ContractError, match="image_hash_mismatch"):
        checker.sft(sft_row(path), "0" * 64)
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"garbage")
    with pytest.raises(contract.ContractError, match="miles_media"):
        checker.sft(sft_row(corrupt))


@needs_miles
def test_length_limits_reject_instead_of_truncating(tmp_path, checker):
    from k2hv.data import contract

    stored = images.prepare(png(photo(4, (1600, 1600))), tmp_path)
    path = tmp_path / stored["image_path"]
    tight = contract.Checker.__new__(contract.Checker)
    tight.__dict__.update(checker.__dict__)
    tight.limits = {**checker.limits, "max_total_tokens": 600}
    with pytest.raises(contract.ContractError, match="limit_total_tokens"):
        tight.sft(sft_row(path), stored["image_sha256"])
    tight.limits = {**checker.limits, "max_supervised_tokens": 3}
    with pytest.raises(contract.ContractError, match="limit_supervised_tokens"):
        tight.sft(sft_row(path, answer="a much longer answer than three tokens"), stored["image_sha256"])
    assert checker.sft(sft_row(path), stored["image_sha256"])["image_tokens"] == 1024


@needs_miles
def test_rl_contract(tmp_path, checker):
    from k2hv.data import contract

    stored = images.prepare(png(chart([5, 9, 3])), tmp_path)
    row = {"problem": "<image>\nHow many bars?\n\n" + scoring.MATH_SUFFIX, "answer": "3", "images": [str(tmp_path / stored["image_path"])], "metadata": {"data_source": scoring.MATH, "answer_type": "numeric"}}
    res = checker.rl(row, stored["image_sha256"])
    assert res["image_tokens"] == images.image_tokens(480, 320, checker.image_config)[0] and res["prompt_tokens"] > res["image_tokens"]
    with pytest.raises(contract.ContractError, match="row_selector"):
        checker.rl({**row, "metadata": {"data_source": "geo3k"}})


# ------------------------------------------------------------------- portability and resume


def test_portable_hash_and_materialize_rewrite_only_the_root(tmp_path):
    record = {"kind": "sft", "image_path": "images/ab/abc.png", "question": "Q?", "answer": "A", "record_id": "r", "source": "s", "task": "t", "image_sha256": "abc"}
    a, b = pipeline.miles_row(record, Path("/data/one")), pipeline.miles_row(record, Path("/mnt/two"))
    assert a["images"] == ["/data/one/images/ab/abc.png"] and pipeline.portable_hash(a) == pipeline.portable_hash(b) == pipeline.portable_hash(pipeline.miles_row(record, None))
    ctx = SimpleNamespace(root=tmp_path, profile_name="p", export_dir=tmp_path / "exports/p")
    portable = tmp_path / "exports/p/portable/default/sft/train/part-00000.parquet"
    portable.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([pipeline.miles_row(record, None)]), portable)
    pipeline.materialize(ctx, tmp_path / "target")
    rows = pq.read_table(tmp_path / "exports/p/miles/default/sft/train/part-00000.parquet").to_pylist()
    assert rows[0]["images"] == [str((tmp_path / "target").resolve() / "images/ab/abc.png")]
    assert pipeline.portable_hash(rows[0]) == pipeline.portable_hash(a)


class LocalAdapter:
    """Reads a local Parquet of (image bytes, question, answer); optionally fails after N rows."""

    dataset = "local"
    fail_after = None

    def shards(self, ctx, src):
        return [{"file": "local.parquet", "row_groups": None}]

    def rows(self, ctx, src, shard):
        table = pq.read_table(Path(ctx.datasets["local"]["path"]))
        for i, r in enumerate(table.to_pylist()):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("injected interruption")
            yield {"row": i, "upstream_id": str(i), "image": r["image"], "reject": None, "orig_split": "train", "group_hint": None, "external_ids": [], "turns": [{"turn": t, "question": q, "answer": a, "provenance": {}} for t, (q, a) in enumerate(zip(r["questions"], r["answers"]))]}


def local_recipe(tmp_path):
    rows = [{"image": png(photo(k)), "questions": ["What is shown?", "And what about the previous answer?"], "answers": ["A pattern.", "no"]} for k in range(6)]
    rows.append({"image": b"corrupt", "questions": ["What?"], "answers": ["x"]})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "local.parquet")
    recipe = RECIPE.read_text().replace("[datasets.finevision]", f'[datasets.local]\nrepo = "local"\nrevision = "0"\npath = "{tmp_path / "local.parquet"}"\n\n[datasets.finevision]')
    recipe += '\n[sources.local_test]\nkind = "sft"\nadapter = "local"\ntask = "test"\nterms = "mit"\nterms_note = "test"\nstatus = "include"\nquota = 10\nmax_qa_per_image = 1\n'
    path = tmp_path / "recipe.toml"
    path.write_text(recipe)
    return path


def test_scan_is_atomic_and_resume_reproduces_outputs(tmp_path, monkeypatch):
    adapter = LocalAdapter()
    monkeypatch.setitem(sources.ADAPTERS, "local", adapter)
    recipe = local_recipe(tmp_path)
    shard = {"file": "local.parquet", "row_groups": None}

    clean = build.Context(recipe, tmp_path / "clean", "full")
    marker = build.scan_shard(clean.args, "local_test", shard)
    out, rej, done = build.scan_paths(clean, "local_test", shard)
    assert marker["counts"]["accepted"] == 6 and marker["counts"]["rejected"] == {"question_follow_up": 6, "image_corrupt": 1}
    assert {r["answer"] for r in pq.read_table(out).to_pylist()} == {"A pattern."}

    interrupted = build.Context(recipe, tmp_path / "resumed", "full")
    adapter.fail_after = 3
    with pytest.raises(RuntimeError, match="injected"):
        build.scan_shard(interrupted.args, "local_test", shard)
    out2, rej2, done2 = build.scan_paths(interrupted, "local_test", shard)
    assert not out2.exists() and not done2.exists() and not list(out2.parent.glob(".*.tmp"))
    adapter.fail_after = None
    resumed = build.scan_shard(interrupted.args, "local_test", shard)
    assert out2.read_bytes() == out.read_bytes() and rej2.read_bytes() == rej.read_bytes()
    assert resumed["candidates_sha256"] == marker["candidates_sha256"]
    assert sorted(p.relative_to(tmp_path / "clean") for p in (tmp_path / "clean/images").rglob("*")) == sorted(p.relative_to(tmp_path / "resumed") for p in (tmp_path / "resumed/images").rglob("*"))

    adapter.fail_after = 0
    assert build.scan_shard(interrupted.args, "local_test", shard) == resumed, "a completed shard is reused, not reread"


def test_recipe_is_complete_and_consistent():
    recipe = tomllib.loads(RECIPE.read_text())
    for key, src in recipe["sources"].items():
        assert src["status"] in ("include", "pending", "exclude"), key
        assert src["adapter"] in sources.ADAPTERS and sources.ADAPTERS[src["adapter"]].dataset in recipe["datasets"], key
        assert src["terms"] and src["terms_note"] and src["kind"] in ("sft", "rl"), key
        if src["status"] == "include":
            assert src["terms"] != "unverified" and src.get("redistribution"), key
            if "license:" in src["terms_note"] and "metadata" in src["terms_note"]:
                assert src["terms_note"].startswith("Creator's Hub declaration"), f"{key}: a Hub tag counts only when the dataset creator set it"
        if src["status"] == "pending":
            assert src.get("missing_evidence"), key
    assert recipe["profiles"]["full"]["statuses"] == ["include"]
    assert len(recipe["media"]["coco_instances_train2017"]["sha256"]) == 64
    for ds in list(recipe["datasets"].values()) + list(recipe["evals"].values()):
        assert len(ds["revision"]) == 40
    assert set(recipe["evals"]) == {"mmmu_val", "mathvista_testmini", "chartqa_test", "docvqa_val", "textvqa_val", "mmbench_dev", "pope_test"}


# ------------------------------------------------------------------- v2 pipeline paths


class FakeCoco:
    names = {"person", "tennis racket", "cat", "keyboard", "bench", "handbag"}

    def __init__(self, boxes):
        self.boxes = {1: boxes}

    category = sources.CocoInstances.category


SPATIAL_Q = "In the image, where is the {a} positioned relative to the {b}? Select from the following choices.\n(A) {x}\n(B) {y}"


def test_nemotron_spatial_rule_keeps_consistent_rows_and_holds_instance_dependent_ones():
    person, racket = [10, 100, 40, 200], [30, 20, 30, 30]
    q = SPATIAL_Q.format(a="person", b="tennis racket", x="below", y="above")
    assert sources.spatial_status(FakeCoco({"person": [person], "tennis racket": [racket]}), 1, q, "(A)") is None, "overlap alone is not an error"
    assert sources.spatial_status(FakeCoco({"person": [person], "tennis racket": [racket]}), 1, q, "(B)") == "spatial_label_contradicts_boxes"
    two = {"person": [person, [200, 0, 40, 10]], "tennis racket": [racket]}
    assert sources.spatial_status(FakeCoco(two), 1, q, "(A)") == "spatial_instance_dependent"
    agree = {"person": [person, [200, 150, 40, 100]], "tennis racket": [racket]}
    assert sources.spatial_status(FakeCoco(agree), 1, q, "(A)") is None
    assert sources.spatial_status(FakeCoco({"person": [person]}), 1, q, "(A)") == "spatial_objects_not_annotated"
    assert sources.spatial_status(FakeCoco({}), 1, "Is it nice?", "(A)") == "spatial_question_unparsed"
    lr = "Where is the cat in relation to the keyboard in this image? Select from the following choices.\n(A) left\n(B) right"
    assert sources.spatial_status(FakeCoco({"cat": [[0, 0, 10, 10]], "keyboard": [[50, 0, 10, 10]]}), 1, lr, "(A)") is None


def test_split_checks_separate_forbidden_from_permitted_sharing():
    rows = [{"bucket": "default/sft/train", "record_id": "a", "group_id": "g1"}, {"bucket": "pending/sft/train", "record_id": "b", "group_id": "g1"},
            {"bucket": "default/sft/sweep50k", "record_id": "a", "group_id": "g1"}, {"bucket": "default/rl/pool", "record_id": "c", "group_id": "g2"}]
    forbidden, permitted, sweep = pipeline.split_checks(rows)
    assert forbidden == {} and permitted == {"default/sft/train & pending/sft/train": 1} and sweep
    forbidden, _, _ = pipeline.split_checks(rows + [{"bucket": "default/sft/validation", "record_id": "d", "group_id": "g1"}, {"bucket": "default/rl/heldout", "record_id": "e", "group_id": "g1"}])
    assert set(forbidden) == {"default/rl/heldout & default/sft/train", "default/rl/heldout & default/sft/validation", "default/sft/train & default/sft/validation",
                              "default/rl/heldout & pending/sft/train", "default/sft/validation & pending/sft/train"}
    assert not pipeline.split_checks(rows + [{"bucket": "default/sft/sweep50k", "record_id": "zz", "group_id": "g9"}])[2]


@needs_miles
def test_verify_fails_on_missing_or_altered_materialized_rows(tmp_path):
    stored = images.prepare(png(photo(5, (64, 64))), tmp_path)
    record = {"kind": "sft", "image_path": stored["image_path"], "question": "Q?", "answer": "A", "record_id": "r1", "source": "s", "task": "t", "image_sha256": stored["image_sha256"], "status": "include", "provenance": "{}"}
    edir = tmp_path / "exports/p"
    portable = edir / "portable/default/sft/train/part-00000.parquet"
    portable.parent.mkdir(parents=True)
    row = pipeline.miles_row(record, None)
    pq.write_table(pa.Table.from_pylist([row]), portable)
    (edir / "index").mkdir()
    pq.write_table(pa.Table.from_pylist([{"record_id": "r1", "portable_hash": pipeline.portable_hash(row), "group_id": "g", "image_sha256": stored["image_sha256"]}]), edir / "index/default__sft__train.parquet")
    pq.write_table(pa.Table.from_pylist([{"image_sha256": stored["image_sha256"], "path": stored["image_path"]}]), edir / "images.parquet")
    files = {"portable/default/sft/train/part-00000.parquet": {}, "index/default__sft__train.parquet": {}, "images.parquet": {}}
    (edir / "manifest.json").write_text(json.dumps({"files": files}))
    (edir / "SHA256SUMS").write_text("".join(f"{runtime.file_sha256(edir / k)}  {k}\n" for k in files))
    ctx = SimpleNamespace(root=tmp_path, profile_name="p", export_dir=edir, recipe={"workers": {"validate": 1}})
    assert not pipeline.verify(ctx, tmp_path), "no materialized rows"
    pipeline.materialize(ctx, tmp_path)
    assert pipeline.verify(ctx, tmp_path)
    miles_file = edir / "miles/default/sft/train/part-00000.parquet"
    bad = {**pq.read_table(miles_file).to_pylist()[0], "metadata": {**row["metadata"], "image_sha256": "0" * 64}}
    pq.write_table(pa.Table.from_pylist([bad]), miles_file)
    assert not pipeline.verify(ctx, tmp_path)


def test_scan_image_geometry_comes_from_the_processor(checker):
    ctx = SimpleNamespace(recipe=tomllib.loads(RECIPE.read_text()))
    assert build.processor_image_config(ctx) == checker.image_config


@needs_miles
def test_validation_reuses_only_unchanged_rows_under_the_same_fingerprint(tmp_path, monkeypatch):
    rows = []
    for k in range(3):
        stored = images.prepare(png(photo(k)), tmp_path)
        rows.append(pipeline.miles_row({"kind": "sft", "image_path": stored["image_path"], "question": "What is shown?", "answer": f"pattern {k}", "record_id": f"r{k}", "source": "s", "task": "t", "image_sha256": stored["image_sha256"]}, tmp_path))
    chunk = tmp_path / "chunk-sft-00000.input.parquet"
    pq.write_table(pa.Table.from_pylist(rows), chunk)
    args = (str(RECIPE), str(tmp_path), "full")
    fresh = pipeline.validate_chunk(args, chunk, tmp_path / "fresh.done.json", "sft")
    first = pq.read_table(tmp_path / "fresh.done.parquet").to_pylist()
    assert fresh["reused"] == 0 and all(r["ok"] for r in first)

    sentinel = {**first[0], "total_tokens": 7}
    stale = {**first[1], "portable_hash": "0" * 64, "total_tokens": 7}
    marker = pipeline.validate_chunk(args, chunk, tmp_path / "again.done.json", "sft", [sentinel, stale])
    again = pq.read_table(tmp_path / "again.done.parquet").to_pylist()
    assert marker["reused"] == 1 and again[0] == sentinel, "a matching record_id and portable hash is reused as is"
    assert again[1] == first[1] and again[2] == first[2], "a changed row is checked again"
    assert marker["fingerprint"] == fresh["fingerprint"]

    ctx = build.Context(*args)
    sft, rl = pipeline._validate_fingerprint(ctx, "sft"), pipeline._validate_fingerprint(ctx, "rl")
    real = runtime.file_sha256
    monkeypatch.setattr(runtime, "file_sha256", lambda p: "changed" if Path(p).name == "reward.py" else real(p))
    assert pipeline._validate_fingerprint(ctx, "sft") == sft and pipeline._validate_fingerprint(ctx, "rl") != rl, "only RL validation depends on the scorer"


def test_blank_rl_images_are_held_as_pending():
    blank = np.full(32 * 32, 255, np.uint8).tobytes()
    noisy = np.random.default_rng(0).integers(0, 255, 32 * 32, dtype=np.uint8).tobytes()
    assert pipeline.blank_thumb(blank) and pipeline.blank_thumb(np.full(1024, 3, np.uint8).tobytes())
    assert not pipeline.blank_thumb(noisy) and not pipeline.blank_thumb(None)
    table = pa.table({"record_id": ["a", "b"], "status": ["include", "include"], "provenance": [build.canonical({"upstream_source": "DeepScaleR"}), build.canonical({})]})
    held = pipeline.hold_rows(table, {"a"}, "image_blank").to_pylist()
    assert held[0]["status"] == "pending" and json.loads(held[0]["provenance"]) == {"upstream_source": "DeepScaleR", "pending_reason": "image_blank"}
    assert held[1] == table.to_pylist()[1]
    assert pipeline.hold_rows(table, set(), "image_blank") is table


def test_select_trace_records_each_decision_and_the_edges_behind_groups(tmp_path):
    # Images 0-1 share pixels, 1-2 are a confirmed near-duplicate pair, 3 joins 0 through a group hint: one group g0.
    rids = ["rl", "rl_cut", "sft_res", "sft_dup", "sft_cap", "sft_q", "sft_ok", "sft_ev", "sft_lim", "sft_s1", "sft_s3"]
    kind = ["rl", "rl"] + ["sft"] * 9
    img = ["s0", "s9", "s2", "s2", "s4", "s4", "s5", "s6", "s7", "s1", "s3"]
    c = {"record_id": rids, "kind": kind, "source": ["r", "r"] + ["s"] * 9, "image_sha256": img, "question": ["q"] * 3 + ["Q"] + ["a", "b", "q", "q", "q", "c", "d"], "answer": ["1"] * 11}
    group = ["g0", "g9", "g0", "g0", "g4", "g4", "g5", "g6", "g7", "g0", "g0"]
    u = {"sha": ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s9"], "pixel_sha256": ["p0", "p0", "p2", "p3", "p4", "p5", "p6", "p7", "p9"]}
    by_pixel = {p: k for k, p in reversed(list(enumerate(u["pixel_sha256"])))}
    candidates, metrics = [(1, 2), (4, 5)], [{"edge": 0.99}, {"edge": 0.2}]
    winner = {pipeline.dedup_key(kind[i], group[i], c["question"][i], c["answer"][i]): i for i in (0, 1, 2, 4, 5, 6, 9, 10)}
    is_winner = np.array([i in winner.values() for i in range(11)])
    excluded = np.array([i == 7 for i in range(11)])
    limit = [("limit_total_tokens_est" if i == 8 else None) for i in range(11)]
    pipeline.write_trace(tmp_path, c, group, u, by_pixel, [(1, 2)], metrics, candidates, [(0, 3, "doc:x")], excluded, limit, winner, is_winner, set(), [0, 6], {"g0"}, {4})
    rows = {r["record_id"]: r for r in pq.read_table(tmp_path / "candidates.parquet").to_pylist()}
    assert {k: v["decision"] for k, v in rows.items()} == {
        "rl": "selected", "rl_cut": "rl_quota", "sft_res": "sft_reserved_for_rl", "sft_dup": "duplicate", "sft_cap": "sft_per_image_cap",
        "sft_q": "sft_quota", "sft_ok": "selected", "sft_ev": "eval_excluded", "sft_lim": "limit_total_tokens_est",
        "sft_s1": "sft_reserved_for_rl", "sft_s3": "sft_reserved_for_rl"}
    assert c["question"][3] == "Q" and rows["sft_dup"]["duplicate_of"] == "sft_res" and rows["sft_res"]["duplicate_of"] is None
    edges = pq.read_table(tmp_path / "edges.parquet").to_pylist()
    assert [(e["a"], e["b"], e["via"], e["group_id"]) for e in edges] == [("s0", "s1", "pixel_sha256", "g0"), ("s1", "s2", "near_duplicate", "g0"), ("s0", "s3", "group_hint", "g0")]
    assert json.loads(edges[1]["metrics"]) == {"edge": 0.99} and edges[2]["hint"] == "doc:x"


def test_temp_paths_are_unique():
    assert len({runtime.temp_path("/x/y.bin") for _ in range(100)}) == 100
