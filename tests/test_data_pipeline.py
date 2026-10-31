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


@needs_rl360
@pytest.mark.parametrize("problem,reference,answer_type,target,selector", [
    ("<image>\nFind x.", "\\boxed{2016}", "numeric", "2016", scoring.MATH),
    ("<image>\nPerimeter?", "\\boxed{20 cm}", "numeric", "20", scoring.MATH),
    ("<image>\nAngle?", "60°", "numeric", "60", scoring.MATH),
    ("<image>\nCost?", "$1,234", "numeric", "1234", scoring.MATH),
    ("<image>\nShare?", "35.294117647058826", "numeric_long_decimal", "35.294117647058826", scoring.PENDING_NUMERIC),
    ("<image>\nFind b.", "2 \\sqrt { 22 }", "symbolic", "2 \\sqrt { 22 }", scoring.MATH),
    ("<image>\nWhich?\nA. cat\nB. dog", "B", "choice", "B", scoring.CHOICE),
    ("<image>\nWhich?\n(A) cat\n(B) dog", "dog", "choice", "B", scoring.CHOICE),
])
def test_rl_targets_are_classified_and_admitted(problem, reference, answer_type, target, selector):
    info = scoring.classify(problem, reference)
    assert (info["answer_type"], info["target"], info["selector"]) == (answer_type, target, selector)
    assert scoring.admit(problem, info) is None


@needs_rl360
@pytest.mark.parametrize("reference,reason", [("Yes", "answer_yes_no"), ("5.48e+04", "answer_scientific"), ("Consecutive interior angles are supplementary.", "answer_free_form"), ("", "answer_empty")])
def test_unscorable_rl_targets_are_excluded(reference, reason):
    assert scoring.classify("<image>\nQ?", reference)["reason"] == reason
    assert scoring.classify("<image>\nWhich one?", "C")["reason"] == "choice_options_unparsed"
    assert scoring.classify("<image>\nWhich?\nA. cat\nB. dog", "fish")["reason"] == "choice_unmatched"


@needs_rl360
def test_pinned_scorer_behavior_is_characterized():
    ok = scoring.scorer_correct
    assert ok(scoring.MATH, "Answer: 1.5", "1.5") and ok(scoring.MATH, "Answer: \\boxed{1.5}", "1.5") and ok(scoring.MATH, "Answer: 1.5 cm", "1.5")
    assert not ok(scoring.MATH, "\\boxed{1.5}", "1.5"), "math_dapo reads only the Answer: line"
    assert not ok(scoring.MATH, "Answer: 1.50", "1.5") and not ok(scoring.MATH, "Answer: 3/2", "1.5"), "no numeric equivalence"
    assert not ok(scoring.MATH, "Answer: 20%", "20"), "percent signs are not normalized"
    assert not ok(scoring.MATH, "Answer: 2.5", "1.5")
    for response in ("Answer: B", "Answer: (B)", "\\boxed{B}", "B", "The answer is B.", "Answer: b"):
        assert ok(scoring.CHOICE, response, "B"), response
    assert not ok(scoring.CHOICE, "Answer: C", "B")
    assert ok(scoring.PENDING_NUMERIC, "Answer: 35.29", "35.294117647058826") and not ok(scoring.PENDING_NUMERIC, "Answer: 36", "35.294117647058826")
    with pytest.raises(ValueError, match="Unsupported"):
        ok("geo3k", "Answer: 1", "1")


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
    assert res["total_tokens"] == len(ids) and res["bos_tokens"] == 2, "current prepare_sample adds a second BOS (reported in step 5)"

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
    row = {"problem": "<image>\nHow many bars?\n\n" + scoring.MATH_SUFFIX, "answer": "3", "images": [str(tmp_path / stored["image_path"])], "metadata": {"data_source": "math_dapo"}}
    res = checker.rl(row, stored["image_sha256"])
    assert res["image_tokens"] == images.image_tokens(480, 320, checker.image_config)[0] and res["prompt_tokens"] > res["image_tokens"]
    with pytest.raises(contract.ContractError, match="row_selector"):
        checker.rl({**row, "metadata": {"data_source": "geo3k"}})


# ------------------------------------------------------------------- portability and resume


def test_portable_hash_and_materialize_rewrite_only_the_root(tmp_path):
    record = {"kind": "sft", "image_path": "images/ab/abc.png", "question": "Q?", "answer": "A", "record_id": "r", "source": "s", "task": "t", "image_sha256": "abc"}
    a, b = pipeline.miles_row(record, Path("/data/one")), pipeline.miles_row(record, Path("/mnt/two"))
    assert a["images"] == ["/data/one/images/ab/abc.png"] and pipeline.portable_hash(a) == pipeline.portable_hash(b) == pipeline.portable_hash(pipeline.miles_row(record, None))
    ctx = SimpleNamespace(root=tmp_path, profile_name="p")
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
        assert src["status"] in ("include", "addon", "pending", "exclude"), key
        assert src["adapter"] in sources.ADAPTERS and sources.ADAPTERS[src["adapter"]].dataset in recipe["datasets"], key
        assert src["terms"] and src["terms_note"] and src["kind"] in ("sft", "rl"), key
        if src["status"] == "addon":
            assert src["terms"] in recipe["terms"]["review"], key
        if src["status"] == "include":
            assert src["terms"] not in recipe["terms"]["review"], key
    for ds in list(recipe["datasets"].values()) + list(recipe["evals"].values()):
        assert len(ds["revision"]) == 40
    assert set(recipe["evals"]) == {"mmmu_val", "mathvista_testmini", "chartqa_test", "docvqa_val", "textvqa_val", "mmbench_dev", "pope_test"}
