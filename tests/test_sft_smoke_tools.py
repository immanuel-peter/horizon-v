import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from k2hv import sft_monitor, sft_resume_check

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "hf/K2-Horizon-V-init"
needs_miles = pytest.mark.skipif(not (ROOT / "third_party/miles/.git").exists() or not (HF / "tokenizer.json").exists(), reason="pinned Miles checkout and tokenizer.json required")


def load_selector():
    spec = importlib.util.spec_from_file_location("select_subset", ROOT / "scripts/sft/select_subset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def index_rows(n):
    rng = np.random.default_rng(0)
    tasks, sources = ["chart", "count", "doc"], ["a", "b", "c", "d"]
    return [
        {"record_id": f"r{i}", "source": sources[i % 4], "task": tasks[i % 3], "total_tokens": int(rng.integers(40, 1140)), "image_tokens": int(rng.integers(16, 1025)), "supervised_tokens": int(rng.integers(2, 40))}
        for i in range(n)
    ]


def test_subset_selection_is_deterministic_stratified_and_keeps_extremes():
    selector = load_selector()
    rows = index_rows(400)
    first = selector.choose(rows, 0, 60, 2, 3)
    assert first == selector.choose(list(reversed(rows)), 0, 60, 2, 3)
    assert first != selector.choose(rows, 1, 60, 2, 3)
    ids = {r["record_id"] for r in first}
    assert len(first) == 60 == len(ids)
    for key in ("total_tokens", "image_tokens", "supervised_tokens"):
        assert max(rows, key=lambda r: r[key])["record_id"] in ids
    assert {r["source"] for r in first} == {r["source"] for r in rows}
    assert {(r["task"], selector.band(r["total_tokens"])) for r in first} == {(r["task"], selector.band(r["total_tokens"])) for r in rows}
    with pytest.raises(ValueError, match="exceed"):
        selector.choose(rows, 0, 10, 4, 3)


@pytest.fixture(scope="module")
def prepared():
    from k2hv import miles_runtime as runtime
    from k2hv.sft_rollout import prepare_sample

    m = runtime.miles()
    processor = runtime.load_processor(HF)
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (84, 112, 3), dtype=np.uint8))
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "What is shown?"}]}, {"role": "assistant", "content": "a cat", "think": ""}]
    sample = prepare_sample(processor, m.Sample(prompt=messages, metadata={"record_id": "x"}, multimodal_inputs={"images": [image]}))
    return processor, sample


@needs_miles
def test_inspect_sample_accepts_prepared_sample_and_rejects_corruption(prepared):
    processor, sample = prepared
    ids = sft_monitor.token_ids(processor, HF)
    row = sft_monitor.inspect_sample(processor, ids, sample)
    assert row["record_id"] == "x" and row["supervised_tokens"] == sample.response_length and row["images"] == 1
    assert row["image_tokens"] == int(sample.multimodal_train_inputs["image_grid_thw"].prod(-1).sum()) // 4
    corruptions = {
        "one_leading_bos": lambda s: setattr(s, "tokens", [ids["bos"]] + s.tokens),
        "mask_is_response": lambda s: setattr(s, "loss_mask", [1] * (s.response_length + 1)),
        "image_pads_match_grid": lambda s: setattr(s, "tokens", [t for i, t in enumerate(s.tokens) if not (t == ids["pad"] and i == s.tokens.index(ids["pad"]))]),
    }
    for name, corrupt in corruptions.items():
        bad = types.SimpleNamespace(**{k: getattr(sample, k) for k in ("tokens", "response_length", "loss_mask", "multimodal_train_inputs", "multimodal_inputs", "metadata")})
        bad.tokens, bad.loss_mask = list(bad.tokens), list(bad.loss_mask)
        corrupt(bad)
        with pytest.raises(ValueError, match=name):
            sft_monitor.inspect_sample(processor, ids, bad)


def test_parse_log_reads_miles_step_perf_and_checkpoint_events(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "x - step 0: {'train/loss': 2.5, 'train/grad_norm': 1.0, 'train/lr-pg_0': 0.001}\n"
        "y - perf 0: {'perf/step_time': 3.0}\n"
        "z - perf 0: {'rollout/num_training_samples': 4}\n"
        "w - [FSDP] Loaded optimizer from /c/iter_0000002/optimizer\n"
    )
    steps, perf, events = sft_monitor.parse_log(log)
    assert steps == {0: {"train/loss": 2.5, "train/grad_norm": 1.0, "train/lr-pg_0": 0.001}}
    assert perf[0] == {"perf/step_time": 3.0, "rollout/num_training_samples": 4}
    assert events == ["w - [FSDP] Loaded optimizer from /c/iter_0000002/optimizer"]


def test_compare_state_separates_exact_bounded_and_structural_differences():
    a = {"state": {"p": {"exp_avg": torch.ones(4), "step": torch.tensor(3.0)}}, "param_groups": [{"lr": 0.1}]}
    same = json.loads(json.dumps({"param_groups": [{"lr": 0.1}]}))
    b = {"state": {"p": {"exp_avg": torch.ones(4), "step": torch.tensor(3.0)}}, **same}
    assert sft_resume_check.compare_state(a, b, 1e-3) == {"passed": True, "exact": True, "entries": 3, "worst": {"max_abs": 0.0, "rel_frobenius": 0.0}, "mismatched": []}
    b["state"]["p"]["exp_avg"] = torch.ones(4) + 1e-6
    close = sft_resume_check.compare_state(a, b, 1e-3)
    assert close["passed"] and not close["exact"] and close["worst"]["max_abs"] > 0
    assert not sft_resume_check.compare_state(a, b, None)["passed"]
    b["state"]["p"]["step"] = torch.tensor(4.0)
    assert sft_resume_check.compare_state(a, b, 1e-3)["mismatched"] == ["state/p/step"]
    b["param_groups"][0]["extra"] = 1
    assert not sft_resume_check.compare_state(a, b, 1e-3)["passed"]
