import hashlib
import multiprocessing
import pickle
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from k2hv import lazy_data

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "hf/K2-Horizon-V-init"
needs_miles = pytest.mark.skipif(not (ROOT / "third_party/miles/.git").exists() or not (HF / "tokenizer.json").exists(), reason="pinned Miles checkout and tokenizer.json required")
KEYS = {"image": "images"}


def write_rows(tmp_path, n, parts=1):
    rows = []
    for i in range(n):
        rng = np.random.default_rng(i)
        size = (64 + 28 * (i % 5), 56 + 28 * (i % 3))
        path = tmp_path / f"img{i}.png"
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
        rows.append({"messages": [{"role": "user", "content": f"<image>\nQuestion {i}?", "think": None}, {"role": "assistant", "content": f"answer {i}", "think": ""}],
                     "images": [str(path)], "metadata": {"record_id": f"r{i}"}})
    out = tmp_path / "rows"
    out.mkdir()
    step = -(-n // parts)
    for p in range(parts):
        pq.write_table(pa.Table.from_pylist(rows[p * step : (p + 1) * step]), out / f"part-{p:05d}.parquet")
    return out, rows


def args_for(path, **kw):
    base = dict(rollout_global_dataset=True, prompt_data=str(path), input_key="messages", multimodal_keys=KEYS, label_key=None, metadata_key="metadata",
                tool_key=None, apply_chat_template=False, apply_chat_template_kwargs=None, rollout_max_prompt_len=None, rollout_seed=7,
                rollout_shuffle=True, n_samples_per_prompt=1, hf_checkpoint=str(HF), dump_details=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def digest(sample):
    return [hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in sample.multimodal_inputs["images"]]


def same(a, b):
    return a.prompt == b.prompt and a.metadata == b.metadata and a.label == b.label and digest(a) == digest(b)


@pytest.fixture(scope="module")
def pinned():
    from k2hv import miles_runtime as runtime

    m = runtime.miles()
    proc = runtime.load_processor(HF)
    loading = runtime.miles_loading()
    counter = {"n": 0}

    def counted(prompt, processor):
        counter["n"] += 1
        return m.process_vision_info(prompt, processor)

    fns = types.SimpleNamespace(build_messages=m.build_messages, process_vision_info=counted, Sample=m.Sample)

    def source(args):
        cls = type("Lazy", (lazy_data.LazyInit, loading.RolloutDataSource), {"miles": fns, "load_processor": staticmethod(lambda a: proc)})
        return cls(args)

    return types.SimpleNamespace(m=m, proc=proc, loading=loading, fns=fns, counter=counter, source=source)


@needs_miles
def test_construction_decodes_nothing_and_matches_pinned_eager_dataset(tmp_path, pinned):
    path, rows = write_rows(tmp_path, 9)
    pinned.counter["n"] = 0
    lazy = lazy_data.LazyDataset(path, pinned.proc, prompt_key="messages", multimodal_keys=KEYS, seed=7, miles=pinned.fns)
    assert pinned.counter["n"] == 0 and lazy.decoded == 0 and len(lazy) == 9
    eager = pinned.loading.Dataset(str(path / "part-00000.parquet"), None, pinned.proc, None, prompt_key="messages", multimodal_keys=KEYS, seed=7)
    for epoch in (0, 1, 2):
        eager.shuffle(epoch)
        lazy.shuffle(epoch)
        got = lazy.samples[:]
        assert len(got) == len(eager.samples) and all(same(a, b) for a, b in zip(got, eager.samples))
    assert pinned.counter["n"] == 27


@needs_miles
def test_get_samples_decodes_only_the_requested_rows(tmp_path, pinned):
    path, rows = write_rows(tmp_path, 10, parts=3)
    pinned.counter["n"] = 0
    src = pinned.source(args_for(path))
    assert pinned.counter["n"] == 0
    batch = src.get_samples(4)
    assert pinned.counter["n"] == 4 and len(batch) == 4
    by_id = {r["metadata"]["record_id"]: r for r in rows}
    for (sample,) in batch:
        row = by_id[sample.metadata["record_id"]]
        # Message/media alignment: the image item sits where the marker was and is the row's file.
        content = sample.prompt[0]["content"]
        assert [c["type"] for c in content] == ["image", "text"] and content[0]["image"] == row["images"][0]
        assert content[1]["text"] == row["messages"][0]["content"].replace("<image>", "")
        with Image.open(row["images"][0]) as im:
            assert sample.multimodal_inputs["images"][0].size[0] % 28 == 0 and abs(sample.multimodal_inputs["images"][0].size[0] - im.size[0]) <= 28


@needs_miles
def test_epochs_and_checkpoint_resume_match_uninterrupted(tmp_path, pinned):
    path, _ = write_rows(tmp_path, 7)
    ref = pinned.source(args_for(path))
    run = pinned.source(args_for(path))
    expected = [[g[0].metadata["record_id"] for g in ref.get_samples(3)] for _ in range(5)]
    got = [[g[0].metadata["record_id"] for g in run.get_samples(3)] for _ in range(2)]
    run.save(tmp_path / "ckpt")
    resumed = pinned.source(args_for(path))
    resumed.load(tmp_path / "ckpt")
    got += [[g[0].metadata["record_id"] for g in resumed.get_samples(3)] for _ in range(3)]
    assert got == expected and len({tuple(b) for b in expected}) > 1
    eager_args = args_for(path / "part-00000.parquet")
    eager = pinned.loading.RolloutDataSource.__new__(pinned.loading.RolloutDataSource)
    eager.__dict__.update(args=eager_args, epoch_id=0, sample_group_index=0, sample_index=0, sample_offset=0, metadata={})
    eager.dataset = pinned.loading.Dataset(str(path / "part-00000.parquet"), None, pinned.proc, None, prompt_key="messages", multimodal_keys=KEYS, seed=7)
    eager.dataset.shuffle(0)
    assert [[g[0].metadata["record_id"] for g in eager.get_samples(3)] for _ in range(5)] == expected


def _child(blob, index, queue):
    from k2hv import miles_runtime as runtime

    m = runtime.miles()
    ds = pickle.loads(blob)
    ds.processor = runtime.load_processor(HF)
    ds._miles = types.SimpleNamespace(build_messages=m.build_messages, process_vision_info=m.process_vision_info, Sample=m.Sample)
    s = ds.samples[index]
    queue.put((s.metadata["record_id"], digest(s)))


@needs_miles
def test_pickled_dataset_rebuilds_same_samples_in_spawned_worker(tmp_path, pinned):
    path, _ = write_rows(tmp_path, 6)
    ds = lazy_data.LazyDataset(path, pinned.proc, prompt_key="messages", multimodal_keys=KEYS, seed=7, miles=pinned.fns)
    ds.shuffle(3)
    expected = ds.samples[2]
    ds.processor = None
    blob = pickle.dumps(ds)
    assert len(blob) < 4096, "the pickle carries paths and indices, not rows or pixels"
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(blob, 2, queue))
    proc.start()
    got = queue.get(timeout=120)
    proc.join(60)
    assert proc.exitcode == 0 and got == (expected.metadata["record_id"], digest(expected))


@needs_miles
def test_unsupported_configurations_raise(tmp_path, pinned):
    path, _ = write_rows(tmp_path, 2)
    for kw in ({"apply_chat_template": True}, {"rollout_max_prompt_len": 2048}, {"tool_key": "tools"}):
        with pytest.raises(ValueError, match="does not support"):
            pinned.source(args_for(path, **kw))
    with pytest.raises(FileNotFoundError):
        lazy_data.LazyDataset(tmp_path / "missing", None, prompt_key="messages", multimodal_keys=KEYS)
