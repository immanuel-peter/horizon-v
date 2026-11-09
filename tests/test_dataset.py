import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from k2hv import dataset


def write_export(root):
    edir = root / "exports" / "vision-pt-v1"
    sft = [{"messages": [{"role": "user", "content": "<image>\nQ?", "think": None}, {"role": "assistant", "content": "A", "think": ""}], "images": ["images/ab/abc.png"], "metadata": {"record_id": "r1", "source": "fv_vqav2", "task": "general_vqa", "image_sha256": "ab", "pending_reason": None}}]
    rl = [{"problem": "<image>\nFind x.", "answer": "1", "images": ["images/ab/abc.png"], "metadata": {"data_source": "math_v1", "record_id": "r2", "source": "rl_geometry3k", "dataset": "hiyouga/geometry3k", "upstream_id": "train:1", "reference": "1", "answer_type": "numeric", "task": "geometry", "image_sha256": "ab", "answer_decimals": None, "pending_reason": None}}]
    files = {}
    for bucket, rows in [("default/sft/train", sft), ("default/rl/pool", rl)]:
        dest = edir / "portable" / bucket / "part-00000.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), dest)
        files[f"portable/{bucket}/part-00000.parquet"] = dest
    manifest = {"counts": {"default/sft/train": 1, "default/rl/pool": 1}, "files": {}}
    lines = []
    for rel, dest in files.items():
        data = dest.read_bytes()
        manifest["files"][rel] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
    (edir / "manifest.json").write_text(json.dumps(manifest) + "\n")
    (edir / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return edir


def test_check_and_materialize(tmp_path):
    edir = write_export(tmp_path)
    out = dataset.check(tmp_path)
    assert out["buckets"] == {"default/rl/pool": 1, "default/sft/train": 1}
    assert out["selectors"] == ["math_v1"]
    written = dataset.materialize(tmp_path)
    assert len(written) == 2
    row = pq.read_table(edir / "miles/default/sft/train/part-00000.parquet").to_pylist()[0]
    assert row["images"] == [str(tmp_path.resolve() / "images/ab/abc.png")]


def test_check_rejects_unknown_selector(tmp_path):
    edir = write_export(tmp_path)
    path = edir / "portable/default/rl/pool/part-00000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["metadata"]["data_source"] = "someone_elses_selector"
    pq.write_table(pa.Table.from_pylist(rows), path)
    sums = []
    manifest = json.loads((edir / "manifest.json").read_text())
    for rel in manifest["files"]:
        data = (edir / rel).read_bytes()
        sums.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
    (edir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    with pytest.raises(ValueError, match="unknown RL selectors"):
        dataset.check(tmp_path)
