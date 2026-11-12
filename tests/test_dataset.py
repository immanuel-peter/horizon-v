import hashlib
import json
from pathlib import Path

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
    out = dataset.check(tmp_path, buckets={"sft_train": "default/sft/train", "rl_pool": "default/rl/pool"})
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
        dataset.check(tmp_path, buckets={"sft_train": "default/sft/train", "rl_pool": "default/rl/pool"})


def test_check_rejects_missing_configured_bucket(tmp_path):
    write_export(tmp_path)
    with pytest.raises(ValueError, match="configured buckets missing"):
        dataset.check(tmp_path, buckets={"sft_validation": "default/sft/validation"})


def test_resolve_root_ignores_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pin = dataset.load_pin()
    assert dataset.resolve_root(pin) == Path(__file__).resolve().parents[1] / "data" / "vision-pt"


def make_images_tree(root, payload=b"fake-image-bytes"):
    import io
    import tarfile

    shard_dir = root / "images" / "shards"
    shard_dir.mkdir(parents=True)
    shard = shard_dir / "images-00000-of-00001.tar"
    with tarfile.open(shard, "w") as tf:
        info = tarfile.TarInfo("images/ab/abc.png")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    (root / "images" / "SHARDS.json").write_text(json.dumps({"images": 1, "shards": [{"shard": "images/shards/images-00000-of-00001.tar", "sha256": digest, "bytes": shard.stat().st_size, "members": 1}]}) + "\n")
    return shard


def test_unpack_images_verifies_and_removes_shard(tmp_path):
    shard = make_images_tree(tmp_path)
    dataset.unpack_images(tmp_path)
    assert (tmp_path / "images" / "ab" / "abc.png").read_bytes() == b"fake-image-bytes"
    assert not shard.exists()


def test_unpack_images_rejects_corrupt_shard(tmp_path):
    shard = make_images_tree(tmp_path)
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="shard checksum"):
        dataset.unpack_images(tmp_path)
    assert shard.exists()
    assert not (tmp_path / "images" / "ab" / "abc.png").exists()
