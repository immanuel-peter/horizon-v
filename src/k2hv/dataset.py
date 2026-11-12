"""Consumer side of the standalone Vision Post-Training Dataset.

This repo does not build training data. It pins one dataset release in
``configs/dataset.toml``, downloads it, rewrites the portable image paths for
the local machine, and runs a light integrity check. Building, overlap
exclusion and the full verification live with the dataset, not here.

Usage:
  python -m k2hv.dataset fetch       download the pinned release, verify each image shard against
                                     images/SHARDS.json, unpack it, then remove the shard tar
  python -m k2hv.dataset materialize write Miles Parquet with absolute image paths
  python -m k2hv.dataset check       export checksums, bucket counts, selector values
"""

import argparse
import hashlib
import json
import tarfile
import tomllib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ALLOWED_SELECTORS = {"math_v1", "choice_v1", "decimal_v1", "pending_decimal_precision"}


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    pin_path = Path(path) if path else REPO_ROOT / "configs" / "dataset.toml"
    with open(pin_path, "rb") as f:
        return tomllib.load(f)


def load_pin(path=None):
    return load_config(path)["dataset"]


def resolve_root(pin, root=None):
    if root:
        return Path(root)
    local = Path(pin["local_root"])
    return local if local.is_absolute() else REPO_ROOT / local


def export_root(root, pin):
    return Path(root) / "exports" / pin["export_dir"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack_images(root):
    root = Path(root)
    index = root / "images" / "SHARDS.json"
    if not index.exists():
        raise FileNotFoundError(f"downloaded tree at {root} has no images/SHARDS.json")
    shards = json.loads(index.read_text())["shards"]
    for entry in shards:
        shard = root / entry["shard"]
        if not shard.exists() or shard.stat().st_size != entry["bytes"] or sha256_file(shard) != entry["sha256"]:
            raise ValueError(f"shard checksum mismatch: {entry['shard']}")
        with tarfile.open(shard) as tf:
            tf.extractall(root, filter="data")
        shard.unlink()
    return len(shards)


def fetch(root, pin):
    from huggingface_hub import snapshot_download

    root = Path(root)
    snapshot_download(repo_id=pin["repo"], repo_type="dataset", revision=pin["revision"], local_dir=root)
    unpack_images(root)
    return root


def materialize(root, target_root=None, pin=None):
    pin = pin or load_pin()
    edir = export_root(root, pin)
    target = Path(target_root or root).resolve()
    written = {}
    for path in sorted((edir / "portable").rglob("*.parquet")):
        table = pq.read_table(path)
        imgs = table.column("images").to_pylist()
        table = table.set_column(table.schema.get_field_index("images"), "images", pa.array([[str(target / p) for p in row] for row in imgs], pa.list_(pa.string())))
        dest = edir / "miles" / path.relative_to(edir / "portable")
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest)
        written[str(dest.relative_to(edir))] = hashlib.sha256(dest.read_bytes()).hexdigest()
    (edir / "miles" / "materialized.json").write_text(json.dumps({"target_root": str(target), "files": written}, indent=1) + "\n")
    return written


def check(root, pin=None, buckets=None):
    pin = pin or load_pin()
    if buckets is None:
        buckets = load_config().get("buckets", {})
    edir = export_root(root, pin)
    bad = []
    for line in (edir / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        f = edir / rel
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            bad.append(rel)
    if bad:
        raise ValueError(f"export checksum failures: {bad[:5]}")
    manifest = json.loads((edir / "manifest.json").read_text())
    missing = sorted(set(buckets.values()) - set(manifest["counts"]))
    if missing:
        raise ValueError(f"configured buckets missing from dataset manifest: {missing}")
    counts = {}
    selectors = set()
    for bucket, expected in manifest["counts"].items():
        total = 0
        for path in sorted((edir / "portable" / bucket).glob("*.parquet")):
            total += pq.ParquetFile(path).metadata.num_rows
            if "rl" in bucket:
                selectors.update(m["data_source"] for m in pq.read_table(path, columns=["metadata"]).column("metadata").to_pylist())
        counts[bucket] = total
        if total != expected:
            raise ValueError(f"bucket {bucket}: {total} rows, manifest says {expected}")
    unknown = selectors - ALLOWED_SELECTORS
    if unknown:
        raise ValueError(f"unknown RL selectors in dataset: {sorted(unknown)}")
    return {"buckets": counts, "selectors": sorted(selectors)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m k2hv.dataset", description=__doc__)
    parser.add_argument("command", choices=["fetch", "materialize", "check"])
    parser.add_argument("--root", help="local dataset root (default: the pinned local_root)")
    parser.add_argument("--target-root", help="materialize: root that holds images/ on this machine")
    args = parser.parse_args(argv)
    config = load_config()
    pin = config["dataset"]
    root = resolve_root(pin, args.root)
    if args.command == "fetch":
        fetch(root, pin)
        print(f"fetched {pin['repo']}@{pin['revision'][:12]} into {root}")
    elif args.command == "materialize":
        written = materialize(root, args.target_root, pin)
        print(f"materialized {len(written)} files under {export_root(root, pin) / 'miles'}")
    else:
        print(json.dumps(check(root, pin, config.get("buckets", {})), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
