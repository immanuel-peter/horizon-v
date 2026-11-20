"""Select a deterministic, stratified SFT smoke subset from the materialized dataset buckets.

Training rows come from the configured ``sft_sweep`` bucket (a subset of ``sft_train``); held-out rows
come from ``sft_validation``. Rows are stratified by task and expanded-length band, every source is
covered, and the longest rows (total, image and supervised tokens) are always included. Ranks are
``sha256(f"{seed}:{record_id}")``, so the same seed and dataset pin give the same rows.

Writes ``train.parquet``, ``train_eval.parquet`` (every fifth training row) and ``validation.parquet``
(materialized Miles rows, unchanged) and
``selection.json`` (seed, pin, input hashes, record IDs and token lengths).
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from k2hv.dataset import export_root, load_config, resolve_root  # noqa: E402

BANDS = (0, 256, 512, 1024)
INDEX_COLUMNS = ["record_id", "source", "task", "total_tokens", "image_tokens", "supervised_tokens", "width", "height"]


def band(total):
    return sum(total >= edge for edge in BANDS) - 1


def rank(seed, record_id):
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def choose(rows, seed, size, per_stratum, extremes):
    rows = sorted(rows, key=lambda r: rank(seed, r["record_id"]))
    chosen = {}

    def take(row, why):
        chosen.setdefault(row["record_id"], {**row, "why": why})

    for key, why in (("total_tokens", "longest_total"), ("image_tokens", "most_image_tokens"), ("supervised_tokens", "longest_answer")):
        for row in sorted(rows, key=lambda r: (-r[key], rank(seed, r["record_id"])))[:extremes]:
            take(row, why)
    strata = Counter()
    for row in rows:
        stratum = (row["task"], band(row["total_tokens"]))
        if strata[stratum] < per_stratum:
            strata[stratum] += 1
            take(row, "stratum")
    for source in sorted({r["source"] for r in rows}):
        if not any(r["source"] == source for r in chosen.values()):
            take(next(r for r in rows if r["source"] == source), "source")
    for row in rows:
        if len(chosen) >= size:
            break
        take(row, "fill")
    if len(chosen) > size:
        raise ValueError(f"required rows ({len(chosen)}) exceed the requested size {size}")
    return sorted(chosen.values(), key=lambda r: (r["task"], band(r["total_tokens"]), rank(seed, r["record_id"])))


def miles_rows(directory, ids):
    files = sorted(Path(directory).glob("*.parquet"))
    table = pa.concat_tables([pq.read_table(f) for f in files])
    record_ids = pc.struct_field(table.column("metadata"), "record_id")
    table = table.filter(pc.is_in(record_ids, value_set=pa.array(sorted(ids))))
    by_id = {r["metadata"]["record_id"]: r for r in table.to_pylist()}
    if set(by_id) != set(ids):
        raise ValueError(f"materialized rows missing for {sorted(set(ids) - set(by_id))[:5]}")
    return by_id, {str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def summary(rows):
    return {
        "rows": len(rows),
        "tasks": dict(sorted(Counter(r["task"] for r in rows).items())),
        "sources": dict(sorted(Counter(r["source"] for r in rows).items())),
        "length_bands": {f">={BANDS[b]}": n for b, n in sorted(Counter(band(r["total_tokens"]) for r in rows).items())},
        "total_tokens": {"min": min(r["total_tokens"] for r in rows), "max": max(r["total_tokens"] for r in rows), "sum": sum(r["total_tokens"] for r in rows)},
        "image_tokens_max": max(r["image_tokens"] for r in rows),
        "supervised_tokens": {"max": max(r["supervised_tokens"] for r in rows), "sum": sum(r["supervised_tokens"] for r in rows)},
        "why": dict(Counter(r["why"] for r in rows)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-size", type=int, default=320)
    p.add_argument("--validation-size", type=int, default=64)
    p.add_argument("--root", help="dataset root (default: the pinned local_root)")
    a = p.parse_args()
    config = load_config()
    pin, buckets = config["dataset"], config["buckets"]
    edir = export_root(resolve_root(pin, a.root), pin)
    out = {"seed": a.seed, "dataset": pin, "buckets": {}, "materialized": json.loads((edir / "miles" / "materialized.json").read_text())["target_root"]}
    a.out.mkdir(parents=True, exist_ok=True)
    for name, key, size, per_stratum, extremes in (("train", "sft_sweep", a.train_size, 4, 8), ("validation", "sft_validation", a.validation_size, 1, 2)):
        bucket = buckets[key]
        index = edir / "index" / (bucket.replace("/", "__") + ".parquet")
        rows = pq.read_table(index, columns=INDEX_COLUMNS).to_pylist()
        chosen = choose(rows, a.seed, size, per_stratum, extremes)
        by_id, files = miles_rows(edir / "miles" / bucket, [r["record_id"] for r in chosen])
        pq.write_table(pa.Table.from_pylist([by_id[r["record_id"]] for r in chosen]), a.out / f"{name}.parquet")
        if name == "train":
            # Every fifth training row: a fixed slice for initial/final training loss.
            pq.write_table(pa.Table.from_pylist([by_id[r["record_id"]] for r in chosen[::5]]), a.out / "train_eval.parquet")
        out["buckets"][name] = {
            "bucket": bucket,
            "index": str(index.relative_to(ROOT)) if index.is_relative_to(ROOT) else str(index),
            "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            "materialized_files": files,
            "summary": summary(chosen),
            "rows": [{k: r[k] for k in ("record_id", "source", "task", "total_tokens", "image_tokens", "supervised_tokens", "why")} for r in chosen],
        }
    train_ids = {r["record_id"] for r in out["buckets"]["train"]["rows"]}
    if train_ids & {r["record_id"] for r in out["buckets"]["validation"]["rows"]}:
        raise ValueError("validation rows overlap training rows")
    (a.out / "selection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({name: b["summary"] for name, b in out["buckets"].items()}, indent=1))


if __name__ == "__main__":
    main()
