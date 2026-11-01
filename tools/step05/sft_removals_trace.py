"""Explain every v2 SFT row that is absent from full-v3, from the selection trace of a `select --trace` rerun.

For each removed row: the v3 select decision, its v3 group, the RL rows in that group, and the shortest chain of
edges (identical pixels, verified near-duplicate pair, shared group hint) from the row's image to an image carrying
a selected RL row. The rerun must reproduce full-v3's selected.parquet byte for byte, so the trace describes the
selection that produced the exports.

Writes OUT_JSON (summary, plus full detail for rows the v3 report could not attribute to an exported RL row in
their v2 group, i.e. those explained earlier only by phash distance) and OUT_PARQUET (one line per removed row).

usage: python tools/step05/sft_removals_trace.py DATA_ROOT TRACE_ROOT OUT_JSON OUT_PARQUET
"""

import glob
import hashlib
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

root, trace_root, out_json, out_parquet = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
work, twork = root / "work" / "full-v3", trace_root / "work" / "full-v3"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


reproduced = {name: (sha256(work / "select" / name), sha256(twork / "select" / name)) for name in ("selected.parquet", "exclusions.parquet", "select.json")}
assert all(a == b for a, b in reproduced.values()), reproduced


def index_rows(build, kind):
    rows = {}
    for f in sorted(glob.glob(str(root / "exports" / build / "index" / f"*__{kind}__*.parquet"))):
        bucket = Path(f).stem
        if bucket.endswith("sweep50k"):
            continue
        for r in pq.read_table(f, columns=["record_id", "source", "dataset", "revision", "file", "row", "turn", "upstream_id", "group_id", "split", "image_sha256"]).to_pylist():
            rows[r["record_id"]] = {**r, "bucket": bucket}
    return rows


v2, v3 = index_rows("full-v2", "sft"), index_rows("full-v3", "sft")
v3_rl = index_rows("full-v3", "rl")
removed = sorted(rid for rid in v2 if rid not in v3)
rl_groups_exported = {r["group_id"] for r in v3_rl.values()}

by_rid = {r["record_id"]: r for r in pq.read_table(twork / "select" / "trace" / "candidates.parquet").to_pylist()}
selected_rl = [r for r in by_rid.values() if r["kind"] == "rl" and r["decision"] == "selected"]
rl_by_group = defaultdict(list)
for r in selected_rl:
    rl_by_group[r["group_id"]].append(r)
rl_images = {r["image_sha256"] for r in selected_rl}
sel = pq.read_table(work / "select" / "selected.parquet", columns=["record_id", "status", "split", "provenance"]).to_pylist()
sel = {r["record_id"]: r for r in sel}

group_size = Counter()
for x in {r["image_sha256"]: r["group_id"] for r in by_rid.values()}.items():
    group_size[x[1]] += 1
edges = pq.read_table(twork / "select" / "trace" / "edges.parquet").to_pylist()
adj = defaultdict(list)
for e in edges:
    adj[e["a"]].append((e["b"], e))
    adj[e["b"]].append((e["a"], e))

# pHash of every candidate image, for the distances the v3 report used.
phash = {}
for f in glob.glob(str(work / "scan" / "*" / "*.parquet")):
    if f.endswith(".rejects.parquet"):
        continue
    t = pq.read_table(f, columns=["image_sha256", "phash"])
    for s, h in zip(t.column("image_sha256").to_pylist(), t.column("phash").to_numpy().view(np.uint64)):
        phash[s] = int(h)


def hamming(a, b):
    return bin(phash[a] ^ phash[b]).count("1")


def chain(start):
    """Shortest edge path from ``start`` to an image with a selected RL row (BFS over group edges)."""
    prev = {start: None}
    queue = deque([start])
    while queue:
        x = queue.popleft()
        if x in rl_images:
            path = []
            while prev[x] is not None:
                y, e = prev[x]
                path.append({"from": y, "to": x, "via": e["via"], "hint": e["hint"], "metrics": json.loads(e["metrics"]) if e["metrics"] else None, "phash_distance": hamming(y, x)})
                x = y
            return path[::-1]
        for y, e in adj[x]:
            if y not in prev:
                prev[y] = (x, e)
                queue.append(y)
    return None


rows = []
for rid in removed:
    old = v2[rid]
    t = by_rid.get(rid)
    decision = t["decision"] if t is not None else "not_a_v3_candidate"
    g = t["group_id"] if t is not None else None
    path = chain(old["image_sha256"]) if decision in ("sft_reserved_for_rl", "duplicate") else None
    rl = rl_by_group.get(g, [])
    rows.append({
        "record_id": rid, "source": old["source"], "v2_bucket": old["bucket"], "v2_group_id": old["group_id"], "image_sha256": old["image_sha256"],
        "v3_decision": decision, "v3_group_id": g, "duplicate_of": t["duplicate_of"] if t is not None else None,
        "v2_group_holds_exported_v3_rl": old["group_id"] in rl_groups_exported,
        "rl_rows_in_group": len(rl), "rl_sources_in_group": sorted({r["source"] for r in rl}),
        "rl_exported_in_group": sum(r["record_id"] in v3_rl for r in rl), "rl_pending_in_group": sum(sel[r["record_id"]]["status"] == "pending" for r in rl),
        "chain_length": len(path) if path is not None else None, "chain_via": [p["via"] for p in path] if path else None,
        "min_phash_distance_to_rl_image": min((hamming(old["image_sha256"], r["image_sha256"]) for r in rl), default=None),
        "chain": path, "identity": {k: old[k] for k in ("dataset", "revision", "file", "row", "turn", "upstream_id")}, "rl_rows": rl,
    })


def rl_detail(r):
    s = sel[r["record_id"]]
    return {"record_id": r["record_id"], "source": r["source"], "image_sha256": r["image_sha256"], "status": s["status"], "split": s["split"],
            "pending_reason": json.loads(s["provenance"]).get("pending_reason"), "exported_bucket": v3_rl[r["record_id"]]["bucket"] if r["record_id"] in v3_rl else None}


def counts(values):
    """Counter of small ints (or None) with string keys in numeric order, so the JSON sorts."""
    return {str(k): v for k, v in sorted(Counter(values).items(), key=lambda kv: (kv[0] is None, kv[0] or 0))}


late = [r for r in rows if not r["v2_group_holds_exported_v3_rl"]]
# The v3 report checked these rows only by pHash distance to the nearest selected RL image anywhere; nine of their
# images were more than 2 away. Recompute that distance to name those images, then explain them from the trace.
rl_hashes = np.array(sorted({phash[x] for x in rl_images}), dtype=np.uint64)
late_images = sorted({r["image_sha256"] for r in late})
bits = np.unpackbits((rl_hashes[None, :] ^ np.array([phash[x] for x in late_images], dtype=np.uint64)[:, None]).view(np.uint8).reshape(len(late_images), -1, 8), axis=2).sum(axis=2)
global_min = dict(zip(late_images, bits.min(axis=1).tolist()))
far = [r for r in late if global_min[r["image_sha256"]] > 2]
far_images = sorted({r["image_sha256"] for r in far})
dup_winners = {r["record_id"]: by_rid[r["duplicate_of"]] for r in rows if r["v3_decision"] == "duplicate"}
report = {
    "method": "select --trace rerun of full-v3 in a scratch root linked to the full-v3 scan, evaluation index and image store; "
              "reproduced selected.parquet, exclusions.parquet and select.json byte for byte",
    "reproduced_sha256": {k: v[0] for k, v in reproduced.items()},
    "removed_from_sft": len(removed),
    "v3_decision": dict(Counter(r["v3_decision"] for r in rows).most_common()),
    "v2_group_holds_exported_v3_rl": sum(r["v2_group_holds_exported_v3_rl"] for r in rows),
    "not_attributed_by_v2_group": {
        "rows": len(late), "images": len({r["image_sha256"] for r in late}), "decision": dict(Counter(r["v3_decision"] for r in late)),
        "chain_length": counts(r["chain_length"] for r in late),
        "chain_via": dict(Counter(" > ".join(r["chain_via"] or []) for r in late).most_common()),
        "group_rl_pending_only": sum(r["rl_exported_in_group"] == 0 and r["rl_pending_in_group"] > 0 for r in late),
        "rl_sources": dict(Counter(s for r in late for s in r["rl_sources_in_group"]).most_common()),
        "min_phash_distance_images": counts(r["min_phash_distance_to_rl_image"] for r in {r["image_sha256"]: r for r in late}.values()),
    },
    "duplicates": {"rows": len(dup_winners), "by_source": dict(Counter(r["source"] for r in rows if r["v3_decision"] == "duplicate")),
                   "winner_decision": dict(Counter(w["decision"] for w in dup_winners.values())),
                   "winner_same_source": sum(w["source"] == v2[rid]["source"] for rid, w in dup_winners.items()),
                   "winner_exported": sum(w["record_id"] in v3 or w["record_id"] in v3_rl for w in dup_winners.values()),
                   "winner_same_v3_group": sum(w["group_id"] == by_rid[rid]["group_id"] for rid, w in dup_winners.items())},
    "nine_images_over_phash_2": {
        "selection": "removed v2 SFT rows whose v2 group held no exported v3 RL row and whose image is more than pHash distance 2 from every selected RL image (the v3 report's unexplained set)",
        "images": len(far_images), "rows": len(far), "global_min_phash_distance": dict(Counter(global_min[x] for x in far_images)),
        "rows_detail": [
            {**{k: r[k] for k in ("record_id", "source", "identity", "v2_bucket", "v2_group_id", "image_sha256", "v3_decision", "v3_group_id", "duplicate_of", "min_phash_distance_to_rl_image", "chain")},
             "global_min_phash_distance_to_rl_image": global_min[r["image_sha256"]], "group_images": group_size[r["v3_group_id"]],
             "duplicate_winner": {k: dup_winners[r["record_id"]][k] for k in ("record_id", "image_sha256", "decision")} if r["record_id"] in dup_winners else None,
             "rl_rows_in_group": [rl_detail(x) for x in r["rl_rows"]]} for r in far]},
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False, default=str) + "\n")
flat = [{**{k: v for k, v in r.items() if k not in ("chain", "identity", "rl_rows", "chain_via", "rl_sources_in_group")},
         "chain": json.dumps(r["chain"], sort_keys=True), "chain_via": " > ".join(r["chain_via"] or []), "rl_sources_in_group": ",".join(r["rl_sources_in_group"]),
         "rl_record_ids_in_group": ",".join(x["record_id"] for x in r["rl_rows"])} for r in rows]
out_parquet.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pylist(flat), out_parquet, compression="zstd")
print(json.dumps({k: report[k] for k in ("removed_from_sft", "v3_decision", "v2_group_holds_exported_v3_rl", "not_attributed_by_v2_group")}, indent=1, default=str))
print(json.dumps({k: report[k] for k in ("duplicates",)}, indent=1))
print(json.dumps({k: v for k, v in report["nine_images_over_phash_2"].items() if k != "rows_detail"}, indent=1))
