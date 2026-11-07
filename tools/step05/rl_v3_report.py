"""Per-source accounting for the RL sources admitted in recipe v3, and the SFT difference from v2.

For each RL source: scan rejects by reason, evaluation-overlap exclusions, limit rejects, duplicates,
pending holds by reason, validation failures and exported rows by bucket, plus answer types,
selectors, ViRL39K categories and upstream sources, and the share of prompts in Chinese or Japanese.
For SFT: rows removed from v2 (with the reason when it is an RL reservation), rows added, and rows
whose exported content changed.

usage: python tools/step05/rl_v3_report.py DATA_ROOT OUT_JSON
"""

import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

root, out = Path(sys.argv[1]), Path(sys.argv[2])
NEW = ("rl_virl39k", "rl_vwi_verified", "rl_thinklite")
RL = NEW + ("rl_chartverse", "rl_geometry3k")
CJK = re.compile(r"[぀-ヿ一-鿿]")


def table(paths, columns=None):
    paths = [p for p in paths if Path(p).exists()]
    return pa.concat_tables([pq.read_table(p, columns=columns) for p in paths]) if paths else None


def prov(r):
    return json.loads(r["provenance"]) if r.get("provenance") else {}


def index_rows(build):
    rows = []
    for f in sorted(glob.glob(str(root / "exports" / build / "index" / "*.parquet"))):
        bucket = Path(f).stem
        if bucket.endswith("sweep50k"):
            continue
        for r in pq.read_table(f, columns=["record_id", "source", "split", "answer_type", "selector", "group_id", "image_sha256", "portable_hash", "provenance"]).to_pylist():
            rows.append({**r, "bucket": bucket})
    return rows


work = root / "work" / "full-v3"
sel_stats = json.loads((work / "select" / "select.json").read_text())
selected = pq.read_table(work / "select" / "selected.parquet", columns=["record_id", "source", "kind", "status", "provenance"]).to_pylist()
exclusions = pq.read_table(work / "select" / "exclusions.parquet").to_pylist()
validated = table(sorted(glob.glob(str(work / "validate" / "chunk-*.done.parquet"))), ["record_id", "ok", "reason"]).to_pylist()
failed = {r["record_id"]: r["reason"] for r in validated if not r["ok"]}
v3 = index_rows("full-v3")
v2 = index_rows("full-v2")

report = {"sources": {}}
for key in RL:
    scan = [json.loads(p.read_text()) for p in sorted((work / "scan" / key).glob("*.done.json"))]
    counts = Counter()
    rejected = Counter()
    for m in scan:
        counts["rows"] += m["counts"].get("rows", 0)
        counts["turns"] += m["counts"].get("turns", 0)
        counts["accepted"] += m["counts"].get("accepted", 0)
        rejected.update(m["counts"].get("rejected", {}))
    cand = table([p for p in glob.glob(str(work / "scan" / key / "*.parquet")) if not p.endswith(".rejects.parquet")], ["question", "answer_type", "selector", "provenance"]).to_pylist()
    sel = [r for r in selected if r["source"] == key]
    exported = [r for r in v3 if r["source"] == key]
    ups = Counter(prov(r).get("upstream_source") for r in exported)
    cats = Counter(prov(r).get("category") for r in exported)
    report["sources"][key] = {
        "scan": {**counts, "rejected": dict(rejected.most_common())},
        "candidates": len(cand),
        "candidate_selectors": dict(Counter(r["selector"] for r in cand)),
        "eval_exclusions": dict(Counter(f"{x['eval']}|{x['method']}|{x['status']}" for x in exclusions if x.get("source") == key)),
        "limit_rejects": {k.split("|", 1)[1]: v for k, v in sel_stats["limit_rejects"].items() if k.startswith(key + "|")},
        "duplicates": sel_stats["duplicates"].get(key, 0),
        "held_image_blank": sel_stats.get("held_image_blank", {}).get(key, 0),
        "selected": len(sel),
        "selected_pending": dict(Counter(prov(r).get("pending_reason") for r in sel if r["status"] == "pending")),
        "validation_failures": dict(Counter(failed[r["record_id"]] for r in sel if r["record_id"] in failed)),
        "exported": dict(Counter(r["bucket"] for r in exported)),
        "exported_selectors": dict(Counter(r["selector"] for r in exported if r["bucket"].startswith("default"))),
        "exported_answer_types": dict(Counter(r["answer_type"] for r in exported if r["bucket"].startswith("default"))),
        "exported_upstream_sources": dict(ups.most_common()) if any(ups) else None,
        "exported_categories": dict(cats.most_common()) if any(cats) else None,
    }

# Chinese/Japanese prompt share among exported rows, read from the portable rows.
for key in NEW:
    rows = []
    for f in glob.glob(str(root / "exports" / "full-v3" / "portable" / "*" / "rl" / "*" / "*.parquet")):
        t = pq.read_table(f, columns=["problem", "metadata"]).to_pylist()
        rows.extend(r for r in t if r["metadata"]["source"] == key)
    report["sources"][key]["exported_cjk_prompts"] = sum(1 for r in rows if CJK.search(r["problem"]))

# SFT difference from v2.
old = {r["record_id"]: r for r in v2 if "__sft__" in r["bucket"]}
new = {r["record_id"]: r for r in v3 if "__sft__" in r["bucket"]}
removed = [rid for rid in old if rid not in new]
groups_rl = {r["group_id"] for r in v3 if "__rl__" in r["bucket"]}
removed_reserved = [rid for rid in removed if old[rid]["group_id"] in groups_rl]
report["sft_vs_v2"] = {
    "v2_rows": len(old), "v3_rows": len(new),
    "removed": len(removed), "removed_group_now_rl": len(removed_reserved),
    "removed_by_source": dict(Counter(old[rid]["source"] for rid in removed).most_common()),
    "removed_group_now_rl_by_source": dict(Counter(old[rid]["source"] for rid in removed_reserved).most_common()),
    "added": sum(1 for rid in new if rid not in old),
    "added_by_source": dict(Counter(new[rid]["source"] for rid in new if rid not in old).most_common()),
    "same_record_changed_content": sum(1 for rid in new if rid in old and new[rid]["portable_hash"] != old[rid]["portable_hash"]),
    "same_record_moved_bucket": sum(1 for rid in new if rid in old and new[rid]["bucket"] != old[rid]["bucket"]),
}
report["rl_totals"] = {
    "v2": dict(Counter(r["bucket"] for r in v2 if "__rl__" in r["bucket"])),
    "v3": dict(Counter(r["bucket"] for r in v3 if "__rl__" in r["bucket"])),
    "v3_default_by_source": dict(Counter(r["source"] for r in v3 if r["bucket"].startswith("default__rl")).most_common()),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
print(json.dumps({k: {kk: v[kk] for kk in ("candidates", "selected", "exported", "held_image_blank")} for k, v in report["sources"].items()}, indent=1))
print(json.dumps(report["sft_vs_v2"], indent=1))
