"""Re-run RL classification/admission on exported RL rows (raw references) and compare with their export.

usage: python tools/step05/rl_recheck.py EXPORT_DIR OUT_JSON
"""

import collections
import glob
import json
import multiprocessing
import re
import sys
import time

import pyarrow.parquet as pq

OLD_SUFFIX = re.compile(r"\n\n(Solve the problem\.|Choose the correct option\.) The last line of your response.*$", re.S)
DELIM = re.compile(r"\\\(|\\\)|\\\[|\\\]|^\$|\$$")


def check(row):
    from k2hv.data import scoring

    problem = OLD_SUFFIX.sub("", row["problem"]).replace("<image>", "", 1)
    ref = row["metadata"]["reference"]
    t = time.time()
    info = scoring.classify(problem, ref)
    reason = info.get("reason") or (scoring.admit(problem, info) if info.get("selector") else None)
    return {"bucket": row["bucket"], "source": row["metadata"]["source"], "old_selector": row["metadata"]["data_source"], "old_target": row["answer"],
            "selector": None if reason else info["selector"], "target": None if reason else info["target"], "reason": reason, "pending": info.get("pending"),
            "old_delimited": bool(DELIM.search(row["answer"])), "new_delimited": bool(not reason and DELIM.search(info["target"] or "")), "seconds": time.time() - t,
            "reference": ref, "record_id": row["metadata"]["record_id"]}


def main():
    export, out = sys.argv[1], sys.argv[2]
    rows = []
    for f in sorted(glob.glob(f"{export}/portable/default/rl*/*/*.parquet")):
        bucket = "/".join(f.split("/")[-3:-1])
        for r in pq.read_table(f, columns=["problem", "answer", "metadata"]).to_pylist():
            rows.append({**r, "bucket": bucket})
    t0 = time.time()
    with multiprocessing.get_context("spawn").Pool(15) as pool:
        res = pool.map(check, rows, chunksize=200)
    c = collections.Counter
    summary = {
        "rows": len(res), "wall_seconds": round(time.time() - t0, 1), "max_row_seconds": round(max(r["seconds"] for r in res), 3),
        "old_delimited_targets": sum(r["old_delimited"] for r in res), "new_delimited_targets": sum(r["new_delimited"] for r in res),
        "old_selector": dict(c(r["old_selector"] for r in res)), "new_selector": dict(c(str(r["selector"]) for r in res)),
        "transitions": dict(c(f'{r["old_selector"]} -> {r["selector"] or "rejected:" + r["reason"]}' for r in res)),
        "new_pending_reasons": dict(c(r["pending"] for r in res if r["selector"] == "pending_decimal_precision")),
        "target_changed": sum(1 for r in res if r["selector"] and r["target"] != r["old_target"]),
        "by_source": {s: dict(c(str(r["selector"] or r["reason"]) for r in res if r["source"] == s)) for s in sorted({r["source"] for r in res})},
    }
    examples = collections.defaultdict(list)
    for r in res:
        key = r["reason"] or ("target_changed" if r["selector"] and r["target"] != r["old_target"] else None)
        if key and len(examples[key]) < 6:
            examples[key].append({k: r[k] for k in ("record_id", "reference", "old_target", "target", "old_selector", "selector")})
    summary["examples"] = examples
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
