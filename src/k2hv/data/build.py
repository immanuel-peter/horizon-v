"""Resumable build stages: scan sources into candidates and index evaluation sets."""

import hashlib
import json
import os
import re
import sys
import time
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import hub, images, runtime, scoring, sources, text

CANDIDATE_SCHEMA = pa.schema([
    ("record_id", pa.string()), ("kind", pa.string()), ("source", pa.string()), ("task", pa.string()), ("terms", pa.string()), ("status", pa.string()),
    ("dataset", pa.string()), ("revision", pa.string()), ("file", pa.string()), ("row", pa.int64()), ("turn", pa.int32()), ("upstream_id", pa.string()), ("orig_split", pa.string()),
    ("question", pa.string()), ("answer", pa.string()), ("answer_orig", pa.string()), ("answer_type", pa.string()), ("selector", pa.string()),
    ("image_sha256", pa.string()), ("image_path", pa.string()), ("image_format", pa.string()), ("image_mode", pa.string()), ("image_bytes", pa.int64()), ("width", pa.int32()), ("height", pa.int32()),
    ("phash", pa.int64()), ("dhash", pa.int64()), ("thumb", pa.binary()), ("pixel_sha256", pa.string()), ("image_tokens_est", pa.int32()),
    ("group_hint", pa.string()), ("external_ids", pa.list_(pa.string())), ("url", pa.string()), ("provenance", pa.string()),
])
REJECT_SCHEMA = pa.schema([("record_id", pa.string()), ("source", pa.string()), ("reason", pa.string()), ("question", pa.string()), ("answer", pa.string())])
EVAL_IMAGE_SCHEMA = pa.schema([("eval", pa.string()), ("item_id", pa.string()), ("item", pa.int64()), ("image_index", pa.int32()), ("image_sha256", pa.string()), ("pixel_sha256", pa.string()), ("phash", pa.int64()), ("thumb", pa.binary()), ("width", pa.int32()), ("height", pa.int32()), ("question", pa.string()), ("external_ids", pa.list_(pa.string()))])


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def write_table(path, table):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=4096)
    os.replace(tmp, path)


def write_json(path, obj):
    hub.atomic_write_text(path, json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Context:
    def __init__(self, recipe_path, root, profile):
        self.recipe_path = Path(recipe_path)
        self.recipe_bytes = self.recipe_path.read_bytes()
        self.recipe = tomllib.loads(self.recipe_bytes.decode())
        self.root = Path(root).resolve()
        self.profile_name = profile
        self.profile = self.recipe["profiles"][profile]
        self.pilot = profile == "pilot"
        self.datasets = self.recipe["datasets"]
        self.work = self.root / "work" / profile
        statuses = set(self.profile["statuses"])
        self.sources = {k: v for k, v in self.recipe["sources"].items() if v["status"] in statuses}

    @property
    def args(self):
        return (str(self.recipe_path), str(self.root), self.profile_name)

    def source_fingerprint(self, key):
        src = self.sources[key]
        ds = self.datasets[sources.ADAPTERS[src["adapter"]].dataset]
        return digest({"source": src, "dataset": ds, "rules": self.recipe["rules"], "profile": self.profile, "code": code_fingerprint()})


SCAN_VERSION = 2  # bump when scan_shard/sft_turn/rl_turn change their output


def code_fingerprint():
    here = Path(__file__).parent
    files = ("sources.py", "images.py", "text.py", "scoring.py")
    return digest({"scan_version": SCAN_VERSION, **{name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in files}})


# ------------------------------------------------------------------- scan


def shard_id(shard):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", shard["file"])
    if shard.get("row_groups") is not None:
        name += "__rg" + "-".join(map(str, shard["row_groups"]))
    return name


def plan_scan(ctx):
    jobs = []
    for key, src in sorted(ctx.sources.items()):
        adapter = sources.ADAPTERS[src["adapter"]]
        for shard in adapter.shards(ctx, src):
            jobs.append((ctx.args, key, shard))
    return jobs


def scan_paths(ctx, key, shard):
    base = ctx.work / "scan" / key / shard_id(shard)
    return base.with_suffix(".parquet"), base.with_suffix(".rejects.parquet"), base.with_suffix(".done.json")


def rank(seed, record_id, salt=""):
    """Deterministic order key used for every sample/quota decision."""
    return hashlib.sha256(f"{seed}:{salt}:{record_id}".encode()).hexdigest()


def has_choices(question):
    return bool(re.search(r"(?m)^\s*\(?[A-Ja-j][\).:]\s|\([A-Ja-j]\)\s*\S|\bchoices\b|\boptions\b", question, re.I))


def sft_turn(src, rules, question, answer):
    q, a = text.clean(question or ""), text.clean(answer or "")
    if src.get("question_exclude") and re.search(src["question_exclude"], q):
        return q, a, "text", "question_excluded_by_source_rule"
    letter = re.fullmatch(r"(?:answer\s*:\s*)?\(?([A-Ja-j])\)?\.?", a, re.I)
    if letter and has_choices(q):
        return q, letter.group(1), "choice", text.sft_reasons(q, letter.group(1), rules)
    if src["adapter"] == "finevision":
        a = text.strip_period(a)
    if text.is_number(a):
        if text.decimals(a) > src.get("max_answer_decimals", 99):
            return q, a, "number", "answer_unrounded_computation"
        kind = "number"
    else:
        kind = "yes_no" if text.key(a) in ("yes", "no") else "text"
    return q, a, kind, text.sft_reasons(q, a, rules)


def rl_turn(problem, reference, max_decimals=3):
    p = text.clean(problem or "")
    markers = p.count("<image>")
    if markers > 1:
        return p, None, {"reason": "multi_image_marker"}
    if markers == 0:
        p = "<image>\n" + p
    bare = p.replace("<image>", "", 1)
    if text.reserved(bare):
        return p, None, {"reason": "problem_reserved_marker"}
    if reference is None or not str(reference).strip():
        return p, None, {"reason": "answer_empty"}
    info = scoring.classify(bare, str(reference), max_decimals=max_decimals)
    if info["selector"] is None:
        return p, None, info
    failure = scoring.admit(bare, info)
    if failure:
        return p, None, {**info, "reason": failure}
    return scoring.rl_problem(p, info), info["target"], info


def scan_shard(args, key, shard):
    ctx = Context(*args)
    runtime.limit_threads(1)
    out, rej, done = scan_paths(ctx, key, shard)
    fingerprint = ctx.source_fingerprint(key)
    if done.exists() and json.loads(done.read_text()).get("fingerprint") == fingerprint:
        return json.loads(done.read_text())
    src = ctx.sources[key]
    adapter = sources.ADAPTERS[src["adapter"]]
    ds = ctx.datasets[adapter.dataset]
    rules = {**ctx.recipe["rules"], **{k: src[k] for k in ("max_answer_words", "max_answer_sentences") if k in src}}
    zips = sources.ZipMedia()
    image_config = {"patch_size": 14, "merge_size": 2, "temporal_patch_size": 2, "min_image_tokens": 16, "max_image_tokens": ctx.recipe["limits"]["max_image_tokens"]}
    accepted, rejected, counts = [], [], {"rows": 0, "turns": 0, "accepted": 0, "rejected": {}}
    pending_urls = []

    def reject(rid, reason, q="", a=""):
        counts["rejected"][reason] = counts["rejected"].get(reason, 0) + 1
        rejected.append({"record_id": rid, "source": key, "reason": reason, "question": (q or "")[:300], "answer": (a or "")[:300]})

    def finish(row, ok, data_or_error):
        if isinstance(data_or_error, Exception):
            reason = str(data_or_error) if isinstance(data_or_error, (images.ImageError, hub.UrlError)) else f"image_{type(data_or_error).__name__}"
            for rid, q, a, *_ in ok:
                reject(rid, reason, q, a)
            return
        try:
            img = images.prepare(data_or_error, ctx.root)
        except images.ImageError as e:
            for rid, q, a, *_ in ok:
                reject(rid, str(e), q, a)
            return
        tokens, _ = images.image_tokens(img["width"], img["height"], image_config)
        prov_row = row.get("provenance") or {}
        for rid, q, a, a_orig, a_type, selector, turn, prov in ok:
            provenance = {**prov_row, **prov}
            if img["image_transform"]:
                provenance["image_transform"] = img["image_transform"]
            if row.get("category"):
                provenance["category"] = row["category"]
            accepted.append({
                "record_id": rid, "kind": src["kind"], "source": key, "task": src["task"], "terms": src["terms"], "status": src["status"],
                "dataset": ds["repo"], "revision": ds["revision"], "file": shard["file"], "row": row["row"], "turn": turn, "upstream_id": row.get("upstream_id"), "orig_split": row.get("orig_split"),
                "question": q, "answer": a, "answer_orig": a_orig, "answer_type": a_type, "selector": selector,
                "image_sha256": img["image_sha256"], "image_path": img["image_path"], "image_format": img["image_format"], "image_mode": img["image_mode"], "image_bytes": img["image_bytes"], "width": img["width"], "height": img["height"],
                "phash": img["phash"], "dhash": img["dhash"], "thumb": img["thumb"], "pixel_sha256": img["pixel_sha256"], "image_tokens_est": tokens,
                "group_hint": row.get("group_hint"), "external_ids": row.get("external_ids") or [], "url": row.get("url"), "provenance": canonical(provenance),
            })
            counts["accepted"] += 1

    def flush_urls():
        from concurrent.futures import ThreadPoolExecutor

        def fetch(item):
            row, ok = item
            _, url, expected = row["image"]
            try:
                return row, ok, hub.fetch_url(ctx.root, url, expected)
            except hub.UrlError as e:
                return row, ok, e

        with ThreadPoolExecutor(ctx.recipe["workers"]["url_threads"]) as pool:
            for row, ok, data in pool.map(fetch, pending_urls):
                finish(row, ok, data)
        pending_urls.clear()

    for row in adapter.rows(ctx, src, shard):
        counts["rows"] += 1
        ok = []
        for t in row["turns"]:
            counts["turns"] += 1
            rid = f"{key}:{shard['file']}:{row['row']}:{t['turn']}"
            if src["kind"] == "sft":
                q, a, a_type, reason = sft_turn(src, rules, t["question"], t["answer"])
                a_orig, selector = text.clean(t["answer"] or ""), None
            else:
                q, a, info = rl_turn(t["problem"], t["reference"])
                reason, a_type, selector, a_orig = info.get("reason"), info.get("answer_type"), info.get("selector"), str(t["reference"])
                if info.get("mapped_from"):
                    t["provenance"]["choice_mapped_from"] = info["mapped_from"]
            reason = row.get("reject") or reason
            if reason:
                reject(rid, reason, q, a or a_orig)
            else:
                ok.append((rid, q, a, a_orig, a_type, selector, t["turn"], t.get("provenance") or {}))
        if not ok:
            continue
        keep = src.get("max_qa_per_image", 0) * 3
        if keep and len(ok) > keep:
            ok = sorted(ok, key=lambda item: rank(ctx.recipe["recipe"]["seed"], item[0]))[:keep]
            counts["capped_turns"] = counts.get("capped_turns", 0) + (len(row["turns"]) - keep)
        spec = row["image"]
        if isinstance(spec, tuple) and spec[0] == "url":
            pending_urls.append((row, ok))
            if len(pending_urls) >= 512:
                flush_urls()
            continue
        if isinstance(spec, tuple) and spec[0] == "zip":
            try:
                spec = zips.read(spec[1], spec[2])
            except hub.UrlError as e:
                finish(row, ok, e)
                continue
        finish(row, ok, spec if spec else images.ImageError("image_missing"))
    flush_urls()
    write_table(out, pa.Table.from_pylist(accepted, schema=CANDIDATE_SCHEMA))
    write_table(rej, pa.Table.from_pylist(rejected, schema=REJECT_SCHEMA))
    marker = {"source": key, "shard": shard, "fingerprint": fingerprint, "counts": counts, "candidates_sha256": runtime.file_sha256(out), "rejects_sha256": runtime.file_sha256(rej)}
    write_json(done, marker)
    return marker


def scan(ctx):
    jobs = plan_scan(ctx)
    log(f"scan: {len(jobs)} shards over {len(ctx.sources)} sources")
    results = []
    for marker in _imap(scan_shard, jobs, ctx.recipe["workers"]["scan"]):
        results.append(marker)
        c = marker["counts"]
        log(f"  {marker['source']} {marker['shard']['file']}: rows={c['rows']} turns={c['turns']} accepted={c['accepted']} ({len(results)}/{len(jobs)})")
    return results


def _imap(fn, jobs, workers):
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(_guard, fn, job) for job in jobs]
        for future in as_completed(futures):
            ok, value = future.result()
            if not ok:
                print(value, file=sys.stderr, flush=True)
                raise RuntimeError("worker failed; see traceback above")
            yield value


def _guard(fn, job):
    try:
        return True, fn(*job)
    except Exception:
        return False, f"job {job[1:]} failed:\n{traceback.format_exc()}"


def preserved_splits(ctx):
    """Official validation/test files of candidate sources that are never read for training."""
    out = {}
    for key, src in ctx.recipe["sources"].items():
        adapter = sources.ADAPTERS[src["adapter"]]
        if hasattr(adapter, "preserved"):
            out[key] = adapter.preserved(ctx, src)
    return out


# ------------------------------------------------------------------- evaluation index


def eval_index_one(args, name):
    ctx = Context(*args)
    runtime.limit_threads(1)
    spec = ctx.recipe["evals"][name]
    out = ctx.root / "work" / "evals" / f"{name}.parquet"
    done = out.with_suffix(".done.json")
    fingerprint = digest({"spec": spec, "code": code_fingerprint(), "eval_version": 2})
    if done.exists() and json.loads(done.read_text()).get("fingerprint") == fingerprint:
        return json.loads(done.read_text())
    rows, items, unavailable = [], 0, []
    for file, item in sources.eval_rows(ctx, name, spec):
        if not item["images"]:
            unavailable.append({"item": items, "id": item["id"], "reason": "no_image"})
        for k, data in enumerate(item["images"]):
            try:
                image, _ = images.decode(data)
                f = images.features(image)
            except images.ImageError as e:
                unavailable.append({"item": items, "id": item["id"], "image_index": k, "reason": str(e)})
                continue
            eval_sha, _ = images.write_asset(ctx.root / "evals", data, images.FORMATS[image.format] if image.format in images.FORMATS else "img")
            rows.append({"eval": name, "item_id": item["id"], "item": items, "image_index": k, "image_sha256": eval_sha, "pixel_sha256": f["pixel_sha256"], "phash": f["phash"], "thumb": f["thumb"], "width": image.size[0], "height": image.size[1], "question": item["question"], "external_ids": item["external_ids"]})
        items += 1
    write_table(out, pa.Table.from_pylist(rows, schema=EVAL_IMAGE_SCHEMA))
    marker = {"eval": name, "repo": spec["repo"], "revision": spec["revision"], "items": items, "images": len(rows), "unique_images": len({r["image_sha256"] for r in rows}), "unavailable": unavailable, "fingerprint": fingerprint, "sha256": runtime.file_sha256(out)}
    write_json(done, marker)
    return marker


def eval_index(ctx):
    jobs = [(ctx.args, name) for name in sorted(ctx.recipe["evals"])]
    results = list(_imap(eval_index_one, jobs, len(jobs)))
    for m in sorted(results, key=lambda m: m["eval"]):
        log(f"  eval {m['eval']}: items={m['items']} images={m['images']} unique={m['unique_images']} unavailable={len(m['unavailable'])}")
    return results
