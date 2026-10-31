"""Selection, validation, export, materialization and verification stages."""

import hashlib
import html
import json
import math
import os
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import build, contract, hub, images, overlap, runtime, scoring, sources, text
from .build import canonical, digest, log, rank, write_json, write_table

SFT_QUOTA_KEY = "sft_quota"
CHUNK = 2000


def _read_dir(paths, columns=None):
    tables = [pq.read_table(p, columns=columns) for p in paths]
    tables = [t for t in tables if t.num_rows]
    return pa.concat_tables(tables) if tables else None


def candidate_files(ctx):
    return sorted(p for p in (ctx.work / "scan").glob("*/*.parquet") if not p.name.endswith(".rejects.parquet") and p.parent.name in ctx.sources)


def frac(seed, key, salt):
    return int(rank(seed, key, salt)[:15], 16) / 16**15


def assign_split(seed, group, kind, split_cfg):
    """Split from the image/document group alone, so every record sharing a group lands together."""
    if kind == "sft":
        return "validation" if frac(seed, group, "sft-split") < split_cfg["sft_validation_fraction"] else "train"
    return "heldout" if frac(seed, group, "rl-split") < split_cfg["rl_heldout_fraction"] else "pool"


def dedup_key(kind, group, question, answer):
    """Same image group and the same normalized question and answer; a shared generic question alone is not a duplicate."""
    return (kind, group, overlap.core_question(question), text.key(answer))


def quota(ctx, src):
    if ctx.pilot:
        return ctx.profile["sft_quota"] if src["kind"] == "sft" else ctx.profile["rl_quota"]
    return src["quota"]


# ------------------------------------------------------------------- inventory


def inventory(ctx):
    """Pin listings and record schema, row counts and sizes for every recipe dataset and evaluation set."""
    out = {"datasets": {}, "evals": {}, "preserved_splits": {}}
    for name, ds in sorted(ctx.datasets.items()):
        files = hub.listing(ctx.root, ds["repo"], ds["revision"])["files"]
        data_files = sorted(f for f in files if f.endswith((".parquet", ".jsonl", ".zip", ".tar")))
        entry = {"repo": ds["repo"], "revision": ds["revision"], "files": len(data_files), "bytes": sum(files[f]["size"] or 0 for f in data_files)}
        first = next((f for f in data_files if f.endswith(".parquet")), None)
        if first:
            pf = pq.ParquetFile(hub.remote(ds["repo"], ds["revision"], first))
            entry["schema_file"] = first
            entry["schema"] = {f.name: str(f.type) for f in pf.schema_arrow}
        out["datasets"][name] = entry
    for key, src in ctx.recipe["sources"].items():
        if src["adapter"] == "finevision":
            ds = ctx.datasets["finevision"]
            files = sources.parquet_files(ctx, ds, src["config"] + "/")
            listing = hub.listing(ctx.root, ds["repo"], ds["revision"])["files"]
            out["datasets"][f"finevision/{src['config']}"] = {"files": len(files), "bytes": sum(listing[f]["size"] for f in files), "rows": sum(pq.ParquetFile(hub.remote(ds["repo"], ds["revision"], f)).metadata.num_rows for f in files)}
    for name, spec in sorted(ctx.recipe["evals"].items()):
        files = hub.listing(ctx.root, spec["repo"], spec["revision"])["files"]
        out["evals"][name] = {"repo": spec["repo"], "revision": spec["revision"], "bytes": sum((files[f]["size"] or 0) for f in files if f.endswith(".parquet"))}
    out["preserved_splits"] = build.preserved_splits(ctx)
    write_json(ctx.root / "inventory" / "inventory.json", out)
    log(f"inventory: {len(out['datasets'])} dataset entries, {len(out['evals'])} evaluation sets")
    return out


# ------------------------------------------------------------------- select


def _tokenizer(ctx):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(ctx.recipe["recipe"]["processor"], trust_remote_code=True)


def _overheads(ctx):
    """Template tokens outside question/answer/image pads, measured once with the real processor."""
    from PIL import Image

    processor = runtime.load_processor(ctx.recipe["recipe"]["processor"])
    tok = processor.tokenizer
    image = Image.new("RGB", (56, 56))
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "\nq"}]}, {"role": "assistant", "content": [{"type": "text", "text": "a"}], "think": ""}]
    ids = processor(text=processor.apply_chat_template(msgs, tokenize=False), images=[image])["input_ids"][0]
    pads = int((ids == processor.image_token_id).sum())
    sft = len(ids) - pads - len(tok.encode("\nq", add_special_tokens=False)) - len(tok.encode("a", add_special_tokens=False))
    gen = processor(text=processor.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True), images=[image])["input_ids"][0]
    rl = len(gen) - pads - len(tok.encode("\nq", add_special_tokens=False))
    return sft, rl


def select(ctx):
    seed = ctx.recipe["recipe"]["seed"]
    cfg = ctx.recipe["overlap"]
    limits = ctx.recipe["limits"]
    cols = ["record_id", "kind", "source", "status", "question", "answer", "answer_orig", "answer_type", "selector", "image_sha256", "pixel_sha256", "phash", "thumb", "group_hint", "external_ids", "image_tokens_est"]
    table = _read_dir(candidate_files(ctx), cols)
    n = table.num_rows
    c = {name: table.column(name).to_pylist() for name in cols if name not in ("phash", "thumb")}
    phash_all = table.column("phash").to_numpy()
    thumb_all = table.column("thumb").to_pylist()
    log(f"select: {n} candidates")

    # unique images
    img_of = {}
    img_rows = []
    for i, sha in enumerate(c["image_sha256"]):
        if sha not in img_of:
            img_of[sha] = len(img_rows)
            img_rows.append(i)
    rec_img = np.array([img_of[s] for s in c["image_sha256"]])
    u = {"sha": [c["image_sha256"][i] for i in img_rows], "pixel_sha256": [c["pixel_sha256"][i] for i in img_rows], "phash": phash_all[img_rows], "thumb": [thumb_all[i] for i in img_rows]}
    log(f"  unique images {len(img_rows)}")

    # evaluation index
    ev_files = sorted((ctx.root / "work" / "evals").glob("*.parquet"))
    if len(ev_files) != len(ctx.recipe["evals"]):
        raise RuntimeError("Run the evals stage first; evaluation index is incomplete")
    ev = pa.concat_tables([pq.read_table(p) for p in ev_files])
    e = {k: ev.column(k).to_pylist() for k in ("eval", "item_id", "item", "image_sha256", "pixel_sha256", "thumb", "question", "external_ids")}
    e["phash"] = ev.column("phash").to_numpy()
    matches = image_matches(ctx, u, e, cfg)
    log(f"  eval image matches {len(matches)} ({sum(m['status'] == 'confirmed' for m in matches.values())} confirmed)")
    eval_ids = defaultdict(set)
    for k, ids in enumerate(e["external_ids"]):
        for x in ids:
            eval_ids[x].add(k)
    eval_q = defaultdict(set)
    for k, q in enumerate(e["question"]):
        cq = overlap.core_question(q or "")
        if len(cq) >= cfg["eval_question_min_chars"]:
            eval_q[cq].add((e["eval"][k], e["item"][k]))
    rec_q = [overlap.core_question(q or "") for q in c["question"]]
    eval_item_images = defaultdict(list)
    for k in range(len(e["eval"])):
        eval_item_images[(e["eval"][k], e["item"][k])].append(k)
    text_matches = []

    exclusions = []
    excluded = np.zeros(n, dtype=bool)
    for i in range(n):
        m = matches.get(rec_img[i])
        hit = None
        if m:
            j = m["eval_index"]
            hit = {"eval": e["eval"][j], "eval_item": e["item_id"][j], "eval_image_sha256": e["image_sha256"][j], **{k: v for k, v in m.items() if k != "eval_index"}}
        else:
            for x in c["external_ids"][i] or ():
                if x in eval_ids:
                    j = min(eval_ids[x])
                    hit = {"method": "external_id", "status": "confirmed", "eval": e["eval"][j], "eval_item": e["item_id"][j], "eval_image_sha256": e["image_sha256"][j], "external_id": x}
                    break
        if hit is None and rec_q[i] in eval_q:
            ev_name, item = min(eval_q[rec_q[i]])
            sims = [images.thumb_similarity(thumb_all[i], e["thumb"][k]) for k in eval_item_images[(ev_name, item)]]
            specific = len(rec_q[i]) >= cfg["eval_question_specific_chars"] or max(sims, default=0) >= cfg["eval_question_image_similarity"]
            text_hit = {"record_id": c["record_id"][i], "source": c["source"][i], "kind": c["kind"][i], "eval": ev_name, "eval_item": str(item), "question": c["question"][i][:300], "thumb_similarity": round(max(sims, default=0), 4), "excluded": specific}
            text_matches.append(text_hit)
            if specific:
                hit = {"method": "question_text", "status": "uncertain", "eval": ev_name, "eval_item": e["item_id"][eval_item_images[(ev_name, item)][0]], "eval_image_sha256": e["image_sha256"][eval_item_images[(ev_name, item)][0]], "thumb_similarity": text_hit["thumb_similarity"]}
        if hit:
            excluded[i] = True
            exclusions.append({"record_id": c["record_id"][i], "source": c["source"][i], "kind": c["kind"][i], "image_sha256": c["image_sha256"][i], "question": c["question"][i][:300], **hit})

    # grouping: identical pixels, near-duplicates, shared document/rendering hints
    uf = overlap.UnionFind(len(img_rows))
    by_pixel = {}
    for k, p in enumerate(u["pixel_sha256"]):
        if p in by_pixel:
            uf.union(by_pixel[p], k)
        else:
            by_pixel[p] = k
    candidates, oversized = overlap.near_duplicate_pairs(u["phash"], u["thumb"], cfg)
    pair_metrics = verify_pairs(ctx, [(ctx.root, u["sha"][a], ctx.root, u["sha"][b]) for a, b in candidates])
    pairs = [(a, b) for (a, b), m in zip(candidates, pair_metrics) if m and overlap.classify_pair(m, cfg) == "confirmed"]
    for a, b in pairs:
        uf.union(a, b)
    by_hint = {}
    for i, h in enumerate(c["group_hint"]):
        if h:
            if h in by_hint:
                uf.union(by_hint[h], rec_img[i])
            else:
                by_hint[h] = rec_img[i]
    members = defaultdict(list)
    for k in range(len(img_rows)):
        members[uf.find(k)].append(u["sha"][k])
    group_of_root = {r: min(v) for r, v in members.items()}
    group = [group_of_root[uf.find(rec_img[i])] for i in range(n)]
    cross_source = Counter()
    sources_in_group = defaultdict(set)
    for i in range(n):
        sources_in_group[group[i]].add(c["source"][i])
    for g, s in sources_in_group.items():
        if len(s) > 1:
            cross_source[" + ".join(sorted(s))] += 1
    log(f"  groups {len(members)} from {len(img_rows)} images; near-dup pairs {len(pairs)}; oversized buckets {oversized}")

    # token estimates
    tok = _tokenizer(ctx)
    sft_overhead, rl_overhead = _overheads(ctx)
    q_text = [("\n" + q.replace("<image>", "", 1)) if k == "rl" else ("\n" + q) for q, k in zip(c["question"], c["kind"])]
    q_tokens = [len(x) for x in tok(q_text, add_special_tokens=False)["input_ids"]]
    a_tokens = [len(x) for x in tok(c["answer"], add_special_tokens=False)["input_ids"]]
    est_total, est_super, limit_reason = [], [], []
    for i in range(n):
        img_t = c["image_tokens_est"][i] + 2
        if c["kind"][i] == "sft":
            total, sup = sft_overhead + q_tokens[i] + a_tokens[i] + img_t, a_tokens[i] + 1
            reason = "limit_supervised_tokens_est" if sup > limits["max_supervised_tokens"] else "limit_total_tokens_est" if total > limits["max_total_tokens"] else None
        else:
            total, sup = rl_overhead + q_tokens[i] + img_t, 0
            reason = "limit_prompt_tokens_est" if total > limits["rl_max_prompt_tokens"] else None
        if c["image_tokens_est"][i] > limits["max_image_tokens"]:
            reason = "limit_image_tokens_est"
        est_total.append(total)
        est_super.append(sup)
        limit_reason.append(reason)

    # deduplication: image group + normalized question + normalized answer; keep the higher-priority source
    priority = {k: p for p, k in enumerate(ctx.recipe["sources"])}
    ranks = [rank(seed, rid) for rid in c["record_id"]]
    winner = {}
    for i in range(n):
        if excluded[i] or limit_reason[i]:
            continue
        key = dedup_key(c["kind"][i], group[i], c["question"][i], c["answer"][i])
        prev = winner.get(key)
        if prev is None or (priority[c["source"][i]], ranks[i]) < (priority[c["source"][prev]], ranks[prev]):
            winner[key] = i
    is_winner = np.zeros(n, dtype=bool)
    is_winner[list(winner.values())] = True
    duplicates = Counter(c["source"][i] for i in range(n) if not excluded[i] and not limit_reason[i] and not is_winner[i])

    # RL selection first; its image groups are reserved away from SFT
    by_source = defaultdict(list)
    for i in range(n):
        if is_winner[i]:
            by_source[c["source"][i]].append(i)
    oversample = 1.0 if ctx.pilot else ctx.recipe["split"]["oversample"]
    selected = []
    for key in sorted(by_source, key=lambda k: priority[k]):
        if ctx.sources[key]["kind"] != "rl":
            continue
        pool = sorted(by_source[key], key=lambda i: ranks[i])
        selected.extend(pool[: math.ceil(quota(ctx, ctx.sources[key]) * (oversample if not ctx.pilot else 1.5))])
    rl_groups = {group[i] for i in selected}
    reserved = Counter()
    for key in sorted(by_source, key=lambda k: priority[k]):
        src = ctx.sources[key]
        if src["kind"] != "sft":
            continue
        per_group = Counter()
        take = []
        want = math.ceil(quota(ctx, src) * (oversample if not ctx.pilot else 1.5))
        for i in sorted(by_source[key], key=lambda i: ranks[i]):
            if group[i] in rl_groups:
                reserved[key] += 1
                continue
            if per_group[group[i]] >= src.get("max_qa_per_image", 1 << 30):
                continue
            per_group[group[i]] += 1
            take.append(i)
            if len(take) >= want:
                break
        selected.extend(take)
    selected.sort(key=lambda i: (c["kind"][i], priority[c["source"][i]], ranks[i]))

    split_cfg = ctx.recipe["split"]
    keep = [name for name in build.CANDIDATE_SCHEMA.names if name != "thumb"]
    full = _read_dir(candidate_files(ctx), keep)
    index = {rid: k for k, rid in enumerate(full.column("record_id").to_pylist())}
    picked = full.take(pa.array([index[c["record_id"][i]] for i in selected], type=pa.int64()))
    split = [assign_split(seed, group[i], c["kind"][i], split_cfg) for i in selected]
    picked = picked.append_column("group_id", pa.array([group[i] for i in selected])).append_column("split", pa.array(split)).append_column("rank", pa.array([ranks[i] for i in selected]))
    picked = picked.append_column("est_total_tokens", pa.array([est_total[i] for i in selected], pa.int32())).append_column("est_supervised_tokens", pa.array([est_super[i] for i in selected], pa.int32()))
    out = ctx.work / "select"
    write_table(out / "selected.parquet", picked)
    write_table(out / "exclusions.parquet", pa.Table.from_pylist(exclusions) if exclusions else pa.table({"record_id": pa.array([], pa.string())}))
    write_table(out / "question_text_matches.parquet", pa.Table.from_pylist(text_matches) if text_matches else pa.table({"record_id": pa.array([], pa.string())}))

    rl_sft_overlap = Counter()
    sft_groups = Counter(group[i] for i in range(n) if c["kind"][i] == "sft" and is_winner[i])
    for g in rl_groups:
        if g in sft_groups:
            rl_sft_overlap["groups"] += 1
            rl_sft_overlap["sft_records"] += sft_groups[g]
    stats = {
        "candidates": n, "unique_images": len(img_rows), "groups": len(members), "near_duplicate_pairs": len(pairs), "oversized_buckets": oversized,
        "cross_source_groups": dict(cross_source.most_common()),
        "excluded": dict(Counter(f"{x['source']}|{x['eval']}|{x['method']}|{x['status']}" for x in exclusions)),
        "limit_rejects": dict(Counter(f"{c['source'][i]}|{limit_reason[i]}" for i in range(n) if limit_reason[i])),
        "duplicates": dict(duplicates), "reserved_for_rl": dict(reserved), "rl_sft_overlap_before_reservation": dict(rl_sft_overlap),
        "selected": dict(Counter(c["source"][i] for i in selected)), "split": dict(Counter(f"{c['kind'][i]}|{s}" for i, s in zip(selected, split))),
        "overheads": {"sft": sft_overhead, "rl": rl_overhead},
        "eval_question_keys": len(eval_q),
        "question_text_matches": {"listed": len(text_matches), "excluded": sum(t["excluded"] for t in text_matches), "by_source_eval": dict(Counter(f"{t['source']}|{t['eval']}|{'excluded' if t['excluded'] else 'listed'}" for t in text_matches))},
    }
    write_json(out / "select.json", stats)
    log(f"  selected {len(selected)}; excluded {int(excluded.sum())}; duplicates {sum(duplicates.values())}; reserved for RL {sum(reserved.values())}")
    return stats


def _pair_metrics(jobs):
    from PIL import Image

    out = []
    for root_a, sha_a, root_b, sha_b in jobs:
        try:
            out.append(overlap.aligned_metrics(Image.open(images.find_asset(root_a, sha_a)), Image.open(images.find_asset(root_b, sha_b))))
        except Exception:
            out.append(None)
    return out


def verify_pairs(ctx, jobs, chunk=500):
    """Stage-2 metrics for (root, sha, root, sha) pairs, computed in parallel and returned in order."""
    if not jobs:
        return []
    parts = [jobs[k : k + chunk] for k in range(0, len(jobs), chunk)]
    results = {}
    with build.ProcessPoolExecutor(max_workers=min(ctx.recipe["workers"]["validate"], len(parts)), mp_context=build.get_context("spawn")) as pool:
        for k, res in enumerate(pool.map(_pair_metrics, parts)):
            results[k] = res
    return [m for k in range(len(parts)) for m in results[k]]


def image_matches(ctx, u, e, cfg):
    """Best evaluation match per unique candidate image: exact pixels, or stage-2 verified pHash candidates."""
    best = {}
    eval_pixels = defaultdict(list)
    for j, p in enumerate(e["pixel_sha256"]):
        eval_pixels[p].append(j)
    for i, p in enumerate(u["pixel_sha256"]):
        if p in eval_pixels:
            best[i] = {"method": "pixel_sha256", "status": "confirmed", "eval_index": eval_pixels[p][0], "distance": 0}
    pending = [(i, j, d, sim) for i, lst in overlap.eval_candidates(u, e, cfg).items() if i not in best for j, d, sim in lst]
    metrics = verify_pairs(ctx, [(ctx.root, u["sha"][i], ctx.root / "evals", e["image_sha256"][j]) for i, j, _, _ in pending])
    for (i, j, d, sim), m in zip(pending, metrics):
        status = overlap.classify_pair(m, cfg) if m else None
        if status is None:
            continue
        prev = best.get(i)
        if prev is None or (status == "confirmed", m["edge"]) > (prev["status"] == "confirmed", prev.get("edge", 0)):
            best[i] = {"method": "phash_aligned", "status": status, "eval_index": j, "distance": d, "thumb_similarity": sim, **m}
    log(f"  stage-2 eval pairs {len(pending)}")
    return best


# ------------------------------------------------------------------- validate


def miles_row(r, root):
    image = str(Path(root) / r["image_path"]) if root is not None else r["image_path"]
    if r["kind"] == "sft":
        return {"messages": [{"role": "user", "content": "<image>\n" + r["question"], "think": None}, {"role": "assistant", "content": r["answer"], "think": ""}], "images": [image], "metadata": {"record_id": r["record_id"], "source": r["source"], "task": r["task"], "image_sha256": r["image_sha256"]}}
    return {"problem": r["question"], "answer": r["answer"], "images": [image], "metadata": {"data_source": r["selector"], "record_id": r["record_id"], "source": r["source"], "dataset": r["dataset"], "upstream_id": r["upstream_id"], "reference": r["answer_orig"], "answer_type": r["answer_type"], "task": r["task"], "image_sha256": r["image_sha256"]}}


def portable_hash(row):
    """Row hash with image paths reduced to their portable images/<..> form."""
    row = json.loads(json.dumps(row))
    row["images"] = [p[p.rindex("images/") :] for p in row["images"]]
    return contract.row_hash(row)


def validate_chunk(args, chunk_path, done_path):
    ctx = build.Context(*args)
    checker = contract.Checker(ctx.recipe["recipe"]["processor"], ctx.recipe["limits"])
    rows = list(checker.miles.read_file(str(chunk_path)))
    results = []
    for data in rows:
        meta = data["metadata"]
        rid = meta["record_id"]
        try:
            res = checker.sft(data, meta["image_sha256"]) if "messages" in data else checker.rl(data, meta["image_sha256"])
            results.append({"record_id": rid, "ok": True, "reason": None, "portable_hash": portable_hash(data), **{k: res.get(k) for k in ("total_tokens", "image_tokens", "supervised_tokens", "prompt_tokens", "bos_tokens")}})
        except contract.ContractError as err:
            results.append({"record_id": rid, "ok": False, "reason": str(err), "portable_hash": portable_hash(data)})
    schema = pa.schema([("record_id", pa.string()), ("ok", pa.bool_()), ("reason", pa.string()), ("portable_hash", pa.string()), ("total_tokens", pa.int32()), ("image_tokens", pa.int32()), ("supervised_tokens", pa.int32()), ("prompt_tokens", pa.int32()), ("bos_tokens", pa.int32())])
    out = Path(done_path).with_suffix(".parquet")
    write_table(out, pa.Table.from_pylist(results, schema=schema))
    marker = {"chunk": str(chunk_path), "chunk_sha256": runtime.file_sha256(chunk_path), "rows": len(results), "failures": sum(not r["ok"] for r in results), "fingerprint": _validate_fingerprint(ctx)}
    write_json(done_path, marker)
    return marker


def _validate_fingerprint(ctx):
    here = Path(__file__).parent
    code = {name: runtime.file_sha256(here / name) for name in ("contract.py", "images.py", "runtime.py")}
    code["sft_rollout.py"] = runtime.file_sha256(here.parent / "sft_rollout.py")
    return digest({"limits": ctx.recipe["limits"], "processor": runtime.processor_fingerprint(ctx.recipe["recipe"]["processor"]), "code": code, "miles": runtime.MILES_COMMIT})


def validate(ctx):
    selected = pq.read_table(ctx.work / "select" / "selected.parquet").to_pylist()
    vdir = ctx.work / "validate"
    jobs = []
    fingerprint = _validate_fingerprint(ctx)
    chunks = []
    for kind in ("rl", "sft"):
        rows = [r for r in selected if r["kind"] == kind]
        chunks.extend((kind, k // CHUNK, rows[k : k + CHUNK]) for k in range(0, len(rows), CHUNK))
    for kind, number, chunk in chunks:
        chunk_path = vdir / f"chunk-{kind}-{number:05d}.input.parquet"
        done_path = vdir / f"chunk-{kind}-{number:05d}.done.json"
        rows = [miles_row(r, ctx.root) for r in chunk]
        table = pa.Table.from_pylist(rows)
        content = digest([portable_hash(r) for r in rows])
        if done_path.exists():
            marker = json.loads(done_path.read_text())
            if marker.get("fingerprint") == fingerprint and marker.get("content") == content and chunk_path.exists():
                continue
        write_table(chunk_path, table)
        jobs.append((ctx.args, str(chunk_path), str(done_path), content))
    log(f"validate: {len(selected)} rows in {len(chunks)} chunks, {len(jobs)} to run")
    for marker in build._imap(_validate_job, jobs, ctx.recipe["workers"]["validate"]):
        log(f"  {Path(marker['chunk']).name}: rows={marker['rows']} failures={marker['failures']}")
    results = _read_dir(sorted(vdir.glob("chunk-*.done.parquet")))
    failures = Counter(r for r in results.column("reason").to_pylist() if r)
    log(f"  validated {results.num_rows}; failures {dict(failures)}")
    return results


def _validate_job(args, chunk_path, done_path, content):
    marker = validate_chunk(args, chunk_path, done_path)
    marker["content"] = content
    write_json(done_path, marker)
    return marker


# ------------------------------------------------------------------- export


def _percentiles(values):
    if not values:
        return {}
    a = np.asarray(values)
    return {"n": int(a.size), "min": int(a.min()), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)), "max": int(a.max()), "mean": round(float(a.mean()), 2)}


def export(ctx):
    seed = ctx.recipe["recipe"]["seed"]
    sel = pq.read_table(ctx.work / "select" / "selected.parquet").to_pylist()
    val = {r["record_id"]: r for r in _read_dir(sorted((ctx.work / "validate").glob("chunk-*.done.parquet"))).to_pylist()}
    missing = [r["record_id"] for r in sel if r["record_id"] not in val]
    if missing:
        raise RuntimeError(f"{len(missing)} selected rows were not validated; run validate")
    ok = [r for r in sel if val[r["record_id"]]["ok"]]
    by_source = defaultdict(list)
    for r in ok:
        by_source[r["source"]].append(r)
    final = []
    for key, rows in by_source.items():
        rows.sort(key=lambda r: r["rank"])
        final.extend(rows[: quota(ctx, ctx.sources[key])])
    final.sort(key=lambda r: (r["kind"], r["rank"]))
    for r in final:
        r.update({k: val[r["record_id"]][k] for k in ("total_tokens", "image_tokens", "supervised_tokens", "prompt_tokens", "bos_tokens", "portable_hash")})

    def bucket(r):
        variant = {"include": "default", "addon": "addon", "pending": "pending", "exclude": "excluded"}[r["status"]]
        if r["kind"] == "sft":
            return f"{variant}/sft/{r['split']}"
        pool = "rl_pending_numeric" if r["selector"] == scoring.PENDING_NUMERIC else "rl"
        return f"{variant}/{pool}/{r['split']}"

    groups = defaultdict(list)
    for r in final:
        groups[bucket(r)].append(r)
    train = sorted(groups.get("default/sft/train", []), key=lambda r: rank(seed, r["record_id"], "sweep"))
    sweep_n = ctx.recipe["split"]["sweep_size"] if not ctx.pilot else min(200, len(train))
    groups["default/sft/sweep50k" if not ctx.pilot else "default/sft/sweep"] = sorted(train[:sweep_n], key=lambda r: (r["kind"], r["rank"]))

    edir = ctx.root / "exports" / ctx.profile_name
    files = {}
    for name, rows in sorted(groups.items()):
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["rank"])
        portable = pa.Table.from_pylist([miles_row(r, None) for r in rows])
        index_cols = ["record_id", "source", "task", "terms", "dataset", "revision", "file", "row", "turn", "upstream_id", "orig_split", "answer_orig", "answer_type", "selector", "image_sha256", "image_path", "width", "height", "group_id", "split", "rank", "total_tokens", "image_tokens", "supervised_tokens", "prompt_tokens", "bos_tokens", "portable_hash", "provenance"]
        index = pa.Table.from_pylist([{k: r[k] for k in index_cols} for r in rows])
        for part, start in enumerate(range(0, len(rows), 100_000)):
            path = edir / "portable" / name / f"part-{part:05d}.parquet"
            write_table(path, portable.slice(start, 100_000))
            files[str(path.relative_to(edir))] = {"rows": min(100_000, len(rows) - start), "sha256": runtime.file_sha256(path), "bytes": path.stat().st_size}
        ipath = edir / "index" / f"{name.replace('/', '__')}.parquet"
        write_table(ipath, index)
        files[str(ipath.relative_to(edir))] = {"rows": len(rows), "sha256": runtime.file_sha256(ipath), "bytes": ipath.stat().st_size}

    image_rows = {}
    for r in final:
        image_rows.setdefault(r["image_sha256"], {"image_sha256": r["image_sha256"], "path": r["image_path"], "bytes": r["image_bytes"], "width": r["width"], "height": r["height"]})
    images_table = pa.Table.from_pylist(sorted(image_rows.values(), key=lambda x: x["image_sha256"]))
    ipath = edir / "images.parquet"
    write_table(ipath, images_table)
    files["images.parquet"] = {"rows": images_table.num_rows, "sha256": runtime.file_sha256(ipath), "bytes": ipath.stat().st_size}
    sums = "".join(f"{v['sha256']}  {k}\n" for k, v in sorted(files.items()))
    hub.atomic_write_text(edir / "SHA256SUMS", sums)

    report = quality_report(ctx, sel, val, final, groups)
    write_json(edir / "reports" / "quality.json", report)
    manifest = build_manifest(ctx, files, image_rows, report)
    write_json(edir / "manifest.json", manifest)
    log(f"export: {len(final)} rows, {len(image_rows)} images, {sum(r['bytes'] for r in image_rows.values()) / 1e9:.2f} GB -> {edir}")
    for name, rows in sorted(groups.items()):
        log(f"  {name}: {len(rows)}")
    return manifest


def quality_report(ctx, sel, val, final, groups):
    scan_markers = [json.loads(p.read_text()) for p in sorted((ctx.work / "scan").glob("*/*.done.json")) if p.parent.name in ctx.sources]
    scanned = defaultdict(lambda: {"rows": 0, "turns": 0, "accepted": 0, "capped_turns": 0, "rejected": Counter()})
    for m in scan_markers:
        s = scanned[m["source"]]
        for k in ("rows", "turns", "accepted"):
            s[k] += m["counts"][k]
        s["capped_turns"] += m["counts"].get("capped_turns", 0)
        s["rejected"].update(m["counts"]["rejected"])
    select_stats = json.loads((ctx.work / "select" / "select.json").read_text())
    failures = Counter(f"{r['source']}|{val[r['record_id']]['reason']}" for r in sel if not val[r["record_id"]]["ok"])
    per_source = {}
    for key in ctx.sources:
        rows = [r for r in final if r["source"] == key]
        per_source[key] = {
            "status": ctx.sources[key]["status"], "kind": ctx.sources[key]["kind"], "task": ctx.sources[key]["task"], "terms": ctx.sources[key]["terms"],
            "scan": {**{k: v for k, v in scanned[key].items() if k != "rejected"}, "rejected": dict(scanned[key]["rejected"].most_common())},
            "selected": select_stats["selected"].get(key, 0), "validated_failures": {k.split("|", 1)[1]: v for k, v in failures.items() if k.startswith(key + "|")},
            "exported": len(rows), "unique_images": len({r["image_sha256"] for r in rows}), "unique_groups": len({r["group_id"] for r in rows}),
            "answer_types": dict(Counter(r["answer_type"] for r in rows)), "selectors": dict(Counter(r["selector"] for r in rows if r["selector"])),
            "total_tokens": _percentiles([r["total_tokens"] or r["prompt_tokens"] for r in rows]), "image_tokens": _percentiles([r["image_tokens"] for r in rows]),
            "supervised_tokens": _percentiles([r["supervised_tokens"] for r in rows if r["supervised_tokens"]]), "answer_words": _percentiles([text.words(r["answer"]) for r in rows]),
            "image_sides": _percentiles([max(r["width"], r["height"]) for r in rows]),
        }
    exports = {name: {"rows": len(rows), "unique_images": len({r["image_sha256"] for r in rows}), "unique_qa": len({(r["group_id"], overlap.core_question(r["question"]), text.key(r["answer"])) for r in rows}), "by_task": dict(Counter(r["task"] for r in rows))} for name, rows in groups.items()}
    leak = {}
    split_groups = defaultdict(set)
    for name, rows in groups.items():
        if "sweep" in name:
            continue
        for r in rows:
            split_groups[name].add(r["group_id"])
    names = sorted(split_groups)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            shared = split_groups[names[a]] & split_groups[names[b]]
            if shared:
                leak[f"{names[a]} & {names[b]}"] = len(shared)
    sweep = groups.get("default/sft/sweep50k") or groups.get("default/sft/sweep") or []
    train_ids = {r["record_id"] for r in groups.get("default/sft/train", [])}
    evals = {json.loads(p.read_text())["eval"]: {k: v for k, v in json.loads(p.read_text()).items() if k not in ("fingerprint",)} for p in sorted((ctx.root / "work" / "evals").glob("*.done.json"))}
    return {
        "profile": ctx.profile_name, "per_source": per_source, "exports": exports,
        "split_group_leakage": leak, "sweep_subset_of_train": all(r["record_id"] in train_ids for r in sweep),
        "selection": select_stats, "evaluation_sets": evals, "preserved_splits": build.preserved_splits(ctx),
        "bos_tokens": dict(Counter(str(r["bos_tokens"]) for r in final if r["kind"] == "sft")),
    }


def build_manifest(ctx, files, image_rows, report):
    import transformers

    from importlib.metadata import version

    return {
        "recipe": {"path": str(ctx.recipe_path), "sha256": hashlib.sha256(ctx.recipe_bytes).hexdigest(), "profile": ctx.profile_name, "seed": ctx.recipe["recipe"]["seed"], "limits": ctx.recipe["limits"], "split": ctx.recipe["split"]},
        "sources": {k: {"status": v["status"], "kind": v["kind"], "dataset": ctx.datasets[sources.ADAPTERS[v["adapter"]].dataset], "config": v.get("config"), "terms": v["terms"], "terms_note": v["terms_note"]} for k, v in ctx.sources.items()},
        "evals": ctx.recipe["evals"],
        "runtime": {"miles_commit": runtime.MILES_COMMIT, "miles_sources": runtime.MILES_SOURCES, "rl360_commit": runtime.RL360_COMMIT, "rl360_sources": runtime.RL360_SOURCES, "qwen_vl_utils": runtime.QWEN_VL_UTILS, "transformers": transformers.__version__, "pillow": version("pillow"), "pyarrow": version("pyarrow"), "numpy": version("numpy"), "python": platform.python_version()},
        "processor": {"path": ctx.recipe["recipe"]["processor"], "files": runtime.processor_fingerprint(ctx.recipe["recipe"]["processor"])},
        "images": {"count": len(image_rows), "bytes": sum(r["bytes"] for r in image_rows.values()), "layout": "images/<sha256[:2]>/<sha256>.<ext>, sha256 of the stored bytes"},
        "files": files,
        "counts": {name: v["rows"] for name, v in report["exports"].items()},
    }


# ------------------------------------------------------------------- materialize / verify


def materialize(ctx, target_root):
    """Write Miles Parquet whose image paths are absolute under target_root (which must contain images/)."""
    target_root = Path(target_root).resolve()
    edir = ctx.root / "exports" / ctx.profile_name
    out = {}
    for path in sorted((edir / "portable").rglob("*.parquet")):
        table = pq.read_table(path)
        imgs = table.column("images").to_pylist()
        table = table.set_column(table.schema.get_field_index("images"), "images", pa.array([[str(target_root / p) for p in row] for row in imgs], pa.list_(pa.string())))
        dest = edir / "miles" / path.relative_to(edir / "portable")
        write_table(dest, table)
        out[str(dest.relative_to(edir))] = runtime.file_sha256(dest)
    write_json(edir / "miles" / "materialized.json", {"target_root": str(target_root), "files": out})
    log(f"materialize: {len(out)} files with image root {target_root}")


def verify(ctx, target_root):
    """Every exported image re-hashed and decoded; every Miles row equals the row that passed validation."""
    target_root = Path(target_root).resolve()
    edir = ctx.root / "exports" / ctx.profile_name
    miles = runtime.miles()
    sums = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in (edir / "SHA256SUMS").read_text().splitlines()}
    bad_files = [k for k, v in sums.items() if runtime.file_sha256(edir / k) != v]
    expected = {}
    for path in sorted((edir / "index").glob("*.parquet")):
        for r in pq.read_table(path, columns=["record_id", "portable_hash"]).to_pylist():
            expected[r["record_id"]] = r["portable_hash"]
    rows, mismatched, paths = 0, [], set()
    for path in sorted((edir / "miles").rglob("*.parquet")):
        for data in miles.read_file(str(path)):
            rows += 1
            if portable_hash(data) != expected.get(data["metadata"]["record_id"]):
                mismatched.append(data["metadata"]["record_id"])
            for p in data["images"]:
                if not p.startswith(str(target_root) + "/"):
                    mismatched.append(data["metadata"]["record_id"])
                paths.add(p)
    jobs = sorted(paths)
    bad_images = []
    for res in build._imap(_verify_images, [(jobs[k : k + 5000],) for k in range(0, len(jobs), 5000)], ctx.recipe["workers"]["validate"]):
        bad_images.extend(res)
    report = {"target_root": str(target_root), "files_checked": len(sums), "bad_files": bad_files, "rows": rows, "row_mismatches": len(mismatched), "mismatch_examples": mismatched[:20], "images_checked": len(jobs), "bad_images": bad_images[:50], "bad_image_count": len(bad_images), "passed": not bad_files and not mismatched and not bad_images}
    write_json(edir / "reports" / "verify.json", report)
    log(f"verify: rows={rows} mismatches={len(mismatched)} images={len(jobs)} bad_images={len(bad_images)} bad_files={len(bad_files)} passed={report['passed']}")
    return report["passed"]


def _verify_images(paths):
    bad = []
    for p in paths:
        try:
            data = Path(p).read_bytes()
            if hashlib.sha256(data).hexdigest() != Path(p).name.split(".")[0]:
                bad.append({"path": p, "reason": "sha256"})
                continue
            images.decode(data)
        except (OSError, images.ImageError) as e:
            bad.append({"path": p, "reason": str(e)})
    return bad


# ------------------------------------------------------------------- inspection sheet


def inspect(ctx, per_source=4, out=None):
    """HTML sheet of exact model-input images (inverted pixel_values) and decoded supervision."""
    out = Path(out or Path("outputs/step05") / ctx.profile_name / "inspect")
    (out / "img").mkdir(parents=True, exist_ok=True)
    edir = ctx.root / "exports" / ctx.profile_name
    checker = contract.Checker(ctx.recipe["recipe"]["processor"], ctx.recipe["limits"])
    tok = checker.processor.tokenizer
    cards = []
    for path in sorted((edir / "miles").glob("*/sft/train/*.parquet")) + sorted((edir / "miles").glob("*/rl/pool/*.parquet")):
        by_source = defaultdict(list)
        for data in checker.miles.read_file(str(path)):
            by_source[data["metadata"]["source"]].append(data)
        for key, rows in sorted(by_source.items()):
            rows.sort(key=lambda d: rank(0, d["metadata"]["record_id"], "inspect"))
            for data in rows[:per_source]:
                rid = data["metadata"]["record_id"]
                name = hashlib.sha256(rid.encode()).hexdigest()[:16] + ".png"
                if "messages" in data:
                    res = checker.sft(data, data["metadata"]["image_sha256"], keep_pixels=True)
                    pixels, grid, ids, start = res["pixels"]
                    images.unpatchify(pixels, grid, checker.processor.image_processor).save(out / "img" / name)
                    prompt = tok.decode([t for t in ids[:start] if t != checker.ids["pad"]], skip_special_tokens=False)
                    supervised = tok.decode(ids[start:], skip_special_tokens=False)
                    body = f"<p class=p>{html.escape(prompt)}</p><p class=s>{html.escape(supervised)}</p><p>{res['total_tokens']} tokens, {res['image_tokens']} image, {res['supervised_tokens']} supervised; model input {res['model_input_size'][0]}x{res['model_input_size'][1]}</p>"
                else:
                    res = checker.rl(data, data["metadata"]["image_sha256"])
                    from PIL import Image

                    mm = checker.miles.process_vision_info(checker.miles.build_messages(json.loads(json.dumps(data)), "problem", True, {"image": "images"}), checker.processor)
                    feat = checker.processor.image_processor(images=mm["images"], return_tensors="np")
                    images.unpatchify(feat["pixel_values"], feat["image_grid_thw"][0], checker.processor.image_processor).save(out / "img" / name)
                    body = f"<p class=p>{html.escape(data['problem'])}</p><p class=s>answer: {html.escape(data['answer'])} ({html.escape(data['metadata']['data_source'])}; reference {html.escape(str(data['metadata']['reference']))})</p><p>{res['prompt_tokens']} prompt tokens, {res['image_tokens']} image</p>"
                cards.append(f"<div class=c><h3>{html.escape(key)}</h3><img src='img/{name}'><code>{html.escape(rid)}</code>{body}</div>")
    page = "<html><head><meta charset=utf-8><style>body{font:13px sans-serif}.c{display:inline-block;vertical-align:top;width:460px;margin:6px;border:1px solid #ccc;padding:6px}.c img{max-width:448px;max-height:448px}.p{white-space:pre-wrap;color:#444}.s{white-space:pre-wrap;background:#e8f5e9;font-weight:bold}</style></head><body>"
    page += f"<h2>{ctx.profile_name}: exact model inputs (inverted pixel_values) and supervised spans</h2><p>Green: decoded tokens under the loss mask. Prompt shown without image-pad tokens.</p>" + "".join(cards) + "</body></html>"
    (out / "index.html").write_text(page)
    log(f"inspect: {len(cards)} cards -> {out / 'index.html'}")
    return out


def exclusion_sheets(ctx, per_category=12, out=None):
    """Side-by-side candidate/evaluation pairs for each (source, eval, method, status) category."""
    from PIL import Image, ImageDraw

    out = Path(out or Path("outputs/step05") / ctx.profile_name / "exclusions")
    out.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(ctx.work / "select" / "exclusions.parquet").to_pylist()
    cats = defaultdict(list)
    for r in rows:
        if r.get("eval_image_sha256"):
            cats[(r["source"], r["eval"], r["method"], r["status"])].append(r)
    written = []
    for (src, ev, method, status), items in sorted(cats.items()):
        seen, picked = set(), []
        for r in sorted(items, key=lambda r: rank(0, r["image_sha256"], "sheet")):
            if r["image_sha256"] not in seen:
                seen.add(r["image_sha256"])
                picked.append(r)
        picked = picked[:per_category]
        sheet = Image.new("RGB", (2 * 260, 290 * len(picked)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, r in enumerate(picked):
            for col, (root, sha) in enumerate(((ctx.root, r["image_sha256"]), (ctx.root / "evals", r["eval_image_sha256"]))):
                path = images.find_asset(root, sha)
                if path:
                    sheet.paste(images.fit(images.to_rgb(Image.open(path)), 256), (col * 260, 290 * k))
            draw.text((2, 290 * k + 258), f"d={r.get('distance')} sim={r.get('similarity')} {r['question'][:60]}", fill="black")
        name = f"{src}__{ev}__{method}__{status}.png"
        sheet.save(out / name)
        written.append({"category": [src, ev, method, status], "count": len(items), "unique_images": len(seen), "sheet": name})
    write_json(out / "index.json", written)
    log(f"exclusion sheets: {len(written)} categories -> {out}")
    return written


def evidence(ctx, out):
    """Compact, path-free summary of a profile's reports for docs/evidence/step05."""
    edir = ctx.root / "exports" / ctx.profile_name
    quality = json.loads((edir / "reports" / "quality.json").read_text())
    manifest = json.loads((edir / "manifest.json").read_text())
    verify_report = json.loads((edir / "reports" / "verify.json").read_text())
    sel = quality["selection"]
    validated = _read_dir(sorted((ctx.work / "validate").glob("chunk-*.done.parquet")))
    exclusions = pq.read_table(ctx.work / "select" / "exclusions.parquet").to_pylist()
    per_source = {}
    for key, v in quality["per_source"].items():
        scan = v["scan"]
        per_source[key] = {
            "status": v["status"], "kind": v["kind"], "task": v["task"], "terms": v["terms"],
            "scanned_rows": scan["rows"], "scanned_turns": scan["turns"], "capped_turns": scan["capped_turns"], "accepted_candidates": scan["accepted"],
            "rejected": scan["rejected"], "excluded_eval": sum(1 for x in exclusions if x["source"] == key),
            "duplicates": sel["duplicates"].get(key, 0), "reserved_for_rl": sel["reserved_for_rl"].get(key, 0),
            "limit_rejects": {k.split("|", 1)[1]: n for k, n in sel["limit_rejects"].items() if k.startswith(key + "|")},
            "selected": v["selected"], "validation_failures": v["validated_failures"], "exported": v["exported"],
            "unique_images": v["unique_images"], "unique_groups": v["unique_groups"], "answer_types": v["answer_types"], "selectors": v["selectors"],
            "total_tokens": v["total_tokens"], "image_tokens": v["image_tokens"], "supervised_tokens": v["supervised_tokens"], "answer_words": v["answer_words"],
        }
    examples = defaultdict(list)
    for x in exclusions:
        k = f"{x['source']}|{x['eval']}|{x['method']}|{x['status']}"
        if len(examples[k]) < 3:
            examples[k].append({key: x.get(key) for key in ("record_id", "eval_item", "distance", "edge", "max_block", "aspect_diff", "thumb_similarity", "external_id") if x.get(key) is not None})
    summary = {
        "profile": ctx.profile_name,
        "recipe_sha256": manifest["recipe"]["sha256"], "seed": manifest["recipe"]["seed"], "limits": manifest["recipe"]["limits"],
        "runtime": manifest["runtime"], "processor_files_sha256": manifest["processor"]["files"],
        "exports": quality["exports"], "images": manifest["images"],
        "files": {k: v for k, v in manifest["files"].items() if not k.startswith("index/")},
        "per_source": per_source,
        "selection": {k: sel[k] for k in ("candidates", "unique_images", "groups", "near_duplicate_pairs", "oversized_buckets", "cross_source_groups", "rl_sft_overlap_before_reservation", "question_text_matches", "overheads")},
        "eval_exclusions": {"by_category": sel["excluded"], "examples": dict(examples)},
        "evaluation_sets": {k: {kk: vv for kk, vv in v.items() if kk not in ("sha256",)} for k, v in quality["evaluation_sets"].items()},
        "preserved_splits": quality["preserved_splits"],
        "validation": {"rows": validated.num_rows, "failures": dict(Counter(r for r in validated.column("reason").to_pylist() if r)), "bos_tokens": quality["bos_tokens"]},
        "split_group_leakage": quality["split_group_leakage"], "sweep_subset_of_train": quality["sweep_subset_of_train"],
        "verify": {k: verify_report[k] for k in ("files_checked", "bad_files", "rows", "row_mismatches", "images_checked", "bad_image_count", "passed")},
    }
    out = Path(out)
    write_json(out / f"{ctx.profile_name}.json", summary)
    log(f"evidence -> {out / (ctx.profile_name + '.json')}")
    return summary
