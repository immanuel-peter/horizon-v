from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .checkpoint_io import (
    INDEX_NAME,
    LM_PREFIX,
    MANIFEST_NAME,
    PROJECTOR_PREFIX,
    VISUAL_PREFIX,
    TensorInfo,
    hash_tree,
    local_dir_revision,
    read_manifest,
    read_tensor_infos,
    sha256_file,
    split_glm_keys,
)
from .build_init import CODE_FILES, IMAGE_TOKENS, PACKAGE_HF_DIR, STAGING_DIR, tensor_sha256

MAX_SHARD_BYTES = 5_000_000_000
EXPECTED_AUTO_MAP = {
    "AutoConfig": "configuration_k2_horizon_v.K2HorizonVConfig",
    "AutoModelForImageTextToText": "modeling_k2_horizon_v.K2HorizonVForConditionalGeneration",
}
SHIPPED_CODE = CODE_FILES
COPIED_FROM_K2 = ("configuration_k2_horizon.py", "modeling_k2_horizon.py", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "chat_template.jinja", "generation_config.json")
COPIED_FROM_GLM = ("preprocessor_config.json",)
PARITY_PROMPT = "The capital of France is Paris. The capital of Japan is"


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = "", **data) -> bool:
        self.checks.append({"name": name, "ok": ok, "detail": detail, **data})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)
        return ok

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)


def check_layout(out: Path, k2: Path, glm: Path, report: Report, token_ids: dict[str, int] = IMAGE_TOKENS) -> dict:
    required = [*SHIPPED_CODE, "config.json", INDEX_NAME, "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "chat_template.jinja", "generation_config.json", "preprocessor_config.json",
                "NOTICE", "README.md", "build_info.json", MANIFEST_NAME,
                "licenses/Apache-2.0.txt", "licenses/MIT-GLM-5.3-Flash-Vision.txt"]
    missing = [name for name in required if not (out / name).is_file()]
    report.add("layout: all required files present", not missing, f"missing: {missing}" if missing else f"{len(required)} files")

    stale = [n for n in SHIPPED_CODE if (out / n).read_bytes() != (PACKAGE_HF_DIR / n).read_bytes()]
    report.add("layout: code equals the code k2hv.hf imports (the code the schema came from)", not stale, str(stale) if stale else f"{len(SHIPPED_CODE)} files")
    report.add("layout: no leftover staging directory", not (out / STAGING_DIR).exists())

    processor_files = sorted(p.name for p in out.iterdir() if p.name.startswith("processing_") or p.name == "processor_config.json")
    report.add("layout: no processor artifacts advertised", not processor_files, str(processor_files) if processor_files else "")

    config = json.loads((out / "config.json").read_text())
    report.add("config: auto_map is exactly AutoConfig + AutoModelForImageTextToText", config.get("auto_map") == EXPECTED_AUTO_MAP,
               json.dumps(config.get("auto_map")))
    prefixed = [m.group(0) for m in re.finditer(r'"[^"]*--[^"]*\.[A-Za-z_]+"', (out / "config.json").read_text())]
    report.add("config: no Hub-prefixed remote-code references anywhere in config.json", not prefixed, str(prefixed[:3]))

    build_info = json.loads((out / "build_info.json").read_text())
    ids = build_info["token_ids"]
    start, pad, end = (token_ids[t] for t in ("<image>", "reserved_special_token_12", "</image>"))
    report.add(
        f"config: image token ids are {start} / {pad} / {end}",
        (config["image_start_token_id"], config["image_token_id"], config["image_end_token_id"]) == (start, pad, end)
        and ids == token_ids,
        f"start={config['image_start_token_id']} pad={config['image_token_id']} end={config['image_end_token_id']}",
    )
    report.add("config: embeddings untied", config.get("tie_word_embeddings") is False and config["text_config"].get("tie_word_embeddings") is False)

    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig

    from .hf.configuration_k2_horizon import K2HorizonConfig

    def same(cls, ours: dict, theirs: dict) -> bool:
        a, b = cls.from_dict(dict(ours)).to_dict(), cls.from_dict(dict(theirs)).to_dict()
        for key in ("_name_or_path", "transformers_version"):
            a.pop(key, None), b.pop(key, None)
        return a == b

    report.add("config: text_config equals K2's config.json", same(K2HorizonConfig, config["text_config"], json.loads((k2 / "config.json").read_text())))
    report.add("config: vision_config equals GLM's config.json", same(Glm5NextVisionConfig, config["vision_config"], json.loads((glm / "config.json").read_text())))

    checked = [(n, k2) for n in COPIED_FROM_K2] + [(n, glm) for n in COPIED_FROM_GLM]
    bad = [n for n, source in checked if (out / n).read_bytes() != (source / n).read_bytes()]
    mit_ok = (out / "licenses/MIT-GLM-5.3-Flash-Vision.txt").read_bytes() == (glm / "LICENSE").read_bytes()
    bad += [] if mit_ok else ["licenses/MIT-GLM-5.3-Flash-Vision.txt"]
    report.add("assets: mirrored upstream files (K2 code, tokenizer, chat template, generation config, preprocessor config, MIT license) "
               "are byte-identical to the sources", not bad, str(bad) if bad else f"{len(checked) + 1} files")

    revisions = {name: build_info["sources"][name]["revision"] for name in ("k2", "glm")}
    docs = {n: (out / n).read_text() for n in ("README.md", "NOTICE")}
    stale_docs = [f"{n} lacks {name} {rev[:8]}" for n, text in docs.items() for name, rev in revisions.items()
                  if rev != "unknown" and rev not in text]
    report.add("docs: README.md and NOTICE name the source revisions this build used", not stale_docs,
               "; ".join(stale_docs) if stale_docs else ", ".join(f"{k}={v[:8]}" for k, v in revisions.items()))
    return config


def check_shards(out: Path, report: Report) -> dict[str, TensorInfo]:
    index = json.loads((out / INDEX_NAME).read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    on_disk = sorted(p.name for p in out.glob("*.safetensors"))
    report.add("shards: every shard in the index exists and no extra .safetensors files", shard_names == on_disk,
               f"{len(shard_names)} in index, {len(on_disk)} on disk")
    count = len(shard_names)
    report.add("shards: names are consecutive model-0000i-of-0000N", shard_names == [f"model-{i:05d}-of-{count:05d}.safetensors" for i in range(1, count + 1)])
    sizes = {n: (out / n).stat().st_size for n in shard_names}
    report.add(f"shards: each file <= {MAX_SHARD_BYTES / 1e9:.0f} GB", max(sizes.values()) <= MAX_SHARD_BYTES,
               ", ".join(f"{n.split('-')[1]}={s / 1e9:.2f}GB" for n, s in sizes.items()))
    infos = read_tensor_infos(out)
    report.add("shards: index entries == shard header entries (both directions)", True, f"{len(infos)} tensors")
    total = sum(i.nbytes for i in infos.values())
    report.add("shards: index total_size equals the tensor bytes", index["metadata"]["total_size"] == total, f"{total}")
    return infos


def check_key_coverage(out_infos: dict[str, TensorInfo], k2: Path, glm: Path, config: dict, report: Report):
    k2_infos, glm_infos = read_tensor_infos(k2), read_tensor_infos(glm)
    glm_kept, glm_excluded = split_glm_keys(glm_infos)
    d_vit, d_llm = config["vision_config"]["hidden_size"], config["text_config"]["hidden_size"]
    projector = {
        "norm.weight": (d_vit,), "norm.bias": (d_vit,),
        "fc1.weight": (d_llm, 4 * d_vit), "fc1.bias": (d_llm,),
        "fc2.weight": (d_llm, d_llm), "fc2.bias": (d_llm,),
    }
    expected = {LM_PREFIX + k: ("K2", k) for k in k2_infos}
    expected |= {VISUAL_PREFIX + k: ("GLM", k) for k in glm_kept}
    expected |= {PROJECTOR_PREFIX + k: ("NEW", k) for k in projector}

    report.add("keys: output key set == K2 + GLM(minus downsample/merger) + projector, exactly",
               set(out_infos) == set(expected),
               f"{len(out_infos)} = {len(k2_infos)} K2 + {len(glm_kept)} GLM + {len(projector)} projector"
               + (f"; missing {sorted(set(expected) - set(out_infos))[:3]}, extra {sorted(set(out_infos) - set(expected))[:3]}" if set(out_infos) != set(expected) else ""))
    leaked = [k for k in out_infos if k.startswith(VISUAL_PREFIX) and k[len(VISUAL_PREFIX):].startswith(("downsample.", "merger."))]
    excluded_params = sum(glm_infos[k].numel for k in glm_excluded)
    report.add("keys: GLM downsample/merger are absent and fully accounted for", not leaked and len(glm_excluded) == 8,
               f"{len(glm_excluded)} tensors, {excluded_params / 1e6:.2f}M params excluded: {sorted(glm_excluded)}")

    bad = []
    for key, (source, name) in expected.items():
        info = out_infos.get(key)
        if info is None:
            continue
        if source == "NEW":
            if (info.shape, info.dtype) != (projector[name], "BF16"):
                bad.append(f"{key}: {info.shape} {info.dtype}")
        else:
            src = (k2_infos if source == "K2" else glm_infos)[name]
            if (info.shape, info.dtype, info.nbytes) != (src.shape, src.dtype, src.nbytes):
                bad.append(f"{key}: out {info.shape}/{info.dtype}, source {src.shape}/{src.dtype}")
    report.add("keys: every shape and dtype matches its source (no dtype conversion)", not bad, str(bad[:3]) if bad else
               f"dtypes in output: {sorted({i.dtype for i in out_infos.values()})}")
    return k2_infos, glm_infos


def _compare_ranges(a: TensorInfo, b: TensorInfo, chunk: int = 64 << 20) -> bool:
    with open(a.file, "rb") as fa, open(b.file, "rb") as fb:
        fa.seek(a.start_offset), fb.seek(b.start_offset)
        remaining = a.nbytes
        while remaining:
            n = min(chunk, remaining)
            if fa.read(n) != fb.read(n):
                return False
            remaining -= n
    return True


def check_frozen_bytes(out_infos, k2_infos, glm_infos, report: Report, workers: int = 4) -> None:
    jobs = []
    for key, info in out_infos.items():
        if key.startswith(LM_PREFIX):
            jobs.append((key, info, k2_infos[key[len(LM_PREFIX):]]))
        elif key.startswith(VISUAL_PREFIX):
            jobs.append((key, info, glm_infos[key[len(VISUAL_PREFIX):]]))
    started = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda j: _compare_ranges(j[1], j[2]), jobs))
    mismatched = [j[0] for j, ok in zip(jobs, results, strict=True) if not ok]
    total = sum(j[1].nbytes for j in jobs)
    report.add("frozen: every language_model.* and visual.* tensor is byte-identical to its source tensor",
               not mismatched, f"{len(jobs)} tensors, {total / 1e9:.2f} GB compared in {time.perf_counter() - started:.0f}s"
               + (f"; MISMATCH {mismatched[:3]}" if mismatched else ""))


def check_projector(out: Path, out_infos, config: dict, report: Report) -> None:
    from safetensors import safe_open

    from .hf.modeling_k2_horizon_v import build_projector

    build_info = json.loads((out / "build_info.json").read_text())
    seed = build_info["projector"]["seed"]
    args = (config["vision_config"]["hidden_size"], config["text_config"]["hidden_size"], config["vision_config"]["spatial_merge_size"])

    def regenerate(global_seed):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(global_seed)
            p = build_projector(*args, seed=seed, dtype=torch.bfloat16)
        return {PROJECTOR_PREFIX + k: v.detach().contiguous() for k, v in p.state_dict().items()}

    first, second = regenerate(12345), regenerate(67890)
    report.add("projector: regeneration is bitwise identical with different global RNG seeds",
               all(torch.equal(first[k].view(torch.int16), second[k].view(torch.int16)) for k in first))
    stored = {}
    for key in first:
        with safe_open(str(out_infos[key].file), "pt") as f:
            stored[key] = f.get_tensor(key)
    same = all(torch.equal(first[k].view(torch.int16), stored[k].view(torch.int16)) for k in first)
    report.add(f"projector: stored weights == regenerated from seed {seed}", same)
    digests = {k: tensor_sha256(v) for k, v in sorted(stored.items())}
    report.add("projector: SHA-256s match build_info.json", digests == build_info["projector"]["sha256"])
    d_vit, d_llm = args[0], args[1]
    params = sum(v.numel() for v in stored.values())
    expected = 2 * d_vit + (4 * d_vit * d_llm + d_llm) + (d_llm * d_llm + d_llm)
    report.add("projector: parameter count follows from the dims", params == expected, f"{params:,}")
    if (d_vit, d_llm) == (1024, 4096):
        report.add("projector: 33,564,672 parameters (~33.6M, as planned)", params == 33_564_672)


def check_manifest(out: Path, report: Report) -> None:
    recorded = read_manifest(out / MANIFEST_NAME)
    actual = hash_tree(out)
    diff = sorted({k for k in recorded.keys() | actual.keys() if recorded.get(k) != actual.get(k)})
    report.add("manifest: SHA256SUMS matches every file in the checkpoint (none missing, none extra)", not diff,
               f"{len(actual)} files" + (f"; differences: {diff[:5]}" if diff else ""))


def check_source_integrity(k2: Path, glm: Path, report: Report) -> None:
    bad, n = [], 0
    for directory in (k2, glm):
        meta_dir = directory / ".cache" / "huggingface" / "download"
        for shard in sorted(directory.glob("*.safetensors")):
            meta = meta_dir / (shard.name + ".metadata")
            if not meta.is_file():
                bad.append(f"{shard.name}: no download metadata")
                continue
            n += 1
            if sha256_file(shard) != meta.read_text().splitlines()[1]:
                bad.append(shard.name)
    report.add("sources: safetensors files match their Hub LFS sha256", not bad, f"{n} files checked" + (f"; BAD {bad}" if bad else ""))


_LOAD_CHILD = textwrap.dedent(
    """
    import glob, json, os, resource, sys, time
    sys.modules["k2hv"] = None
    sys.modules["k2hv_sglang"] = None
    out, glm, ids_path, result_path, threads = sys.argv[1:6]
    import numpy as np, torch
    torch.set_num_threads(int(threads))
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    res = {}

    t = time.time()
    cfg = AutoConfig.from_pretrained(out, trust_remote_code=True, local_files_only=True)
    res["config_class"] = f"{type(cfg).__module__}.{type(cfg).__name__}"
    res["config_ok"] = (cfg.model_type, cfg.image_token_id, cfg.image_start_token_id, cfg.image_end_token_id) == ("k2_horizon_v", 250062, 250031, 250032)
    res["config_seconds"] = round(time.time() - t, 1)

    t = time.time()
    model, info = AutoModelForImageTextToText.from_pretrained(
        out, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        output_loading_info=True)
    res["model_class"] = f"{type(model).__module__}.{type(model).__name__}"
    res["load_seconds"] = round(time.time() - t, 1)
    res["loading_info"] = {k: (sorted(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v) for k, v in info.items()}
    res["loading_info_clean"] = all(not v for v in info.values())
    res["k2hv_imported"] = sys.modules.get("k2hv") is not None

    comp = {}
    for prefix in ("language_model", "visual", "projector"):
        ps = [p for n, p in model.named_parameters() if n.startswith(prefix + ".")]
        comp[prefix] = {"params": sum(p.numel() for p in ps), "dtypes": sorted({str(p.dtype) for p in ps}),
                        "trainable": sum(p.numel() for p in ps if p.requires_grad)}
    res["components"] = comp
    res["trainable_names"] = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    res["untied"] = model.language_model.lm_head.weight.data_ptr() != model.language_model.model.embed_tokens.weight.data_ptr()
    res["attn_impl"] = [model.config._attn_implementation, model.config.vision_config._attn_implementation, model.config.text_config._attn_implementation]

    state, seen, bad = model.state_dict(), set(), []
    for shard in sorted(glob.glob(os.path.join(out, "model-*.safetensors"))):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                a, b = f.get_tensor(key), state[key].detach().contiguous()
                seen.add(key)
                if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.view(torch.int16), b.view(torch.int16)):
                    bad.append(key)
    res["loaded_equals_files"] = {"tensors": len(seen), "all_keys_covered": seen == set(state), "mismatches": bad[:5]}

    ids = torch.load(ids_path)
    with torch.inference_mode():
        t = time.time()
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        res["text_forward_seconds"] = round(time.time() - t, 1)
    torch.save(logits, result_path + ".logits.pt")

    try:
        tok = AutoTokenizer.from_pretrained(out, local_files_only=True, trust_remote_code=False)
        res["tokenizer"] = {"class": type(tok).__name__, "len": len(tok),
                            "ids": [tok.convert_tokens_to_ids(x) for x in ("<image>", "reserved_special_token_12", "</image>")]}
    except Exception as exc:
        res["tokenizer"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}

    from transformers.models.glm5_next.image_processing_pil_glm5_next import Glm5NextImageProcessorPil
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel
    pre = json.load(open(os.path.join(out, "preprocessor_config.json")))
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (448, 448, 3), dtype=np.uint8).astype(np.float32) / 255.0
    image = ((image - np.array(pre["image_mean"], dtype=np.float32)) / np.array(pre["image_std"], dtype=np.float32)).transpose(2, 0, 1)
    flat, gh, gw = Glm5NextImageProcessorPil().patchify(image, patch_size=14, merge_size=2, temporal_patch_size=2)
    pixels = torch.from_numpy(np.ascontiguousarray(flat))
    grid = torch.tensor([[1, gh, gw]])
    res["image"] = {"pixel_values": list(pixels.shape), "grid": grid.tolist()}

    native = Glm5NextVisionModel.from_pretrained(glm, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    captured = {}
    native.post_layernorm.register_forward_hook(lambda m, i, o: captured.update(x=o.detach()))
    with torch.inference_mode():
        t = time.time()
        native(pixels.to(torch.bfloat16), grid_thw=grid)
        ours = model.visual(pixels.to(torch.bfloat16), grid_thw=grid)
        res["image"]["vit_seconds"] = round((time.time() - t) / 2, 1)
        res["image"]["tower_equals_native_through_post_layernorm"] = bool(torch.equal(ours.view(torch.int16), captured["x"].view(torch.int16)))
        feats = model.get_image_features(pixels, grid)
    res["image"]["vit_std"] = float(ours.float().std())
    res["image"]["feature_shape"] = list(feats.shape)
    res["image"]["features_finite"] = bool(torch.isfinite(feats).all())
    del native

    n_tokens = int(grid.prod() // 4)
    text_ids = ids[0, :6].tolist()
    seq = torch.tensor([[0, 250031] + [250062] * n_tokens + [250032] + text_ids])
    with torch.inference_mode():
        t = time.time()
        out_logits = model(input_ids=seq, attention_mask=torch.ones_like(seq), pixel_values=pixels, image_grid_thw=grid, logits_to_keep=1).logits
        res["image"]["forward_seconds"] = round(time.time() - t, 1)
    res["image"]["tokens"] = int(seq.shape[1])
    res["image"]["logits_shape"] = list(out_logits.shape)
    res["image"]["logits_finite"] = bool(torch.isfinite(out_logits).all())

    res["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    json.dump(res, open(result_path, "w"))
    """
)

_K2_CHILD = textwrap.dedent(
    """
    import json, resource, sys, time
    k2, ids_path, result_path, threads = sys.argv[1:5]
    import torch
    torch.set_num_threads(int(threads))
    from transformers import AutoModelForCausalLM
    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(k2, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    ids = torch.load(ids_path)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    torch.save(logits, result_path + ".logits.pt")
    json.dump({"seconds": round(time.time() - t, 1), "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}, open(result_path, "w"))
    """
)


def _offline_env(scratch: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not any(s in k for s in ("TOKEN", "HF_", "GH_", "GITHUB", "HUGGING"))}
    env.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(scratch / "hf_home"),
        HF_MODULES_CACHE=str(scratch / "hf_modules"), HF_HUB_CACHE=str(scratch / "hf_hub"), XDG_CACHE_HOME=str(scratch / "xdg"),
        TOKENIZERS_PARALLELISM="false",
    )
    return env


def _run_child(script: str, args: list[str], env: dict, result_path: Path, timeout: int) -> tuple[dict, str]:
    proc = subprocess.run([sys.executable, "-I", "-c", script, *args], env=env, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if proc.returncode or not result_path.is_file():
        raise RuntimeError(f"child failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}")
    return json.loads(result_path.read_text()), proc.stderr


def check_fresh_process(out: Path, k2: Path, glm: Path, report: Report, *, threads: int, atol: float, timeout: int = 7200) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(k2, trust_remote_code=False)
    ids = tokenizer(PARITY_PROMPT, return_tensors="pt")["input_ids"]

    with tempfile.TemporaryDirectory(prefix="k2hv-verify-") as tmp:
        scratch = Path(tmp)
        (scratch / "hf_modules").mkdir()
        env = _offline_env(scratch)
        ids_path = scratch / "ids.pt"
        torch.save(ids, ids_path)

        started = time.perf_counter()
        try:
            res, _ = _run_child(_LOAD_CHILD, [str(out), str(glm), str(ids_path), str(scratch / "load.json"), str(threads)], env, scratch / "load.json", timeout)
        except Exception as exc:
            report.add("load: fresh offline process loads AutoConfig + AutoModelForImageTextToText", False, str(exc)[:1500])
            return {}
        elapsed = time.perf_counter() - started

        cached = sorted(p.relative_to(scratch / "hf_modules").as_posix() for p in (scratch / "hf_modules").rglob("*.py"))
        report.add("load: remote code came from the fresh module cache and covers all four shipped modules",
                   all(any(c.endswith(name) for c in cached) for name in SHIPPED_CODE), f"{len(cached)} cached modules")
        report.add("load: AutoConfig(trust_remote_code, local_files_only) in a fresh HF_HUB_OFFLINE=1 process", res["config_ok"],
                   f"{res['config_class']} in {res['config_seconds']}s")
        report.add("load: AutoModelForImageTextToText(trust_remote_code, local_files_only) succeeds",
                   res["model_class"].endswith("K2HorizonVForConditionalGeneration") and not res["k2hv_imported"],
                   f"{res['model_class']} in {res['load_seconds']}s; project package not imported")
        report.add("load: no missing / unexpected / mismatched keys", res["loading_info_clean"], json.dumps(res["loading_info"]))
        comps = res["components"]
        report.add("model: only the projector is trainable (6 tensors, 33,564,672 params)",
                   res["trainable_names"] == sorted(f"projector.{n}" for n in ("norm.weight", "norm.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"))
                   and comps["projector"]["trainable"] == 33_564_672 and comps["language_model"]["trainable"] == 0 and comps["visual"]["trainable"] == 0,
                   "params " + ", ".join(f"{k}={v['params']:,}" for k, v in comps.items()))
        report.add("model: all parameters bfloat16, embeddings untied", all(c["dtypes"] == ["torch.bfloat16"] for c in comps.values()) and res["untied"])
        eq = res["loaded_equals_files"]
        report.add("load: every loaded tensor equals its stored bytes", eq["all_keys_covered"] and not eq["mismatches"], f"{eq['tensors']} tensors")
        tok = res["tokenizer"]
        report.add("tokenizer: AutoTokenizer loads from the checkpoint with the planned image-token ids",
                   tok.get("ids") == [250031, 250062, 250032], json.dumps(tok))

        image = res["image"]
        report.add("image smoke (model level): truncated tower == native GLM vision model through post_layernorm, real weights",
                   image["tower_equals_native_through_post_layernorm"], f"{image['pixel_values']} patches, vit std {image['vit_std']:.3f}")
        report.add("image smoke (model level): 448x448 -> 256 finite features; hand-built <image>+256 pads+</image> forward is finite",
                   image["feature_shape"] == [256, 4096] and image["features_finite"] and image["logits_finite"],
                   f"features {image['feature_shape']}, {image['tokens']} tokens, vit {image['vit_seconds']}s, forward {image['forward_seconds']}s")

        try:
            k2_res, _ = _run_child(_K2_CHILD, [str(k2), str(ids_path), str(scratch / "k2.json"), str(threads)], env, scratch / "k2.json", timeout)
        except Exception as exc:
            report.add("parity: stock K2 reference run", False, str(exc)[:1500])
            return res
        ours = torch.load(scratch / "load.json.logits.pt")
        ref = torch.load(scratch / "k2.json.logits.pt")
        diff = (ours.float() - ref.float()).abs()
        max_diff = float(diff.max())
        report.add(
            f"parity: text-only logits equal stock K2 (bf16, sdpa, {ids.shape[1]} tokens, {threads} threads)",
            ours.dtype == ref.dtype and ours.shape == ref.shape and max_diff <= atol,
            f"max |diff| = {max_diff:g}, mean |diff| = {float(diff.mean()):g}, tolerance atol = {atol:g}, bitwise equal = {torch.equal(ours, ref)}, "
            f"logits range [{float(ref.float().min()):.2f}, {float(ref.float().max()):.2f}], argmax equal = {bool((ours.argmax(-1) == ref.argmax(-1)).all())}",
            max_abs_diff=max_diff, atol=atol,
        )
        res["k2_reference"] = k2_res
        res["wall_seconds"] = round(time.perf_counter() - started, 1)
        res["load_wall_seconds"] = round(elapsed, 1)
    return res


def disk_usage(out: Path) -> dict:
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    free = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
    return {"checkpoint_bytes": size, "free_bytes_on_volume": free}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a K2-Horizon-V-init checkpoint.")
    parser.add_argument("--k2", required=True, type=Path)
    parser.add_argument("--glm", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="write a JSON report here")
    parser.add_argument("--skip-bytes", action="store_true", help="skip the byte-for-byte frozen tensor comparison")
    parser.add_argument("--skip-load", action="store_true", help="skip the fresh-process load, image smoke and parity checks")
    parser.add_argument("--source-hashes", action="store_true", help="also hash the source safetensors against the Hub's sha256")
    parser.add_argument("--threads", type=int, default=4, help="torch threads in the load/parity subprocesses")
    parser.add_argument("--parity-atol", type=float, default=0.0, help="max allowed |logit difference| vs stock K2 (default: exact)")
    args = parser.parse_args(argv)

    out, k2, glm = (p.resolve() for p in (args.out, args.k2, args.glm))
    report, extra = Report(), {}
    started = time.perf_counter()
    config = check_layout(out, k2, glm, report)
    out_infos = check_shards(out, report)
    k2_infos, glm_infos = check_key_coverage(out_infos, k2, glm, config, report)
    if not args.skip_bytes:
        check_frozen_bytes(out_infos, k2_infos, glm_infos, report)
    check_projector(out, out_infos, config, report)
    check_manifest(out, report)
    if args.source_hashes:
        check_source_integrity(k2, glm, report)
    if not args.skip_load:
        extra["fresh_process"] = check_fresh_process(out, k2, glm, report, threads=args.threads, atol=args.parity_atol)

    summary = {
        "ok": report.ok,
        "checks": report.checks,
        "sources": {"k2_revision": local_dir_revision(k2), "glm_revision": local_dir_revision(glm)},
        "disk": disk_usage(out),
        "verify_seconds": round(time.perf_counter() - started, 1),
        "verify_parent_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        **extra,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2))
    passed = sum(c["ok"] for c in report.checks)
    print(f"\n{passed}/{len(report.checks)} checks passed in {summary['verify_seconds']}s; "
          f"checkpoint {summary['disk']['checkpoint_bytes'] / 1e9:.2f} GB, {summary['disk']['free_bytes_on_volume'] / 1e9:.0f} GB free")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
