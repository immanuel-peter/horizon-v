"""Build the K2-Horizon-V-init checkpoint.

    uv run python -m k2hv.build_init \\
        --k2 models/K2-Horizon-7B \\
        --glm models/GLM-5.3-Flash-Vision \\
        --out models/K2-Horizon-V-init

The result is K2-Horizon-7B's weights (``language_model.*``), the GLM vision tower through
``post_layernorm`` (``visual.*``) and a deterministically initialized projector (``projector.*``) in one
self-contained, offline-loadable Hugging Face checkpoint. Source tensors are copied byte-for-byte; the
large modules are never instantiated (the expected key schema comes from a meta-device model) and
weights stream shard by shard, so memory stays near one output shard.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import resource
import shutil
import sys
import time
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.dynamic_module_utils import custom_object_save
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig

from .checkpoint_io import (
    INDEX_NAME,
    LM_PREFIX,
    PROJECTOR_PREFIX,
    VISUAL_PREFIX,
    CheckpointError,
    TensorInfo,
    local_dir_revision,
    parse_size,
    read_tensor_infos,
    split_glm_keys,
    write_manifest,
)
from .configuration_k2_horizon import K2HorizonConfig
from .configuration_k2_horizon_v import K2HorizonVConfig
from .modeling_k2_horizon_v import K2HorizonVForConditionalGeneration, build_projector

PACKAGE_DIR = Path(__file__).resolve().parent

K2_REPO = "IFM/K2-Horizon-7B"
GLM_REPO = "immanuelpeter/GLM-5.3-Flash-Vision"

# Plan token ids, verified against the tokenizer that ships with the K2 weights.
IMAGE_TOKENS = {"<image>": 250031, "</image>": 250032, "reserved_special_token_12": 250062}

# K2's remote-code files are vendored byte-for-byte in this package and must match the source.
K2_CODE_FILES = ("configuration_k2_horizon.py", "modeling_k2_horizon.py")
K2_ASSET_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json")
GLM_ASSET_FILES = ("preprocessor_config.json",)

SAFETENSORS_DTYPE = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}


class BuildError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[build_init {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def check_vendored_k2_code(k2_dir: Path) -> None:
    """The package's copy of K2's modeling/config code must be byte-identical to the source checkpoint's."""
    for name in K2_CODE_FILES:
        source = k2_dir / name
        if not source.is_file():
            raise BuildError(f"{source} not found")
        if source.read_bytes() != (PACKAGE_DIR / name).read_bytes():
            raise BuildError(
                f"{source} differs from the copy vendored in src/k2hv/{name}. The vendored files are pinned to a "
                "specific K2 revision; update them deliberately (and rerun the tests) before building from another."
            )


def verify_token_ids(tokenizer_json: Path, expected: dict[str, int], vocab_size: int) -> None:
    data = json.loads(tokenizer_json.read_text())
    found = {t["content"]: t["id"] for t in data.get("added_tokens", [])}
    vocab = data.get("model", {}).get("vocab", {})
    if isinstance(vocab, dict):
        for token, token_id in vocab.items():
            found.setdefault(token, token_id)
    for token, token_id in expected.items():
        if found.get(token) != token_id:
            raise BuildError(f"tokenizer maps {token!r} to {found.get(token)}, the plan expects {token_id}")
        if token_id >= vocab_size:
            raise BuildError(f"{token!r} id {token_id} is outside vocab_size {vocab_size}")


def compare_schema(name: str, expected: dict[str, tuple[tuple[int, ...], str]], infos: dict[str, TensorInfo]) -> None:
    """Strict two-way key/shape/dtype comparison; any dtype difference is an error (no silent casts)."""
    problems = []
    if missing := sorted(set(expected) - set(infos)):
        problems.append(f"{len(missing)} expected tensors missing from source, e.g. {missing[:3]}")
    if extra := sorted(set(infos) - set(expected)):
        problems.append(f"{len(extra)} source tensors the model does not use, e.g. {extra[:3]}")
    for key in sorted(set(expected) & set(infos)):
        shape, dtype = expected[key]
        if infos[key].shape != shape:
            problems.append(f"{key}: source shape {infos[key].shape}, model shape {shape}")
        if infos[key].dtype != dtype:
            problems.append(
                f"{key}: source dtype {infos[key].dtype}, checkpoint dtype {dtype}. Converting would break bitwise "
                "preservation; resolve the dtype explicitly instead of casting."
            )
    if problems:
        raise BuildError(f"{name} does not match the model schema:\n  " + "\n  ".join(problems[:20]))


def expected_schema(config: K2HorizonVConfig, dtype: torch.dtype):
    """Model on the meta device: the authoritative key/shape/dtype schema, without allocating any weights."""
    with torch.device("meta"):
        model = K2HorizonVForConditionalGeneration._from_config(copy.deepcopy(config), dtype=dtype)
    schema = {k: (tuple(v.shape), SAFETENSORS_DTYPE[v.dtype]) for k, v in model.state_dict().items()}
    return model, schema


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------


def header_bound(key: str, ndim: int) -> int:
    """Upper bound on one tensor's entry in a safetensors header."""
    return len(key.encode()) + 96 + 22 * ndim


def plan_shards(entries: list[tuple[str, int, int]], limit: int) -> list[list[str]]:
    """Greedy, order-preserving packing of (key, nbytes, ndim) into shards of at most `limit` bytes (header included)."""
    base = 4096  # header length prefix, __metadata__, alignment padding
    shards: list[list[str]] = []
    current: list[str] = []
    used = base
    for key, nbytes, ndim in entries:
        cost = nbytes + header_bound(key, ndim)
        if base + cost > limit:
            raise BuildError(f"{key} ({nbytes} bytes) cannot fit in a {limit}-byte shard")
        if current and used + cost > limit:
            shards.append(current)
            current, used = [], base
        current.append(key)
        used += cost
    shards.append(current)
    return shards


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def render_notice(info: dict) -> str:
    k2, glm = info["sources"]["k2"], info["sources"]["glm"]
    return f"""K2-Horizon-V-init: third-party components and attribution

This checkpoint combines the following works. Each keeps its own license; no overall license is asserted.

1. K2-Horizon-7B (IFM)  --  Apache License 2.0
   Source:   https://huggingface.co/{k2['repo']} @ {k2['revision']}
   Used for: all language_model.* weights (unmodified), tokenizer files, chat_template.jinja,
             generation_config.json, configuration_k2_horizon.py, modeling_k2_horizon.py.
   The two Python files are copied unchanged; their headers credit "The Qwen team, Alibaba Group and the
   HuggingFace Inc. team" (Apache-2.0). Full license text: licenses/Apache-2.0.txt
   Citation: IFM Team, "Introducing K2 Horizon: Frontier Performance, Radically Open", 2026, https://ifm.ai/blog/k2/

2. GLM-5.3-Flash vision tower  --  MIT License, Copyright (c) 2026 Z.AI Co., Ltd
   Source:   https://huggingface.co/{glm['repo']} @ {glm['revision']}
             (vision tensors extracted from https://huggingface.co/zai-org/GLM-5.3-Flash)
   Used for: all visual.* weights (unmodified; GLM's downsample and merger modules are not included)
             and preprocessor_config.json. Full license text: licenses/MIT-GLM-5.3-Flash-Vision.txt

3. projector.* weights are new: randomly initialized from seed {info['projector']['seed']}, not trained.

The GLM vision model code is not copied; it is imported from the installed `transformers` package
(Apache-2.0, Copyright The HuggingFace Inc. team).
"""


def render_readme(info: dict) -> str:
    k2, glm = info["sources"]["k2"], info["sources"]["glm"]
    return f"""# K2-Horizon-V-init

Initialization checkpoint for K2-Horizon-V. **It is not a trained model**: the projector is random
(seed {info['projector']['seed']}) and the model cannot interpret images until it is trained.

| Module | Weights | Trainable |
|---|---|---|
| `language_model` | {k2['repo']} @ `{k2['revision']}`, unmodified | no |
| `visual` | GLM vision tower through `post_layernorm`, from {glm['repo']} @ `{glm['revision']}`, unmodified | no |
| `projector` | `LayerNorm(1024)`, 2x2 merge, `Linear(4096,4096)`, GELU, `Linear(4096,4096)`; new | yes |

Load (offline, model code ships in this directory):

```python
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained(path, trust_remote_code=True, dtype="bfloat16")
```

## Not done yet

- No `AutoProcessor` / `K2HorizonVProcessor`: `AutoProcessor.from_pretrained` is not supported and untested.
- `chat_template.jinja` is K2's original template, which rejects image content. The image-part patch is pending.
- `preprocessor_config.json` is GLM's, kept for the processor work. Multimodal prompt formatting is untested.
- Only CPU validation has been done (see the repository handoff notes); no GPU or SGLang checks.

See `NOTICE` and `licenses/` for upstream licenses and attribution.
"""


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------


def build(
    k2_dir: Path,
    glm_dir: Path,
    out_dir: Path,
    *,
    seed: int = 0,
    max_shard_size: int = 5_000_000_000,
    k2_repo: str = K2_REPO,
    glm_repo: str = GLM_REPO,
    token_ids: dict[str, int] | None = None,
) -> dict:
    started = time.perf_counter()
    token_ids = token_ids or IMAGE_TOKENS
    k2_dir, glm_dir, out_dir = Path(k2_dir), Path(glm_dir), Path(out_dir)
    tmp_dir = out_dir.with_name(out_dir.name + ".building")

    if out_dir.exists():
        raise BuildError(f"{out_dir} already exists; refusing to overwrite it. Move or delete it first.")
    if tmp_dir.exists():
        raise BuildError(f"{tmp_dir} exists from an earlier failed build; delete it first.")
    for directory in (k2_dir, glm_dir):
        if not directory.is_dir():
            raise BuildError(f"{directory} is not a directory")

    # -- inputs and schema ------------------------------------------------------------------
    check_vendored_k2_code(k2_dir)
    k2_config_dict = json.loads((k2_dir / "config.json").read_text())
    glm_config_dict = json.loads((glm_dir / "config.json").read_text())
    dtype_name = k2_config_dict.get("dtype")
    if dtype_name != glm_config_dict.get("dtype") or not hasattr(torch, str(dtype_name)):
        raise BuildError(f"K2 dtype {dtype_name!r} and GLM dtype {glm_config_dict.get('dtype')!r} must match and be valid")
    dtype = getattr(torch, dtype_name)
    verify_token_ids(k2_dir / "tokenizer.json", token_ids, k2_config_dict["vocab_size"])
    log(f"token ids verified against tokenizer.json: {token_ids}")

    config = K2HorizonVConfig(
        text_config=K2HorizonConfig.from_dict(k2_config_dict),
        vision_config=Glm5NextVisionConfig.from_dict(glm_config_dict),
        image_start_token_id=token_ids["<image>"],
        image_end_token_id=token_ids["</image>"],
        image_token_id=token_ids["reserved_special_token_12"],
        tie_word_embeddings=False,
        dtype=dtype,
    )
    meta_model, schema = expected_schema(config, dtype)
    unknown = {k for k in schema if not k.startswith((LM_PREFIX, VISUAL_PREFIX, PROJECTOR_PREFIX))}
    if unknown:
        raise BuildError(f"model has unexpected top-level modules: {sorted(unknown)[:5]}")

    k2_infos, glm_infos = read_tensor_infos(k2_dir), read_tensor_infos(glm_dir)
    glm_kept, glm_excluded = split_glm_keys(glm_infos)
    compare_schema("K2 checkpoint", {k[len(LM_PREFIX):]: v for k, v in schema.items() if k.startswith(LM_PREFIX)}, k2_infos)
    compare_schema(
        "GLM checkpoint (excluding downsample/merger)",
        {k[len(VISUAL_PREFIX):]: v for k, v in schema.items() if k.startswith(VISUAL_PREFIX)},
        {k: glm_infos[k] for k in glm_kept},
    )
    excluded_params = sum(glm_infos[k].numel for k in glm_excluded)
    log(
        f"K2: {len(k2_infos)} tensors | GLM: {len(glm_kept)} kept, {len(glm_excluded)} excluded "
        f"({excluded_params / 1e6:.2f}M params: downsample.* + merger.*)"
    )

    # -- projector --------------------------------------------------------------------------
    projector = build_projector(
        config.vision_config.hidden_size,
        config.text_config.hidden_size,
        config.vision_config.spatial_merge_size,
        seed=seed,
        dtype=dtype,
    )
    projector_tensors = {PROJECTOR_PREFIX + k: v.detach().contiguous() for k, v in projector.state_dict().items()}
    compare_projector = {k: (tuple(t.shape), SAFETENSORS_DTYPE[t.dtype]) for k, t in projector_tensors.items()}
    if compare_projector != {k: v for k, v in schema.items() if k.startswith(PROJECTOR_PREFIX)}:
        raise BuildError("projector tensors do not match the model's projector schema")

    # -- plan -------------------------------------------------------------------------------
    sizes: dict[str, tuple[int, int]] = {}  # key -> (nbytes, ndim)
    for key, (shape, _) in schema.items():
        if key.startswith(LM_PREFIX):
            sizes[key] = (k2_infos[key[len(LM_PREFIX):]].nbytes, len(shape))
        elif key.startswith(VISUAL_PREFIX):
            sizes[key] = (glm_infos[key[len(VISUAL_PREFIX):]].nbytes, len(shape))
        else:
            t = projector_tensors[key]
            sizes[key] = (t.numel() * t.element_size(), len(shape))
    shards = plan_shards([(key, nbytes, ndim) for key, (nbytes, ndim) in sizes.items()], max_shard_size)
    total_size = sum(nbytes for nbytes, _ in sizes.values())
    free = shutil.disk_usage(out_dir.parent).free
    if free < total_size * 1.02 + (1 << 30):
        raise BuildError(f"need ~{total_size / 1e9:.1f} GB in {out_dir.parent}, only {free / 1e9:.1f} GB free")
    log(f"{len(schema)} tensors, {total_size / 1e9:.2f} GB -> {len(shards)} shards (max {max_shard_size / 1e9:.1f} GB each)")

    # -- write ------------------------------------------------------------------------------
    tmp_dir.mkdir(parents=True)
    try:
        handles: dict[Path, object] = {}

        def read(key: str) -> torch.Tensor:
            if key.startswith(PROJECTOR_PREFIX):
                return projector_tensors[key]
            prefix, infos = (LM_PREFIX, k2_infos) if key.startswith(LM_PREFIX) else (VISUAL_PREFIX, glm_infos)
            info = infos[key[len(prefix):]]
            if info.file not in handles:
                handles[info.file] = safe_open(str(info.file), framework="pt")
            return handles[info.file].get_tensor(key[len(prefix):])

        weight_map: dict[str, str] = {}
        shard_report = []
        for number, keys in enumerate(shards, 1):
            name = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
            save_file({key: read(key) for key in keys}, tmp_dir / name, metadata={"format": "pt"})
            (tmp_dir / name).chmod(0o644)  # safetensors creates files 0600; other users must be able to read shards
            weight_map.update({key: name for key in keys})
            size = (tmp_dir / name).stat().st_size
            if size > max_shard_size:
                raise BuildError(f"{name} is {size} bytes, over the {max_shard_size}-byte limit")
            shard_report.append({"file": name, "bytes": size, "tensors": len(keys)})
            log(f"wrote {name}: {len(keys)} tensors, {size / 1e9:.2f} GB")
        handles.clear()

        total_parameters = sum(torch.Size(shape).numel() for shape, _ in schema.values())
        (tmp_dir / INDEX_NAME).write_text(
            json.dumps(
                {"metadata": {"total_size": total_size, "total_parameters": total_parameters}, "weight_map": weight_map},
                indent=2,
            )
        )

        # Remote code + config. The classes come from the installed package, so auto_map has no Hub prefix.
        K2HorizonVConfig.register_for_auto_class()
        K2HorizonVForConditionalGeneration.register_for_auto_class("AutoModelForImageTextToText")
        config.architectures = [K2HorizonVForConditionalGeneration.__name__]
        custom_object_save(meta_model, tmp_dir, config=config)
        config.save_pretrained(tmp_dir)

        for name in K2_CODE_FILES:  # shipped copies must equal the source files exactly
            if (tmp_dir / name).read_bytes() != (k2_dir / name).read_bytes():
                raise BuildError(f"{name} in the output differs from the K2 source")
        for name in K2_ASSET_FILES:
            shutil.copyfile(k2_dir / name, tmp_dir / name)
        for name in GLM_ASSET_FILES:
            shutil.copyfile(glm_dir / name, tmp_dir / name)
        (tmp_dir / "licenses").mkdir()
        shutil.copyfile(PACKAGE_DIR / "licenses" / "Apache-2.0.txt", tmp_dir / "licenses" / "Apache-2.0.txt")
        shutil.copyfile(glm_dir / "LICENSE", tmp_dir / "licenses" / "MIT-GLM-5.3-Flash-Vision.txt")

        info = {
            "format_version": 1,
            "sources": {
                "k2": {"repo": k2_repo, "revision": local_dir_revision(k2_dir) or "unknown", "tensors": len(k2_infos)},
                "glm": {
                    "repo": glm_repo,
                    "revision": local_dir_revision(glm_dir) or "unknown",
                    "tensors": len(glm_infos),
                    "excluded_tensors": sorted(glm_excluded),
                    "excluded_parameters": excluded_params,
                },
            },
            "token_ids": token_ids,
            "dtype": dtype_name,
            "dtype_conversions": 0,
            "tensors": len(schema),
            "parameters": total_parameters,
            "total_size_bytes": total_size,
            "shards": shard_report,
            "projector": {
                "seed": seed,
                "init": "float32 draws from torch.Generator().manual_seed(seed), cast to the checkpoint dtype",
                "parameters": sum(t.numel() for t in projector_tensors.values()),
                "sha256": {k: tensor_sha256(t) for k, t in sorted(projector_tensors.items())},
            },
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "build_seconds_before_manifest": round(time.perf_counter() - started, 1),
            "peak_rss_bytes_before_manifest": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        (tmp_dir / "NOTICE").write_text(render_notice(info))
        (tmp_dir / "README.md").write_text(render_readme(info))
        (tmp_dir / "build_info.json").write_text(json.dumps(info, indent=2) + "\n")

        check_written(tmp_dir, schema, max_shard_size)
        log("hashing output files")
        write_manifest(tmp_dir)
        tmp_dir.rename(out_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    info["build_seconds_total"] = round(time.perf_counter() - started, 1)
    info["peak_rss_bytes_total"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return info


def check_written(directory: Path, schema: dict, max_shard_size: int) -> None:
    """Header-level check of the written checkpoint against the schema (full validation is `verify_init`)."""
    try:
        infos = read_tensor_infos(directory)
    except CheckpointError as exc:
        raise BuildError(f"written checkpoint is inconsistent: {exc}") from exc
    compare_schema("written checkpoint", schema, infos)
    for shard in {i.file for i in infos.values()}:
        if shard.stat().st_size > max_shard_size:
            raise BuildError(f"{shard.name} exceeds the shard size limit")
    config = json.loads((directory / "config.json").read_text())
    auto_map = config.get("auto_map", {})
    expected = {
        "AutoConfig": "configuration_k2_horizon_v.K2HorizonVConfig",
        "AutoModelForImageTextToText": "modeling_k2_horizon_v.K2HorizonVForConditionalGeneration",
    }
    if auto_map != expected:
        raise BuildError(f"unexpected auto_map: {auto_map}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the K2-Horizon-V-init checkpoint.")
    parser.add_argument("--k2", required=True, type=Path, help="K2-Horizon-7B checkpoint directory")
    parser.add_argument("--glm", required=True, type=Path, help="GLM-5.3-Flash-Vision checkpoint directory")
    parser.add_argument("--out", required=True, type=Path, help="output directory (must not exist)")
    parser.add_argument("--seed", type=int, default=0, help="projector init seed (default: 0)")
    parser.add_argument("--max-shard-size", default="5GB", help="maximum shard file size (default: 5GB)")
    args = parser.parse_args(argv)

    try:
        info = build(args.k2, args.glm, args.out, seed=args.seed, max_shard_size=parse_size(args.max_shard_size))
    except (BuildError, CheckpointError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    log(
        f"done: {args.out} | {info['total_size_bytes'] / 1e9:.2f} GB, {info['tensors']} tensors, "
        f"{info['build_seconds_total']}s, peak RSS {info['peak_rss_bytes_total'] / 2**30:.1f} GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
