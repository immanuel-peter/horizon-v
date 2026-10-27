"""Framework-free helpers for reading safetensors checkpoints and hashing files.

    uv run python -m k2hv.checkpoint_io manifest hf/K2-Horizon-V-init   # rewrite SHA256SUMS after editing tracked files

Shared by the init-checkpoint builder and its independent verifier. Headers are parsed directly from
the file bytes, so sizes and offsets do not depend on any tensor library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"
MANIFEST_NAME = "SHA256SUMS"

LM_PREFIX = "language_model."
VISUAL_PREFIX = "visual."
PROJECTOR_PREFIX = "projector."
# GLM's own token-compression modules; the projector replaces them.
EXCLUDED_GLM_PREFIXES = ("downsample.", "merger.")

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    file: Path
    dtype: str  # safetensors dtype string, e.g. "BF16"
    shape: tuple[int, ...]
    start: int  # absolute byte offsets of the payload within `file`
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def numel(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n


def read_header(path: Path) -> tuple[dict, int]:
    """Return ``(tensor_entries, data_start)`` for one ``.safetensors`` file (``__metadata__`` dropped)."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


def read_tensor_infos(directory: str | Path) -> dict[str, TensorInfo]:
    """Tensor headers of a sharded (index.json) or unsharded (model.safetensors) checkpoint.

    Raises if the index references a missing shard, disagrees with the shard headers, or a key repeats.
    """
    directory = Path(directory)
    index_path, single_path = directory / INDEX_NAME, directory / SINGLE_NAME
    weight_map = None
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        files = sorted(set(weight_map.values()))
    elif single_path.is_file():
        files = [SINGLE_NAME]
    else:
        raise CheckpointError(f"{directory} has neither {INDEX_NAME} nor {SINGLE_NAME}")

    infos: dict[str, TensorInfo] = {}
    for name in files:
        path = directory / name
        if not path.is_file():
            raise CheckpointError(f"{index_path.name} references missing shard {name}")
        header, data_start = read_header(path)
        for key, entry in header.items():
            if key in infos:
                raise CheckpointError(f"tensor {key} appears in both {infos[key].file.name} and {name}")
            begin, end = entry["data_offsets"]
            infos[key] = TensorInfo(path, entry["dtype"], tuple(entry["shape"]), data_start + begin, data_start + end)

    if weight_map is not None:
        if set(weight_map) != set(infos):
            missing, extra = set(weight_map) - set(infos), set(infos) - set(weight_map)
            raise CheckpointError(
                f"{index_path.name} and shard headers disagree: {len(missing)} indexed tensors absent from shards "
                f"(e.g. {sorted(missing)[:3]}), {len(extra)} shard tensors absent from index (e.g. {sorted(extra)[:3]})"
            )
        wrong = [k for k, name in weight_map.items() if infos[k].file.name != name]
        if wrong:
            raise CheckpointError(f"{index_path.name} maps {len(wrong)} tensors to the wrong shard (e.g. {wrong[:3]})")
    return infos


def split_glm_keys(keys) -> tuple[list[str], list[str]]:
    """Split GLM vision tensor names into (kept, excluded) by the ``downsample``/``merger`` rule."""
    kept, excluded = [], []
    for key in keys:
        (excluded if key.startswith(EXCLUDED_GLM_PREFIXES) else kept).append(key)
    return kept, excluded


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(directory: Path, *, exclude: tuple[str, ...] = (MANIFEST_NAME,), workers: int = 4) -> dict[str, str]:
    """SHA-256 of every regular file under ``directory`` (relative POSIX paths, sorted).

    Hidden entries (``.cache/`` from `hf download`, ``.building/``) and ``__pycache__`` (bytecode Python writes next
    to the sources it imports) are not part of the checkpoint and are skipped.
    """
    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and p.relative_to(directory).as_posix() not in exclude
        and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(directory).parts)
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        digests = list(pool.map(sha256_file, files))
    return {p.relative_to(directory).as_posix(): d for p, d in zip(files, digests, strict=True)}


def write_manifest(directory: Path) -> Path:
    """Write ``SHA256SUMS`` in `sha256sum -c` format."""
    path = directory / MANIFEST_NAME
    path.write_text("".join(f"{digest}  {rel}\n" for rel, digest in hash_tree(directory).items()))
    return path


def read_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        digest, _, rel = line.partition("  ")
        entries[rel] = digest
    return entries


def parse_size(text: str | int) -> int:
    """``"5GB"`` -> 5_000_000_000 (decimal units, as transformers' ``max_shard_size``)."""
    if isinstance(text, int):
        return text
    units = {"KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
    text = text.strip().upper()
    for suffix, factor in units.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * factor)
    return int(text)


def local_dir_revision(directory: Path) -> str | None:
    """Commit hash recorded by ``snapshot_download(local_dir=...)``, if all downloaded files agree."""
    meta_dir = directory / ".cache" / "huggingface" / "download"
    revisions = {p.read_text().splitlines()[0] for p in meta_dir.rglob("*.metadata") if p.stat().st_size}
    return revisions.pop() if len(revisions) == 1 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint file helpers.")
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest", help="rewrite SHA256SUMS for an existing checkpoint directory")
    manifest.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if not any(args.directory.glob("*.safetensors")):
        print(f"error: {args.directory} holds no .safetensors files; build the checkpoint first", flush=True)
        return 1
    path = write_manifest(args.directory)
    print(f"wrote {path} ({len(read_manifest(path))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
