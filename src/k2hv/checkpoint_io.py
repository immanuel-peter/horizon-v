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
EXCLUDED_GLM_PREFIXES = ("downsample.", "merger.")

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    file: Path
    dtype: str
    shape: tuple[int, ...]
    start_offset: int
    end_offset: int

    @property
    def nbytes(self) -> int:
        return self.end_offset - self.start_offset

    @property
    def numel(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n


def read_header(path: Path) -> tuple[dict, int]:
    """Return ``(tensor_entries, data_start)`` for a safetensors file.

    ``data_start`` is an absolute byte offset. Omit ``__metadata__`` from the entries.
    """
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


def read_tensor_infos(directory: str | Path) -> dict[str, TensorInfo]:
    """Read tensor metadata from indexed shards or ``model.safetensors``.

    Raise ``CheckpointError`` if the index references a missing shard, disagrees with the shard
    headers, or a key repeats.
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
    """Return file SHA-256s keyed by sorted relative POSIX paths.

    Skip paths in ``exclude``, hidden entries and ``__pycache__`` directories.
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
    """Parse byte counts with decimal KB, MB, GB or TB suffixes."""
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
