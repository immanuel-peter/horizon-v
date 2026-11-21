import hashlib
import importlib.metadata
import importlib.util
import subprocess
from datetime import datetime, timezone
from pathlib import Path

INIT_CHECKPOINT = Path("hf/K2-Horizon-V-init")


def deepgemm_header():
    spec = importlib.util.find_spec("deep_gemm")
    if spec is None:
        return None
    return Path(spec.submodule_search_locations[0], "include/deep_gemm/impls/sm120_bf16_gemm.cuh")


def capture():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    sources = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for root in ("src", "hf/K2-Horizon-V-init", "scripts/gates")
        for p in Path(root).rglob("*")
        if p.is_file() and p.suffix in (".py", ".sh", ".patch") and "__pycache__" not in p.parts
    }
    versions = {}
    for name in ("torch", "transformers", "sglang", "triton", "ray", "pillow", "safetensors", "flashinfer-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    header = deepgemm_header()
    versions["deep_gemm_sm120_bf16_header_sha256"] = hashlib.sha256(header.read_bytes()).hexdigest() if header and header.is_file() else None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")), "utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sources, "packages": versions}
