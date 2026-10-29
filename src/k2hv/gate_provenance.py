import hashlib
import importlib.metadata
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def capture():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    sources = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for root in ("src", "hf/K2-Horizon-V-init", "tools/step04")
        for p in Path(root).rglob("*")
        if p.is_file() and p.suffix in (".py", ".sh", ".patch") and "__pycache__" not in p.parts
    }
    versions = {}
    for name in ("torch", "transformers", "sglang", "triton", "ray", "pillow", "safetensors", "flashinfer-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")), "utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sources, "packages": versions}
