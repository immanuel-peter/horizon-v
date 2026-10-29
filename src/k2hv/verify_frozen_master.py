"""Compare actual Miles FP32 frozen checkpoint tensors with promoted init tensors."""

import argparse
import hashlib
import json
import resource
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader, load

from .gate_provenance import capture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--init", type=Path, default=Path("hf/K2-Horizon-V-init"))
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    index = json.loads((a.init / "model.safetensors.index.json").read_text())["weight_map"]
    metadata = FileSystemReader(a.checkpoint / "model").read_metadata().state_dict_metadata
    tensors = {name: torch.empty(metadata["model_state.model." + name].size, dtype=torch.float32) for name in index if not name.startswith("projector.")}
    load(state_dict={"model_state": {"model": tensors}}, checkpoint_id=str(a.checkpoint / "model"))
    digest = hashlib.sha256()
    checked = 0
    for shard in sorted(set(index.values())):
        with safe_open(str(a.init / shard), framework="pt", device="cpu") as file:
            for name in sorted(file.keys()):
                if name not in tensors:
                    continue
                expected = file.get_tensor(name).float()
                actual = tensors.pop(name)
                assert torch.equal(actual, expected), f"Frozen FP32 master changed: {name}"
                digest.update(name.encode())
                digest.update(actual.contiguous().numpy().tobytes())
                checked += 1
    assert checked == 666 and not tensors, checked

    report = {
        "provenance": capture(),
        "passed": True,
        "frozen_tensors_checked": checked,
        "comparison": "exact FP32 checkpoint values versus archived BF16 values promoted to FP32, before any downcast",
        "sha256": digest.hexdigest(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}))


if __name__ == "__main__":
    main()
