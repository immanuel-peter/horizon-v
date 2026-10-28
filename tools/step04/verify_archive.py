"""Verify generated checkpoint assets against the separately downloaded archive manifest."""
import argparse
import hashlib
import json
from pathlib import Path

ASSETS = {"build_info.json", "tokenizer.json", "model.safetensors.index.json"}


def verify(checkpoint, manifest):
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("* ").removeprefix("./")
        if name in ASSETS or name.endswith(".safetensors"):
            if Path(name).name != name:
                raise ValueError(f"Unexpected nested asset: {name}")
            entries[name] = digest
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    required = ASSETS | set(index["weight_map"].values())
    if set(entries) != required:
        raise ValueError(f"Archive manifest coverage differs: {set(entries) ^ required}")
    results = {}
    for name, expected in sorted(entries.items()):
        digest = hashlib.file_digest((checkpoint / name).open("rb"), "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
        results[name] = digest
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("hf/K2-Horizon-V-init"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/step04/archive/SHA256SUMS"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.checkpoint, args.manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
