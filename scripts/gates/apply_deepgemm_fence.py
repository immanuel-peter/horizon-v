"""Apply sgl-project/DeepGEMM 22c7937a (fence before the SM120 BF16 GEMM releases a shared-memory stage) to the installed sgl-deep-gemm 0.2.2 header."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from k2hv.gate_provenance import deepgemm_header

PATCH = Path(__file__).with_name("deepgemm-fence.patch")
STATES = {
    "43b13d7b457079ae2bc84a452e40953af2ade83a7302f9c7370aef5b280a577e": "original",
    "25c4d2040e8601661ea0a0bb15a7320b9bb43b81e040a9386f4a9cf292743e06": "patched",
}


def state(path):
    return STATES.get(hashlib.sha256(path.read_bytes()).hexdigest(), "unknown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("apply", "revert", "check"))
    parser.add_argument("--header", type=Path)
    args = parser.parse_args()
    path = args.header or deepgemm_header()
    current = state(path)
    print(f"deepgemm-fence: {path} is {current}", file=sys.stderr)
    if current == "unknown":
        sys.exit("deepgemm-fence: unknown header hash, refusing to use or modify it")
    target = "original" if args.mode == "revert" else "patched"
    if current == target:
        return
    if args.mode == "check":
        sys.exit("deepgemm-fence: header lacks the sgl-project/DeepGEMM 22c7937a fence; run scripts/gates/activate.sh")
    direction = "--reverse" if args.mode == "revert" else "--forward"
    with PATCH.open() as patch:
        subprocess.run(["patch", direction, "--silent", "--no-backup-if-mismatch", str(path)], stdin=patch, check=True)
    if state(path) != target:
        sys.exit(f"deepgemm-fence: {args.mode} did not produce the {target} header")
    print(f"deepgemm-fence: {path} is {target}", file=sys.stderr)


if __name__ == "__main__":
    main()
