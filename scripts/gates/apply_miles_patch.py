from pathlib import Path

from apply_source_patch import apply

apply(Path("third_party/miles"), Path("scripts/gates/miles-source-audit.json"), Path("scripts/gates/miles-gate.patch"))
