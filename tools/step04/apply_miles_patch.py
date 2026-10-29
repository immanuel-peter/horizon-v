from pathlib import Path

from apply_source_patch import apply

apply(Path("third_party/miles"), Path("tools/step04/miles-source-audit.json"), Path("tools/step04/miles-gate.patch"))
