"""Capture the audited runtime JSON separately from upstream import diagnostics."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--report", type=Path, required=True)
a = p.parse_args()
a.report.write_text(json.dumps({"passed": False, "reason": "runtime check has not completed"}) + "\n")
result = subprocess.run([sys.executable, "-m", "k2hv_sglang.check_runtime"], capture_output=True, text=True)
a.report.with_suffix(".raw.log").write_text(result.stdout + result.stderr)
result.check_returncode()
report = json.loads(result.stdout[result.stdout.index("{") :])
assert report["reference_revision"] == "b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf"
assert report["verified_source_files"] == 19 and report["reasoning_parser"] == "k2_horizon"
a.report.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
