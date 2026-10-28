import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_pinned_upstream_discovers_package_without_explicit_adapter_imports():
    source = Path(os.environ.get("K2HV_SGLANG_SOURCE", "third_party/sglang-source")).resolve()
    if not source.is_dir():
        pytest.skip("Set K2HV_SGLANG_SOURCE to the pinned sglang/srt source directory")
    from k2hv_sglang.runtime import REGISTRATION_ENVIRONMENT

    env = dict(os.environ, **REGISTRATION_ENVIRONMENT, HF_HUB_OFFLINE="1")
    result = subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("sglang_discovery_probe.py")), str(source)],
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
