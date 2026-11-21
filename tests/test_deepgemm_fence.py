import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/gates/apply_deepgemm_fence.py"


@pytest.mark.parametrize("mode", ["apply", "revert", "check"])
def test_unknown_header_is_refused_and_left_untouched(tmp_path, mode):
    header = tmp_path / "sm120_bf16_gemm.cuh"
    header.write_text("not the pinned header\n")
    result = subprocess.run([sys.executable, str(SCRIPT), mode, "--header", str(header)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "refusing" in result.stderr
    assert header.read_text() == "not the pinned header\n"


def test_patch_only_adds_the_upstream_fence():
    hunk = SCRIPT.with_name("deepgemm-fence.patch").read_text().splitlines()[3:]
    assert not [line for line in hunk if line.startswith("-")]
    assert [line[1:].strip() for line in hunk if line.startswith("+") and line[1:].strip()][-1] == "cutlass::arch::fence_view_async_shared();"
