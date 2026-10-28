#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export PATH="/root/.local/bin:$PATH"
export UV_NO_SYNC=1 SGLANG_BUILD_RUST_EXTS=none
[[ $(git -C third_party/miles rev-parse HEAD) == 5078ff7cd6387f553db010bb38148a59e9e12cb0 ]]
[[ $(git -C third_party/sglang-source rev-parse HEAD) == b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf ]]
runtime_site=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
if [[ ! -x .venv-container/bin/python ]]; then
    uv venv --system-site-packages --python "$(command -v python3)" .venv-container
fi
source .venv-container/bin/activate
venv_site=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
printf '%s\n' "$runtime_site" "$runtime_site/nvidia_cutlass_dsl/dsl_packages" > "$venv_site/image-runtime.pth"
uv pip install --no-deps -e third_party/miles -e .
uv pip install setuptools setuptools-rust setuptools-scm wheel
uv pip install --no-deps --no-build-isolation -e third_party/sglang-source/python
export SGLANG_EXTERNAL_MODEL_PACKAGE=k2hv_sglang
export SGLANG_EXTERNAL_MM_MODEL_ARCH=K2HorizonVForConditionalGeneration
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=k2hv_sglang
exec "$@"
