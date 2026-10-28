#!/usr/bin/env bash
set -euo pipefail
cd /workspace
source .venv-container/bin/activate
export UV_NO_SYNC=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=k2hv_sglang
export SGLANG_EXTERNAL_MM_MODEL_ARCH=K2HorizonVForConditionalGeneration
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=k2hv_sglang
exec python -m sglang.launch_server --model-path /workspace/hf/K2-Horizon-V-init \
    --trust-remote-code --enable-multimodal --image-processor-backend pil --reasoning-parser k2_horizon \
    --tp-size 1 --dtype bfloat16 --attention-backend triton --mm-attention-backend sdpa \
    --context-length 2048 --chunked-prefill-size 256 --max-running-requests 1 --max-total-tokens 4096 \
    --random-seed 0 --mem-fraction-static 0.65 --disable-cuda-graph --host 127.0.0.1 --port 30000
