#!/usr/bin/env bash
set -euo pipefail
cd /workspace
source .venv-container/bin/activate
export UV_NO_SYNC=1 K2HV_MILES_GATE=/workspace/outputs/step04/miles-sft
export SGLANG_EXTERNAL_MODEL_PACKAGE=k2hv_sglang
export SGLANG_EXTERNAL_MM_MODEL_ARCH=K2HorizonVForConditionalGeneration
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=k2hv_sglang
export WANDB_MODE=disabled
exec python third_party/miles/train_async.py \
    --train-backend fsdp --actor-num-nodes 1 --actor-num-gpus-per-node 1 \
    --hf-checkpoint /workspace/hf/K2-Horizon-V-init --load /workspace/hf/K2-Horizon-V-init \
    --prompt-data /workspace/outputs/step04/sft-fixture/fixture.parquet --input-key messages \
    --multimodal-keys '{"image":"images"}' --rollout-function-path k2hv.sft_rollout.generate_rollout \
    --rollout-shuffle --seed 0 --num-rollout 256 --rollout-batch-size 4 --global-batch-size 4 --micro-batch-size 4 \
    --loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns --debug-train-only \
    --attn-implementation sdpa --true-on-policy-mode --gradient-checkpointing \
    --optimizer adam --lr 0.001 --lr-decay-style constant --weight-decay 0 \
    --save /workspace/outputs/step04/miles-sft/checkpoints --save-interval 256 "$@"
