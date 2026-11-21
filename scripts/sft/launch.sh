#!/usr/bin/env bash
# Real-data projector SFT through pinned Miles FSDP, LazyRolloutDataSource and k2hv.sft_rollout.
# Run inside the step 4 container after scripts/gates/activate.sh has bootstrapped .venv-container.
#
# The caller passes the run-specific flags: --prompt-data, --load, --save, --save-interval,
# --num-rollout, --rollout-batch-size, --global-batch-size, --micro-batch-size, --seed,
# --rollout-seed and the LR schedule. K2HV_GPUS sets the actor GPU count (default 1).
# K2HV_SFT_MONITOR=<dir> enables k2hv.sft_monitor instrumentation (no change to the computation).
# It refuses to start unless the DeepGEMM SM120 BF16 header carries the fence that activate.sh applies.
set -euo pipefail
cd /workspace
source .venv-container/bin/activate
# The step 4 capture hooks write fixture tensors; they must never run on real data.
unset K2HV_MILES_GATE
export UV_NO_SYNC=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=k2hv_sglang
export SGLANG_EXTERNAL_MM_MODEL_ARCH=K2HorizonVForConditionalGeneration
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=k2hv_sglang
python scripts/gates/apply_deepgemm_fence.py check
rollout=k2hv.sft_rollout.generate_rollout
if [[ -n ${K2HV_SFT_MONITOR:-} ]]; then
    rollout=k2hv.sft_monitor.generate_rollout
fi
exec python third_party/miles/train_async.py \
    --train-backend fsdp --actor-num-nodes 1 --actor-num-gpus-per-node "${K2HV_GPUS:-1}" \
    --hf-checkpoint /workspace/hf/K2-Horizon-V-init \
    --input-key messages --multimodal-keys '{"image":"images"}' \
    --data-source-path k2hv.lazy_data.LazyRolloutDataSource --rollout-function-path "$rollout" \
    --rollout-shuffle --loss-type sft_loss --calculate-per-token-loss \
    --disable-compute-advantages-and-returns --debug-train-only \
    --attn-implementation sdpa --true-on-policy-mode --gradient-checkpointing \
    --optimizer adam --weight-decay 0 "$@"
