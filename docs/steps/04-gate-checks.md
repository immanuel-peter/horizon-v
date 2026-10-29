# Step 4: GPU gates

Run these checks before full SFT or RL. The 64-example projector overfit is part of the step, and CPU regressions alone cannot certify it. Results are in [the handoff](../handoff/04-gate-checks.md) and [the structured report](../evidence/step04/report.json).

| Gate | Required measurement |
|---|---|
| CUDA/load | Real SM120 BF16 matmul and attention backward; HF composite; audited native SGLang; text, Miles input IDs plus image data, and chat image URLs |
| Text | Stock K2 versus image-free composite logits; aligned HF/native token logprobs at short and 769-token lengths; chunked and unchunked prefill |
| Vision | Independent pinned source `post_layernorm` versus HF truncation; actual native truncation and projected features versus HF |
| Preprocessing/insertion | Exact pixels, grids, patch order, original text IDs, expansion counts, per-image hash-padding offsets and decoder embedding insertion; malformed requests rejected |
| Packing | Actual Miles `get_batch`, FSDP2 actor inputs and SFT loss; concatenated documents with position resets versus unpadded separate forwards; projector gradients and document isolation |
| Cache/update | Repeated/multiple images, equal patch tensors with different grids; supported scheduler/worker projector update; feature invalidation; fresh updated engine equivalence and init reload |
| Overfit | Real Miles FSDP SFT on 64 deterministic image-QA examples; every per-example loss <0.05; teacher forcing and free generation measured separately; frozen tensors unchanged |

## Runtime and numerical policy

Use the image, source revisions and patch recipes from the handoff. Keep the image's CUDA torch stack. Activate the child GPU venv, install editable with `--no-deps`, set `UV_NO_SYNC=1`, and never sync the development lock.

The native runs use TP=1, BF16, one GPU, context 2048, max running requests 1 and KV capacity 4096. Run the heavy HF, Miles and SGLang stages one at a time.

- SGLang uses `torch_native` text attention and `sdpa` vision attention. HF uses SDPA math. Both use the pinned batch-invariant matrix kernels with the BMM override off.
- `TORCH_COMPILE_DISABLE=1` keeps the archived vision RoPE in eager rounding.
- The native path covers the scheduler, KV cache, model, processor registration and the `k2_horizon` reasoning parser. It never sends requests to an HF inference engine.
- The native adapter keeps the source Conv3d patch embedding, the vision Q/K normalization epsilon and BF16 rounding. The K2 decoder keeps HF RoPE and materialized SiLU rounding.
- Source audits cover the original interfaces plus the attention and update interfaces added in this step.
- A small pinned tokenizer-manager patch calls the adapter validator before the image-free fast path, so image markers without media fail there too.

Miles uses the `K2HorizonVAdapter`, which passes document boundaries into document-sized SDPA calls inside one concatenated decoder row. Linear layers stay packed. This removes the shape dependence of BF16 attention backward reductions. FP32 projector master leaves sum gradients from separate images before any BF16 rounding, while forward weights and activations stay BF16. The pinned Miles patch registers the adapter and lets `apply_fsdp2` honor its per-module precision. The generic adapter's same-shape isolation result stays a separate diagnostic.

### Tolerances

| Quantity | Limit |
|---|---|
| Stock/HF logits, source/HF vision outputs | Exact |
| Pixels, grids, IDs, counts, offsets, native embedding insertion | Exact |
| HF/native tower, projector and HF embedding insertion | `atol=0.05`, `rtol=0.03` |
| Token logprob error (natural log) | max 0.2, mean 0.03 |
| Packed loss | absolute error 0.02 |
| Projector gradient | `atol=0.005`, `rtol=0.03` |

Image-pad prediction targets are excluded from logprob comparisons because their IDs become hashes. Every other target ID and causal prediction position is checked. Never change a limit to fit a result.

## Rerun commands

Use the retained container, or recreate it from the exact digest and bind mount in the handoff. The following helper runs commands in its activated venv without synchronizing dependencies:

```bash
export UV_NO_SYNC=1
mkdir -p outputs/step04/final
gpu() {
  docker exec k2hv-step04 bash -c \
    'source .venv-container/bin/activate; export UV_NO_SYNC=1; exec "$@"' bash "$@"
}
```

Verify the archive and source patches first. Download only the assets the processor handoff allows, and keep the tracked HF code, configs and original projector. Run `tools/step04/activate.sh` once to bootstrap, never during a training or serving stage.

```bash
docker exec k2hv-step04 bash tools/step04/activate.sh python -m k2hv_sglang.check_runtime
gpu python tools/step04/verify_archive.py --report outputs/step04/final/archive.json
gpu python -m k2hv.checkpoint_io manifest hf/K2-Horizon-V-init
gpu python tools/step04/probe.py > outputs/step04/final/cuda-probe.json
gpu python -m k2hv.gpu_gates --report outputs/step04/final/hf.json
gpu python -m k2hv.miles_packing_gate --report outputs/step04/final/packing.json
```

The HF component reports `packing_controlled` separately. The full packing gate is `miles_packing_gate`, which uses real FSDP2 and unpadded separate samples. None of these commands starts a native engine.

Capture HF references, then release the process before launching SGLang:

```bash
gpu python -m k2hv_sglang.logprob_gate capture --batch-invariant \
  --reference outputs/step04/final/text-reference.json
gpu python -m k2hv_sglang.native_gates capture --directory outputs/step04/final/native
gpu python tools/step04/capture_runtime.py --report outputs/step04/final/runtime.json

docker exec -e K2HV_GATE_TRACE=/workspace/outputs/step04/final/native-trace \
  -e K2HV_GATE_WEIGHT_STATE=1 k2hv-step04 bash tools/step04/serve.sh
# Wait for "The server is fired up"; run clients in another terminal.
gpu python -m k2hv_sglang.smoke --model-path /workspace/hf/K2-Horizon-V-init \
  --report outputs/step04/final/smoke.json
gpu python -m k2hv_sglang.logprob_gate evaluate --reference outputs/step04/final/text-reference.json \
  --report outputs/step04/final/text-chunked.json
gpu python -m k2hv_sglang.native_gates evaluate --directory outputs/step04/final/native \
  --trace outputs/step04/final/native-trace
gpu python -m k2hv_sglang.cache_gate update --directory outputs/step04/final/cache \
  --trace outputs/step04/final/native-trace --cases outputs/step04/final/native
```

Stop only this launched server using `docker exec k2hv-step04 pkill -TERM -f '^python -m sglang.launch_server'`. Launch a fresh engine for the cache experiment:

```bash
docker exec -e K2HV_GATE_TRACE=/workspace/outputs/step04/final/fresh-trace \
  -e K2HV_GATE_WEIGHT_STATE=1 k2hv-step04 bash tools/step04/serve.sh \
  --model-path /workspace/outputs/step04/final/cache/updated-checkpoint
# After readiness, from another terminal:
gpu python -m k2hv_sglang.cache_gate fresh --directory outputs/step04/final/cache \
  --trace outputs/step04/final/fresh-trace --cases outputs/step04/final/native
```

This uses HTTP pause/begin/update/end/continue through the real tokenizer manager, scheduler and worker. The final stage reloads the init checkpoint and verifies its features. Stop that server, then launch `tools/step04/serve.sh --chunked-prefill-size -1` with a new trace directory and run `logprob_gate evaluate --reference outputs/step04/final/text-reference.json --report outputs/step04/final/text-unchunked.json`. Stop it before training.

Prepare only the small gate fixture and run real Miles SFT:

```bash
gpu python tools/step04/prepare_sft.py --directory outputs/step04/sft-fixture
docker exec k2hv-step04 bash tools/step04/miles_sft.sh --num-rollout 512 --save-interval 512 \
  --save /workspace/outputs/step04/final/miles-checkpoints \
  > outputs/step04/final/miles-from-init.log 2>&1
gpu python -m k2hv.verify_frozen_master \
  --checkpoint outputs/step04/final/miles-checkpoints/iter_0000512 \
  --report outputs/step04/final/frozen-master.json
gpu python -m k2hv.miles_sft_evaluate \
  --checkpoint outputs/step04/final/miles-checkpoints/iter_0000512 \
  --report outputs/step04/final/overfit.json
gpu python -m k2hv.miles_training_report --logs outputs/step04/final/miles-from-init.log \
  --report outputs/step04/final/training-summary.json
```

The fixture asks only for color and shape. Four colors, four shapes and four locations give 64 varied images without leaking the answer. The captured Miles tensors and decoded spans show that the answer and turn end are supervised, while prompt, image and empty-thinking tokens are not. Only the frozen vision tower runs under `no_grad`. The frozen decoder passes gradients to the projector. Keep experiment checkpoints separate from init.

To resume, add `--load /workspace/outputs/step04/final/miles-checkpoints`, raise `--num-rollout` (528 adds one pass over the 64 samples) and use a new save directory. Miles restores optimizer, RNG and data-source state. The criterion is every example, not the average. Keep failed measurements.

Finally run the CPU regressions and assemble every required component:

```bash
K2HV_SGLANG_SOURCE=$PWD/third_party/sglang-source/python/sglang/srt \
  env -u LD_LIBRARY_PATH .venv-gpu/bin/python -m pytest -q \
  --junitxml=outputs/step04/final/regression.xml > outputs/step04/final/pytest.log 2>&1
gpu python -m k2hv.step04_report --root outputs/step04/final \
  --report outputs/step04/final/report.json
```

Provenance records a source census. Assembly checks the hashes of the model, processor and training compute sources and of each stage's runner, so editing an evidence CLI that no stage ran leaves finished computations valid. Pinned external source audits stay mandatory. The runner exits nonzero on missing or unrun components, failed checks or stale source hashes. Keep large tensor traces, image payloads, checkpoints and logs in the ignored `outputs/`, and commit only the sanitized report and compact evidence.
