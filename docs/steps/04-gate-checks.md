# Step 4: GPU gates

Complete these checks before full SFT or RL. The authorized 64-example projector overfit is part of this step. CPU regressions alone cannot certify it. Results and machine-only artifact locations are in [the handoff](../handoff/04-gate-checks.md) and [the structured report](../evidence/step04/report.json).

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

Use the exact image, source revisions and patch recipes in the handoff. Preserve the image's CUDA torch stack. Activate the child GPU venv, use editable `--no-deps` installs and `UV_NO_SYNC=1`; do not sync the development lock. TP=1, BF16, one GPU, context 2048, max running requests 1 and KV capacity 4096 bound the native runs. Heavy HF, Miles and SGLang stages run sequentially.

The supported SM120 profile selects native SGLang `torch_native` text attention and `sdpa` vision attention. HF uses SDPA math. Both use the pinned SGLang batch-invariant matrix kernels with BMM override disabled. `TORCH_COMPILE_DISABLE=1` preserves the archived vision RoPE's eager rounding. This is the native scheduler/KV/model path, including external model and processor registration and `k2_horizon` reasoning parsing. It does not dispatch requests to an HF inference engine.

The native adapter preserves the source Conv3d patch embedding, vision Q/K normalization epsilon and BF16 rounding boundaries. The K2 decoder preserves HF RoPE and materialized SiLU rounding. Source audits retain the original audited interfaces and include the newly used attention/update interfaces. A small pinned tokenizer-manager patch invokes the optional adapter validator before the image-free fast path; image markers without media must fail there as well.

For Miles, the narrow `k2_horizon_v` adapter passes explicit document boundaries into document-sized SDPA calls within a truly concatenated decoder row. Linear layers remain packed. This avoids shape-dependent BF16 attention backward reductions. FP32 projector master leaves accumulate gradients from separate images before any BF16 rounding; forward weights and activations remain BF16. The pinned Miles patch registers this adapter and lets `apply_fsdp2` honor its per-module precision policy. The generic adapter's controlled same-shape isolation result remains a separate diagnostic.

Tolerances are fixed: stock/HF logits and source/HF vision outputs exact; pixels, grids, IDs, counts, offsets, native embedding insertion exact; HF/native tower, projector and HF embedding insertion `atol=0.05, rtol=0.03`; token logprobs max error 0.2 and mean error 0.03 natural-log units; packed loss absolute error 0.02 and projector gradient `atol=0.005, rtol=0.03`. Image-pad prediction targets are excluded from logprob comparisons because their IDs become hashes. Every retained target ID and causal prediction position is checked. Thresholds are never adjusted in response to results.

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

First verify the archive and source patches. Download only the assets allowed in the processor handoff; retain current tracked HF code/configs and the original projector. Bootstrap once with `tools/step04/activate.sh`, not during a running training/serving stage.

```bash
docker exec k2hv-step04 bash tools/step04/activate.sh python -m k2hv_sglang.check_runtime
gpu python tools/step04/verify_archive.py --report outputs/step04/final/archive.json
gpu python -m k2hv.checkpoint_io manifest hf/K2-Horizon-V-init
gpu python tools/step04/probe.py > outputs/step04/final/cuda-probe.json
gpu python -m k2hv.gpu_gates --report outputs/step04/final/hf.json
gpu python -m k2hv.miles_packing_gate --report outputs/step04/final/packing.json
gpu python -m k2hv.text_prefix_diagnostic --report outputs/step04/final/prefix-diagnostic.json
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

For a bounded resume, add `--load /workspace/outputs/step04/final/miles-checkpoints`, increase `--num-rollout` (for example to 528 for one more pass over 64 samples), and use a separate save directory. Miles restores optimizer, RNG and data-source state. The criterion is every example, not an average. Continue bounded resumes if needed and preserve failed measurements. The generic question asks only for color and shape; 4 colors × 4 shapes × 4 locations produce 64 varied images without answer leakage. Actual Miles-prepared tensors and decoded answer-only spans are captured. The answer and turn end are supervised; prompt, image and empty-thinking tokens are excluded. Only the frozen vision tower uses `no_grad`; the frozen decoder passes gradients to the projector. Experiment checkpoints are separate from init.

Finally run the CPU regressions and assemble every required component:

```bash
K2HV_SGLANG_SOURCE=$PWD/third_party/sglang-source/python/sglang/srt \
  env -u LD_LIBRARY_PATH .venv-gpu/bin/python -m pytest -q \
  --junitxml=outputs/step04/final/regression.xml > outputs/step04/final/pytest.log 2>&1
gpu python -m k2hv.step04_report --root outputs/step04/final \
  --report outputs/step04/final/report.json
```

Provenance records a source census. Assembly verifies the model/processor/training compute sources and each stage's actual runner; an edit to an unused evidence CLI does not change a completed computation. Pinned external source audits remain mandatory. The aggregate runner rejects missing/unrun components, failed checks and stale executed source hashes, and returns nonzero until all required gates pass. Component success is not whole-step completion. Large tensor traces, image payloads, checkpoints and logs stay in ignored `outputs/`; commit the sanitized report and compact evidence only.
