# Step 04 GPU handoff

Step 04 is complete. Every required gate passed on the local SM120 GPU through the intended HF, native SGLang and real Miles FSDP paths. [The sanitized report](../evidence/step04/report.json) records results, execution commits, source hashes, thresholds and artifact hashes. [Exact rerun commands](../steps/04-gate-checks.md) are separate from the fast CPU suite.

## Checkout and runtime

Resumed PR #4 on `step04-gpu-gates` at `a9d5195a14b6a52bd82e545e1078771a0712a157`, preserving the checkout and cached assets. A fresh fetch still resolved main to `2b00cf5f32e66741dbe36316c5e785e66ac99ab2`. No AGENTS.md was present. The final from-init training used model code at `4febe97f52dab29680747ec6c4811b5ce6f3bd69`; later commits add verification utilities and fix snapshot selection, without changing model computation. Per-stage execution commits and matching compute-source hashes are in the report.

Rechecked hardware: RTX PRO 6000 Blackwell Server Edition, SM120, 97887 MiB, driver 595.112.07; x86_64, 16 CPUs, 124 GiB RAM. This is neither Hopper nor SM100.

- Image: `lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae`.
- Python 3.12.15, torch 2.14.1+cu130, CUDA 13.0, Transformers 5.19.0, Triton 3.8.0, FlashInfer 0.7.1, flash-attn-4 4.0.0b35; FA2/FA3 absent.
- Miles `5078ff7cd6387f553db010bb38148a59e9e12cb0` plus the committed registration/per-module precision patch; nine source files audited.
- SGLang `b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf` plus the committed optional tokenizer validation hook; 19 source files audited. No newer revision or CUDA dependency replacement was needed.
- HF SDPA math, native text `torch_native`, native vision `sdpa`; pinned batch-invariant matrix kernels, BMM override disabled, eager vision RoPE. Observed launch profiles confirm TP=1, unquantized BF16, context 2048, chunk sizes 256/-1, max running requests 1, KV capacity 4096, PIL and reasoning parser `k2_horizon`.

The child venv explicitly exposes the image's `/opt/sglang` packages. Bootstrap with `tools/step04/activate.sh`, the pinned CPU/orchestration requirements and `--no-deps` editable installs; use `UV_NO_SYNC=1`. Do not sync the development lock into this environment. Build the pinned Rust tree-cache extension with `SGLANG_RUST_BUILD_MODE=auto` and Cargo on PATH when recreating the container. The retained container already has that build. Host commands need `env -u LD_LIBRARY_PATH` to avoid the inherited cuDNN conflict.

## Results

| Gate | Actual result |
|---|---|
| CUDA/load | Passed BF16 matmul and attention backward; full HF composite; audited native engine; text, Miles pre-expanded image and chat image_url smoke requests |
| Stock/HF text | Exact logits at 12 and 769 tokens |
| Native text/logprobs | Passed with and without chunking; worst error 3.45611e-6, mean long-text error 1.01542e-6, versus unchanged limits 0.2/0.03 |
| HF vision/source | Exact truncated output versus independent pinned source post_layernorm hook, five geometries |
| Native vision/preprocessing/insertion | All 12 expanded/raw cases passed; pixels, grids, tower outputs, projected features and decoder embeddings exact; counts, original text IDs, hash offsets and full insertion coverage checked; three malformed cases rejected with HTTP 400 |
| Packing | Actual Miles get_batch → FSDP2 → actor inputs → SFT loss, packed length 128 versus unpadded length-39 samples: loss error 1.58946e-7; all six projector gradients exact. Controlled same-shape isolation is separately reported |
| Cache/update | Real pause/begin/tensor-update/end/continue worker session, KV flush, repeated/multiple images and identical pixels with different grids passed. Frozen digest unchanged; feature change 0.2509766; fresh-engine features/logprobs exact; init weights/features reloaded |
| Real Miles overfit | Fresh 512-update run from init: batch loss 4.66667 → 7.94729e-8; every example <0.05, worst 1.58946e-7; teacher forcing 64/64 and free generation 64/64 |
| Training invariants | Six projector tensors trainable; all 512 gradient norms finite/nonzero; 2,048 image-feature backward checks passed; all 666 frozen FP32 checkpoint tensors exactly equal promoted init values before downcast |
| CPU regression | 163 passed, separate from GPU certification |

Peak GPU allocations: native 22.41 GB, SFT 44.38 GB, full packing verification 75.50 GB. Heavy stages ran sequentially. No tolerance was increased. No required gate remains failed, blocked or unrun.

The [optimizer loss CSV](../evidence/step04/miles-loss.csv), [64-example losses and generations](../evidence/step04/overfit.csv), and [fixture dimensions/hashes](../evidence/step04/fixtures.json) are committed. Actual Miles inputs contain 40 tokens, including 16 image tokens and three supervised answer/end tokens. Decoded spans include, for example, `blue circle<|ifm|im_end|>`; prompt, image and empty-thinking tokens are excluded. Only vision runs under no_grad; frozen LM operations propagate gradients to image features.

## Diagnoses

[Retained before/after evidence](../evidence/step04/diagnosis.json) explains the original failures. Long-text divergence was already present at position 2 and survived disabling chunks. Shape-dependent BF16 kernels affected HF prefixes themselves. Batch-invariant matrices and supported native Torch attention remove the gap without changing alignment or thresholds.

Vision first diverged at the native Conv3d-to-linear optimization; Q/K RMSNorm also used 1e-6 instead of source 1e-5. Preserving convolution, epsilon, eager RoPE and BF16 norm/activation rounding makes native features exact. Explicit document-sized SDPA fixes packing's BF16 backward shape dependence, with FP32 projector master leaves accumulating image gradients.

The pinned CUDA mean override treats an empty dimension list incorrectly for instrumentation. Error metrics therefore use detached CPU FP32 tensors; the actual supervised objective is explicitly token-weighted. The real packing mismatch remained after correcting that measurement and was then fixed. A constant-color cache fixture was replaced because constant values can legitimately yield identical outputs on different grids. A final verifier bug selected a stale snapshot from unordered directory enumeration; latest-capture selection and a reversed-order regression fix it. The complete update/fresh/reload cycle was rerun successfully. Original failures remain in ignored logs.

## Persistent assets and rerun

Run `docker start k2hv-step04` on this machine, or recreate the container:

```bash
docker run --name k2hv-step04 --gpus device=0 --ipc=host -d \
  --mount type=bind,src="$PWD",dst=/workspace -w /workspace \
  lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae sleep infinity
docker exec k2hv-step04 git config --global --add safe.directory '*'
docker exec k2hv-step04 bash tools/step04/activate.sh python -m k2hv_sglang.check_runtime
```

Clone radixark/miles and sgl-project/sglang under the ignored `third_party/` at the pins above. The patch recipes check the revision, patch digest and working-file hashes, and refuse unexpected edits. The [step guide](../steps/04-gate-checks.md) has the sequential commands.

The archive revision is still `fc1f1c887facf728e2e02a13f9bf9a920af44552`. After the experiments, all seven downloaded assets matched the separate archive manifest. The tracked HF code, configs and original random projector stay attached, and the combined 23-file local manifest was regenerated. Independent sources stay pinned to IFM/K2-Horizon-7B `4622f94a6632384784471fd674d5b78f8d527798` and immanuelpeter/GLM-5.3-Flash-Vision `44a87fd4e6e31f5722a375238b7aedce3e0ad2db`.

`report.json` summarizes the machine-only artifacts under `outputs/step04/final/` (file count, total bytes and a digest of the sorted path and SHA256 list). The final model file is `outputs/step04/final/miles-checkpoints/iter_0000512/model/__0_0.distcp` (37,748,956,064 bytes, SHA256 `bf58f6a3962a59e9e5b08671e3b86ccc0104f4adbfce7b7817dd688f6b0caf2f`). Full raw reports and reference payloads stay in the same directory.

Additional machine-only assets: init weight assets under `hf/`, source checkpoints under `models/`, both venvs, exact source checkouts, Docker image/container and Rust cache; earlier diagnostic logs and integration/warmup/resume checkpoints under `outputs/step04/miles-sft/{checkpoints,final-checkpoints,resumed-checkpoints,pre-final}`; older cache experiments under `outputs/step04/cache-debug2/`. These historical checkpoints are retained, not substituted for the final proof. Experimental projectors are separate; no model was uploaded. Repository visibility was unchanged (GitHub already reported PUBLIC; the HF archive remains private). Full SFT, RL and bulk preparation were not run.

The task servers and container are stopped, and the container is kept. The Brev instance is still running.
