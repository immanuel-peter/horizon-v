# SGLang integration handoff

The external model and processor are implemented in `src/k2hv_sglang/`. CPU tests cover the adapter boundaries. An actual SGLang engine, CUDA kernels, full-size vision parity and logprob parity have not been run on this Mac. Training has not run.

## Runtime pin

Miles at `5078ff7` uses the `sglang-miles` branch, with an optional `SGLANG_COMMIT` build argument. It does not pin that branch by default. This implementation targets [sgl-project/sglang at b8d46f2](https://github.com/sgl-project/sglang/tree/b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf), resolved while implementing this step. The reference Docker base is `lmsysorg/sglang:v0.5.22` and the project requires `transformers==5.19.0`.

`runtime.py` records SHA-256s for 17 source files defining the interfaces used here. Audit an existing source tree without importing SGLang:

```bash
uv run python -m k2hv_sglang.check_runtime --source-dir /path/to/sglang/python/sglang/srt
```

On a GPU machine, use the Miles image and its CUDA stack as described in [the environment guide](../steps/00-environment.md). Pin the image build with `SGLANG_COMMIT=b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf`, or update its existing SGLang checkout before serving:

```bash
git -C /sgl-workspace/sglang fetch origin sglang-miles
git -C /sgl-workspace/sglang checkout b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf
SGLANG_BUILD_RUST_EXTS=none uv pip install --no-deps -e /sgl-workspace/sglang/python
uv pip install --no-deps -e .
```

Keep the GPU venv activated with `UV_NO_SYNC=1`; the image provides the CUDA dependencies. `check_runtime` without `--source-dir` checks the installed source hashes, Transformers version, external class registration and reasoning-parser availability. It is intended for that GPU environment and is not a replacement for launching the engine.

## Checkpoint and launch

Use the current tracked checkpoint files from this checkout. The pinned private HF archive is still the step 1 snapshot; download only its generated weights, index, tokenizer and build metadata, following the [processor handoff](02-processor-and-template.md#checkpoint-integration). A full old snapshot download would overwrite the newer processor configs. Refresh `SHA256SUMS` after attaching the weights. No checkpoint weights are changed by this integration.

Set all three external registration variables before importing SGLang:

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=k2hv_sglang
export SGLANG_EXTERNAL_MM_MODEL_ARCH=K2HorizonVForConditionalGeneration
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=k2hv_sglang

uv run python -m k2hv_sglang.check_runtime
uv run python -m sglang.launch_server \
  --model-path /path/to/horizon-v/hf/K2-Horizon-V-init \
  --trust-remote-code --enable-multimodal --image-processor-backend pil \
  --reasoning-parser k2_horizon --tp-size 1 --port 30000
```

Start with one GPU, tensor parallel size 1 and the unquantized checkpoint. Quantization and the data-parallel vision-encoder path are explicitly rejected. Tensor/pipeline parallel execution, speculative decoding, encoder disaggregation and precomputed-embedding inputs are outside this step's validated scope.

The native K2 decoder handles text attention, grouped RMSNorm, 1D positions and logits. The wrapper supplies the embedding interface expected by `prepare_mm_inputs`. The native GLM vision tower stops after `post_layernorm`; its original downsample and merger parameters are absent. The projector is the same `Projector` class used by HF. Checkpoint loading routes contiguous component groups explicitly: language tensors go to the native K2 loader, while vision/projector tensors load directly. This preserves streaming for interleaved inputs without accumulating the checkpoint in a list or relying on non-language side effects inside the language iterator.

The pinned SGLang parser is named `k2_horizon`, not the older RL360 selector `k2_v3`. It already supports all three IFM think-tag variants. No parser is needed on the raw `/generate` path used by Miles.

## Image/token contract

The processor uses SGLang's native media loader with GPU JPEG decoding disabled, then GLM's PIL preprocessing from the HF processor. This keeps decoding, resizing, normalization, token budget and patch ordering consistent with the HF path. The default cap remains 1024 image tokens. Server image-process config may override only `min_image_tokens` and `max_image_tokens`; the training side must use the same budget.

Text prompts are encoded once. Token-ID inputs are neither decoded nor re-tokenized. Every image must have a complete start/pad/end block. A single pad is expanded to the grid's feature count; an already-expanded block must match that count exactly. Non-image token IDs are preserved. Each image gets its own `MultimodalDataItem`, patch slice, `[1, 3]` grid and inclusive token offsets. K2 uses ordinary 1D positions, so no M-RoPE metadata is returned.

Image hashes include the grid as well as normalized patch bytes. Otherwise uniformly colored images with equal patch counts but different aspect ratios could share a hash even though their positional embeddings differ. SGLang's native padding pattern replaces each image's pad positions with that item's cache value while preserving image boundaries.

Projector-only `load_weights` calls update only projector parameters and clear the projected-image embedding cache. The raw media/preprocessing cache does not need clearing. Weight updates must be synchronized with requests and must use the scheduler's normal KV-cache flush. Cached-versus-fresh output equivalence after a projector update remains a GPU gate.

Raw image inputs are supported. Video, audio, processor-output dictionaries, image frame lists and precomputed embeddings are rejected. CPU preprocessing runs in the processor's I/O executor; the generic multimodal processor-clone pool is disabled because this adapter owns its preprocessing path.

## Validation

All 150 CPU tests pass locally with the pinned source files present, including the existing model/processor suite and the token-layout and runtime-boundary tests. The SGLang runtime classes are replaced with explicit doubles in those adapter tests; this does not prove that native CUDA execution works.

The fresh-process discovery regression executes the pinned upstream registry and processor-discovery code without explicitly importing the adapter modules. The pinned runtime scans package submodules, so the package root need not re-export `EntryClass` or import GPU-dependent code. Set `K2HV_SGLANG_SOURCE` to an `sglang/srt` source directory to run this regression elsewhere; it defaults to the ignored `third_party/sglang-source` audit directory and skips when no source tree is available. The GPU-side runtime probe also avoids explicit adapter imports before checking registration.

The tests cover unexpanded and Miles-expanded prompts, preservation of input IDs, exact HF pixel/grid parity, per-image offsets, mixed image sizes, geometry-aware hashes, truncated-tower output layout, 1D decoder positions, native weight-name mapping, projector-only updates, image-cache invalidation and errors for invalid inputs or unsupported configuration.

The 17 pinned source hashes were checked. The actual upstream model/processor registration functions and image-padding implementation were also exercised on CPU with runtime boundary doubles. The upstream reasoning detector was exercised directly for all three effort levels, both non-streaming and with one-character streaming chunks. Those source-level checks are separate from importing or starting the real engine.

## GPU smoke check and gates

After the server starts, run:

```bash
uv run python -m k2hv_sglang.smoke \
  --model-path /path/to/horizon-v/hf/K2-Horizon-V-init \
  --url http://127.0.0.1:30000 --report outputs/sglang/smoke.json
```

This client sends a text `/generate` request, a Miles-style pre-expanded-ID image request and an OpenAI-style image chat request. It checks that responses exist, that the HF image count is 256 for 448x448 and that the Miles prompt length is unchanged. The client and GPU server have not been run here. A successful smoke check establishes request compatibility, not image understanding or numerical parity.

Before training, complete [the gate checks](../steps/04-gate-checks.md): native text-only parity with stock K2; native truncated-vision and projected-feature parity with HF; token-count and pixel/grid parity across actual request paths; chunked prefill, multiple images and cache hits; HF/SGLang logprob parity; and cache invalidation after a projector-only update. Packing and the 64-example overfit remain training-side gates. Record the actual image, SGLang revision, package versions, attention backend, dtype and tolerances with those results.
