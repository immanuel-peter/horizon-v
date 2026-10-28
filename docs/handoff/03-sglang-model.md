# SGLang integration handoff

`src/k2hv_sglang/` holds the external model and processor. CPU tests cover the adapter boundaries. Nothing has run on a GPU yet: no SGLang engine, CUDA kernels, full-size vision parity or logprob parity. Training has not run either.

## Runtime pin

Miles at `5078ff7` installs SGLang from the `sglang-miles` branch and only pins it if the build sets `SGLANG_COMMIT`. This adapter targets [sgl-project/sglang at b8d46f2](https://github.com/sgl-project/sglang/tree/b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf). The reference Docker base is `lmsysorg/sglang:v0.5.22` and the project requires `transformers==5.19.0`.

`runtime.py` records SHA-256s of the SGLang source files this package depends on. Audit a source tree without importing SGLang:

```bash
uv run python -m k2hv_sglang.check_runtime --source-dir /path/to/sglang/python/sglang/srt
```

On a GPU machine, use the Miles image and its CUDA stack from [the environment guide](../steps/00-environment.md). Build the image with `SGLANG_COMMIT=b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf`, or move an existing checkout to that commit before serving:

```bash
git -C /sgl-workspace/sglang fetch origin sglang-miles
git -C /sgl-workspace/sglang checkout b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf
SGLANG_BUILD_RUST_EXTS=none uv pip install --no-deps -e /sgl-workspace/sglang/python
uv pip install --no-deps -e .
```

Keep the GPU venv activated with `UV_NO_SYNC=1`, since the image provides the CUDA dependencies. Without `--source-dir`, `check_runtime` checks the installed source hashes, the Transformers version, the external class registration and the reasoning parser. Run it in that GPU environment before launching the engine.

## Checkpoint and launch

Use the tracked checkpoint files from this checkout. The pinned private HF archive is still the step 1 snapshot. Download only its generated weights, index, tokenizer and build metadata, following the [processor handoff](02-processor-and-template.md#checkpoint-integration), because a full snapshot download overwrites the newer processor configs. Refresh `SHA256SUMS` after attaching the weights. This integration changes no checkpoint weights.

Set the three registration variables before importing SGLang. The launch commands in [step 7](../steps/07-eval.md) and [step 9](../steps/09-release.md) assume them.

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

Start with one GPU, tensor parallel size 1 and the unquantized checkpoint. The model rejects quantization and the data-parallel vision-encoder path. Tensor and pipeline parallelism, speculative decoding, encoder disaggregation and precomputed-embedding inputs are unvalidated.

SGLang's K2 decoder handles text attention, grouped RMSNorm, 1D positions and logits. The wrapper adds the embedding interface that `prepare_mm_inputs` expects. The GLM vision tower stops after `post_layernorm`, and its downsample and merger parameters are removed. The projector is the HF `Projector` class. `load_weights` routes contiguous groups of checkpoint tensors by component: language tensors go to the K2 loader and vision/projector tensors load directly. Interleaved inputs stream without being collected into a list.

SGLang's parser for this model is `k2_horizon`, not the RL360 selector `k2_v3`. It supports all three IFM think-tag variants. Miles's `/generate` path on raw tokens needs no parser.

## Image/token contract

The processor loads media with SGLang's loader (GPU JPEG decoding off), then preprocesses with the HF processor's PIL path. Decoding, resizing, normalization, token budget and patch order match the HF path. The default cap is 1024 image tokens. Server image-process config may override only `min_image_tokens` and `max_image_tokens`, and training must use the same budget.

The processor encodes text prompts once and never decodes or re-tokenizes token-ID inputs. Every image needs a complete start/pad/end block. A single pad expands to the grid's feature count, and an already-expanded block must match that count. Other token IDs pass through unchanged. Each image gets its own `MultimodalDataItem` with a patch slice, a `[1, 3]` grid and inclusive token offsets. K2 uses 1D positions, so the processor returns no M-RoPE metadata.

Image hashes cover the grid and the patch bytes. Without the grid, uniformly colored images with the same patch count but different aspect ratios would share a hash despite different positional embeddings. `MultiModalityDataPaddingPatternMultimodalTokens` replaces each image's pad positions with that item's cache value and keeps the image boundaries.

Every `load_weights` call clears the projected-image embedding cache, including projector-only updates. The raw media cache stays. Synchronize weight updates with requests and flush the KV cache through the scheduler's normal path. Whether cached and fresh outputs match after a projector update is still a GPU gate.

Raw image inputs work. Video, audio, processor-output dictionaries, image frame lists and precomputed embeddings raise errors. Preprocessing runs in the processor's I/O executor. The generic processor-clone pool is off because this adapter owns its preprocessing.

## Validation

All CPU tests pass with the pinned source files present. They replace the SGLang runtime classes with doubles, so they do not show that CUDA execution works.

A fresh-process regression runs the pinned upstream registry and processor-discovery code without importing the adapter modules. SGLang scans the package's submodules, so the package root neither re-exports `EntryClass` nor imports GPU-dependent code. Set `K2HV_SGLANG_SOURCE` to an `sglang/srt` directory to run it elsewhere. It defaults to the git-ignored `third_party/sglang-source` and skips when no source exists. The GPU-side `check_runtime` probe also avoids importing the adapter before checking registration.

The tests cover unexpanded and Miles-expanded prompts, preserved input IDs, HF pixel/grid parity, per-image offsets, mixed image sizes, geometry-aware hashes, truncated-tower output layout, 1D decoder positions, weight-name mapping, projector-only updates, image-cache invalidation, and errors for invalid inputs or unsupported configuration.

On CPU I also ran the pinned source-hash audit, the upstream model/processor registration functions, the image-padding code, and the reasoning detector (all three effort levels, whole and in one-character streaming chunks). None of this imports or starts the real engine.

## GPU smoke check and gates

After the server starts:

```bash
uv run python -m k2hv_sglang.smoke \
  --model-path /path/to/horizon-v/hf/K2-Horizon-V-init \
  --url http://127.0.0.1:30000 --report outputs/sglang/smoke.json
```

The client sends a text `/generate` request, a Miles-style pre-expanded image request and an OpenAI-style image chat request. It checks that each response exists, that the HF processor yields 256 image tokens for 448x448, and that the server counts the same prompt length. Passing shows request compatibility, not image understanding or numerical parity.

Before training, complete [the gate checks](../steps/04-gate-checks.md):

- text-only parity with stock K2
- truncated-vision and projected-feature parity with HF
- token-count and pixel/grid parity across real request paths
- chunked prefill, multiple images and cache hits
- HF/SGLang logprob parity
- cache invalidation after a projector-only update

Packing and the 64-example overfit remain training-side gates. Record the image, SGLang revision, package versions, attention backend, dtype and tolerances with the results.
