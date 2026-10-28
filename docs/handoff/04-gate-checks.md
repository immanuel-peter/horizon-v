# Step 04 GPU handoff

Step 04 is incomplete. See `docs/evidence/step04/report.json` for sanitized
results. CPU tests and native request compatibility do not establish parity.

## Checkout and hardware

Fetched main and started from `2b00cf5f32e66741dbe36316c5e785e66ac99ab2` with a
clean checkout. Work is on `step04-gpu-gates`. No AGENTS.md was present in the
checkout or its checked parent directories.

The machine was rechecked locally: RTX PRO 6000 Blackwell Server Edition,
SM120, 97887 MiB, driver 595.112.07, x86_64, 16 CPUs, 124 GiB RAM. Initial free
root disk was 317 GB. GPU and Docker access required execution outside the
filesystem sandbox. Both credentials worked in `bash -lic`; neither was passed
as a literal argument or copied into a container. GitHub reported this checkout's
repository PUBLIC. Visibility was not changed. The private HF archive was not
modified or uploaded.

## Runtime

- Miles `5078ff7cd6387f553db010bb38148a59e9e12cb0`.
- SGLang `b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf`, editable with no dependency
  resolution and no build isolation. All ten audited source files match.
- Base `lmsysorg/sglang:v0.5.22` resolved to
  `sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae`.
- Container Python 3.12.15; host diagnostic Python 3.12.16.
- torch 2.14.1+cu130, CUDA 13.0, Transformers 5.19.0, Triton 3.8.0.
- Container FlashInfer 0.7.1 and flash-attn-4 4.0.0b35. FA2/FA3 are absent;
  neither Hopper FA3 nor SM100-specific kernels were selected.
- HF uses SDPA; packing isolation explicitly selects SDPA math. Native text uses
  Triton and native vision uses SDPA. TP=1, unquantized BF16, seed 0, context
  2048, chunk size 256, max running requests 1, KV capacity 4096, no CUDA graphs.

The image itself stores Python packages in `/opt/sglang`, a venv. A new
`--system-site-packages` venv inherits the underlying interpreter, not this
parent venv. `tools/step04/activate.sh` explicitly adds the image package
location to the child venv's search path, preserving its CUDA stack.
The pinned checkout also needs its own Rust tree-cache extension; build it with
`SGLANG_RUST_BUILD_MODE=auto` and Cargo on PATH. The image defaults to `never`
and the new checkout does not contain the old bundled extension. This is a
source build of the exact pin, not an audit exemption.

The host's inherited LD_LIBRARY_PATH selected incompatible cuDNN sublibraries.
`env -u LD_LIBRARY_PATH` fixes the measured vision convolution failure with the
venv's cuDNN 9.24.0.43. The real BF16 matmul and attention forward/backward probe
passed on SM120 in both environments.

## Weights

Only safetensors, their index, tokenizer.json and build_info.json were attached
from archive `fc1f1c887facf728e2e02a13f9bf9a920af44552`. Each asset matched the
separately fetched archive SHA256SUMS; the seven hashes are committed separately.
The combined current-code manifest was regenerated. No projector rebuild or
archive config overwrite occurred. Source references are under ignored models/:

- IFM/K2-Horizon-7B `4622f94a6632384784471fd674d5b78f8d527798`.
- immanuelpeter/GLM-5.3-Flash-Vision `44a87fd4e6e31f5722a375238b7aedce3e0ad2db`.

Concurrent Docker extraction and Xet writes caused high storage I/O pressure.
An HF process remained in uninterruptible `rq_qos_wait` after termination. A
retry with `HF_HUB_DISABLE_XET=1` eventually completed and checksum verification
passed. The partial-download diagnosis did not imply missing authentication.

## Findings and remaining gates

HF text logits match stock K2 exactly at 12 and 769 tokens. Truncated HF vision
matches the independent source post_layernorm exactly for five image geometries.
Native SGLang loads and the existing text, pre-expanded image, and image_url
smoke requests pass. Chunked image prefill is visible in the server log.

Native teacher-forced logprob parity fails the predeclared limits. Short text
passes; bounded long text and all four image cases fail. Raw results are in the
committed report. No tolerance was widened. Direct native-tower/projector
numerical capture, real-path pixel/grid/hash-padding and embedding insertion
checks, multi-image/repeat-cache geometry checks, and worker/scheduler projector
updates against a fresh engine remain unimplemented and unrun.

The real loader rejected our adapter because fast_load_mm_data requires
multimodal_tokens. The adapter and CPU double now follow that signature.

Packed-versus-unpadded BF16 loss fails. The image diagnostic found identical
patch embeddings, with divergence beginning in the first vision block and
growing through later blocks. HF and SGLang now encode/project each image
independently, preserving feature kernel shapes across request batching.
This reduces the loss discrepancy but does not close the unpadded test.
With matching tensor shapes and document offsets, true packed versus isolated
image documents have matching losses and exactly matching projector gradients.
This confirms document isolation in that controlled test. A trial explicit
Miles mask adapter did not fix the original failure and was removed. K2's
vendored upstream code remains unchanged. Full Miles overfit is still blocked. The bounded diagnostic completed 256 updates,
with unchanged frozen hashes, changed projector weights, and 64/64 correct
teacher-forced answer sequences. It failed its declared every-example loss
threshold: worst loss 0.1954 exceeds 0.05. Free generation was not evaluated.
It is explicitly labeled and is not certification.

## Rerun

Clone the two source repositories under third_party/, checkout the full pins
above, and attach only the archive assets as described in the processor handoff.
Run `uv run --no-project --python 3.12 python tools/step04/verify_archive.py
--report outputs/step04/archive-verification.json` before regenerating the local
manifest. Preserve the archive manifest in outputs/step04/archive/.

```bash
docker run --name k2hv-step04 --gpus device=0 --ipc=host -d \
  --mount type=bind,src="$PWD",dst=/workspace -w /workspace \
  lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae sleep infinity
docker exec k2hv-step04 git config --global --add safe.directory '*'
docker exec k2hv-step04 bash tools/step04/activate.sh python -m k2hv_sglang.check_runtime
docker exec -e SGLANG_RUST_BUILD_MODE=auto k2hv-step04 bash -c \
  'source .venv-container/bin/activate; export PATH=/root/.cargo/bin:$PATH; python -c "import sglang.srt.mem_cache.rust_tree_core.extension"'
docker exec k2hv-step04 bash tools/step04/serve.sh
# In another shell, after the engine reports ready:
docker exec k2hv-step04 bash -c \
  'source .venv-container/bin/activate; python -m k2hv_sglang.smoke --model-path /workspace/hf/K2-Horizon-V-init --report outputs/step04/smoke.json'
```

Stop the launched server before any heavyweight HF runner. Use the two-stage
logprob commands in the step document; the reference JSON contains synthetic
image data and belongs under ignored outputs/. Run `python -m pytest -q` with
`K2HV_SGLANG_SOURCE=$PWD/third_party/sglang-source/python/sglang/srt` in the
activated host venv. GPU checks are separate and always return nonzero while
any required gate is failed, unselected or unimplemented.

## Machine-only artifacts

Ignored outputs/step04/ contains raw downloads/build/server logs, detailed JSON
runs, HF teacher-forced references, the failed mask-adapter attempt and diagnostic
loss data. models/ contains both source checkpoints; hf/K2-Horizon-V-init holds
the original archived weight assets and current manifest. third_party/ contains
the exact source checkouts. .venv-gpu/ and .venv-container/ are machine-specific.
The task container k2hv-step04 was stopped and retained. The Docker image and
Rust build cache remain local. No model was uploaded.
