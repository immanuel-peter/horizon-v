# Handoff: init checkpoint (plan step 1) built and verified

`K2-Horizon-V-init` is built and verified on the Linux CPU box (8 vCPU, 62 GiB RAM). The processor, chat template, SGLang model and every GPU check are **not** done. Branch: `build-init-checkpoint`.

## Result

| | |
|---|---|
| Path (this machine only, git-ignored, not published) | `/home/ubuntu/horizon-v/models/K2-Horizon-V-init` |
| Size | 18,894,997,595 bytes (18.89 GB, 17.60 GiB): 4 shards of 4.94 / 4.98 / 5.00 / 3.95 GB (limit 5 GB) plus code and assets |
| Contents | 672 tensors, 9,436,974,080 parameters, all BF16: 327 `language_model.*` + 339 `visual.*` + 6 `projector.*` |
| Parameters | `language_model` 8,999,178,240, `visual` 404,231,168, `projector` 33,564,672 (only trainable part) |
| Checksum manifest | `models/K2-Horizon-V-init/SHA256SUMS` (`sha256sum -c` format; manifest itself: `192e28a98728c8fabd7d3d340543972ab98fed3b97bf82ef48c168b14aab9499`) |
| Build record | `models/K2-Horizon-V-init/build_info.json` (revisions, excluded GLM tensors, projector SHA-256s, versions) |
| Logs | `outputs/init_checkpoint/{build.log,verify.log,verify_report.json}` (git-ignored) |

Shard SHA-256 (any rebuild must reproduce these):

```text
beb374d9e5c3db7e1e51b49c6eb14f884d5c396d4d5aa5df06f9adc4e813c1a0  model-00001-of-00004.safetensors
6bcddfdf7f873b78d9bc5c075e7aa56226f8fbb24b8fec2319bf24af6540b2fd  model-00002-of-00004.safetensors
99be344787ea797d1d49be3093b4839564255685c8486ad94b6cbb7734c2effb  model-00003-of-00004.safetensors
43de4cd7bbbd6c3af4e5e0c479153a4f5ca4e207aade044f3badb52308bf5f75  model-00004-of-00004.safetensors
23253f5ff85eff7bee805e8623879cc6bc84d8a557baf10ab5ba09a51f317e9a  model.safetensors.index.json
```

Projector (seed 0) tensor SHA-256, first 16 hex digits: `norm.weight 95fbbe9f3324b7d9`, `norm.bias e5a00aa9991ac8a5`, `fc1.weight 20f78ab19b3ec462`, `fc1.bias 276bb46c8e473e95`, `fc2.weight d26c3e6b80f82c17`, `fc2.bias 815c54b0770fc068`.

## Sources

| | Repo | Revision | License |
|---|---|---|---|
| Language model | `IFM/K2-Horizon-7B` | `4622f94a6632384784471fd674d5b78f8d527798` (2026-10-14) | Apache-2.0 |
| Vision tower | `immanuelpeter/GLM-5.3-Flash-Vision` | `44a87fd4e6e31f5722a375238b7aedce3e0ad2db` (2026-09-17) | MIT |

Revisions were resolved before any weight download, and the downloads record them (`build_info.json` reads them back). All 37 source `.safetensors` match the Hub's LFS sha256. K2's `configuration_k2_horizon.py` / `modeling_k2_horizon.py` match the hashes in K2's own `migration_manifest.json`. Both source checkpoints are all-BF16, so **no dtype conversion happened**: every frozen tensor is byte-identical to its source.

## Commands

```bash
uv sync --locked

# One copy of each source, straight into models/ (no duplicate in the HF cache). Public repos; a token is optional.
uv run hf download IFM/K2-Horizon-7B --revision 4622f94a6632384784471fd674d5b78f8d527798 \
  --local-dir models/K2-Horizon-7B \
  --include "*.json" --include "*.py" --include "*.jinja" --include "README.md" --include "model-*.safetensors"
uv run hf download immanuelpeter/GLM-5.3-Flash-Vision --revision 44a87fd4e6e31f5722a375238b7aedce3e0ad2db \
  --local-dir models/GLM-5.3-Flash-Vision \
  --include "*.json" --include "*.py" --include "LICENSE" --include "README.md" --include "model.safetensors"

uv run python -m k2hv.build_init \
  --k2 models/K2-Horizon-7B --glm models/GLM-5.3-Flash-Vision --out models/K2-Horizon-V-init

uv run python -m k2hv.verify_init \
  --k2 models/K2-Horizon-7B --glm models/GLM-5.3-Flash-Vision --out models/K2-Horizon-V-init \
  --source-hashes --report outputs/init_checkpoint/verify_report.json

uv run pytest
```

(The downloads were actually done with `huggingface_hub.snapshot_download(..., local_dir=..., allow_patterns=...)`; the `hf download` form above is the equivalent and was checked with `--dry-run`: nothing left to fetch.) The builder refuses to write into an existing `--out` and builds in `<out>.building`, renamed only on success.

Dependencies: `torch` resolves to the CPU wheel on Linux through `[tool.uv.sources]` / `[[tool.uv.index]]` in `pyproject.toml` (no CUDA packages in `uv.lock`; macOS still uses PyPI). `transformers==5.19.0` unchanged. `pytest` is in the `dev` group.

## Measurements

| | |
|---|---|
| Build time | 43.3 s in-process (47.4 s wall including `uv run` startup) |
| Peak memory | 17.9 GiB RSS (`ru_maxrss`; `/usr/bin/time -v`: 18,803,832 KB). The builder never instantiates the model, but RSS counts the memory-mapped source pages |
| Disk | checkpoint 18.89 GB; sources 18.02 GB (K2) + 1.13 GB (GLM); `models/` 36 GiB; `.venv` 1.0 GB; 351 GB free of 400 GB |
| Verification | 225 s; fresh-process peak 22.9 GiB (checkpoint), 15.2 GiB (stock K2) |
| Reproducibility | Two independent builds produced identical SHA-256 for every shard, the index, `config.json` and all code files. Only `build_info.json` (timings) differs |

## What was verified (all passing)

`verify_init` (37 checks) and `pytest` (47 tests). Everything below ran on CPU.

- **Layout**: 18 required files; no `processing_*.py`, no `processor_config.json`; `auto_map` is exactly `AutoConfig` + `AutoModelForImageTextToText`, with no Hub prefix anywhere in `config.json`; token IDs 250031 / 250062 / 250032 (also confirmed against K2's `tokenizer.json`); untied embeddings; `text_config` / `vision_config` equal the sources' configs; tokenizer files, chat template, generation config, K2 code, `preprocessor_config.json` and the MIT license are byte-identical to the sources.
- **Shards**: every indexed shard exists, none extra, each ≤ 5 GB, index and shard headers agree in both directions, `total_size` matches.
- **Keys**: exactly 327 K2 + 339 GLM + 6 projector. The 8 excluded GLM tensors (`downsample.{weight,bias}` and `merger.{proj,post_projection_norm.{weight,bias},gate_proj,up_proj,down_proj}`, 159,395,840 params) are absent and listed in `build_info.json`. Every shape and dtype matches its source.
- **Frozen tensors**: all 666 `language_model.*` / `visual.*` tensors (18.81 GB) are byte-for-byte equal to their source ranges. Corrupting one byte in a copy is caught (test).
- **Projector**: regenerating from seed 0 is bitwise identical to the stored tensors, twice, independent of the global RNG. Only these 6 tensors are trainable.
- **Offline load, fresh process**: `HF_HUB_OFFLINE=1`, `local_files_only=True`, empty `HF_MODULES_CACHE`, `HF_HOME` and hub cache, `python -I`, and `import k2hv` blocked. `AutoConfig` and `AutoModelForImageTextToText` (`trust_remote_code=True`) load with no missing / unexpected / mismatched keys. All 672 loaded tensors equal their stored bytes. After the real `from_pretrained`, trainable params = the projector's 33,564,672 only.
- **Stock-K2 parity**: text-only logits of the checkpoint vs stock `AutoModelForCausalLM` K2, same 13 input ids, bf16, sdpa, 4 threads: **max |diff| = 0, mean |diff| = 0, bitwise equal**; the pass tolerance was exact (`atol = 0`). Logits range [-10.00, 26.62].
- **Model-level image smoke** (hand-built `<image>` + 256 × 250062 + `</image>`; pixels from upstream `Glm5NextImageProcessorPil.patchify`, no processor, no template): the truncated tower equals the native `Glm5NextVisionModel` through `post_layernorm` on the real weights; 448×448 → 1024 patches → 256 features of 4096-d; the 265-token forward is finite.
- **Unit tests** (tiny configs): composition and key layout, projector 2×2 patch order against upstream `patchify`, shapes (33,564,672 params at 1024/4096), deterministic init, text-only path identical to K2, features land only on pad positions, placeholder-count errors in every direction, grid validation, gradients reach the projector and nothing else (also with gradient checkpointing, reentrant and not), two images per sequence, `generate` with an image, `from_pretrained` keeps frozen modules frozen, builder end-to-end (sharded and unsharded GLM, multi-shard splitting, byte preservation, determinism, refusal paths).

## Not done (deferred; none of this was tested or is claimed)

- `AutoProcessor` / `K2HorizonVProcessor`: no processor code, no `processor_config.json`, no `AutoProcessor` in `auto_map`. `AutoProcessor.from_pretrained` on the checkpoint is unsupported and was not exercised.
- Chat template: `chat_template.jinja` is K2's original, which rejects image parts. No image-template or prompt-expansion test exists.
- SGLang model package, datasets, SFT/RL, GPU parity, flash-attention / flex-attention (see below).

## Differences from `docs/steps/01-init-checkpoint.md`

1. **`from_pretrained` unfreezes everything.** transformers 5.19 rewraps every loaded float tensor as `nn.Parameter(requires_grad=True)`, discarding the flags set in `__init__`. Miles builds `AdamW(model.parameters())` with no freeze logic, so a loaded model would have trained all 9.4B parameters. `K2HorizonVForConditionalGeneration.from_pretrained` re-freezes after loading (regression-tested). Any other load path (`load_state_dict(assign=True)`, FSDP re-materialization) must call `model.freeze_pretrained_modules()` and check that only `projector.*` requires grad.
2. **The GLM ViT is native** in 5.19 (`transformers.models.glm5_next`), so no vendored GLM module ships. Its `forward` is `(hidden_states, grid_thw)`, not `pixel_values`. `GlmVisionTower` (key names identical to GLM's) wraps the native blocks and stops after `post_layernorm`. Consequence: **any environment that loads this checkpoint needs `transformers==5.19.0`** (Miles' pin) because the remote code imports `transformers.models.glm5_next` and `transformers.vision_utils`.
3. **K2's code is vendored** in `src/k2hv/` (byte-identical, pinned by test to IFM's hashes) so `K2HorizonVConfig` imports without the weights. The builder fails if `--k2`'s copies differ and ships the source's bytes.
4. **Build procedure**: no full model is constructed and `save_pretrained` is not used. A meta-device model supplies the key/shape/dtype schema (checked both ways against both sources); source bytes stream into ≤ 5 GB shards; `custom_object_save` writes the code and `auto_map`. Peak is 17.9 GiB, not ~35 GB.
5. **Patch order verified**, not assumed. The GLM image processor's `patchify` (permute to `(h/2, w/2, 2, 2, …)`), `get_vision_position_ids` ("block-major over merge windows") and GLM's own `view(-1, 2, 2, C)` + `Conv2d(k=2, s=2)` all agree: every 4 consecutive rows are one 2×2 window in `(0,0),(0,1),(1,0),(1,1)` order, so `reshape(-1, 4*C)` is correct.
6. **Projector init** is defined as: float32 draws from `torch.Generator().manual_seed(0)` in the order `fc1.weight, fc1.bias, fc2.weight, fc2.bias` with `U(±1/√fan_in)` (torch's `nn.Linear` default), `LayerNorm` weight 1 / bias 0, then cast to BF16. It does not touch the global RNG.
7. **Forward behavior choices**: input ids containing 250062 with no `pixel_values` raise at prefill (the likeliest symptom of a dropped multimodal kwarg); a lone `image_grid_thw` is ignored because `generate()` drops `pixel_values` after prefill but keeps the grid; `pixel_values` requires `input_ids`, not `inputs_embeds`.
8. **Attention backends**: only `eager`/`sdpa` are declared supported and tested. The GLM tower's flash path and K2's flash/flex support are unvalidated here; decide the flags in step 4 on a GPU.
9. **Size**: the step doc's "~17 GB" is 17.6 GiB (18.9 GB).
10. Importing K2's modeling file logs `[ERROR] cache_position is part of … not documented`. This is upstream `@auto_docstring` noise, harmless, and the file is kept unmodified.

## What remains for the processor and template step (`docs/steps/02-processor-and-template.md`)

1. Write `K2HorizonVProcessor` (`processing_k2_horizon_v.py`): GLM image processor + K2 tokenizer + template; expand each `reserved_special_token_12` to `prod(grid) // 4` copies; return only `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`.
2. Register it so `processor_config.json` carries an `AutoProcessor` `auto_map`, then ship it. `build_init` copies assets and code, so extend it to include the processor, rebuild (45 s here; shards come out bit-identical) and re-run `verify_init`. Update `verify_init`'s "no processor artifacts" check at the same time.
3. Patch the template: K2's `render_text_content` macro is `chat_template.jinja:893-909` and the image branch goes before the inner `else` on line 902, as the step doc says (checked against revision `4622f94`). Ship it as `chat_template.jinja` and keep the thinking tags, generation prompt and `{% generation %}` span unchanged.
4. Pick the per-image token cap. GLM's `preprocessor_config.json` uses `min_image_tokens: 16` / `max_image_tokens: 8000`, **not** the `min_pixels` / `max_pixels` named in the step doc, and the processor's budget math uses `factor=28` and `temporal_factor=2`; confirm the unit before choosing the doc's example cap of 1024.
5. **Image-processor backend**: torchvision is not a project dependency, so `AutoImageProcessor.from_pretrained` falls back to the PIL backend (`Glm5NextImageProcessorPil`) here, while the torchvision backend is the default elsewhere. Decide whether to add torchvision, and compare both backends' `pixel_values` (resize is bicubic) on sample images before they feed training and SGLang.
6. Tests: expansion count = `prod(grid) // 4` per image, multi-image and no-image prompts, the three components agreeing on `<image>` + N × 250062 + `</image>` (template, processor, later `pad_input_ids`), scrubbing a stray literal `<image>` from text, then an offline `AutoProcessor.from_pretrained(out, trust_remote_code=True)` smoke.
