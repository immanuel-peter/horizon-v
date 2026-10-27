# Init checkpoint handoff

`K2-Horizon-V-init` is built and verified on a Linux CPU machine with 8 vCPUs
and 62 GiB RAM. The processor, image chat template, SGLang model and training
code are planned. GPU execution is untested.

## Files and builds

The Hugging Face checkpoint directory is `hf/K2-Horizon-V-init/`.
Git tracks its Python code, docs, licenses, `config.json` and small upstream
assets. Weights, the index, `tokenizer.json`, `build_info.json` and `SHA256SUMS`
are git-ignored. The `src/k2hv/hf` symlink imports the same code as `k2hv.hf.*`.
The builder checks that the output code matches the code it imports.

Hatch follows the symlink, so `pyproject.toml` excludes weight files and build
outputs by filename. The root `.gitignore` paths alone do not exclude files
reached through `src/k2hv/hf`.

To fetch the tokenizer without downloading weights:

```bash
uv run hf download IFM/K2-Horizon-7B --revision 4622f94a6632384784471fd674d5b78f8d527798 \
  --include tokenizer.json --local-dir hf/K2-Horizon-V-init
```

After editing tracked checkpoint files, refresh the manifest:

```bash
uv run python -m k2hv.checkpoint_io manifest hf/K2-Horizon-V-init
```

Code and asset edits need no weight rebuild if the tensor schema and
initialization stay unchanged. Run the verifier after an edit.
When uploading, pass `--exclude "__pycache__/*" --exclude ".cache/*"` to
`hf upload`. Python and `hf download` create these directories; the manifest
skips them.

## Checkpoint contents

The checkpoint is archived in the private
[immanuelpeter/K2-Horizon-V-init](https://huggingface.co/immanuelpeter/K2-Horizon-V-init/tree/fc1f1c887facf728e2e02a13f9bf9a920af44552)
model repository at revision `fc1f1c887facf728e2e02a13f9bf9a920af44552`,
from source commit `edb83ce4d3ee993070d1b5cfd55dbbc0101e616e`. It was built at
`/home/ubuntu/horizon-v/hf/K2-Horizon-V-init` and contains about 18.89 GB of
files and 672 BF16 tensors, totaling 9,436,974,080 parameters.

The archive includes the build and verification evidence under `validation/`.
Its final report, `cleanup_verify_report.json`, passes all 39 checks. All 27
manifest entries were verified after upload; the four shard hashes below
are unchanged.

| Module | Tensors | Parameters | Trainable |
|---|---|---|---|
| `language_model` | 327 | 8,999,178,240 | No |
| `visual` | 339 | 404,231,168 | No |
| `projector` | 6 | 33,564,672 | Yes |

Four shards hold about 4.94, 4.98, 5.00 and 3.95 GB each, within the 5 GB
limit. Rebuilding from the same sources and seed must reproduce these SHA-256s:

```text
beb374d9e5c3db7e1e51b49c6eb14f884d5c396d4d5aa5df06f9adc4e813c1a0  model-00001-of-00004.safetensors
6bcddfdf7f873b78d9bc5c075e7aa56226f8fbb24b8fec2319bf24af6540b2fd  model-00002-of-00004.safetensors
99be344787ea797d1d49be3093b4839564255685c8486ad94b6cbb7734c2effb  model-00003-of-00004.safetensors
43de4cd7bbbd6c3af4e5e0c479153a4f5ca4e207aade044f3badb52308bf5f75  model-00004-of-00004.safetensors
23253f5ff85eff7bee805e8623879cc6bc84d8a557baf10ab5ba09a51f317e9a  model.safetensors.index.json
a011a78cfc2a5805bd0e709341efa7a8029bbcc7ef655721d496f011a8e3c423  config.json
```

The seed-0 projector tensor hashes begin with `norm.weight 95fbbe9f3324b7d9`,
`norm.bias e5a00aa9991ac8a5`, `fc1.weight 20f78ab19b3ec462`,
`fc1.bias 276bb46c8e473e95`, `fc2.weight d26c3e6b80f82c17` and
`fc2.bias 815c54b0770fc068`.

`build_info.json` records source revisions, excluded GLM tensors, projector
hashes and package versions. `SHA256SUMS` covers the checkpoint files in
`sha256sum -c` format. Logs and JSON reports are in the git-ignored
`outputs/init_checkpoint/` directory.

## Sources

| Component | Repository | Revision | License |
|---|---|---|---|
| Language model | `IFM/K2-Horizon-7B` | `4622f94a6632384784471fd674d5b78f8d527798` | Apache-2.0 |
| Vision tower | `immanuelpeter/GLM-5.3-Flash-Vision` | `44a87fd4e6e31f5722a375238b7aedce3e0ad2db` | MIT |

The downloads record their revisions in local metadata. The builder reads
that metadata; the verifier checks that README and NOTICE name the build's
revisions. All 37 source safetensors matched the Hub's LFS SHA-256s.
Both source checkpoints are BF16. All 666 frozen tensors retain their source
bytes, with no dtype conversion.

K2's vendored Python files match IFM's `migration_manifest.json` hashes. The
builder, verifier and tests require those files to remain unchanged, including
their comments and Apache headers. The chat template also matches its source
byte for byte.

## Commands

```bash
uv sync --locked

uv run hf download IFM/K2-Horizon-7B --revision 4622f94a6632384784471fd674d5b78f8d527798 \
  --local-dir models/K2-Horizon-7B \
  --include "*.json" --include "*.py" --include "*.jinja" --include "README.md" --include "model-*.safetensors"
uv run hf download immanuelpeter/GLM-5.3-Flash-Vision --revision 44a87fd4e6e31f5722a375238b7aedce3e0ad2db \
  --local-dir models/GLM-5.3-Flash-Vision \
  --include "*.json" --include "*.py" --include "LICENSE" --include "README.md" --include "model.safetensors"

uv run python -m k2hv.build_init \
  --k2 models/K2-Horizon-7B --glm models/GLM-5.3-Flash-Vision --out hf/K2-Horizon-V-init

uv run python -m k2hv.verify_init \
  --k2 models/K2-Horizon-7B --glm models/GLM-5.3-Flash-Vision --out hf/K2-Horizon-V-init \
  --source-hashes --report outputs/init_checkpoint/verify_report.json

uv run pytest
```

Downloads use `--local-dir` to avoid a second weight copy in the Hub cache.
The builder requires the tracked files in `--out`, stages new files in
`.building/`, then moves them into place. It refuses to overwrite any shard,
index, `build_info.json` or `SHA256SUMS`. Move those outputs before rebuilding.
It reports a changed `config.json` for review and rejects a different
`tokenizer.json` already in place.

The project pins `transformers==5.19.0`. PyPI supplies CUDA torch on Linux
and CPU/MPS torch on macOS. This CPU machine runs `torch 2.14.1+cu130`; its
venv uses about 5.4 GB. GPU nodes reuse the Miles image's torch and do not sync
this lockfile.

## Validation and measurements

The original full-size build took 54.7 seconds in process and 59.2 seconds
wall time, with peak RSS of 18.3 GiB. The first build with the CPU wheel and
a warm page cache took 43.3 seconds. Full verification took 228 seconds;
fresh-process peak RSS was 23.3 GiB for the checkpoint and 15.6 GiB for stock
K2. Memory-mapped source pages count toward the builder's RSS. Source
checkpoints occupy 18.02 GB for K2 and 1.13 GB for GLM.

Four builds across CPU and CUDA torch wheels reproduced the shard, index,
config, code and asset hashes. `build_info.json` changes with timing and
memory measurements.

The verifier checks layout, source revisions, key coverage, tensor shapes
and dtypes, shard limits, frozen bytes, projector regeneration and the
manifest. Its load checks run under `python -I` with offline mode, empty
Hugging Face caches and imports of `k2hv` blocked. Every loaded tensor must
match its stored bytes, and only the projector may require gradients.

Text-only logits matched stock K2 exactly for 13 input IDs with BF16, SDPA
and four CPU threads. Maximum and mean absolute differences were zero;
logits ranged from -10.00 to 26.62. CPU and CUDA torch wheels gave the same
result.

The image smoke test uses upstream `patchify` and hand-built placeholders.
A 448x448 image produces 1,024 patches and 256 features of width 4,096.
The truncated tower matches native GLM through `post_layernorm`, and the
265-token model forward produces finite logits. This does not test a
processor or chat template.

CPU tests use tiny configs to cover patch order, shapes, deterministic
initialization, frozen bytes, text parity, embedding replacement, input
validation and projector gradients. They also cover both gradient
checkpointing modes, multiple images, beam expansion, returned sequences,
last-position logits, freezing after load, offline loading, builder refusal
paths and manifest refreshes.

## Model behavior

The model uses GLM's native vision blocks from `transformers.models.glm5_next`
and stops at `post_layernorm`. It omits the eight `downsample` and `merger`
tensors, totaling 159,395,840 parameters. The checkpoint needs
`transformers==5.19.0` for those imports and `transformers.vision_utils`.

A meta-device model supplies the builder's key, shape and dtype schema.
Weights stream into shards without constructing a full model.

The projector applies LayerNorm, merges each four consecutive patches,
then applies Linear, GELU and Linear. Upstream `patchify`, vision position
IDs and GLM's downsample use the same row-major order within each 2x2 window.
Initialization draws float32 values from a private seed-0 generator in
`fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias` order, using torch's default
uniform bounds. LayerNorm starts with weight 1 and bias 0. The builder then
casts to BF16 without changing the global RNG state.

`from_pretrained` restores the freeze flags after transformers replaces
loaded parameters. Other loading paths, including `load_state_dict(assign=True)`
and FSDP rematerialization, must call `freeze_pretrained_modules()` and check
that only `projector.*` requires gradients.

Image placeholders without pixels raise during prefill. Pixels require
`input_ids` and `image_grid_thw`; supplied `inputs_embeds` are rejected.
A lone grid is ignored because generation retains it after dropping pixels
for decode. `logits_to_keep` is explicit so generation requests only the last
position. Beam expansion repeats whole images per sample. Pixel values and
image features move to the devices of their consuming modules; multiple
devices remain untested. Only eager and SDPA attention are declared supported.

K2's upstream `@auto_docstring` reports an undocumented `cache_position`
parameter during import. The vendored code remains unchanged.

## Next steps

1. Add `processing_k2_horizon_v.py` with `K2HorizonVProcessor`, combining the
   GLM image processor, K2 tokenizer and template. Expand each image placeholder
   to `prod(grid) // 4` tokens. Return `input_ids`, `attention_mask`,
   `pixel_values` and `image_grid_thw`.
2. Add `processor_config.json` and its `AutoProcessor` mapping. Update the
   builder's `CODE_FILES` and `RECIPE_FILES`. Replace the verifier's checks for
   absent processor files with processor checks. When changing mirrored
   assets, replace their byte comparisons with checks of the intended edits.
   Refresh the manifest and run the verifier.
3. Add image parts to `render_text_content` in `chat_template.jinja` at
   lines 893 to 909. Keep thinking tags, the generation prompt and the
   `{% generation %}` span. The original template rejects image parts.
4. Choose the image token cap. GLM uses `min_image_tokens: 16` and
   `max_image_tokens: 8000`, with `factor=28` and `temporal_factor=2`. Confirm
   the budget units before adopting the plan's example cap of 1024.
5. Compare PIL and torchvision image processing before training. Without
   torchvision, `AutoImageProcessor` selects `Glm5NextImageProcessorPil` here.
   Compare bicubic resize outputs between backends before choosing one.
6. Test expansion counts, multiple images, text-only prompts, agreement
   between template and processor, literal `<image>` handling and offline
   `AutoProcessor.from_pretrained` loading. When SFT lands in step 6, assert
   that image placeholders never enter the loss mask.

Datasets, SFT/RL, SGLang integration, GPU parity and flash/flex attention
checks remain in later plan steps.
