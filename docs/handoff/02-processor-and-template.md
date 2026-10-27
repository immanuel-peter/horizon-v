# Processor and chat template handoff

`K2HorizonVProcessor` is implemented in
`hf/K2-Horizon-V-init/processing_k2_horizon_v.py`. The same file is importable
through `k2hv.hf`. It combines K2's original tokenizer with GLM's image
preprocessing. The template adds image and image_url parts in user messages.

## Local use

Fetch the pinned tokenizer, about 20.6 MB, without model weights:

```bash
uv sync --locked
uv run hf download IFM/K2-Horizon-7B --revision 4622f94a6632384784471fd674d5b78f8d527798 \
  --include tokenizer.json --local-dir hf/K2-Horizon-V-init
uv run pytest
```

`AutoProcessor.from_pretrained("hf/K2-Horizon-V-init", trust_remote_code=True,
local_files_only=True)` loads the processor. The README has a complete
image-message example using `apply_chat_template`. Use `tokenize=True` and
`return_dict=True` to obtain model inputs; the default call returns PyTorch
tensors. Mixed-length batches need `processor_kwargs={"padding": True}`.

For an already rendered prompt, call `processor(text=prompt, images=images)`.
Each image must have exactly one unexpanded
`<image>reserved_special_token_12</image>` marker. Flat image lists follow
marker order across the batch. Nested image lists must match each sample's
marker count. The processor rejects incomplete or pre-expanded markers,
missing/extra images and truncation that drops image tokens.

Ordinary calls return only `input_ids`, `attention_mask`, `pixel_values` and
`image_grid_thw`; text-only calls omit the image tensors. Explicit assistant
mask requests also return `assistant_masks`. The inherited HF mask handling
shifts assistant spans after expanding image placeholders. Remove that mask
from the inputs before passing the remaining tensors to the model.

## Image preprocessing

The default is `Glm5NextImageProcessorPil`, including GLM's original rescaling,
CLIP normalization and patch ordering. Patch size is 14, temporal patch size
is 2 and spatial merge size is 2. The processor rejects per-call changes to these dimensions because they change the model's input layout.

The token budget uses GLM's `min_image_tokens` / `max_image_tokens`
units, measured after spatial merging. Defaults are 16 and 1024, compared
with the source checkpoint's cap of 8000. Pass overrides in `images_kwargs`.
A 448x448 image gives a `[1, 32, 32]` grid, 1024 patch rows of width 1176 and
256 image-pad tokens. The processor applies the cap before tokenizing the prompt.

The processor stores its image backend separately from the tokenizer backend.
Passing the image backend through the common HF loading kwargs also changes
the tokenizer's backend label in Transformers 5.19, incorrectly disabling its
assistant-mask support. Loading routes those kwargs to the correct component.

PIL and torchvision were compared on CPU with torch 2.14.1 and torchvision
0.29.1 in an isolated environment. The grids matched for all four images.

| Input size | Maximum normalized patch difference | Mean difference |
|---|---:|---:|
| 448x448 | 2.38e-7 | 6.93e-8 |
| 31x47 | 0.0300 | 8.46e-5 |
| 333x517 | 2.38e-7 | 6.82e-8 |
| 1400x1200 | 0.0150 | 6.97e-5 |

Resizing is not bit-identical between the backends. Use PIL for both
training and rollouts so inputs match. `image_backend="torchvision"`
is a loading override, not a validated training configuration.
Torchvision is not a project dependency. Matching the Qwen resize factor
alone does not establish preprocessing parity; check Miles and SGLang's
whole input paths during integration.

## Checkpoint integration

The composite config, processor config and image-preprocessor config register
`AutoProcessor`. `save_pretrained` exports the custom processor code, tokenizer,
image settings and patched chat template as a directory that loads without the `k2hv` package.

The wheel maps the canonical HF tree into `k2hv/hf`, while editable
imports use the existing symlink. Both the wheel and source distribution exclude
weights and tokenizer data. A wheel built from the source distribution was
checked in isolation for imports, bundled recipe files and offline processor
loading with the separately supplied tokenizer.

The builder includes the new processor files in the checkpoint recipe. The
verifier preserves exact comparisons for unmodified source assets and checks
the template against the specific user-image patch. It checks the image
settings against the source config plus the PIL backend, token cap and
processor metadata. Full model verification loads `AutoProcessor` in a
fresh offline process and uses its tensors for the image forward pass.

The model code, tensor schema and seed-0 projector initialization are unchanged.
The four existing weight shards do not need rebuilding. To attach them to this
checkout, fetch only the generated files from the private step 1 archive:

```bash
uv run hf download immanuelpeter/K2-Horizon-V-init --revision fc1f1c887facf728e2e02a13f9bf9a920af44552 \
  --include "*.safetensors" --include model.safetensors.index.json \
  --include tokenizer.json --include build_info.json --local-dir hf/K2-Horizon-V-init
uv run python -m k2hv.checkpoint_io manifest hf/K2-Horizon-V-init
```

Keep the current tracked files when attaching weights; a full snapshot download
would replace the new processor files with the old step 1 configs. The original
HF snapshot remains pinned and unchanged. Publishing the updated processor to
HF requires uploading the current small files and refreshed manifest alongside
the same weights, after code review.

Run `k2hv.verify_init` with the pinned source checkpoints on a machine that has
the weights. Its existing source/tensor checks still apply. This Mac validation
does not rerun full-size model or GPU checks.

## CPU validation

All 105 CPU tests pass on macOS. They cover:

- pixel and grid parity with GLM's processor, square and non-square images, and the minimum and maximum token budgets
- multiple images and mixed text/image batches
- marker errors and truncation
- unchanged text and tool templates at all reasoning efforts
- assistant masks
- offline save/load with `k2hv` imports blocked and empty HF caches
- a small composite-model forward pass

The real pinned K2 tokenizer is also checked locally for unchanged vocabulary,
exact text-template parity at all three reasoning efforts, 256 tokens for a
448x448 image, assistant masks excluding image pads and identical tensors
after an offline save/load round trip.

The archived weights are still an untrained initialization. SGLang integration,
GPU parity and SFT/RL remain later steps.
