# Step 3: SGLang model

Implemented in `src/k2hv_sglang/` against `sgl-project/sglang` revision `b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf`, resolved from the `sglang-miles` branch. Miles's Dockerfile at `5078ff7` follows that branch unless `SGLANG_COMMIT` is supplied. See the [handoff](../handoff/03-sglang-model.md) for the pinned environment, launch commands, validation and remaining GPU checks.

The adapter composes SGLang's native K2 decoder (`models/xllm.py`) and GLM vision blocks (`models/glm5_next.py`) with the existing HF projector. It is needed for RL rollouts ([step 8](08-rl.md)) and eval serving ([step 7](07-eval.md)). CPU boundary tests pass; the actual GPU engine has not been run here.

## Package

```text
k2hv_sglang/
├── __init__.py
├── k2_horizon_v.py       model class + EntryClass
├── processor.py          multimodal processor
├── input_layout.py       image layout, weight names and geometry-aware hashes
├── runtime.py            source pin and registration environment
├── check_runtime.py      source and GPU-side import/registration checks
└── smoke.py              client for an already-running GPU server
```

It ships inside the project, so the `uv pip install -e .` in [step 0](00-environment.md) installs it. Load it with:

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=k2hv_sglang
export SGLANG_EXTERNAL_MM_MODEL_ARCH=K2HorizonVForConditionalGeneration
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=k2hv_sglang
```

## Model class

The original sketch below describes the composition. The implementation uses the pinned `prepare_mm_inputs` helper to prepare embeddings and then calls the native K2 decoder. The wrapper supplies `get_input_embeddings`, which the current K2 decoder's internal model does not expose.

```python
class K2HorizonVForConditionalGeneration(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.visual = TruncatedGlm5NextVision(config.vision_config)
        self.projector = Projector(1024, 4096)
        self.language_model = K2HorizonForCausalLM(config.text_config, quant_config)

    def pad_input_ids(self, input_ids, mm_inputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items):
        pixel_values = torch.cat([item.feature for item in items])
        grid_thw = torch.cat([item.image_grid_thw for item in items])
        return self.projector(self.visual(pixel_values, grid_thw=grid_thw))

    def forward(self, input_ids, positions, forward_batch, **kwargs):
        # general_mm_embed_routine calls get_image_feature and scatters the
        # features into the pad positions before running the language model
        ...

    def load_weights(self, weights):
        ...


EntryClass = K2HorizonVForConditionalGeneration
```

- `TruncatedGlm5NextVision`: native `Glm5NextVisionModel` with `downsample` and `merger` removed, returning after `post_layernorm`.
- `pad_input_ids` keys on token 250062, matching the processor and template ([step 2](02-processor-and-template.md)).
- No M-RoPE: K2 uses 1D RoPE, so the positions SGLang computes for text also apply to image tokens.

## Multimodal processor

A `BaseMultimodalProcessor` subclass that wraps the HF processor and lists the model class in `models`.

- It must accept the pre-expanded `input_ids` plus `image_data` that Miles sends (`miles/rollout/sglang_rollout.py:167-216`).
- The processor consumes token IDs directly. It expands single-pad image blocks and keeps correctly pre-expanded Miles blocks unchanged; it never decodes and re-tokenizes input IDs.
- It must produce the same `pixel_values` and `image_grid_thw` as the HF processor for the same image.
- CPU PIL decoding and the same HF image processor are used. Each image has its own patch tensor, grid and inclusive token offsets. Hashes include both pixels and grid geometry so different aspect ratios cannot reuse the same cached embedding merely because flat pixels match.

## `load_weights`

- Map `language_model.*`, `visual.*` and `projector.*` from the HF checkpoint.
- Accept a projector-only update. In RL, Miles is patched to send only trainable weights ([step 8](08-rl.md)).
- Language weights stream through the native K2 loader. Vision QKV and gate/up projections map to SGLang's fused parameters. Unknown or dropped head weights are rejected. Any weight update clears the projected-image embedding cache; the caller must still use the scheduler's normal KV-cache flush and synchronize updates with requests.

## Reasoning parser

The pinned upstream revision already includes `K2V3Detector`, registered as **`k2_horizon`**, covering high, medium and low IFM think tags. Launch with `--reasoning-parser k2_horizon`; no parser port or alias is needed. This supersedes the older RL360 selector name `k2_v3` in the original plan.

- RL rollouts do not need it: Miles calls `/generate` on raw tokens.
- Eval does: chat completions must separate the reasoning from the answer.
