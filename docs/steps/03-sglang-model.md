# Step 3: SGLang model

Implemented in `src/k2hv_sglang/` against `sgl-project/sglang` revision `b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf`. The [handoff](../handoff/03-sglang-model.md) has the pinned environment, launch commands, validation and remaining GPU checks.

The adapter combines SGLang's K2 decoder (`models/xllm.py`) and GLM vision blocks (`models/glm5_next.py`) with the HF projector. RL rollouts ([step 8](08-rl.md)) and eval serving ([step 7](07-eval.md)) need it.

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

The `uv pip install -e .` in [step 0](00-environment.md) installs it. The three environment variables that load it are in the [handoff](../handoff/03-sglang-model.md#checkpoint-and-launch).

## Model class

The class below shows the composition. `forward` calls SGLang's `prepare_mm_inputs` to build embeddings, then runs the K2 decoder. The wrapper defines `get_input_embeddings` because the K2 decoder's inner model does not expose it.

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

- `TruncatedGlm5NextVision`: `Glm5NextVisionModel` with `downsample` and `merger` removed, returning after `post_layernorm`.
- `pad_input_ids` keys on token 250062, matching the processor and template ([step 2](02-processor-and-template.md)).
- No M-RoPE: K2 uses 1D RoPE, so the positions SGLang computes for text also apply to image tokens.

## Multimodal processor

A `BaseMultimodalProcessor` subclass that wraps the HF processor and lists the model class in `models`.

- It must accept the pre-expanded `input_ids` plus `image_data` that Miles sends (`miles/rollout/sglang_rollout.py:167-216`).
- It takes token IDs directly. It expands single-pad image blocks, keeps pre-expanded Miles blocks unchanged, and never decodes and re-tokenizes input IDs.
- It must produce the same `pixel_values` and `image_grid_thw` as the HF processor for the same image.
- It decodes on CPU with PIL and uses the HF image processor. Each image gets its own patch tensor, grid and inclusive token offsets. Hashes cover pixels and grid, so two aspect ratios with identical flat pixels do not share a cached embedding.

## `load_weights`

- Map `language_model.*`, `visual.*` and `projector.*` from the HF checkpoint.
- Accept a projector-only update. In RL, Miles is patched to send only trainable weights ([step 8](08-rl.md)).
- Language weights stream through the K2 loader. Vision QKV and gate/up projections map to SGLang's fused parameters. Unknown weights and dropped head weights raise errors. Every weight update clears the projected-image embedding cache. The caller must still flush the KV cache and synchronize updates with requests.

## Reasoning parser

The pinned revision includes `K2V3Detector`, registered as `k2_horizon`, which covers the high, medium and low IFM think tags. Launch with `--reasoning-parser k2_horizon`. No port is needed, and the RL360 name `k2_v3` does not exist here.

- RL rollouts do not need it: Miles calls `/generate` on raw tokens.
- Eval does: chat completions must separate the reasoning from the answer.
