# Step 3: SGLang model

SGLang already supports K2-Horizon (`python/sglang/srt/models/xllm.py`, `configs/k2_horizon.py`) and the GLM vision tower (`Glm5NextVisionModel` in `models/glm5_next.py`). K2-Horizon-V composes the two in an out-of-tree package. You need it for RL rollouts ([step 8](08-rl.md)) and for eval serving ([step 7](07-eval.md)).

## Package

```text
k2hv_sglang/
├── __init__.py
├── k2_horizon_v.py       model class + EntryClass
└── processor.py          multimodal processor
```

It ships inside the project, so the `uv pip install -e .` in [step 0](00-environment.md) installs it. Load it with:

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=k2hv_sglang   # see models/registry.py:154
```

## Model class

Sketch only. Copy the exact signatures from `Glm5NextForConditionalGeneration` (`glm5_next.py:1562-1565` and its `forward`).

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

- `TruncatedGlm5NextVision`: `Glm5NextVisionModel` returning after `post_layernorm`, with no `downsample` and no `merger`.
- `pad_input_ids` keys on token 250062, matching the processor and template ([step 2](02-processor-and-template.md)).
- No M-RoPE: K2 uses 1D RoPE, so the positions SGLang computes for text also apply to image tokens.

## Multimodal processor

A `BaseMultimodalProcessor` subclass that wraps the HF processor and lists the model class in `models`.

- It must accept the pre-expanded `input_ids` plus `image_data` that Miles sends (`miles/rollout/sglang_rollout.py:167-216`).
- Copy how SGLang's Qwen-VL processor handles pre-tokenized input. Do not re-tokenize.
- It must produce the same `pixel_values` and `image_grid_thw` as the HF processor for the same image.

## `load_weights`

- Map `language_model.*`, `visual.*` and `projector.*` from the HF checkpoint.
- Accept a projector-only update. In RL, Miles is patched to send only trainable weights ([step 8](08-rl.md)).

## Reasoning parser

Use the `k2_v3` reasoning parser from RL360's bundled SGLang (`srt/parser/reasoning_parser.py:480`, `K2V3Detector`). It handles all three think-tag variants. Port it if upstream SGLang lacks it.

- RL rollouts do not need it: Miles calls `/generate` on raw tokens.
- Eval does: chat completions must separate the reasoning from the answer.
