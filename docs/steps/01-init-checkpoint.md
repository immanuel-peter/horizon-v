# Step 1: Build the init checkpoint

`K2-Horizon-V-init` is not a trained model. It is K2's weights, the GLM ViT weights and a randomly initialized projector, saved together as one HF checkpoint.

## Why it exists

- **Miles loads from one path.** The FSDP actor calls `AutoModelForImageTextToText.from_pretrained(--hf-checkpoint)`. It has no hook to assemble a model from two repos plus a new module.
- **SGLang loads from one path.** RL rollout engines and the parity checks in [step 4](04-gate-checks.md) must see identical weights, including the same random projector.
- **Reproducibility.** A saved projector init gives every SFT sweep the same starting point.

You need it once, as the SFT starting point. RL starts from the SFT checkpoint instead.

## Repo layout

```text
K2-Horizon-V-init/
├── config.json                       composite config (below)
├── configuration_k2_horizon_v.py     K2HorizonVConfig
├── modeling_k2_horizon_v.py          K2HorizonVForConditionalGeneration, Projector, truncated ViT
├── processing_k2_horizon_v.py        K2HorizonVProcessor (step 2)
├── configuration_k2_horizon.py       copied from IFM/K2-Horizon-7B
├── modeling_k2_horizon.py            copied from IFM/K2-Horizon-7B
├── <glm vision module>.py            only if the GLM ViT class is not native in transformers 5.19
├── model-0000x-of-0000N.safetensors  ~17 GB bf16 total
├── model.safetensors.index.json
├── tokenizer.json, tokenizer_config.json, special_tokens_map.json   from K2, unchanged
├── chat_template.jinja               K2's template + the image-part patch (step 2)
├── preprocessor_config.json          GLM image-processor settings (patch 14, merge 2, min/max pixels)
├── processor_config.json             auto_map → AutoProcessor
└── generation_config.json            from K2
```

## `config.json`

```json
{
  "model_type": "k2_horizon_v",
  "architectures": ["K2HorizonVForConditionalGeneration"],
  "text_config": { "...": "K2's config.json verbatim" },
  "vision_config": { "...": "GLM ViT config" },
  "image_token_id": 250062,
  "image_start_token_id": 250031,
  "image_end_token_id": 250032,
  "auto_map": {
    "AutoConfig": "configuration_k2_horizon_v.K2HorizonVConfig",
    "AutoModelForImageTextToText": "modeling_k2_horizon_v.K2HorizonVForConditionalGeneration",
    "AutoProcessor": "processing_k2_horizon_v.K2HorizonVProcessor"
  }
}
```

`auto_map` must include `AutoModelForImageTextToText`. Otherwise Miles's `_has_image_text_to_text_impl` (`miles/backends/fsdp_utils/actor.py:167`) loads the model as a plain CausalLM and the vision path is dropped.

## Weight keys

| Prefix | Source |
|---|---|
| `language_model.*` | Every K2 tensor, renamed with the prefix |
| `visual.*` | GLM-5.3-Flash-Vision's vision tensors, except `downsample.*` and `merger.*` |
| `projector.{norm,fc1,fc2}.*` | New, initialized with a fixed seed |

## Model code

```python
class K2HorizonVForConditionalGeneration(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.visual = GlmVisionTower(config.vision_config)
        self.projector = Projector(1024, 4096)
        self.language_model = K2HorizonForCausalLM(config.text_config)
        self.visual.requires_grad_(False)
        self.language_model.requires_grad_(False)

    def forward(self, input_ids, position_ids=None, pixel_values=None, image_grid_thw=None, **kwargs):
        embeds = self.language_model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            with torch.no_grad():
                feats = self.visual(pixel_values, grid_thw=image_grid_thw)
            feats = self.projector(feats).to(embeds.dtype)
            mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            embeds = embeds.masked_scatter(mask, feats)
        return self.language_model(inputs_embeds=embeds, position_ids=position_ids, **kwargs)


class Projector(nn.Module):
    def __init__(self, d_vit, d_llm):
        super().__init__()
        self.norm = nn.LayerNorm(d_vit)
        self.fc1 = nn.Linear(4 * d_vit, d_llm)
        self.fc2 = nn.Linear(d_llm, d_llm)

    def forward(self, x):
        x = self.norm(x).view(-1, 4 * x.shape[-1])
        return self.fc2(F.gelu(self.fc1(x)))
```

`GlmVisionTower` is GLM's vision model with `forward` returning right after `post_layernorm`.

- **2×2 merge.** `view(-1, 4096)` merges each 2×2 patch block into one token. GLM's image processor emits patches in merge-window order; GLM's own `downsample` relies on the same order (`view(-1, 2, 2, C)`).
- **`no_grad` on the ViT.** Nothing before the projector needs gradients, so ViT activations are not stored. The LLM is frozen too, but gradients must flow back through its activations to reach the projector. Keep `--gradient-checkpointing` on.
- **Positions.** K2 uses 1D RoPE. Image tokens take sequential positions; no M-RoPE.
- **Optimizer.** Miles builds `AdamW(model.parameters())` (`actor.py:128`). Frozen params never get a `.grad`, so they get no optimizer state and no updates.
- **Size.** Baseten's projector is 49.5M params (4608 → 4608 → 6144). Ours is smaller because the ViT is 1024-d and K2 is 4096-d.

## Build procedure

Implement this as `src/k2hv/build_init.py` and run it in the laptop env ([step 0](00-environment.md)):

```bash
uv run python -m k2hv.build_init \
  --k2 /models/K2-Horizon-7B \
  --glm /models/GLM-5.3-Flash-Vision \
  --out /models/K2-Horizon-V-init
```

Peak memory is about 2× the checkpoint, so use a machine with about 35 GB of RAM.

1. Write the four `*_k2_horizon_v.py` files.
2. Call `torch.manual_seed(0)`, then construct `K2HorizonVForConditionalGeneration(config)` in bf16 on CPU. `__init__` initializes the projector.
3. Load K2: `model.language_model.load_state_dict(k2_state, strict=True)`.
4. Load the ViT: `model.visual.load_state_dict(vit_state, strict=True)`.
   - Use GLM's `model.safetensors.index.json` to find the shards that hold vision keys. Read only those tensors with `safe_open`. Do not load the full GLM model.
   - `strict=True` on both loads verifies every key rename.
5. Register the custom classes, so `save_pretrained` copies the `.py` files and writes `auto_map`:
   - `K2HorizonVConfig.register_for_auto_class()`
   - `K2HorizonVForConditionalGeneration.register_for_auto_class("AutoModelForImageTextToText")`
   - `K2HorizonVProcessor.register_for_auto_class()`
6. Save: `model.save_pretrained(out, max_shard_size="5GB")` and `processor.save_pretrained(out)`.
7. Copy K2's modeling and config files in. Check that no `auto_map` entry keeps a Hub prefix such as `IFM/K2-Horizon-7B--...`. A prefix makes transformers fetch code from the Hub at load time.
8. Smoke test with `HF_HUB_OFFLINE=1`:
   - `AutoModelForImageTextToText.from_pretrained(out, trust_remote_code=True)`
   - `AutoProcessor.from_pretrained(out, trust_remote_code=True)`

## Hosting

The repo can be private, or not on the Hub at all. Miles and SGLang load from a local directory; the Hub only moves the directory between machines.

```bash
uv run hf repo create <you>/K2-Horizon-V-init --private
uv run hf upload <you>/K2-Horizon-V-init ./K2-Horizon-V-init

# on the GPU machine, with HF_TOKEN set
uv run hf download <you>/K2-Horizon-V-init --local-dir /models/K2-Horizon-V-init
```

- Point `--hf-checkpoint`, `--load` and SGLang's model path at the local directory, not the repo ID.
- Set `HF_HUB_OFFLINE=1`. No token is needed at runtime, and a stray Hub reference fails loudly.
- Multi-node: the same path must exist on every node (shared filesystem, or download on each).

After the merge you do not need IFM/K2-Horizon-7B or GLM-5.3-Flash-Vision again.

## Not in the repo, but needed on the machine

1. **Your code**, installed into the project venv ([step 0](00-environment.md)):
   - the SFT rollout ([step 6](06-sft.md)), the reward wrapper and the weight-sync patch ([step 8](08-rl.md));
   - the `k2hv_sglang` package plus `SGLANG_EXTERNAL_MODEL_PACKAGE=k2hv_sglang` ([step 3](03-sglang-model.md)).
2. **Data:** parquet files, with images as local paths or embedded bytes. See [step 5](05-data.md).
