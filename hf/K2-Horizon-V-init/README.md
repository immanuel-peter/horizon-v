# K2-Horizon-V-init

Initialization checkpoint for K2-Horizon-V. The projector has random weights
from seed 0. The model needs training to interpret images.

| Module | Weights | Trainable |
|---|---|---|
| `language_model` | IFM/K2-Horizon-7B @ `4622f94a6632384784471fd674d5b78f8d527798`, unmodified | no |
| `visual` | GLM vision tower through `post_layernorm`, from immanuelpeter/GLM-5.3-Flash-Vision @ `44a87fd4e6e31f5722a375238b7aedce3e0ad2db`, unmodified | no |
| `projector` | `LayerNorm(1024)`, 2x2 merge, `Linear(4096,4096)`, GELU, `Linear(4096,4096)`; new | yes |

Load from a local directory:

```python
from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    path, trust_remote_code=True, local_files_only=True, dtype="bfloat16"
)
```

## Current limits

`AutoProcessor.from_pretrained` is unsupported. `chat_template.jinja` is K2's
original template and rejects image content. `preprocessor_config.json` is
GLM's original config for the planned processor. Multimodal prompt formatting,
GPU execution and SGLang integration are untested. Validation has run on CPU.

See `NOTICE` and `licenses/` for upstream licenses and attribution.
