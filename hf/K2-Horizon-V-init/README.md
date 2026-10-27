# K2-Horizon-V-init

Initialization checkpoint for K2-Horizon-V. **It is not a trained model**: the projector is random
(seed 0) and the model cannot interpret images until it is trained.

| Module | Weights | Trainable |
|---|---|---|
| `language_model` | IFM/K2-Horizon-7B @ `4622f94a6632384784471fd674d5b78f8d527798`, unmodified | no |
| `visual` | GLM vision tower through `post_layernorm`, from immanuelpeter/GLM-5.3-Flash-Vision @ `44a87fd4e6e31f5722a375238b7aedce3e0ad2db`, unmodified | no |
| `projector` | `LayerNorm(1024)`, 2x2 merge, `Linear(4096,4096)`, GELU, `Linear(4096,4096)`; new | yes |

Load (offline, model code ships in this directory):

```python
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained(path, trust_remote_code=True, dtype="bfloat16")
```

## Not done yet

- No `AutoProcessor` / `K2HorizonVProcessor`: `AutoProcessor.from_pretrained` is not supported and untested.
- `chat_template.jinja` is K2's original template, which rejects image content. The image-part patch is pending.
- `preprocessor_config.json` is GLM's, kept for the processor work. Multimodal prompt formatting is untested.
- Only CPU validation has been done (see the repository handoff notes); no GPU or SGLang checks.

See `NOTICE` and `licenses/` for upstream licenses and attribution.
