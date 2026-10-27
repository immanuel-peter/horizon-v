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
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    path, trust_remote_code=True, local_files_only=True, dtype="bfloat16"
)
```

## Current limits

The processor accepts user images and text, with native GLM PIL preprocessing
and a default cap of 1,024 merged image tokens. Each image has one unexpanded
`<image>reserved_special_token_12</image>` marker before processing. The processor
expands it to match the image features. It rejects missing images, mismatched
markers and truncation that removes image tokens. Text-only K2 formatting is
preserved. Batch requests should use `processor_kwargs={"padding": True}` in
`apply_chat_template`. The normal output contains only model-input tensors.

GPU execution and SGLang integration remain untested. Validation has run on CPU.
The projector is untrained, so model outputs do not yet demonstrate image understanding.

See `NOTICE` and `licenses/` for upstream licenses and attribution.
