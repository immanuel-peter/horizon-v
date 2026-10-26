# Key facts

## K2-Horizon-7B

| Property | Value |
|---|---|
| Class | `K2HorizonForCausalLM` (trust_remote_code) |
| Hidden size / layers | 4096 / 36 |
| Attention | 32 heads, 8 KV groups, head dim 128 |
| FFN | SwiGLU, 12288 |
| Norm | Grouped RMSNorm, 4 groups, ε 1e-6 |
| RoPE | 1D, base 1e7 |
| Vocab | 250624, untied embeddings |
| Processor | None shipped (`processor_class` is `None`) |

## Tokens

| String | ID | Use |
|---|---|---|
| `<image>` | 250031 | Image start (trained embedding) |
| `</image>` | 250032 | Image end (trained embedding) |
| `reserved_special_token_12` | 250062 | Image pad, expanded to N per image (embedding RMS ≈ 0.0002, unused) |
| `<\|ifm\|im_end\|>` | 250019 | End of turn |
| `<\|ifm\|begin_of_text\|>` | — | BOS |
| `<\|ifm\|endoftext\|>` | — | EOS / pad |

Think tags by reasoning effort:

| Effort | Open | Close |
|---|---|---|
| high (default) | `<ifm\|think>` | `</ifm\|think>` |
| medium | `<ifm\|think_fast>` | `</ifm\|think_fast>` |
| low | `<ifm\|think_faster>` | `</ifm\|think_faster>` |

## Vision path

| Property | Value |
|---|---|
| Patch size | 14 (temporal patch 2) |
| Spatial merge | 2×2 |
| ViT output | 1024-d per patch, after `post_layernorm` |
| Dropped from GLM | `downsample` + `merger`, 159.4M params |
| Tokens per image | `t · h · w / 4`; 448×448 → 256 |
| Projector | ~33.6M params (Baseten's: 49.5M) |

## Source references

Miles (`radixark/miles` @ `5078ff7`):

| Reference | What |
|---|---|
| `miles/backends/fsdp_utils/actor.py:128` | `AdamW(self.model.parameters())`, no freeze flag |
| `miles/backends/fsdp_utils/actor.py:167` | Loads `AutoModelForImageTextToText` only if `vision_config` + matching `auto_map` |
| `miles/backends/fsdp_utils/actor.py:413` | `multimodal_train_inputs` passed as `forward` kwargs |
| `miles/backends/fsdp_utils/hf_weight_iterator.py:24` | Full state dict sent to SGLang every step |
| `miles/backends/fsdp_utils/adaptations/arch_adapter.py:40` | Stock adapter passes no packing kwargs |
| `miles/rollout/sft_rollout.py` | Text-only SFT rollout (replace it) |
| `miles/rollout/sglang_rollout.py:167-216` | Processor call, expanded `input_ids` + base64 `image_data` |
| `miles/utils/data.py:150-190` | Splits prompt strings on literal `<image>` |
| `miles/utils/data.py:228` | `--apply-chat-template` flattens the prompt to a string |
| `miles/utils/processing_utils.py:168` | `process_vision_info`, `qwen_vl_utils` fallback |
| `miles/utils/arguments.py:2567` | `--loss-mask-type` accepts only qwen variants |
| `tools/convert_fsdp_to_hf.py` | FSDP → HF export, copies assets from `--origin-hf-dir` |
| `requirements.txt:33` | `transformers==5.19.0` |

SGLang:

| Reference | What |
|---|---|
| `srt/models/xllm.py`, `srt/configs/k2_horizon.py` | K2-Horizon support |
| `srt/models/glm5_next.py:1562-1565` | `pad_input_ids`, `get_image_feature` to copy |
| `srt/models/registry.py:154` | `SGLANG_EXTERNAL_MODEL_PACKAGE` |
| RL360 `srt/parser/reasoning_parser.py:480` | `K2V3Detector` (`k2_v3`) |

K2 chat template:

| Reference | What |
|---|---|
| `chat_template.jinja:893-908` | `render_text_content`, raises on non-text parts |
| `chat_template.jinja:902` | Insert the image branch before this `else` |
