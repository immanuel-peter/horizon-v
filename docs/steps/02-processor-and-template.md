# Step 2: Processor and chat template

K2 ships no processor (`processor_class` is `None`), and its chat template rejects images. Both need a small addition. Nothing else about K2's prompt format changes.

## Processor

`K2HorizonVProcessor(ProcessorMixin)` combines:

- GLM's image processor, unchanged: patch 14, merge 2, temporal patch 2, same normalization;
- K2's tokenizer, unchanged;
- the patched chat template.

`__call__(text, images)`:

1. Run the image processor: `pixel_values [ΣN_patch, 1176]`, `image_grid_thw [n_img, 3]`.
2. Replace the i-th `reserved_special_token_12` in `text` with `prod(image_grid_thw[i]) // 4` copies.
3. Tokenize.
4. Return only tensors: `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`.

Miles passes every returned key except `input_ids`, `attention_mask` and `mm_token_type_ids` straight into the model's `forward`, and list-valued fields raise an error (see `examples/geo3k_vlm/rollout.py`). Return nothing else.

Token budget:

- Cap tokens per image with `max_pixels` (for example 1024 merged tokens ≈ 1024 · 28² px) and `min_pixels`.
- Miles's `process_vision_info` falls back to `qwen_vl_utils`, whose resize factor (28) equals 14 × 2. The fallback works as-is.

Register `AutoProcessor` in `processor_config.json` / `preprocessor_config.json` `auto_map`.

## Template patch

K2's `render_text_content` macro raises on any content part that is not text. Add one branch before its `else` (`chat_template.jinja:902`):

```jinja
{%- elif part is mapping and part.type in ['image', 'image_url'] and role == 'user' -%}
{{- '<image>reserved_special_token_12</image>' -}}
```

| Token string | ID | Note |
|---|---|---|
| `<image>` | 250031 | Trained embedding |
| `reserved_special_token_12` | 250062 | Image pad; the processor expands it |
| `</image>` | 250032 | Trained embedding |

All three are single special tokens. No vocabulary or embedding changes.

Keep unchanged:

- **Thinking tags and generation prompt.** High effort ends the prompt with `<|ifm|im_start|>assistant\n<ifm|think>\n`; medium uses `<ifm|think_fast>`; low uses `<ifm|think_faster>`.
- **The assistant `{% generation %}` span.**
- **Tools, system turns, assistant turns.** These stay text-only.

Ship the patched file as `chat_template.jinja` in the model repo. The tokenizer (which Miles uses to render prompts), the HF processor and SGLang's `/v1/chat/completions` all read it.

## Three components must agree

In RL, Miles renders the template to a string and runs your HF processor on it, which expands the pads. It then sends the expanded `input_ids` plus `image_data` (base64 PNG) to SGLang (`miles/rollout/sglang_rollout.py:167-216`). These three must produce the same layout, `<image>` + N × 250062 + `</image>`:

1. the template;
2. the HF processor;
3. `pad_input_ids` in the SGLang model ([step 3](03-sglang-model.md)).

## Data-format consequences

- **`<image>` placeholder.** Miles splits dataset prompt strings on the literal `<image>` to place images (`miles/utils/data.py:150-190`). This is the same string as K2's token. It works because the split happens before templating. Any `<image>` left in plain text tokenizes to a bare 250031 with no pads, so scrub it from text fields.
- **Assistant turns need a thinking field.** The template raises without one. For SFT targets, set `"reasoning_content": ""`. The turn renders as `<ifm|think>\n</ifm|think>answer<|ifm|im_end|>`.
- **Append-only rendering.** Nothing is emitted between `<|ifm|im_end|>` and the next `<|ifm|im_start|>`, and generation stops at `<|ifm|im_end|>` with no trailing byte (confirmed by RL360's `K2V3TITOTokenizer`). Prefix/suffix splits of a rendered conversation are safe.
