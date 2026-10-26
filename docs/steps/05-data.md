# Step 5: Data

Baseten did not name its datasets. These are public stand-ins that match the recipe's shape: short-answer SFT data to teach the projector, then verifiable-answer data for RL.

## SFT

**Sources.** About 500k–1M samples from:

- LLaVA-1.5 `mix665k`, short-answer subsets (VQA, OCR, captions with short targets);
- `HuggingFaceM4/the_cauldron`, short-answer subsets.

**Row format** (parquet):

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nWhat is the total on the receipt?"},
    {"role": "assistant", "content": "$42.18", "reasoning_content": ""}
  ],
  "images": ["/data/sft/img/0000001.jpg"]
}
```

- `reasoning_content: ""` is required. K2's template raises on an assistant turn without a thinking field.
- One `<image>` placeholder per entry in `images`, in order.

**Rules:**

- **Images:** local paths or embedded bytes. Miles's image loading (`qwen_vl_utils`) handles local files, HTTP URLs and base64. It does not handle `s3://`.
- **Length:** filter offline. Do not use `--rollout-max-prompt-len` for SFT; its filter runs the processor on the raw message list.
- **Text:** scrub literal `<image>` strings from text fields ([step 2](02-processor-and-template.md)).
- **Dedupe** against every eval set in [step 7](07-eval.md).

## RL

**Sources:** ViRL39K or ThinkLite-VL-70k, plus geometry3k. Every row needs a verifiable answer.

**Row format** (parquet):

```json
{
  "problem": "<image>\nFind the measure of angle ABC. Put the final answer in \\boxed{}.",
  "answer": "42",
  "images": ["/data/rl/img/000123.png"],
  "metadata": {"data_source": "math_dapo"}
}
```

`metadata.data_source` picks the scorer in RL360's `async_rm` ([step 8](08-rl.md)):

| Answer type | `data_source` |
|---|---|
| Numeric / symbolic math | `math_dapo` |
| Multiple choice | `stem__supergpqa` |

## RL prompt selection

Follow IFM's protocol from RL360 (`src/agent360/harbor/miles/calibration.py`):

1. Take a 10–20k-prompt subset.
2. Sample the SFT checkpoint 16 times per prompt, at the RL temperature and reasoning effort.
3. Keep prompts with a pass rate between 20% and 40%.

Prompts the model always or never solves give every sample the same reward. GRPO assigns them zero advantage, so they waste rollouts.
