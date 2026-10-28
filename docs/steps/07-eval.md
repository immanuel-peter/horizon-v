# Step 7: Evaluate the SFT checkpoint

This checkpoint is the baseline. RL in [step 8](08-rl.md) has to beat it.

## Serve

On a GPU node with the project venv active ([step 0](00-environment.md)) and the registration variables from the [SGLang handoff](../handoff/03-sglang-model.md#checkpoint-and-launch) exported:

```bash
export HF_HUB_OFFLINE=1
python -m sglang.launch_server \
  --model-path /models/K2-Horizon-V-sft \
  --trust-remote-code --enable-multimodal --image-processor-backend pil \
  --reasoning-parser k2_horizon
```

`--reasoning-parser k2_horizon` separates the reasoning from the answer in chat completions. Use the runtime pinned in [step 3](03-sglang-model.md).

## Benchmarks

Run lmms-eval or VLMEvalKit against the OpenAI-compatible endpoint.

| Benchmark | Split | Measures |
|---|---|---|
| MMMU | val | College-level multimodal reasoning |
| MathVista | testmini | Visual math |
| ChartQA | test | Chart reading |
| DocVQA | val | Document OCR + QA |
| TextVQA | val | Scene text |
| MMBench | dev | General perception |
| POPE | — | Object hallucination |

Run one text benchmark too, for example GSM8K. The result must match stock K2 exactly; any difference means the frozen weights or the text path changed.

## Record

For each checkpoint, record:

- benchmark scores at a fixed reasoning effort and max tokens;
- the reasoning effort used;
- mean response length;
- the rate of responses whose think block never closes.

Use the same settings for the RL checkpoints, so the tables compare directly.
