# Step 7: Evaluate the SFT checkpoint

This checkpoint is the baseline. RL in [step 8](08-rl.md) has to beat it. The harness, prompts and scores below are planned. Only serving and the text/image gates from [step 4](04-gate-checks.md) are validated.

## Serve

`tools/step04/serve.sh` supplies the registration variables and tested native SGLang launch on SM120 (TP=1, BF16, `torch_native` text attention, `sdpa` vision attention, batch-invariant kernels, reasoning parser `k2_horizon`). Point `--model-path` at the exported SFT checkpoint ([step 6](06-sft.md#export)). Its 2048-token context limits the generation budget remaining after image/prompt tokens; choose context and KV capacity for the benchmark protocol. Increase concurrency only after measuring memory and throughput. Changed limits, GPUs or attention backends need the [step 4](04-gate-checks.md) text and image gates rerun on the target.

Chat completions separate reasoning from the answer through the parser. Use the runtime pinned in [step 3](03-sglang-model.md).

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
| POPE | Pin selection | Object hallucination |

Pin the harness version, dataset revisions, exact splits (and POPE selection), image budget, reasoning effort, sampling settings and max tokens before the first run.

## Text regression

IFM's [Eval360](https://github.com/ifm-ai/Eval360/blob/30e14ec1d4127dcaf8e07317968c7aee3d313601/model_zoo/k2_horizon_7b_base_full.yaml) has a K2 [suite](https://github.com/ifm-ai/Eval360/blob/30e14ec1d4127dcaf8e07317968c7aee3d313601/group_zoo/k2_horizon_7b_base_full.yaml) with prompts, parsers, BOS settings, task-specific decoding overrides and dataset revisions/hashes (for example [GSM8K](https://github.com/ifm-ai/Eval360/blob/30e14ec1d4127dcaf8e07317968c7aee3d313601/data_zoo/k2_horizon_7b_base_full/gsm8k_cot_zeroshot.yaml)). Use it as a base-model text protocol reference; adapt it for our instruct template and keep image evaluation separate.

Run one text benchmark, for example GSM8K, through stock K2 and Horizon-V with matching token IDs and generation settings. For a deterministic, matched protocol, generated scores should agree; compare token logprobs as well, since equal scores can hide logit changes. Investigate differences in weights, tokenization, template, decoding and numerical backend.

## Record

For each checkpoint, record:

- benchmark scores at a fixed reasoning effort and max tokens;
- the reasoning effort, image budget and generation settings used;
- the checkpoint's provenance, meaning the SFT run, the export hash and the serve command;
- mean response length;
- the rate of responses whose think block never closes.

Use the same settings for the RL checkpoints, so the tables compare directly.
