# K2-Horizon-V post-training plan

Give [IFM/K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B) vision. Attach the vision tower from [GLM-5.3-Flash-Vision](https://huggingface.co/immanuelpeter/GLM-5.3-Flash-Vision) and train only a new projector, following [Baseten's GLM-5.2-with-vision recipe](https://www.baseten.co/blog/glm-52-with-vision): projector SFT, then projector-only RL.

## Architecture

| Part | Source | Trained |
|---|---|---|
| Vision tower, through `post_layernorm` (1024-d per patch) | GLM-5.3-Flash-Vision, minus `downsample` + `merger` (159.4M params dropped) | No |
| Projector: `LayerNorm(1024)` → 2×2 merge → `Linear(4096, 4096)` → GELU → `Linear(4096, 4096)` | New, ~33.6M params | Yes |
| LLM | K2-Horizon-7B (hidden 4096, 36 layers, vocab 250624) | No |

Data flow:

```text
image ─► GLM image processor (patch 14) ─► ViT ─► [N_patch, 1024]
      ─► projector (2×2 merge) ─► [N_patch / 4, 4096]
      ─► scattered into K2's input embeddings at the pad-token positions

prompt: <image> + (N_patch / 4) × reserved_special_token_12 + </image>
        448×448 image → 32×32 patches → 256 image tokens
```

Both frozen parts stay bit-identical, so text-only behavior is exactly K2's.

## Steps

| # | Step | GPUs |
|---|---|---|
| 0 | [Environment (uv)](steps/00-environment.md) | No |
| 1 | [Build the init checkpoint](steps/01-init-checkpoint.md) | No |
| 2 | [Processor and chat template](steps/02-processor-and-template.md) | No |
| 3 | [SGLang model](steps/03-sglang-model.md) | No |
| 4 | [Gate checks](steps/04-gate-checks.md) | 1 GPU |
| 5 | [Data](steps/05-data.md) | Yes, for RL prompt filtering |
| 6 | [Stage 1: projector SFT](steps/06-sft.md) | 1 node |
| 7 | [Evaluate the SFT checkpoint](steps/07-eval.md) | 1 GPU |
| 8 | [Stage 2: projector-only GRPO](steps/08-rl.md) | 1 node |
| 9 | [Release](steps/09-release.md) | No |

The checkpoint is built and verified on CPU. The processor and image template are next.
See the [step 1 handoff](handoff/01-init-checkpoint.md).

Do steps 0–4 before spending GPU hours. Projects like this usually break on token-count or packing mismatches, not on hyperparameters.

## Stack

- **Environment:** uv. Laptop: `uv sync`. GPU nodes: a uv venv on top of Miles's Docker image. See [step 0](steps/00-environment.md).
- **Training:** upstream [Miles](https://github.com/radixark/miles) (commit `5078ff7`), FSDP backend. Pins `transformers==5.19.0`.
- **Rollouts and serving:** SGLang, with an out-of-tree model package.
- **Borrowed from [RL360](https://github.com/ifm-ai/RL360):** K2 GRPO settings, prompt selection, reward functions, the `k2_v3` reasoning parser. See [reference/rl360.md](reference/rl360.md).

## Reference

- [Key facts](reference/key-facts.md): token IDs, dimensions, source file references.
- [Compute budget](reference/compute.md): about 90–150 H100-hours total, including sweeps.
- [RL360](reference/rl360.md): what to borrow, what to ignore.
