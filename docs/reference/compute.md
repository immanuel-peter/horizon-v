# Compute budget

Estimates in H100-hours, assuming about 40% MFU for training and single-GPU SGLang engines for generation. Measure real throughput in the first runs and update this page.

| Item | H100-hours | Basis |
|---|---|---|
| Gate checks (step 4) | 2–5 | Single-GPU parity and overfit runs |
| SFT LR sweep, 2 × 50k samples | 3–5 | ~1.5–2.5 h per run |
| SFT full run, ~600k samples | 15–20 | 0.4B tokens × ~3 forward equivalents ≈ 2e19 FLOPs |
| RL prompt selection | 15–25 | 10–20k prompts × 16 samples × ~3k tokens |
| RL, 1–2 runs of ~300 steps | 40–100 | Generation-dominated; 64 prompts × 8 samples per step |
| Eval, per checkpoint | 1–2 | Seven benchmarks plus a text check |
| **Total** | **~90–150** | Within the 50–200 range estimated earlier |

## Main cost drivers

- **RL response length.** Cost scales almost linearly with thinking length. Medium effort roughly halves it ([step 8](../steps/08-rl.md)).
- **Image tokens per sample.** `max_pixels` sets the cap ([step 2](../steps/02-processor-and-template.md)).
- **Weight sync.** Unpatched, Miles sends ~16 GB per RL step. The projector-only patch sends ~67 MB.

## Why training is cheap

- Optimizer state covers ~33.6M params only.
- No weight gradients for the frozen parts. Backward still runs through the LLM's activations to reach the projector.
- No ViT activations stored (`no_grad`).
- No reference model in RL (KL 0).
