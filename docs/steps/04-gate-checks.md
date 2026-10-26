# Step 4: Gate checks

Pass all of these before any real training run. Each one catches a failure that would otherwise show up as a mysteriously bad loss curve.

| # | Check | Pass criterion |
|---|---|---|
| 0 | Load K2 in the GPU env ([step 0](00-environment.md)) | K2's config and tokenizer load under `transformers==5.19.0` |
| 1 | Text parity | No images: logits match stock K2, in both HF and SGLang |
| 2 | ViT parity | Truncated tower output equals a forward hook on GLM-5.3-Flash-Vision's `post_layernorm` |
| 3 | Token-count parity | `(ids == 250062).sum()` = `pixel_values.shape[0] // 4` = SGLang feature rows |
| 4 | Packing | Loss of a packed batch equals the mean of per-sample losses |
| 5 | Overfit | SFT on 64 samples drives loss to ≈ 0 |
| 6 | Logprob parity | HF vs SGLang logprobs on image prompts (random projector) are close |

## Notes

**0. Load check.** The public checkpoint uses the newer config format (`rope_parameters`) and tokenizer class `TokenizersBackend`. transformers 5.19 reads both natively. If loading fails anyway, apply the compatibility fix from RL360's `recipes/prepare_k2_horizon.py`.

**1. Text parity.** Run a few long text prompts. K2 uses grouped RMSNorm with 4 groups and RoPE base 1e7, so check that SGLang's `xllm.py` and K2's HF code agree, not only your wrapper.

**3. Token counts.** Check several image sizes, including non-square ones and sizes at the `min_pixels` / `max_pixels` limits.

**4. Packing.** Miles packs several samples into one sequence. Its stock arch adapter passes no `cu_seqlens` (`miles/backends/fsdp_utils/adaptations/arch_adapter.py:40`); it relies on HF flash-attention detecting sample boundaries from `position_ids` resets.

- If the losses differ, K2's remote-code attention bypasses HF's attention interface.
- Fix: write a small arch adapter modeled on `adaptations/specs/qwen3_5.py`, keyed on `model_type: k2_horizon_v`.

**5. Overfit.** Proves gradients reach the projector and the loss mask covers the right tokens. Print the masked tokens for one sample and read them.

**6. Logprob parity.** Mismatched train and rollout logprobs make GRPO's importance ratios meaningless. `--use-tis` in [step 8](08-rl.md) corrects small residual gaps, not large ones.
