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

## Explicit GPU runners

The first SM120 run is recorded in [the step 04 handoff](../handoff/04-gate-checks.md).
Step 04 is **not complete**. A native smoke request passes, but numerical parity
fails and several required engine/training checks remain unimplemented.

Run the CUDA checks separately from pytest in an activated GPU venv with
`UV_NO_SYNC=1`. Do not sync `uv.lock` into that environment.

```bash
env -u LD_LIBRARY_PATH python -m k2hv.gpu_gates \
  --only cuda load_hf text_hf vision_hf packing --report outputs/step04/gates.json
```

This runner writes each result immediately and returns nonzero whenever any
required gate fails, is unselected, or has no implementation. Its packing probe
uses the actual pinned Miles actor input method and a truly concatenated row
with position resets. Single-document controls retain the packed tensor shape
and offsets to test isolation without changing BF16 matrix-kernel shapes.
Unpadded separate forwards have also been measured and do not pass the original
loss tolerance. The controlled isolation result alone does not close packing.

Teacher-forced comparisons run in two sequential stages. Stop the SGLang engine
before capture; release the HF model before starting the native engine.

```bash
env -u LD_LIBRARY_PATH python -m k2hv_sglang.logprob_gate capture \
  --reference outputs/step04/hf-reference.json
# Start the native engine using tools/step04/serve.sh inside the container.
python -m k2hv_sglang.logprob_gate evaluate \
  --reference outputs/step04/hf-reference.json --report outputs/step04/logprob.json
```

The initial BF16 limits are maximum error 0.2 and mean error 0.03 in natural-log
units. These were declared before comparing native results. Image-pad positions
are excluded because SGLang replaces their IDs with hashes; every compared
non-image token ID and position is checked. These limits have not been widened.
A comparison failure is a failed gate, even when generation succeeds.

`python -m k2hv.gate_training --report outputs/step04/overfit-diagnostic.json`
runs a bounded 256-update, 64-image, projector-only diagnostic. It uses the real
processor, `k2hv.sft_rollout` preparation and the pinned Miles actor input method,
but **does not execute the Miles SFT actor training loop**. It cannot pass the
required Miles overfit gate. The fixture questions do not contain the answers.
The mask includes the answer and turn end, excludes the empty think block, and
validates the decoded span. Experimental parameters are restored in memory;
no init weight file is changed and no trained model is exported.
