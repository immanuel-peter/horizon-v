# Step 6: Stage 1, projector SFT

Train only the projector on short-answer image data with upstream Miles, FSDP and no rollout engines. [Step 4](04-gate-checks.md) validated this path on the 64-example fixture on one SM120 GPU (real Miles FSDP packing, FP32 projector masters, 512-update overfit). It did not run bulk data, multiple GPUs or any other GPU/attention backend. Everything under "Full run" below is proposed and unmeasured.

## What exists

| Piece | Behavior |
|---|---|
| `src/k2hv/sft_rollout.py` | Processes images and validates one user/assistant pair with explicit `think: ""`. Sets tokens, the answer mask and `multimodal_train_inputs` (`pixel_values`, `image_grid_thw`). |
| `src/k2hv/miles_gate_adapter.py` | `K2HorizonVAdapter` freezes everything except the projector, installs packed attention, keeps projector masters in FP32 and sets its FSDP2 precision. |
| `src/k2hv/packed_attention.py` | Document-sized SDPA calls inside one packed row, with packed linear layers. |
| `tools/step04/miles_sft.sh` | The launch that passed the gate. |

The loss covers only the answer and `<|ifm|im_end|>`, excluding prompt, image and empty-thinking tokens. Only the vision tower runs under `no_grad`; the frozen language model passes gradients back to the projector.

Do not pass `--apply-chat-template`. Miles would turn `sample.prompt` into a generation-prompt string and lose the assistant turn. With `--multimodal-keys` and no template flag, Miles keeps the message list and fills `sample.multimodal_inputs["images"]`. The row format is in [step 5](05-data.md).

### Template compatibility

IFM's SFT rollout (`miles/rollout/sft_rollout.py` in [RL360](https://github.com/ifm-ai/RL360/blob/3e13010e0e47b7d08f19dfb6bdd3ac6d21dc200e/components/miles/miles/rollout/sft_rollout.py)) builds a text loss mask and never produces `multimodal_train_inputs`, so it cannot replace ours. The [process_entry training template](https://github.com/ifm-ai/process_entry/blob/a4745b10f5ce71e15d46021301e1ee717dc09bdc/tokenizer/chat_template.jinja) differs in reasoning newlines and drops non-string content such as images. Keep `hf/K2-Horizon-V-init/chat_template.jinja`.

The rollout is single-turn with empty thinking. Before extending it to multi-turn, nonempty reasoning, other efforts or several images, add tests for rendered bytes, token IDs and supervised spans on each case.

## Full run

Create a bulk launcher from `tools/step04/miles_sft.sh` in the step 4 container/environment. Preserve the registration variables, `UV_NO_SYNC=1`, `--attn-implementation sdpa --true-on-policy-mode --gradient-checkpointing` and the SFT loss/rollout flags. Replace the fixture input, run length and save path with the step 5 dataset and a separate output directory.

Remove its `K2HV_MILES_GATE` export before using new data: it enables capture hooks and writes sample tensors into the step 4 evidence directory. The script's one actor GPU, 4-sample batch and constant 1e-3 LR are the tested gate settings; the full-run settings below need measurement.

Proposed settings to sweep, none yet measured on bulk data:

| Setting | Starting value | Note |
|---|---|---|
| LR | 1e-3, cosine to 1e-5 | LLaVA's projector-pretraining value. Sweep {2e-4, 1e-3} on the 50k subset from [step 5](05-data.md). The gate used a constant 1e-3. |
| Warmup | 3% | Not in `miles_sft.sh`; add the Miles LR flags. |
| Batch | 256 | Gate used 4. Memory at this size is unmeasured. |
| Weight decay | 0 | Matches the gate. |
| Epochs | 1 | |

Measure before committing compute. The gate peaked at 44.4 GB for SFT, 75.5 GB for the full packing check, on 97.9 GB. Time a few hundred steps to get tokens per second, peak memory and the cost of a 256-sample batch, then size the run. Multi-GPU FSDP, other GPUs and `flash_attention_2` are untested here (flash-attn 2/3 are absent from the step 4 image).

## Check the result

- `python -m k2hv.verify_frozen_master --checkpoint <iter dir> --report <path>` compares every frozen FP32 checkpoint tensor with the promoted init value.
- Track the training loss from the Miles log and a held-out loss on the validation split from [step 5](05-data.md). `miles_training_report` and `miles_sft_evaluate` depend on the gate's captured fixture, so they are regression checks on the 64 examples, not bulk-run metrics.
- Confirm the trainable set stays the six projector tensors.

## Export

Miles writes a distributed FP32 checkpoint. The verified readers are `verify_frozen_master` and `miles_sft_evaluate`, which also writes `*.projector.safetensors` with the six projector tensors. A full FSDP-to-HF converter is not validated here, and Miles's `tools/convert_fsdp_to_hf.py` has not been run against this model.

Only the projector changes, so the proposed export is the init HF repo with the projector tensors replaced. Verify it before step 7:

1. Keep every non-weight file (code, tokenizer, template, processor) identical to init, and regenerate the manifest with `python -m k2hv.checkpoint_io manifest <dir>`.
2. Reload through HF and serve it with `tools/step04/serve.sh --model-path <dir>`, then compare projected features and logprobs with the Miles checkpoint, as the cache gate's fresh-engine stage does.

Keep experiment checkpoints separate from the init directory.
