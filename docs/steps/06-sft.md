# Step 6: Stage 1, projector SFT

Train only the projector on short-answer image data with upstream Miles, FSDP and no rollout engines. [Step 4](04-gate-checks.md) validated this path on the 64-example fixture on one SM120 GPU (real Miles FSDP packing, FP32 projector masters, 512-update overfit). The [real-data smoke](../handoff/06-sft-smoke.md) then ran the pinned dataset through the same path on one and two GPUs, including save/restart. No full run, LR sweep or quality evaluation has been done.

## What exists

| Piece | Behavior |
|---|---|
| `src/k2hv/sft_rollout.py` | Processes images and validates one user/assistant pair with explicit `think: ""`. Sets tokens, the answer mask and `multimodal_train_inputs` (`pixel_values`, `image_grid_thw`). |
| `src/k2hv/miles_gate_adapter.py` | `K2HorizonVAdapter` freezes everything except the projector, installs packed attention, keeps projector masters in FP32 and sets its FSDP2 precision. |
| `src/k2hv/packed_attention.py` | Document-sized SDPA calls inside one packed row, with packed linear layers. |
| `scripts/gates/miles_sft.sh` | The launch that passed the gate. |
| `scripts/sft/launch.sh` | Real-data launcher: the gate's flags with `LazyRolloutDataSource`, no `K2HV_MILES_GATE` capture hooks and caller-supplied data, batch, schedule and save flags. |
| `scripts/sft/select_subset.py` | Deterministic stratified subset of the configured SFT buckets for smoke runs. |
| `src/k2hv/sft_monitor.py` | Optional instrumentation (`K2HV_SFT_MONITOR=<dir>`): per-sample BOS/mask/image checks, decode timing, rollout and trainer memory, projector gradients. |
| `src/k2hv/sft_eval.py`, `src/k2hv/sft_resume_check.py` | Held-out loss of a checkpoint's projector; control versus restarted-run comparison. |

The loss covers only the answer and `<|ifm|im_end|>`, excluding prompt, image and empty-thinking tokens. `prepare_sample` tokenizes the rendered template with `add_special_tokens=False`, so a sequence starts with exactly one BOS. The [PR #7 recertification](../handoff/04-gate-checks.md) includes this correction on the 64-example fixture. Only the vision tower runs under `no_grad`; the frozen language model passes gradients back to the projector.

Do not pass `--apply-chat-template`. Miles would turn `sample.prompt` into a generation-prompt string and lose the assistant turn. With `--multimodal-keys` and no template flag, Miles keeps the message list and fills `sample.multimodal_inputs["images"]`. The row format is on the dataset card, linked from [step 5](05-data.md).

### Loading the dataset

On the training machine, run `uv run python -m k2hv.dataset fetch`, then `materialize` and `check`. The pin and bucket mapping are in [`configs/dataset.toml`](../../configs/dataset.toml). Set `--prompt-data` to the materialized `sft_train` bucket, use `sft_sweep` for short sweeps, and measure held-out loss on `sft_validation`. Keep the release's split assignments and exclude all pending buckets. These are the existing dataset-client commands; the builder is not needed.

Pinned Miles `Dataset` decodes every image when it is built and keeps it, about 3 MB per row here, which is terabytes for this export. Pass `--data-source-path k2hv.lazy_data.LazyRolloutDataSource`. It keeps the rows as Arrow text and image paths and decodes a sample's images only when `get_samples` returns it. Batching, epochs, the shuffle permutation and data-source checkpoints are the pinned `RolloutDataSource` code. `--prompt-data` can be a directory of export parts, read in sorted order, or a comma-separated list. It refuses `--apply-chat-template`, `--rollout-max-prompt-len` and rows with a tools column; the dataset release already validated every row's length. Pinned Miles defaults `--tool-key` to `tools`, so the source accepts that default when the rows have no such column, as the pinned `Dataset` does.

CPU evidence (`tests/test_lazy_data.py`; the original comparison run is archived with the dataset build):

- Against the pinned eager `Dataset` on a fixture, the lazy source returns the same prompts, images, metadata and order for three shuffled epochs. Save, restart and load reproduce an uninterrupted run, and a pickled dataset rebuilds the same sample in a spawned process.
- Building it over the 839,928-row v1 training split takes 1.1 s and 614 MB RSS and decodes no image. Each 256-row batch decodes 256 images (about 740 MB of resized RGB) in about 9 s on one thread. After each batch is released, RSS returns to 466 MB above the start across 12 batches, so memory follows the batch, not the corpus.

The [real-data smoke](../handoff/06-sft-smoke.md) ran the source inside Miles's Ray rollout actor with `k2hv.sft_rollout.generate_rollout`. Decoded images equal the samples requested, three batches at most stay live, and single-threaded decoding (about 35 rows/s) is 10× what two GPUs consume.

### Template compatibility

IFM's SFT rollout (`miles/rollout/sft_rollout.py` in [RL360](https://github.com/ifm-ai/RL360/blob/3e13010e0e47b7d08f19dfb6bdd3ac6d21dc200e/components/miles/miles/rollout/sft_rollout.py)) builds a text loss mask and never produces `multimodal_train_inputs`, so it cannot replace ours. The [process_entry training template](https://github.com/ifm-ai/process_entry/blob/a4745b10f5ce71e15d46021301e1ee717dc09bdc/tokenizer/chat_template.jinja) differs in reasoning newlines and drops non-string content such as images. Keep `hf/K2-Horizon-V-init/chat_template.jinja`.

The rollout is single-turn with empty thinking. Before extending it to multi-turn, nonempty reasoning, other efforts or several images, add tests for rendered bytes, token IDs and supervised spans on each case.

## Full run

The [real-data smoke](../handoff/06-sft-smoke.md) ran `scripts/sft/launch.sh` on one and two RTX PRO 6000 GPUs. It exercised lazy loading, packing, frozen masters, save/restart and both two-GPU layouts on 320 subset rows. The launcher keeps the gate's registration variables, `UV_NO_SYNC=1`, `--attn-implementation sdpa --true-on-policy-mode --gradient-checkpointing` and the SFT loss and rollout flags. It never sets `K2HV_MILES_GATE`.

Measured on that subset (medians):

| Layout | Global / micro batch | Samples/s | Peak reserved per GPU |
|---|---|---|---|
| 1 GPU | 64 / 8 | 2.08 | 54.5 GiB |
| 2 GPUs, FSDP shard | 64 / 8 | 2.82 | 49.8 GiB |
| 2 GPUs, `--dp-replicate-size 2` | 64 / 8 | 3.53 | 57.5 GiB |

Replicating the frozen weights beats sharding them because the shard layout re-gathers 9.4B FP32 masters over PCIe for every micro-batch. Miles also runs a forward-only log-prob pass that SFT does not use, about a third of each step. A checkpoint is about 37 GiB. Two uninterrupted runs are not bitwise reproducible; restarting from a checkpoint is.

Settings still to choose, none measured for quality:

| Setting | Starting value | Note |
|---|---|---|
| LR | 1e-3, cosine to 1e-5 | LLaVA's projector-pretraining value. Sweep {2e-4, 1e-3} on the 50k subset from [step 5](05-data.md). The smoke ran these flags only to exercise the scheduler. |
| Warmup | 3% | `--lr-warmup-fraction 0.03`. Warmup starts at LR 0, so update 0 does not move the projector. |
| Batch | 256 | Memory depends on the micro-batch, not the global batch; 64 was the largest global batch run. |
| Weight decay | 0 | Matches the gate. |
| Epochs | 1 | |

Multi-node, other GPUs and `flash_attention_2` are untested here (flash-attn 2/3 are absent from the step 4 image).

## Check the result

- `python -m k2hv.verify_frozen_master --checkpoint <iter dir> --report <path>` compares every frozen FP32 checkpoint tensor with the promoted init value.
- Track the training loss from the Miles log and a held-out loss on the validation split from [step 5](05-data.md). `miles_training_report` and `miles_sft_evaluate` depend on the gate's captured fixture, so they are regression checks on the 64 examples, not bulk-run metrics.
- Confirm the trainable set stays the six projector tensors.
- Record the model/code revisions, dataset pin and bucket mapping, materialization root, any training subset's selected record IDs and sampling seed, and the run settings. A dataset release change is an input change and must be recorded separately.

## Export

Miles writes a distributed FP32 checkpoint. The verified readers are `verify_frozen_master` and `miles_sft_evaluate`, which also writes `*.projector.safetensors` with the six projector tensors. A full FSDP-to-HF converter is not validated here, and Miles's `tools/convert_fsdp_to_hf.py` has not been run against this model.

Only the projector changes, so the proposed export is the init HF repo with the projector tensors replaced. Verify it before step 7:

1. Keep every non-weight file (code, tokenizer, template, processor) identical to init, and regenerate the manifest with `python -m k2hv.checkpoint_io manifest <dir>`.
2. Reload through HF and serve it with `scripts/gates/serve.sh --model-path <dir>`, then compare projected features and logprobs with the Miles checkpoint, as the cache gate's fresh-engine stage does.

Keep experiment checkpoints separate from the init directory.
