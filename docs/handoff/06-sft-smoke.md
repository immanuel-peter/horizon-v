# Step 06 real-data SFT smoke handoff

The real-data projector SFT path runs through pinned Miles, Ray and FSDP2 on one and two SM120 GPUs. It loads the pinned dataset lazily, applies the certified attention, packing and FP32-projector policy, keeps all 666 frozen FP32 masters equal to the promoted init, and saves and restarts exactly. This was a bounded systems smoke of 281 optimizer updates on a 320-row subset. It was not a full run, an LR sweep or a quality evaluation. [The structured report](../evidence/step06/report.json) holds every number below, together with the [selected rows](../evidence/step06/selection.json) and the [control run's per-update log](../evidence/step06/g1-control-steps.csv).

One finding needs attention before a long run. The Miles FSDP path is not bitwise reproducible across processes (see [Reproducibility](#reproducibility)). Restarting from the same checkpoint reproduces the next update bitwise, but two uninterrupted runs drift apart after a few updates.

## Machine, runtime and inputs

- Host `brev-pvj5tzwrv`: 2× RTX PRO 6000 Blackwell Server Edition (SM120, 97,887 MiB), driver 595.91.07 (reports CUDA 13.2), AMD EPYC 9355 (30 vCPUs), 283 GiB RAM, 1.4 TB root disk. No volumes or containers were carried over from the step 4 machine.
- Container `k2hv-sft-smoke`, built from `lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae` with `--gpus all --ipc=host --shm-size=64g` and the repo bind-mounted at `/workspace`. Python 3.12.15, torch 2.14.1+cu130 (CUDA runtime 13.0, not the driver's 13.2), Transformers 5.19.0, Triton 3.8.0, Ray 2.58.0. FA2/FA3 are absent.
- Miles `5078ff7c` with the committed patch (sha256 `f5dec690…`) and SGLang `b8d46f2f` with its patch (`5b238df8…`), applied by `scripts/gates/activate.sh`. All 9 Miles and 19 SGLang audited files were verified. The development lock was not synced into the container.
- The 16 certified compute sources (adapter, rollout, packed attention, numerics, SGLang adapter and HF model code) are byte-identical to the [step 4 report](../evidence/step04/report.json).
- Init assets come from `immanuelpeter/K2-Horizon-V-init@fc1f1c88`: four shards, the index, `tokenizer.json` and `build_info.json`. All seven match the archive manifest, and the shard hashes match the [init handoff](01-init-checkpoint.md). Tracked HF files were not overwritten, the 23-file manifest was regenerated, and the init shards were made read-only.
- The dataset is `immanuelpeter/Vision-PT-Data@87767e09` (`vision-pt-v1`). It was fetched with shard verification, materialized with `--target-root /workspace/data/vision-pt` so Ray workers see the image paths, and passed `check` with all nine bucket counts. All 857,284 SFT image paths resolve inside the container.

## Subset

`scripts/sft/select_subset.py --seed 0` selects 320 training rows from `default/sft/sweep50k` and 64 held-out rows from `default/sft/validation`. Rows are stratified by task and expanded-length band (<256, 256–511, 512–1023, ≥1024 tokens), every source is covered, and the longest rows by total, image and supervised tokens are always included. The training rows cover all 14 tasks and 15 sources. They hold 198,855 tokens, of which 92 rows are at least 1,024 tokens and the longest is 1,139. `train_eval.parquet` (every fifth training row) measures training loss. Reruns produce byte-identical files. The two splits share no rows, and no pending or RL rows are used.

## Results

| Check | Result |
|---|---|
| Real-data launch | PASS after one fix (below). `scripts/sft/launch.sh` uses `LazyRolloutDataSource`, `k2hv.sft_rollout.generate_rollout`, `sft_loss`, SDPA, true-on-policy and gradient checkpointing, without `--apply-chat-template` and without `K2HV_MILES_GATE` |
| Sample construction | PASS. 6,416 prepared samples across 28 runs: exactly one leading BOS, image pads equal to merged grid patches, one start/end marker pair per image, mask covering only the answer and `<\|ifm\|im_end\|>`. All 320 rows' token, image-token and supervised counts equal the dataset index |
| Trainable set and precision | PASS. Exactly the six projector tensors are trainable, as FP32 local shards on every rank. Gradient checkpointing is on |
| Gradients | PASS. Every update's grad norm is finite and nonzero. Every projector-output gradient (one per image, through the frozen LM) is finite and nonzero. Update 0 runs at LR 0 by schedule design (warmup starts at `lr_warmup_init=0`; Miles logs the post-step LR). The other 47 updates changed the projector |
| Frozen masters | PASS. 666/666 exact in 5 checkpoints: 1-GPU control, 1-GPU restart, 1-GPU traced restart, 2-GPU traced restart and 2-GPU replicate. All have the same digest `e7d66180…` |
| Lazy loading and memory | PASS. Decoded images equal batches × batch size exactly. Rollout RSS stays at 3.2–3.4 GiB (gb16) or 7.4–7.5 GiB (gb64), and live PIL images stay at three batches, so memory follows the batch, not the corpus. Single-threaded decode and processing run at about 35 rows/s, 10× what the trainer consumes |
| Trainer host memory | PASS, with a note. The first checkpoint save adds a one-time ~42 GiB CPU staging buffer (5.5 → 48 GiB on 1 GPU; 26 GiB per rank on 2 GPUs). It stays flat across later saves (four saves tested) |
| Save/restart state | PASS. LR scheduler, RNG, meta, data-source cursor and permutation (mid-epoch and across the epoch boundary) and batch record IDs are exact. All loads were logged with no failures or skips. The prefetched restart batch is replayed from the executor snapshot |
| Restart from the same checkpoint | PASS, bitwise. 1 GPU: the restored projector, the first resumed update's loss and grad norm, and the projector after it are identical to the uninterrupted run. 2 GPUs (shard, traced) and 2 GPUs (replicate): every resumed update is identical (4 and 2 updates) |
| Resume chain versus control, declared bounds | FAIL. Loss 1e-3, grad norm 1e-3 relative, projector 1e-4. The restarted chain's first leg already differs from the control because the runs are not reproducible. The bounds were not widened |
| Two-GPU FSDP | PASS. Both ranks train, each writes one shard (`__0_0`, `__1_0`), and memory falls to 26–29 GiB allocated per rank. The first update's grad norm is bitwise equal to 1 GPU on the same batch (475.69561767578125 at gb16, 308.1412353515625 at gb64); loss differs by 0.49% against a declared 1% bound |
| CPU regressions | PASS. 220 passed, 0 failed, 0 skipped (`K2HV_SGLANG_SOURCE` set to the pinned checkout). No stale source-pin failures |

### Throughput and memory

Medians after two warmup updates. Step time includes Miles's forward-only log-prob pass, which SFT does not need and which takes about a third of each step.

| Run | GPUs | Global / micro batch | s/update | Samples/s | Tokens/s | Peak alloc / reserved per GPU (GiB) | NVML peak (MiB) |
|---|---|---|---|---|---|---|---|
| g1-control (48 updates) | 1 | 16 / 4 | 7.76 | 2.06 | 1,278 | 43.5 / 49.9 | 52,519 |
| g1-size | 1 | 64 / 8 | 30.84 | 2.08 | 1,315 | 47.5 / 54.5 | 60,515 |
| g1-size | 1 | 64 / 16 | 31.79 | 2.01 | 1,288 | 54.2 / 66.1 | 75,209 |
| g2-measure, FSDP shard | 2 | 16 / 4 | 6.73 | 2.38 | 1,493 | 25.7 / 34.9 | 36,775 |
| g2-size, FSDP shard | 2 | 64 / 8 | 22.71 | 2.82 | 1,813 | 29.4 / 49.8 | 55,687 |
| g2-size, FSDP shard | 2 | 64 / 16 | 21.09 | 3.04 | 1,968 | 38.5 / 80.6 | 90,143 |
| g2-size, `--dp-replicate-size 2` | 2 | 64 / 8 | 18.15 | 3.53 | 2,259 | 47.1 / 57.5 | 63,605 |

Larger micro-batches do not help on one GPU. Sharding the frozen 9.4B FP32 masters over two GPUs scales only 1.36× because every micro-batch re-gathers them over PCIe. Replicating instead (`--dp-replicate-size 2`, HSDP with replicate 2 and shard 1) avoids the gathers and reaches 1.70×. Micro-batch 16 with sharding peaked at 90 GB of 97.9 GB, too close to the limit. Startup to the first update takes about 1 minute. A checkpoint is 36 GiB of model (FP32, every tensor), 257 MiB of optimizer state and ~10 MiB of rollout replay per sample in the batch. Saving takes 35–45 s on 1 GPU and 21 s on 2 GPUs.

### Losses

These come from a short run on a tiny subset, so they say nothing about generalization. Token-weighted loss on 64 held-out validation rows: init 6.080, after the 48-update 1-GPU control 1.648, after a 24-update 2-GPU schedule 4.249. On the 64-row training slice: 7.964, 1.791 and 5.222. The 1-GPU control's training loss went from 8.68 to 1.71. Grad norms start at 475 (clipped to 1.0) and settle near 1.

## Reproducibility

Two uninterrupted Miles runs with the same init, data, seed and flags match bitwise for the first updates, then differ by about 1e-4 relative in one update's gradients. Adam then amplifies the difference. The divergence step varied from update 0 to 9.

- The same packed micro-batches through HF with the same numerics policy are bitwise reproducible within and across processes.
- `torch.use_deterministic_algorithms(True)` raised no warnings and did not fix it. Neither did `CUDA_LAUNCH_BLOCKING=1` or `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Forward hooks that copy every layer's output to the host make 1-GPU runs reproducible for 9 updates instead of 1–2, but do not remove the problem.
- Packing, token counts and inputs were identical in every case.

The source is somewhere in the pinned Miles trainer path and was not isolated. Training is unaffected. Resume correctness was proven by restarting from the control's own checkpoint (see above), which separates restore errors from this drift. `K2HV_SFT_DETERMINISTIC` and `K2HV_SFT_TRACE` in `src/k2hv/sft_monitor.py` remain as diagnostics.

## Code revision

The runs used the code committed on `sft-real-data-smoke` (branched from `main` at `f1f39da`). `sft_monitor.py` gained its diagnostic switches and the merged `perf` parsing during the session, and `sft_resume_check.py` gained the replay and bitwise first-update checks; neither touches the training computation. All reports were regenerated with the committed versions.

## Fixes and failures

- `LazyRolloutDataSource` refused every launch because pinned Miles defaults `--tool-key` to `tools`. It now refuses only rows that carry that column, matching the pinned `Dataset`. A regression test covers the default and the refusal. Failed log: `outputs/sft-smoke/failed-attempts/shakedown-toolkey/`.
- Two restart runs were first launched with `--save-interval` but no `--save`, and Miles rejected them during argument parsing before any training. Logs are in `failed-attempts/*-save-interval-without-save/`.
- The resume chain against bounded tolerances failed as described above. It is kept, not rerun with looser bounds.

## Code

- `scripts/sft/launch.sh`: the reusable real-data launcher. `K2HV_GPUS` sets the actor GPUs, and the caller passes data, batch, schedule, load and save flags.
- `scripts/sft/select_subset.py`: deterministic stratified subset with `selection.json`.
- `src/k2hv/sft_monitor.py`: enabled by `K2HV_SFT_MONITOR=<dir>`. It wraps the rollout to check and time every sample, and hooks the trainer for per-rank memory and projector gradients. `summarize` and `sample` are CLIs. `k2hv/__init__.py` installs the trainer hooks only when that variable is set, so the certified adapter is unchanged.
- `src/k2hv/sft_eval.py`: token-weighted held-out loss of a checkpoint's FP32 projector.
- `src/k2hv/sft_resume_check.py`: control versus restarted run, covering state, optimizer, scheduler, RNG, meta, data source, logs, batch IDs, load events and bitwise first-update checks.

## Machine-only artifacts

Everything below stays under ignored `outputs/` on this machine. `outputs/sft-smoke/inventory-sha256.txt` hashes every checkpoint file, and `inventory-sizes.txt` lists their sizes.

- `outputs/sft-smoke/checkpoints/`: 381 files, 648,385,727,237 bytes (604 GiB) of DCP checkpoints from 13 runs. The inventory's SHA256 is `e43e1fe0c38f7099ca2deabdadc10096638c056681844349d0ec444cc9990859`; the 1-GPU control's final model shard `iter_0000048/model/__0_0.distcp` is `ba390ee49042a529848c64a6d3bec776ea9a9b31071373d4dd0d7fd4d8b545c4`. The `*-at<N>` directories are hard links to a control's checkpoint, used as restart sources.
- `outputs/sft-smoke/runs/<run>/`: `command.txt`, `train.log`, monitor JSONL and `summary.json` for 28 runs, including the diagnostics.
- `outputs/sft-smoke/{subset,eval,frozen,resume,env}/`, `tolerances.json`, `assemble.py`, `queue*.sh`, `run.sh`, `logs/` and `regression.xml`.
- `data/vision-pt` (74 GB, materialized), the init weights under `hf/`, `third_party/` checkouts, `.venv` and `.venv-container`, the pulled image, and the stopped container `k2hv-sft-smoke`.

Nothing was uploaded. No RL, dataset rebuild or full training was run.
