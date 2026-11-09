# Step 5: Data

This repo does not build training data. It consumes the standalone [Vision Post-Training Dataset](https://huggingface.co/datasets/immanuelpeter/Vision-PT-Data), pinned in [`configs/dataset.toml`](../../configs/dataset.toml). That file is the only place in the repo that names a dataset revision or export directory. Bump the revision there to move to a new dataset release.

The dataset card carries the row format, the source list, the licence terms and the split rules. This page covers only what this repo does with the data.

## Get the data

```bash
uv run python -m k2hv.dataset fetch        # download the pinned release and unpack the images (about 78 GB)
uv run python -m k2hv.dataset materialize  # write Miles Parquet with absolute image paths for this machine
uv run python -m k2hv.dataset check        # export checksums, bucket row counts, RL selector values
```

`fetch` downloads into the `local_root` from the pin (`data/vision-pt`, git-ignored) and unpacks the 20 image tar shards there. `materialize` rewrites the portable `images/...` paths in each row to absolute paths under that root and writes the result under `exports/vision-pt-v1/miles/`. Training reads the `miles/` tree. `check` is a light consumer check. The full verification runs on the dataset side before a release is pinned here.

The pinned revision was verified from a fresh download on 2026-11-08: all archive checksums passed, and the full verify reported 1,003,854 of 1,003,854 rows with 0 mismatches and 627,987 images with 0 bad images.

## What this repo trains on

| Bucket | Used by |
|---|---|
| `default/sft/train` | [Step 6](06-sft.md) projector SFT |
| `default/sft/validation` | Held-out SFT loss in [step 6](06-sft.md) |
| `default/sft/sweep50k` | Short SFT runs and the LR sweep in [step 6](06-sft.md) |
| `default/rl/pool` | [Step 8](08-rl.md) GRPO prompts |
| `default/rl/heldout` | Held-out RL prompts in [step 8](08-rl.md) |

The `pending` buckets are not trained on. Each pending row names its reason in `metadata.pending_reason`.

SFT rows are one user and assistant turn with `think` markers for the empty reasoning block. [Step 6](06-sft.md) documents how Miles loads them through `k2hv.lazy_data.LazyRolloutDataSource`.

## RL

RL rows select a scorer in `metadata.data_source`. The current release uses `math_v1`, `choice_v1` and `decimal_v1`. The first build used the spellings `k2hv_math_v1`, `k2hv_choice_v1` and `k2hv_decimal_v1` for the same three checks. [`src/k2hv/reward.py`](../../src/k2hv/reward.py) accepts both spellings and raises on anything else, so old and new downloads both train. The scorer contract and its tests are the consumer side of the dataset interface and stay in this repo.

`pending_decimal_precision` rows stay untrainable until a question-specific precision rule exists. Do not enable a blanket tolerance.

### RL prompt selection after SFT

This GPU-dependent stage waits for the [SFT checkpoint](06-sft.md). Follow IFM's protocol from RL360 (`src/agent360/harbor/miles/calibration.py`):

1. Take a 10 to 20k prompt subset of `default/rl/pool`.
2. Sample the SFT checkpoint 16 times per prompt, at the RL temperature, image budget and reasoning effort used for training and evaluation.
3. Keep prompts with a pass rate between 20% and 40%.

Use our checkpoint's measured pass rates, not difficulty or pass-rate annotations from other models. Prompts the model always or never solves give every sample the same reward, so GRPO assigns them zero advantage.

## Where the builder lives

The pipeline that built this data (source audit, overlap exclusion, splits, export and the full verify) lives in the separate `vision-pt-builder` repository, copied out of this repo at the commit that built the current release. Dataset releases are versioned there and on the Hub. This repo records only the pin above.

The earlier in-repo build, its source audit and its build results are preserved in the dataset archive at the previous revision and in git history. Nothing in steps 6 to 9 depends on the builder being present.
