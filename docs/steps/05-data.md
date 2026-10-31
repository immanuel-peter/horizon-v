# Step 5: Data

Build reproducible image-QA datasets on a Linux CPU machine. SFT teaches the projector to connect images to short answers; RL later uses questions whose answers a rule-based scorer can verify. Dataset preparation needs no model weights or GPU. RL difficulty calibration needs the SFT checkpoint and runs after [step 6](06-sft.md).

The pipeline is `python -m k2hv.data` (`src/k2hv/data/`), driven by the recipe [`configs/data/step05.toml`](../../configs/data/step05.toml). Every source revision, rule, limit, quota, overlap threshold and split parameter lives in the recipe, and its hash is in every build manifest. Results of the run on 2026-10-31 are under [Build results](#build-results); compact evidence is in [`docs/evidence/step05/`](../evidence/step05/).

A CPU data audit cannot establish model quality. Mixture weights, the value of each capability and the effect of the 1,024-image-token budget on document reading need the SFT sweep and [step 7](07-eval.md) evaluations.

## Setup

```bash
uv sync --locked            # includes the data group: pyarrow, qwen-vl-utils==0.0.14
uv run hf download immanuelpeter/K2-Horizon-V-init --revision fc1f1c887facf728e2e02a13f9bf9a920af44552 \
  --include tokenizer.json --local-dir hf/K2-Horizon-V-init
git clone https://github.com/radixark/miles third_party/miles && git -C third_party/miles checkout 5078ff7cd6387f553db010bb38148a59e9e12cb0
git clone --filter=blob:none https://github.com/ifm-ai/RL360 third_party/rl360 && git -C third_party/rl360 checkout 3e13010e0e47b7d08f19dfb6bdd3ac6d21dc200e
```

`HF_TOKEN` must be in the environment. The data root defaults to `data/step05` (git-ignored; override with `--data-root` or `K2HV_DATA_ROOT`).

### Why the Miles code is loaded selectively

Importing `miles.utils.data` or `miles.utils.processing_utils.load_processor` pulls in SGLang through the rollout and chat-template packages. `k2hv.data.runtime` instead checks that `third_party/miles` is at the pinned commit, verifies the sha256 of `miles/utils/{data,processing_utils,types}.py`, and executes the pinned `_build_messages` and `read_file` source extracted from `data.py`, the pinned `process_vision_info`, and `Sample`. `load_processor` reduces to `AutoProcessor.from_pretrained(path, trust_remote_code=True)` for this checkpoint. `qwen_vl_utils` imports torchvision only for its video readers; a stub exists only during that import, so transformers still sees no torchvision. RL360's `math_dapo` and `supergpqa` scorer modules run verbatim from the pinned checkout, with their process-based timeout decorator replaced by identity.

## Stages

| Stage | Does |
|---|---|
| `inventory` | Pin listings (sizes, LFS sha256), schemas, row counts and preserved official splits of every recipe dataset and evaluation set |
| `scan` | Per source shard: read pinned rows, apply text rules, fetch and decode every image, store it content-addressed, compute hashes; write candidates, rejects and a done marker |
| `evals` | Index every evaluation image and question |
| `select` | Evaluation exclusion, near-duplicate grouping, QA deduplication, RL reservation, quotas and group-level splits |
| `validate` | Run every selected row through the pinned Miles path and the processor/supervision contract |
| `export` | Final quotas over validated rows; portable Parquet, indices, image list, checksums, manifest and quality report |
| `materialize` | Rewrite image paths to absolute paths under a target root |
| `verify` | Re-hash and decode every exported image; check every Miles row equals its validated row |
| `build` | All of the above except `inventory` |
| `inspect` | HTML sheets of exact model inputs with decoded supervision, and candidate/evaluation pair sheets for exclusions |

```bash
uv run python -m k2hv.data build --profile pilot   # ~1,000 SFT examples plus RL samples
uv run python -m k2hv.data build --profile full
```

Resume by rerunning the same command. Every stage writes outputs atomically (temporary file, then rename) followed by a done marker that fingerprints its inputs: the recipe slice and scan code for a shard; the processor files, limits, contract code and Miles pin for a validation chunk. Completed work with a matching fingerprint is reused; anything else is recomputed. Every sampling decision (per-image turn caps, quotas, splits, the sweep subset) orders records by `sha256(seed:salt:record_id)` or by image group, never by processing order. A hard-killed and resumed build therefore reproduces the same records, splits and file bytes ([evidence](#reproducibility)).

Workers are spawned processes with one torch/OMP/MKL thread each: 14 scan workers and 15 validation workers on the 16-vCPU machine, and 12 threads for URL fetches.

## Sources

The audit inspected Hub metadata, cards and a deterministic sample of every candidate, then a 1,000-example real-data pilot. "Terms" is the Hub metadata license where one exists. Licenses marked *secondary* come from the license tables of the Bee paper ([arXiv 2510.13795](https://arxiv.org/abs/2510.13795)) or a CVPRW 2025 supplemental and were not verified against the original projects. FineVision, The Cauldron and the RL sources state that component licenses apply.

Status: **include** (default export), **addon** (built, validated and exported separately; terms need a decision), **pending** (audited, blocked on a decision), **exclude** (audited, not built).

### SFT candidates

| Source @ revision | Component | Status | Evidence |
|---|---|---|---|
| [HuggingFaceM4/FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) `3c380a7` | `vqav2`, `visual7w`, `vsr`, `tallyqa`, `clevr` (13 of 22 files), `textvqa`, `st_vqa`, `docvqa`, `infographic_vqa`, `ai2d_merged`, `plotqa` (6 of 18 files) | include | Single image; independent questions bundled per image; short explicit answers. No upstream IDs (`images[].path` is null), so identity is the image hash. Every row is from the Cauldron `train` packaging, which does not prove original-split hygiene; the overlap stage checks images against the evaluation sets directly. |
| | `dvqa` (3 of 10 files), `iconqa`, `chartqa` | addon | DVQA CC BY-NC 4.0 (secondary; another paper says CC BY 4.0), IconQA CC BY-NC-SA 4.0 (secondary), ChartQA GPL-3.0 (HuggingFaceM4/ChartQA metadata). |
| | `ocrvqa` | exclude | Labels come from Amazon metadata: titles include series text absent from the cover, and "Is this a *category* book?" uses store categories. |
| | `a_okvqa`, `aokvqa` | exclude | Two copies of A-OKVQA inside FineVision; answers mix rationales ("Answer: … Rationale: …") with choices. |
| | `cocoqa`, `tqa`, `scienceqa`, `figureqa` | exclude | Caption-generated ungrammatical questions; answers with appended commentary ("The correct answer is B. 6. This relates to…"); knowledge questions with low visual dependency (also MathVista sources); yes/no only (ThinkLite covers FigureQA for RL). |
| | `CoSyn_400k_*` | exclude | Same data as allenai/CoSyn-400K, which is used as the canonical copy. |
| [allenai/CoSyn-400K](https://huggingface.co/datasets/allenai/CoSyn-400K) `86e46e1`, ODC-BY | `chart` (26 of 51 files), `table`, `document` (26 of 41), `diagram` (20k quota) | include | At this revision every config uses `qa_pairs` (parallel `question`/`explanation`/`answer` lists); the card's `questions` field does not occur. `explanation`, `code` and `data` stay out of model inputs (hashes in provenance). Answers with more than two decimals are rejected as unrounded computations. Rows with identical `data` (alternative renderings) share a split group. Diagram answers are often vague relation words ("Affects", "All actions"), hence the reduced quota. Official `validation` splits preserved. |
| [opendatalab/ChartVerse-SFT-600K](https://huggingface.co/datasets/opendatalab/ChartVerse-SFT-600K) `07b1576`, Apache-2.0 | | exclude | The explicit `answer` is a computed result: 30% of pilot answers have unrounded decimals (`136.1013215859031`), and inspected integer answers are multi-condition computations over dense charts at the token cap. With empty thinking these targets teach guessing. ChartVerse-RL is used instead, which also avoids SFT/RL overlap within the family. 259 GB. |
| [HuggingFaceM4/the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) `847a98a` | | exclude | Row counts and text formatting equal FineVision's (vqav2 82,772; chartqa 18,265); FineVision adds ratings. Alternative packaging, not extra examples. |
| [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) `9d451dc` (mix665k) | | exclude | JSON conversations referencing COCO/GQA/OCR-VQA/TextVQA/VG images that are not bundled; its VQA parts duplicate FineVision components and the rest are long conversational answers. |
| [allenai/pixmo-count](https://huggingface.co/datasets/allenai/pixmo-count) `ebf51cf`, ODC-BY | train | **pending** | 0 of 923 sampled URLs return bytes matching the supplied `image_sha256` (846 mismatched, 41 gone; size variants and decoded-pixel hashes also fail). Counts are detector-derived; the question would be a fixed template. Validation/test preserved. |
| [allenai/pixmo-ask-model-anything](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything) `739ffd8`, ODC-BY | | exclude | 44 of 1,079 sampled pairs pass the short-answer rules, and those are mostly emoji games, refusals and sentence answers. 98% of sampled URLs were available; 18 hashes mismatched. |
| [HuggingFaceM4/Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix) `0725b65`, MIT | | exclude | 58% of sampled turns are on multi-page documents; single-page answers are LLM-written sentences restating the question. No document IDs. 982 GB. |
| [ryokamoi/VisOnlyQA_Train](https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Train) `10d6012`, GPL-3.0 | all 7 tasks | addon | Relicensed to GPL-3.0 (bases: CLEVR/SuperCLEVR BSD, synthetic geometry). Training images are CLEVR train, SuperCLEVR and synthetic geometry; the evaluation repositories are not read. The pilot found one image pixel-identical to a MathVista testmini image. |
| [nvidia/Nemotron-Image-Training-v3.5](https://huggingface.co/datasets/nvidia/Nemotron-Image-Training-v3.5) `70b27e0`, OpenMDW-1.1 | `coco_count_spatial_qa` | include | Counting and above/below choices from COCO annotations. Images are read from COCO `train2017.zip` with HTTP range requests (ETag `62ff7d7f…-2306`; zip CRC checked; COCO's TLS certificate does not match its hostname, so this is plain HTTP). train2017 contains val2014 images, and the pilot found two pixel-identical POPE images. Some counted instances are too small to see at the model's resolution. |
| | `pixmo_count_qa` | exclude | Alternative packaging of allenai/pixmo-count, same unverifiable hashes. |
| | `chartverse_perception_qa` | pending | Short perception answers, but images must be extracted from opendatalab/ChartVerse-SFT-1.8M; not retrieved for this build. |
| | `chartnet_qa_*`, `chartverse_reasoning_*`, `babyvision`, `deepvision`, `vision_language_reasoning` | exclude | Reasoning-trace targets; the final answers presuppose the reasoning. |
| | `visualpuzzles_reasoning` | exclude | Built from the neulab/VisualPuzzles benchmark. |
| | `arxiv_chart_qa_*` | exclude | Needs arXiv source archives plus PDF/PostScript rendering. |
| | detection, grounding, pointing, video, multi-image subsets | exclude | Outside this build's single-image QA objective. |

### RL candidates

| Source @ revision | Reference | Notes |
|---|---|---|
| [TIGER-Lab/ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) `812ec61`, MIT | `answer` (boxed) | `category`, `source`, `qid` and both pass-rate fields kept in provenance only. Images from the verified `images.zip`; multi-image rows rejected. A trailing "Let's think step by step and output the final answer within \boxed{}." is replaced by the scorer format. |
| [TIGER-Lab/VisualWebInstruct-verified](https://huggingface.co/datasets/TIGER-Lab/VisualWebInstruct-verified) `0e54d11`, MIT | `short_answer` | **97,295 rows in 95 files at this revision**, not the 973K in the card prose. Sub-questions (`idx` 3247.1, 3247.2) share a group. The reasoning `answer` is not read. |
| [opendatalab/ChartVerse-RL-40K](https://huggingface.co/datasets/opendatalab/ChartVerse-RL-40K) `0b28d26`, Apache-2.0 | `reward_model.ground_truth` | The system prompt is dropped (recorded). Long-decimal references go to the pending numeric pool. |
| [russwang/ThinkLite-VL-70k](https://huggingface.co/datasets/russwang/ThinkLite-VL-70k) `5c86ea4`, MIT | `ground_truth` | No per-row upstream source. Items resemble FigureQA, TabMWP, Geometry3K and others that feed MathVista, so image and question matching is the only provenance control; the pilot found MathVista and MMBench duplicates. |
| [hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k) `fd21e53`, MIT | `answer` | `train` only; validation/test preserved. The pilot found two training images pixel-identical to MathVista testmini images (removed). |

## Row contracts

### SFT

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nWhat is the payee?\nMake the answer very short.", "think": null},
    {"role": "assistant", "content": "Washington University", "think": ""}
  ],
  "images": ["/data/k2hv/images/2f/2f…c1.png"],
  "metadata": {"record_id": "fv_docvqa:docvqa/train-00000-of-00025.parquet:17:2", "source": "fv_docvqa", "task": "document", "image_sha256": "2f…c1"}
}
```

Train with `--input-key messages --multimodal-keys '{"image":"images"}'` and no `--apply-chat-template`, as in [step 6](06-sft.md). `validate` reads each row with the pinned Miles `read_file`, converts it with `_build_messages`, loads media with `process_vision_info` (which first resizes with `qwen_vl_utils.smart_resize` to multiples of 28 px), then runs `k2hv.sft_rollout.prepare_sample`. It rejects a row unless:

- it has one user/assistant pair, string contents, `think: ""`, exactly one `<image>` and one image path, and no media marker in the answer;
- the image file exists, has the recorded sha256 and loads through Miles;
- `prepare_sample` accepts it, so the decoded supervised span equals `answer + <|ifm|im_end|>` and starts right after the single empty think block;
- the sequence holds exactly one complete `<image>` / pad / `</image>` block, the pad count equals the grid's `t·h·w/4`, no pad lies outside it, and `pixel_values` has `t·h·w` rows;
- the pad count equals the value predicted from the stored image size by the qwen and GLM resize rules;
- no image token is under the loss mask and the mask is all ones over the supervised span;
- image tokens ≤ 1,024, total tokens ≤ 2,048 and supervised tokens ≤ 128. Rows over a limit are rejected, never truncated. `--rollout-max-prompt-len` is not used.

Text rules, applied before any image is fetched: no empty fields; no `<image>`, `reserved_special_token`, template or think markers in text; no code (`import`, `def`, `plt.`, fenced blocks, HTML or TikZ); no follow-up references to earlier turns; single-line answers of at most 24 words and fewer than two sentences. FineVision's packaging adds a trailing period to short answers ("Sweden."); one trailing period is removed from answers of eight words or fewer. Choice answers ("Answer: B") become the bare letter when the question lists options. The original answer stays in the index. Answers are never shortened by truncation.

### RL

```json
{
  "problem": "<image>\nFind x.\n\nSolve the problem. The last line of your response must be of the form \"Answer: X\" (without quotes), where X is the final answer as a number or expression, without units.",
  "answer": "17.1",
  "images": ["/data/k2hv/images/…"],
  "metadata": {"data_source": "math_dapo", "dataset": "hiyouga/geometry3k", "source": "rl_geometry3k", "upstream_id": "train:12", "reference": "17.1", "answer_type": "numeric", "task": "geometry", "record_id": "…", "image_sha256": "…"}
}
```

`metadata.data_source` is only the scorer selector; `dataset`, `source`, `upstream_id` and the original `reference` are separate. The pinned scorers set the admission rules:

- RL360 `math_dapo` (Minerva path) reads only the last `Answer: …` line within the final 300 characters and compares strings after its normalization. A bare `\boxed{1.5}` scores wrong; `1.50`, `3/2` and `20%` do not match `1.5`/`20`; units it strips (`cm`, `degrees`) do match. The earlier example prompt ("Put the final answer in \boxed{}") would therefore score every correct answer given only in a box as wrong. The RL prompt replaces any source format sentence mentioning a box with the "Answer: X" instruction.
- RL360 `stem__supergpqa` extracts one option letter from `<answer>`, a box, an "answer:" line or the bare output; the target must be a letter A–J.

Classification: references with one letter matching parsed options, or text matching exactly one option, become `stem__supergpqa` letters. Plain numbers (thousands separators, currency, `%`, `°` and common units removed into a canonical target) and short symbolic answers (`\frac`, `\sqrt`, π, powers) become `math_dapo`. Numbers with more than three decimals go to the separate `rl_pending_numeric` pool, which needs the 1% relative-tolerance rule in `k2hv.data.scoring.numeric_close` registered as a selector in step 8 before use. Yes/no, scientific-notation, free-form text and unparseable choices are excluded. A row is admitted only if the pinned scorer accepts `Answer: <target>` and rejects a perturbed wrong answer. Publisher verification and pass-rate fields are not treated as scorer compatibility. `tests/test_data_pipeline.py` characterizes each scorer on equivalent and wrong answers, units, tolerances and choice formats. Reward integration, close-tag handling and timeouts remain step 8 work.

Every RL prompt is also rendered with the image-aware template (`add_generation_prompt=True`) and processed; prompts above 1,536 tokens are rejected. Step 8 must still check Miles's own RL chat-template path.

## Overlap and splits

**Evaluation exclusion.** The `evals` stage indexes the exact [step 7](07-eval.md) splits at pinned revisions (lmms-eval packaging): MMMU `validation` (`lmms-lab/MMMU` `364f2e2`), MathVista `testmini` (`AI4Math/MathVista` `2b6ad69`), ChartQA `test` (`lmms-lab/ChartQA` `9e63b7d`), DocVQA `validation` (`lmms-lab/DocVQA` `539088e`), TextVQA `validation` (`lmms-lab/textvqa` `9c0699c`), MMBench `en` `dev` (`lmms-lab/MMBench` `56ba1af`) and POPE (`lmms-lab/POPE` `4db1276`, `default` config, `test` split: adversarial + popular + random, 9,000 questions over 500 COCO val2014 images, matching lmms-eval's `pope` task). Every candidate image is compared with every evaluation image:

1. identical decoded pixels, or external IDs (COCO image IDs for POPE, Flickr photo and Open Images IDs for TextVQA);
2. 64-bit pHash within Hamming distance 10 and 32×32 thumbnail correlation ≥ 0.6, then a decoded comparison at 256 px: gradient (edge) correlation and the largest mean RGB difference over 16×16-px blocks. **Confirmed**: edge ≥ 0.98 (same structure, possibly recolored), or edge ≥ 0.93 with max block difference ≤ 0.10. **Uncertain**: edge ≥ 0.85. Both are excluded; uncertain matches are listed for inspection.
3. A question identical to an evaluation question after removing format lines is excluded when it is at least 120 characters, or when the two images are even loosely similar; otherwise it is listed but kept, since template questions ("What is the date mentioned in this letter?") recur across unrelated images.

The stage-2 thresholds come from inspecting every pilot pair (sheets via `inspect`). The first single-stage rule flagged 233 pilot images; nearly all were same-template Statista charts, TabMWP tables, US maps with a different highlighted state, or different pages of one document. The two-stage rule kept the true duplicates: POPE images inside COCO train2017, Geometry3K/VisOnlyQA/ThinkLite images inside MathVista testmini and MMBench dev, a recolored table, a VSR image. Crops are not detected. Re-encoded or resized copies are (tested).

**Grouping.** Images are joined into groups by identical pixels, stage-2-confirmed near-duplicates (pHash ≤ 3 via 16-bit band buckets), and shared document/rendering keys (CoSyn `data` hash, VisualWebInstruct base `idx`, VisOnlyQA source image). A group's ID is its smallest image sha256. Splits are drawn per group (`sha256(seed:sft-split:group)`), so every question about an image, its copies in other collections and its alternative renderings land in one split.

**Deduplication.** Two records are duplicates when they share an image group, the normalized question (format lines removed) and the normalized answer. The record from the earlier recipe source wins. The same generic question on different images is not a duplicate.

**RL reservation.** RL rows are selected first; any SFT record whose group contains an RL image is removed, and the report counts the overlap.

**Preserved splits.** CoSyn `validation`, PixMo-Count `validation`/`test` and Geometry3K `validation`/`test` are never read for training. The evaluation repositories of VisOnlyQA are never read.

## Outputs and rematerialization

`data/step05/` after a build:

```text
images/<sha[:2]>/<sha>.<ext>   content-addressed original bytes (P/LA transparency composited on white, recorded)
evals/                         evaluation images (exclusion index only)
exports/<profile>/
  portable/<variant>/<split>/part-*.parquet   rows with relative images/… paths
  miles/…                                     same rows with absolute paths (materialize)
  index/*.parquet              per-row provenance: source/revision/file/row/turn, upstream id, terms, group, split, token counts, row hash
  images.parquet               sha256, path, bytes and size of every referenced image
  manifest.json, SHA256SUMS, reports/{quality,verify}.json
work/                          scan candidates/rejects, selection, validation results (resumable state)
hf/, media/, inventory/        verified source downloads and listings
```

Variants: `default/sft/{train,validation,sweep50k}`, `default/rl/{pool,heldout}`, `default/rl_pending_numeric/{pool,heldout}`, `addon/sft/{train,validation}`. The sweep is the first 50,000 default training rows in `sha256(seed:sweep:record_id)` order: a documented subset of train, never validation.

To use the data on another machine, copy `images/` and `exports/<profile>/` (rsync or object storage), then:

```bash
uv run python -m k2hv.data materialize --profile full --data-root /path/to/copy --target-root /data/k2hv
uv run python -m k2hv.data verify --profile full --data-root /path/to/copy --target-root /data/k2hv
```

`verify` checks SHA256SUMS, re-hashes and decodes every referenced image, and checks that each Miles row hashes to its validated portable row. Miles's loader does not read `s3://`, so images must be on a local or mounted path at training time.

### Step 6 dependency: Miles loads every image eagerly

Pinned Miles `Dataset.__init__` calls `process_vision_info` for every row at load time and holds each decoded qwen-resized image in memory: 2.9 MB per row on average (p90 8.9 MB) for this data, about 2.9 TB for a million rows. The full export cannot be loaded as one Miles dataset. Step 6 needs lazy media loading (for example keeping paths in the sample and loading inside `k2hv.sft_rollout.generate_rollout`) or must train over small shards in sequence. The 50k sweep subset would need about 150 GB.

### Step 6 finding: double BOS in the SFT path

`prepare_sample` renders the template, which already begins with `<|ifm|begin_of_text|>`, then calls the processor with default `add_special_tokens=True`, so every training sequence starts with two BOS tokens. The SGLang processor and step 4's gates tokenize with `add_special_tokens=False` (one BOS). The step 4 fixture's 40 tokens include both. This step does not change the certified training path. All reported lengths include the extra token. If step 6 fixes it (`add_special_tokens=False` in `prepare_sample`, then GPU recertification), every row shortens by exactly one token and all exports stay valid.

## Build results

Full build on 2026-10-31 (n2d-standard-16, 16 vCPU, 62 GiB), profile `full`, recipe sha256 `c91f84fe…a807`, seed 20261031. Wall time: scan 62 min, evals 7 min, select 88 min, validate 126 min, export 1 min, materialize 11 s, verify 11 min. Compact summaries: [`full.json`](../evidence/step05/full.json), [`pilot.json`](../evidence/step05/pilot.json); every number below comes from them.

### Funnel

| Stage | Records |
|---|---|
| Candidates after text and image rules (scan) | 2,624,494 over 820,020 unique images, 683,947 image groups |
| Excluded for evaluation overlap | 8,938 |
| Exact QA duplicates (same group, question and answer) | 44,198 (35,341 TallyQA questions already present as VQAv2 rows on the same COCO image) |
| SFT records reserved for RL (image group shared with an RL row) | 13,527 (6,977 groups) |
| Selected with 1.15× oversampling, all validated | 1,174,681; **0 contract failures** |
| Exported after final quotas | 1,102,190 rows, 709,353 images (99.28 GB) |

### Exports

| Bucket | Rows | Unique images | Tokens p50 / p99 / max | Supervised p50 / p99 / max |
|---|---|---|---|---|
| `default/sft/train` | 839,928 | 483,567 | 469 / 1,064 / 1,134 total | 2 / 14 / 101 |
| `default/sft/validation` | 8,605 | 4,954 | 486 / 1,064 / 1,100 total | 2 / 15 / 60 |
| `default/sft/sweep50k` (subset of train) | 50,000 | 48,340 | 487 / 1,064 / 1,102 total | 2 / 14 / 44 |
| `addon/sft/train` | 111,245 | 101,969 | 295 / 1,131 / 1,142 total | 2 / 5 / 27 |
| `addon/sft/validation` | 1,104 | 1,022 | 295 / 1,131 / 1,140 total | 2 / 5 / 9 |
| `default/rl/pool` | 119,413 | 97,537 | 200 / 1,142 / 1,445 prompt | |
| `default/rl/heldout` | 6,409 | 5,307 | 199 / 1,145 / 1,390 prompt | |
| `default/rl_pending_numeric/pool` | 14,706 | 14,498 | 1,090 / 1,122 / 1,190 prompt | |
| `default/rl_pending_numeric/heldout` | 780 | 777 | 1,090 / 1,120 / 1,139 prompt | |

SFT totals: 848,533 default rows and 960,882 with the add-on. Image tokens never exceed 1,024, total SFT tokens never exceed 1,142 and supervised tokens never exceed 101, all well inside the 2,048/128 limits; the limits were not changed. Every SFT row carries two BOS tokens ([above](#step-6-finding-double-bos-in-the-sft-path)). RL selectors: `math_dapo` 98,420 and `stem__supergpqa` 27,402 rows in `rl/`; 15,486 long-decimal rows (15,105 from ChartVerse-RL) wait in `rl_pending_numeric/` for a tolerance selector.

Sources below quota exported every eligible row: AI2D 4,057 of 15,000, VSR 3,065 of 10,000, InfographicVQA 21,306 of 30,000, DocVQA 29,086 of 40,000, TextVQA 34,551 of 40,000, ST-VQA 22,970 of 25,000, Nemotron COCO 14,447 of 15,000, CoSyn chart 119,051 of 120,000; add-on ChartQA 26,081 of 40,000 and IconQA 16,268 of 20,000; RL ThinkLite 41,107 of 70,000, VisualWebInstruct 34,821 of 100,000, ViRL39K 29,793 of 40,000, ChartVerse-RL 34,077 of 40,000, Geometry3K 1,510 of 3,000. Per-source rejection reasons (answer length, multi-line or explanatory answers, free-form or yes/no RL references, scorer admission) are in `per_source.*.scan.rejected` of `full.json`.

### Evaluation exclusion

8,938 records: MMBench dev 3,901, MathVista testmini 2,983, ChartQA test 1,259, POPE 393, DocVQA val 371, TextVQA val 29, MMMU val 2. By method: identical pixels 426, stage-2 confirmed 2,748, stage-2 uncertain 4,078, question text 1,686 (a further 15,412 template-question matches are listed and kept). The largest groups are ThinkLite against MMBench (957 confirmed copies, 1,832 uncertain US-map and chart templates), add-on ChartQA train against ChartQA test (9 identical, 29 confirmed, 1,078 uncertain same-template Statista charts), DVQA questions matching MathVista text (1,063), VQAv2/TallyQA/Visual7W COCO images inside MMBench and POPE, Nemotron COCO images identical to POPE (73) and Geometry3K images identical to MathVista (66). Uncertain matches are excluded, which costs some template look-alikes in exchange for no known leakage. Pair sheets for all 64 categories are written by `inspect` to `outputs/step05/full/exclusions/`.

### Checks

| Check | Scope | Result |
|---|---|---|
| Miles normalization, `prepare_sample` and processor contract | every selected row (1,174,681) | 0 failures |
| Image decode and sha256 | every exported image (709,353), every exported file (29) | 0 bad |
| Miles rows hash to validated portable rows | every exported row incl. sweep (1,152,190) | 0 mismatches |
| Image group shared across splits or between SFT and RL | all buckets | none; groups repeat only between variants of one split (add-on/default SFT train 3,529, validation 36; RL/pending pool 265, heldout 13) |
| Sweep is a subset of train | 50,000 rows | true |
| Visual review of model inputs, supervision and exclusion pairs | sampled: 56 pilot cards plus 6 at native resolution, 20 pilot RL cards, 25 pilot exclusion categories, 51 full-build cards | [findings](../evidence/step05/manual-audit.json) |

The sampled review found no label errors in the inspected cards. It did find limits worth knowing: Nemotron COCO spatial labels follow box geometry and are ambiguous for held or overlapping objects, and `math_dapo` compares normalized strings, so `5/7`, `\dfrac{5}{7}` and `5√3` score wrong against `\frac{5}{7}` and `5\sqrt{3}` targets (spaces do not matter). Step 8 should word the RL prompt or the selector accordingly.

### Reproducibility

The pilot was rebuilt in a fresh root, killed with SIGKILL after 15 of 58 scan shards and again during validation, and resumed. All 58 shard outputs, `selected.parquet` and the export SHA256SUMS were byte-identical to the clean pilot, and stayed identical after the final export change ([evidence](../evidence/step05/reproducibility.json)). A unit test injects a mid-shard failure and checks atomicity and byte-identical resume.

### Artifacts

| Path (under `data/step05/`) | Size | sha256 |
|---|---|---|
| `exports/full/SHA256SUMS` (per-file sha256 of every Parquet file) | 29 files, 617 MB | `6d78bf8bd6a5b1a720ac9b552c5931094d1e7ea663f09ff829c0c2903531fd73` |
| `exports/full/manifest.json` (recipe, pins, runtime, counts, files) | 17 KB | `9af418ba41c05c9014be419b74edd8ff360a204bb03521dc83c75523eb63f4e8` |
| `exports/full/images.parquet` (sha256, path, bytes, size of every image) | 54 MB | `cf6a031d4ce56b7d704a84f743834ead721e9dc787d1e6748d3cd0b8d592db53` |
| `images/` | 99.28 GB referenced (104 GB stored including unselected candidates) | per image, by name |
| `exports/pilot/SHA256SUMS` | 1.3 MB export | `1185e88739fb9271945ee287012aa3a453ff41267d298adb76721b4c22f7a031` |

These artifacts exist only on the CPU instance; nothing was uploaded. The same holds for `hf/` (116 GB verified source downloads), `evals/` (2.3 GB), `work/` (1.4 GB resumable state), `data/step05-resume/` (12 GB, the reproducibility root), `outputs/step05/` (build logs, inspection and exclusion sheets) and the `third_party/` checkouts. To use the data elsewhere, copy `images/` and `exports/full/` and run `materialize` and `verify` as [above](#outputs-and-rematerialization). To rebuild from nothing, follow [Setup](#setup) and run `build --profile full` (about 5 hours and 225 GB on this machine type). The rebuild reproduces these bytes as long as the pinned Hub revisions and the COCO train2017 zip (ETag-pinned) stay available.

### Known limitations

- Cropped copies of evaluation images are not detected; re-encoded, resized and recolored copies are.
- ChartVerse-RL and Nemotron COCO are single-file sources, so their scan runs on one worker and dominates scan time.
- A worker killed during the reproducibility test left one temporary file in `data/step05-resume/images/`; temporary files are never read as assets.
- PixMo-Count stays pending: none of 923 checked URLs matched the published image hashes.
- Terms are unlisted for ST-VQA, DocVQA, InfographicVQA and AI2D (default build, flagged). ChartQA and VisOnlyQA (GPL-3.0), IconQA and DVQA (non-commercial) are confined to `addon/`. Using either set is a licensing decision for the project owner.

## RL prompt selection after SFT

This GPU-dependent stage waits for the [SFT checkpoint](06-sft.md). Follow IFM's protocol from RL360 (`src/agent360/harbor/miles/calibration.py`):

1. Take a 10–20k-prompt subset of `default/rl/pool`.
2. Sample the SFT checkpoint 16 times per prompt, at the RL temperature, image budget and reasoning effort used for training and evaluation.
3. Keep prompts with a pass rate between 20% and 40%.

Use our checkpoint's measured pass rates, not difficulty or pass-rate annotations from other models (ViRL39K's `PassRate_*` fields stay in provenance only). Prompts the model always or never solves give every sample the same reward, so GRPO assigns them zero advantage. This calibration is a dependency for RL, not part of the CPU preparation above.
