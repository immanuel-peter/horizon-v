# Step 5: Data

Build reproducible image-QA datasets on a Linux CPU machine. SFT teaches the projector to connect images to short answers; RL later uses questions with answers that a scorer can verify. No model weights or GPU are needed for dataset preparation. RL difficulty calibration needs the SFT checkpoint and runs after [step 6](06-sft.md).

Baseten did not name its datasets. The sources below are public stand-ins, revised after a Hugging Face survey on 2026-10-29. They are candidates for a pilot, not a claim of improved model quality. The data pipeline and bulk datasets still need implementation.

## SFT sources

Use a balanced mix of ordinary visual QA, OCR/document reading, charts/tables and counting. Start with named source subsets and inspect their retained examples before choosing mixture weights.

| Priority | Source | Initial selection |
|---|---|---|
| Base | [HuggingFaceM4/FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) | Selected single-image, short-answer subsets. Include ordinary visual QA and counting, with eligible training examples from `tallyqa`, `chartqa`, `docvqa` and `textvqa`. Preserve upstream split identity and applicable component terms. |
| Supplement | [allenai/CoSyn-400K](https://huggingface.co/datasets/allenai/CoSyn-400K) | Short-answer pairs from `chart`, `table` and `document`; add diagrams only after the pilot. Images are bundled and each image can have several questions. |
| Targeted supplement | [opendatalab/ChartVerse-SFT-600K](https://huggingface.co/datasets/opendatalab/ChartVerse-SFT-600K) | Sample charts whose evidence remains readable at our image budget. Use the explicit short `answer`, not the long `reasoning` field. |
| Alternatives | [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron), [LLaVA-1.5 mix665k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | Use when they provide a needed source more conveniently. They are alternative packaging, not automatically additional independent examples. |

FineVision already contains CoSyn and other sources in this table. Choose one version of each underlying source, deduplicate across collections, and retain its original IDs. Download selected configurations rather than the full FineVision/FineVisionMax mixture.

The initial full-build target remains roughly 500k–1M usable QA pairs. Set the actual size and source proportions after the pilot and overlap audit; do not repeat or admit poor examples just to reach a count. Count image records and QA pairs separately.

### Deferred sources

- [PixMo-Count](https://huggingface.co/datasets/allenai/pixmo-count) can add natural-image counting. Its training labels are detector-derived and its images require URL downloads with checksum verification. Audit label quality before using these counts for RL rewards.
- [PixMo AskModelAnything](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything) has human-authored questions, but answers are generally long. Admit only short, standalone pairs with available images.
- [Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix) is a document-reading extension. Select genuinely single-page questions with short answers; removing pages from a multi-page question is not valid conversion.
- [VisOnlyQA Train](https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Train) is useful for geometric perception, but its Hub metadata lists GPL-3.0. Resolve the applicable terms before including it in the default mixture; keep its evaluation repositories out of training.
- [Nemotron Image Training v3.5](https://huggingface.co/datasets/nvidia/Nemotron-Image-Training-v3.5) offers counting, spatial, chart and visual-puzzle subsets. Most upstream images are not bundled. Admit selected components only after resolving media, answer fields and component terms; exclude video, grounding-only and long-reasoning tasks from this first build.

## SFT row format

Export Parquet with one user/assistant pair and initially one image per row. Match the tested Miles path in `k2hv.sft_rollout.prepare_sample`, including explicit assistant `think: ""`:

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nWhat is the total on the receipt?"},
    {"role": "assistant", "content": "$42.18", "think": ""}
  ],
  "images": ["/data/k2hv/images/receipt.jpg"]
}
```

The `<image>` string is the raw Miles dataset placeholder. Validate its conversion through Miles's media/message preparation and the actual HF processor, not by passing this raw row directly to the processor. The processor needs the complete start/pad/end layout produced by the image-aware chat template ([step 2](02-processor-and-template.md)).

- Use `think: ""` for this loader. The template's acceptance of `reasoning_content` alone does not satisfy `prepare_sample`.
- Split source records into standalone QA pairs. Reject follow-up questions that need earlier conversation turns. Keep all questions for a shared image in the same data split.
- Supervise only the answer and `<|ifm|im_end|>`. Prompt, image and empty-thinking tokens must be outside the loss mask, as in [step 6](06-sft.md).
- Use a source's explicit final-answer field when available. Reject unsuitable long responses rather than truncating explanations into answer labels. Source reasoning, rendering code and answer-computation code belong in provenance, not model inputs.

## Preparation and validation

Implement a resumable CPU pipeline with source-specific conversion at the input boundary and one shared validation/export path. The following stages describe planned work, not existing CLI commands.

1. **Inventory.** Pin each dataset revision and configuration. Record original split, source terms, image location, schema, answer field and download size before bulk retrieval. Inspect the pinned schema; cards and example code can disagree.
2. **Fetch and normalize.** Materialize selected images, preserve original IDs and verify supplied hashes. Convert eligible records to the SFT or RL schema. Use a stable data root and relative asset identities in the manifest so exports can be rematerialized on another machine.
3. **Validate images and supervision.** Decode every retained image, reject corrupt/missing assets, and validate image counts and marker order. Run the real processor and SFT preparation path to check expanded token counts and decoded supervised spans. Sample rendered chart/OCR inputs after resizing to check that the answer evidence is still visible.
4. **Filter length offline.** Start the pilot with at most 128 supervised answer/end tokens, 2048 total expanded sequence tokens and the existing 1024-image-token cap. Record these limits and the processor/template hashes in the recipe. Revise limits before freezing the full build if the pilot warrants it; reject overlong examples rather than truncating image blocks or targets. Do not use `--rollout-max-prompt-len` as an SFT data filter.
5. **Remove overlap and split.** Apply the exclusions below before emitting final train/validation files. Fix the split seed and group by original image/document/rendering identity, including copies found in other collections.
6. **Export and audit.** Write sharded Parquet, image assets, source manifests, checksums and a quality report. Include retained/rejected counts and reasons per source, unique image/QA counts, token distributions, split sizes and unresolved checks. A resumed build with the same pinned recipe must produce the same records and splits.

Image loading must use a representation supported by pinned Miles. The gate fixture uses local paths under a fixed mount. Keep images available without a training-time web fetch; URL sources should be downloaded during preparation. Miles's current media loader does not handle `s3://` directly. Validate embedded-byte exports through that loader before selecting them as an alternative.

### Duplicates and evaluation exclusions

- Identify exact assets by original source ID and image hash. Detect re-encodes/resizes with image matching where needed, and inspect uncertain matches. Deduplicate QA records using image identity plus normalized question/answer; a generic repeated question alone is not a duplicate.
- Reserve the RL candidate pool separately and audit its image/question overlap with SFT, especially ChartVerse SFT versus ChartVerse RL. Keep derived charts and documents together when selecting holds or splits.
- Exclude the [step 7](07-eval.md) evaluation material: MMMU `val`, MathVista `testmini`, ChartQA `test`, DocVQA `val`, TextVQA `val`, MMBench `dev` and the chosen POPE split/images. Pin the exact POPE selection before building the exclusion index.
- Preserve official validation/test sets. A repackaged split called `train` is not proof that all original evaluation material was excluded. Use original identifiers and image/question matching across collections.
- Record inaccessible evaluation assets and unresolved provenance explicitly. Do not label an incomplete overlap audit as passed.

## RL candidate sources

| Source | Candidate reference | Admission rule |
|---|---|---|
| [TIGER-Lab/ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) | Boxed answer | Keep supported numeric, symbolic or choice targets, retaining original category/source information. |
| [TIGER-Lab/VisualWebInstruct-verified](https://huggingface.co/datasets/TIGER-Lab/VisualWebInstruct-verified) | `short_answer` | Use this instead of `answer`, which contains reasoning. Start with one image and scorer-compatible targets. |
| [opendatalab/ChartVerse-RL-40K](https://huggingface.co/datasets/opendatalab/ChartVerse-RL-40K) | `reward_model.ground_truth` | Preserve the numeric reference and define numeric tolerance/normalization for values with long decimal tails. |
| [ThinkLite-VL-70k](https://huggingface.co/datasets/russwang/ThinkLite-VL-70k), [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k) | Explicit answer/ground-truth fields | Additional candidates after source/split and scorer checks. Use eligible training splits only. |

The 2026-10-29 Hub row endpoint reported 97,295 VisualWebInstruct-Verified rows, despite card prose claiming 973K. Recount on import and record the selected revision. Publisher verification and external difficulty labels do not replace our label audit, scorer tests or checkpoint-specific calibration.

Export RL candidates as Parquet:

```json
{
  "problem": "<image>\nFind the measure of angle ABC. Put the final answer in \\boxed{}.",
  "answer": "42",
  "images": ["/data/k2hv/images/geometry.png"],
  "metadata": {"data_source": "math_dapo"}
}
```

`metadata.data_source` is the scorer selector in RL360's `async_rm` ([step 8](08-rl.md)); record the original dataset name separately in provenance:

| Answer type | `data_source` |
|---|---|
| Numeric / symbolic math | `math_dapo` |
| Multiple choice | `stem__supergpqa` |

Store the original reference alongside any normalized target. Test equivalent answers, wrong answers, units, numeric tolerances and choice normalization before admission. Free-form targets without a reliable rule-based scorer remain excluded. The reward implementation is still part of step 8.

## Build sequence and outputs

1. Implement conversion, validation, overlap indexing and export with small local fixtures.
2. Run a **1,000-example real-data pilot** covering the selected SFT adapters and a small sample of each RL source. Review decoded supervision and resized images, rejection reasons, length distributions and download/storage requirements.
3. Freeze the source revisions, mixture recipe, limits and split seed. Produce a **50k SFT sweep subset** and the larger SFT train/validation exports. The sweep subset is a documented subset of training, not validation.
4. Export the deduplicated RL candidate pool and its held-out split. Keep the data manifest and verification reports with both builds; keep bulk images and datasets out of git.

Expected deliverables are SFT train/validation Parquet, the sweep subset, RL candidate/held-out Parquet, portable image assets, a pinned source/build manifest, checksums, an evaluation-exclusion report and a quality report. Record recipes and compact results in git. Dataset preparation is complete when all exported rows validate and required overlap checks are resolved; merely downloading sources is not completion.

## RL prompt selection after SFT

This GPU-dependent stage waits for the [SFT checkpoint](06-sft.md). Follow IFM's protocol from RL360 (`src/agent360/harbor/miles/calibration.py`):

1. Take a 10–20k-prompt subset.
2. Sample the SFT checkpoint 16 times per prompt, at the RL temperature, image budget and reasoning effort used for training/evaluation.
3. Keep prompts with a pass rate between 20% and 40%.

Use our checkpoint's measured pass rates, not difficulty/pass-rate annotations from other models. Prompts the model always or never solves give every sample the same reward, so GRPO assigns them zero advantage. This calibration is a dependency for RL, not a prerequisite for finishing the CPU preparation work above.
