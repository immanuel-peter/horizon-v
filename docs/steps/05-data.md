# Step 5: Data

Build reproducible image-QA datasets on a Linux CPU machine. SFT teaches the projector to connect images to short answers; RL later uses questions whose answers a rule-based scorer can verify. Dataset preparation needs no model weights or GPU. RL difficulty calibration needs the SFT checkpoint and runs after [step 6](06-sft.md).

The pipeline is `python -m k2hv.data` (`src/k2hv/data/`), driven by the recipe [`configs/data/step05.toml`](../../configs/data/step05.toml). Every source revision, rule, limit, quota, overlap threshold and split parameter lives in the recipe, and its hash is in every build manifest. Results of the v2 build (2026-11-01) and the changes from v1 are under [Build results](#build-results); compact evidence is in [`docs/evidence/step05/`](../evidence/step05/).

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

Importing `miles.utils.data` or `miles.utils.processing_utils.load_processor` pulls in SGLang through the rollout and chat-template packages. `k2hv.data.runtime` instead checks that `third_party/miles` is at the pinned commit, verifies the sha256 of `miles/utils/{data,processing_utils,types}.py`, and executes the pinned `_build_messages` and `read_file` source extracted from `data.py`, the pinned `process_vision_info`, and `Sample`. `load_processor` reduces to `AutoProcessor.from_pretrained(path, trust_remote_code=True)` for this checkpoint. `qwen_vl_utils` imports torchvision only for its video readers; a stub exists only during that import, so transformers still sees no torchvision. RL360's `math_dapo` and `supergpqa` modules are loaded from the pinned checkout only to characterize them in tests.

## Stages

| Stage | Does |
|---|---|
| `inventory` | Pin listings (sizes, LFS sha256), schemas, row counts and preserved official splits of every recipe dataset and evaluation set |
| `scan` | Per source shard: read pinned rows, apply text rules, fetch and decode every image, store it content-addressed, compute hashes; write candidates, rejects and a done marker |
| `evals` | Index every evaluation image and question |
| `select` | Evaluation exclusion, near-duplicate grouping, QA deduplication, RL reservation, quotas and group-level splits; `--trace` also writes each candidate's decision and the edges that formed each group |
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

The audit inspected Hub metadata, cards and a deterministic sample of every candidate, then a 1,000-example real-data pilot.

**Terms policy.** This is a research project building on Baseten's work; future commercial use is not a requirement, so non-commercial terms do not exclude a source. A source is admitted only when an original statement from its creators permits research training use: card or README text, a licence file covering data distributed with it, the release archive's own licence, or the creators' own Hub `license:` declaration on their repository (since v3). A tag set by someone else, such as a mirror, or a code-repository licence alone is not a basis. Explicit upstream restrictions are recorded per source; upstream terms that cannot be found are recorded as missing, not as prohibitions. Permission to train is recorded separately from redistribution, and nothing is redistributed. The recipe holds each source's basis (`terms_note`), `redistribution` and, for pending sources, `missing_evidence`. The fetched documents and their hashes are in [`terms.json`](../evidence/step05/terms.json). Media origin is recorded for every source, but per-image rights of scraped or web images (ChartQA charts, for example) are not independently verified. FineVision redistributes the `fv_*` components and defers to each one's licence.

Status: **include** (built and exported), **pending** (not built until a recorded gap closes: terms evidence or image integrity), **exclude** (audited and rejected for quality or fit). Rows inside an included source can also be pending, with `metadata.pending_reason`, in the `pending/` export buckets.

| Terms decision (v3) | Sources |
|---|---|
| Included, creators' Hub declaration (v3) | ViRL39K, VisualWebInstruct-verified and ThinkLite-VL-70k, all `license: mit` on their creators' repositories. Upstream restrictions, none of which bars research training: ScienceQA, IconQA and TabMWP CC BY-NC-SA 4.0, AI2D no redistribution, ChartQA GPL-3.0, LLaVA-OneVision academic use only ([below](#rl-sources-admitted-in-v3)) |
| Included, original statement found | VQAv2 (CC BY 4.0, visualqa.org), VSR (Apache-2.0, project README; data in repo), TallyQA (Apache-2.0; QA zip in the licensed repo), CLEVR (CC BY 4.0), TextVQA (CC BY 4.0, creator's card), PlotQA (CC BY 4.0, README; v1 recorded MIT), CoSyn (ODC-BY), Nemotron COCO (OpenMDW-1.1), ChartVerse-RL (Apache-2.0), Geometry3K (MIT; data ships in the InterGPS repo) |
| Included, previously add-on | ChartQA (GPL-3.0; data folder in the licensed repo), VisOnlyQA (GPL-3.0 `LICENSE.md`), IconQA (CC BY-NC-SA 4.0 README text; its badge says CC BY-SA) |
| Included with conditions | AI2D: AI2's `license.txt` in `ai2d-all.zip` permits use by academic, non-profit or government-sponsored researchers, not by a for-profit enterprise, and grants no right to redistribute or publicly release the data. AI2's former dataset page and the AWS Open Data listing say CC BY-SA instead; the build follows the bundled licence ([below](#ai2d-terms)) |
| Pending, terms unverified | Visual7W (toolkit MIT covers code; project page unreachable), ST-VQA, DocVQA and InfographicVQA (RRC download terms are behind registration; public pages state none), DVQA (no statement anywhere; GitHub NOASSERTION) |
| Pending, integrity | PixMo-Count ([below](#pixmo-count)) |

### SFT candidates

| Source @ revision | Component | Status | Evidence |
|---|---|---|---|
| [HuggingFaceM4/FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) `3c380a7` | `vqav2`, `vsr`, `tallyqa`, `clevr` (13 of 22 files), `textvqa`, `ai2d_merged`, `plotqa` (6 of 18 files) | include | Single image; independent questions bundled per image; short explicit answers. No upstream IDs (`images[].path` is null), so identity is the image hash. Every row is from the Cauldron `train` packaging, which does not prove original-split hygiene; the overlap stage checks images against the evaluation sets directly. |
| | `iconqa`, `chartqa` | include | Admitted in v2 on their original statements (see the terms table). |
| | `dvqa` (3 of 10 files) | pending | No licence statement located. |
| | `visual7w`, `st_vqa`, `docvqa`, `infographic_vqa` | pending | Included in v1; terms unverified (see the terms table). |
| | `ocrvqa` | exclude | Labels come from Amazon metadata: titles include series text absent from the cover, and "Is this a *category* book?" uses store categories. |
| | `a_okvqa`, `aokvqa` | exclude | Two copies of A-OKVQA inside FineVision; answers mix rationales ("Answer: … Rationale: …") with choices. |
| | `cocoqa`, `tqa`, `scienceqa`, `figureqa` | exclude | Caption-generated ungrammatical questions; answers with appended commentary ("The correct answer is B. 6. This relates to…"); knowledge questions with low visual dependency (also MathVista sources); yes/no only (ThinkLite covers FigureQA for RL). |
| | `CoSyn_400k_*` | exclude | Same data as allenai/CoSyn-400K, which is used as the canonical copy. |
| [allenai/CoSyn-400K](https://huggingface.co/datasets/allenai/CoSyn-400K) `86e46e1`, ODC-BY | `chart` (26 of 51 files), `table`, `document` (26 of 41), `diagram` (20k quota) | include | At this revision every config uses `qa_pairs` (parallel `question`/`explanation`/`answer` lists); the card's `questions` field does not occur. `explanation`, `code` and `data` stay out of model inputs (hashes in provenance). Answers with more than two decimals are rejected as unrounded computations. Rows with identical `data` (alternative renderings) share a split group. Diagram answers are often vague relation words ("Affects", "All actions"), hence the reduced quota. Official `validation` splits preserved. |
| [opendatalab/ChartVerse-SFT-600K](https://huggingface.co/datasets/opendatalab/ChartVerse-SFT-600K) `07b1576`, Apache-2.0 | | exclude | The explicit `answer` is a computed result: 30% of pilot answers have unrounded decimals (`136.1013215859031`), and inspected integer answers are multi-condition computations over dense charts at the token cap. With empty thinking these targets teach guessing. ChartVerse-RL is used instead, which also avoids SFT/RL overlap within the family. 259 GB. |
| [HuggingFaceM4/the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) `847a98a` | | exclude | Row counts and text formatting equal FineVision's (vqav2 82,772; chartqa 18,265); FineVision adds ratings. Alternative packaging, not extra examples. |
| [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) `9d451dc` (mix665k) | | exclude | JSON conversations referencing COCO/GQA/OCR-VQA/TextVQA/VG images that are not bundled; its VQA parts duplicate FineVision components and the rest are long conversational answers. |
| [allenai/pixmo-count](https://huggingface.co/datasets/allenai/pixmo-count) `ebf51cf`, ODC-BY | train | **pending** | No fetched image matches its published `image_sha256` ([PixMo-Count](#pixmo-count)). Counts are detector-derived; the question would be a fixed template. Validation/test preserved. |
| [allenai/pixmo-ask-model-anything](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything) `739ffd8`, ODC-BY | | exclude | 44 of 1,079 sampled pairs pass the short-answer rules, and those are mostly emoji games, refusals and sentence answers. 98% of sampled URLs were available; 18 hashes mismatched. |
| [HuggingFaceM4/Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix) `0725b65`, MIT | | exclude | 58% of sampled turns are on multi-page documents; single-page answers are LLM-written sentences restating the question. No document IDs. 982 GB. |
| [ryokamoi/VisOnlyQA_Train](https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Train) `10d6012`, GPL-3.0 | all 7 tasks | include | Relicensed to GPL-3.0 (bases: CLEVR/SuperCLEVR BSD, synthetic geometry). Training images are CLEVR train, SuperCLEVR and synthetic geometry; the evaluation repositories are not read. The pilot found one image pixel-identical to a MathVista testmini image. |
| [nvidia/Nemotron-Image-Training-v3.5](https://huggingface.co/datasets/nvidia/Nemotron-Image-Training-v3.5) `70b27e0`, OpenMDW-1.1 | `coco_count_spatial_qa` | include | Counting and above/below choices from COCO annotations. Images are read from COCO `train2017.zip` with HTTP range requests (ETag `62ff7d7f…-2306`; zip CRC checked; COCO's TLS certificate does not match its hostname, so this is plain HTTP). train2017 contains val2014 images, and the pilot found two pixel-identical POPE images. Some counted instances are too small to see at the model's resolution. Spatial labels: see [below](#nemotron-coco-spatial-labels). |
| | `pixmo_count_qa` | exclude | Alternative packaging of allenai/pixmo-count, same unverifiable hashes. |
| | `chartverse_perception_qa` | pending | Short perception answers, but images must be extracted from opendatalab/ChartVerse-SFT-1.8M; not retrieved for this build. |
| | `chartnet_qa_*`, `chartverse_reasoning_*`, `babyvision`, `deepvision`, `vision_language_reasoning` | exclude | Reasoning-trace targets; the final answers presuppose the reasoning. |
| | `visualpuzzles_reasoning` | exclude | Built from the neulab/VisualPuzzles benchmark. |
| | `arxiv_chart_qa_*` | exclude | Needs arXiv source archives plus PDF/PostScript rendering. |
| | detection, grounding, pointing, video, multi-image subsets | exclude | Outside this build's single-image QA objective. |

### Nemotron COCO spatial labels

NVIDIA's card describes this subset only as a "public rule-based-conversion" of COCO and documents no rule. Against the pinned COCO 2017 instance boxes (`[media.coco_instances_train2017]`), every single-instance spatial label agrees with a comparison of box centers in the image plane, including when the boxes overlap on the asked axis (a held racket, a worn tie). That is an observed convention, not a documented one, and overlap alone is not treated as an error. When a named category has several instances, the row is kept only if every subject/reference pair gives the labelled answer. Otherwise the intended objects cannot be identified from the image, and the row goes to the `pending` bucket with `pending_reason: spatial_instance_dependent`. Of the 4,777 spatial questions in the source, 4,534 are kept and 243 are pending; none contradict the boxes. A row that contradicts every pair would be rejected. Counting questions are unchanged. Nothing here shows that this subset caused or explains Nemotron's spatial results.

### PixMo-Count

The card's own check is `sha256(requests.get(image_url).content) == image_sha256`, and Molmo's downloader does the same. The v1 pilot's 923 samples gave 830 hash mismatches, 29 gone (404/410), 64 rate-limited (429) and 0 matches. Earlier text that said "846 mismatched, 41 gone" was wrong. A bounded recheck of 60 evenly spaced train rows with the card's exact snippet returned 59 JPEGs and one 404, with no matches. Flickr serves the same bytes on refetch (12 of 12) and with browser or JPEG-only `Accept` headers (12 of 12). 36 of the 59 responses carry a 2019 `Last-Modified` date. Hashes of decoded pixels and of the URL do not match either. That rules out client headers and refetch drift as causes. Two explanations remain hypotheses: the published hashes came from a different representation at crawl time, or Flickr regenerated its size derivatives without changing `Last-Modified`. Hash checks are not waived, so the source stays pending until Ai2's original bytes or a reproducing hash definition is available ([evidence](../evidence/step05/pixmo-count.json)). Coverage from other counting data does not settle this.

### AI2D terms

AI2 has published two sets of terms for AI2D, and no document I found says which one governs ([evidence](../evidence/step05/ai2d-terms.json), collected by `tools/step05/ai2d_terms.py`).

- **Bundled licence.** `ai2d/license.txt` in `ai2d-all.zip`, titled "License for Non-Commercial Use" and marked "(C) 2016". Its zip entry is dated 2016-08-18, and all 28,757 members are dated August or September 2016. It allows "general use by academic or non-profit, or government-sponsored researchers" and evaluation elsewhere. It excludes use "in a for-profit enterprise". It grants no right "to modify and publicly release the data in any form" or "to distribute the data to a third party in any form", only to publish short segments or still images in an academic paper. It must accompany any redistribution.
- **CC BY-SA label.** AI2's dataset page `allenai.org/data/diagrams` showed "License: CC BY-SA", linked to the 4.0 deed. The field is absent from the 2020-09-27 Wayback capture and present in all 11 readable captures from 2021-01-26 to 2024-06-15. Wayback blocked the requests for six more captures in that span. The AWS Open Data listing (`allenai-diagrams.yaml` in `awslabs/open-data-registry`) copies the label.
  - The listing was added by [PR #915](https://github.com/awslabs/open-data-registry/pull/915), opened 2021-04-07 and merged 2021-04-20. Its author, Michal Guerquin (`aimichal`), called the bucket "our S3 bucket" and described the files as specs for the AI2 datasets "listed at https://allenai.org/data which have a license".
  - Every one of the PR's 18 commits had CC BY-SA, and no later commit has touched the file. The AWS Marketplace page shows the same label, with AI2 as provider and manager.

History, and what it does and does not settle:

- **Archive.** The zip has one S3 object version, last modified 2021-02-04T19:48:17Z. The TQA zip has the identical timestamp, which points to a bulk re-upload rather than a revision of AI2D. The archive AI2 serves today still contains the 2016 licence, uploaded after the CC BY-SA label first appeared.
- **Current pages.** Since at least 2024-08-16, `allenai.org/data/diagrams` redirects (HTTP 307) straight to the zip, so AI2's own site no longer shows a licence. The AWS listing that AI2 manages still says CC BY-SA. The PRIOR project page states no licence in any of its 14 readable captures from 2020-08 to 2026-03; 8 were blocked.
- **No explanation found.** No AI2 statement explains the label, withdraws the 2016 licence, or says one supersedes the other. I searched the registry history, the PR discussion, AI2's pages and the web.
- **TQA comparison.** AI2's TQA page gained the same "License: CC BY-SA" field between its 2020-10-23 and 2020-11-09 captures, and PR #915 gave TQA the same label. TQA's archive README says "Creative Commons Attribution – Non-Commerical 3.0", and from November 2020 to at least June 2021 the TQA page showed the CC BY-SA field next to a description saying the data "is distributed under a Creative Commons Attribution – Non-Commerical 3.0 license". The label field contradicted the dataset's own terms on the same AI2 page. That suggests AI2 did not set the field from each dataset's terms. It does not show what AI2 intended for AI2D.
- **Image origin.** The paper (arXiv 1603.07396, p. 9) says the images "are collected by scraping Google Image Search" and annotated on Mechanical Turk. Neither licence can grant rights in the diagrams themselves that AI2 does not hold, and per-image rights are not verified. The v2 recipe's AI2D `terms_note`, which the v2 manifest copies, says "diagrams collected by AI2, covered by the same licence". That clause is wrong. The recipe stays as built, because its bytes are hashed into the v2 manifest and its AI2D entry into the scan fingerprint. The next recipe revision that rebuilds for another reason should fix it.

This record treats neither source as overriding the other. A CC BY-SA 4.0 grant by AI2 would be irrevocable, and a rightsholder may offer data under more than one licence. The open question is whether the page label was a deliberate grant for AI2D or a catalogue default. The bundled licence is the document AI2 ships with the data, and the build follows it as the stricter of the two. Both allow this project's academic research training, so AI2D stays included as before.

Storage and redistribution are separate questions:

| Use | Bundled 2016 licence | CC BY-SA 4.0 label | Status |
|---|---|---|---|
| Copies on this CPU instance's disk (a GCP VM, so the data already sits with a cloud provider), or in a private bucket only the licensed researchers can read | Grants use by academic researchers. Says nothing about where copies are stored and defines neither "third party" nor "distribute" | Permitted | Neither text prohibits it. A cloud provider holding the copy under the project's account is not normally a recipient, but the licence does not say so |
| Read access for collaborators beyond those researchers, such as a shared bucket or a private Hub repo with other members | Not granted if they count as a third party; the licence names no project or institutional scope | Permitted with attribution | Unresolved. It depends on who counts as a third party |
| Public release of AI2D rows, images or modified copies (public Hub dataset, public bucket) | Not granted | Permitted with attribution; adapted material under BY-SA | Not permitted under the terms the build follows. Needs written confirmation from AI2 (`ai2-data@allenai.org` per the project page, `vision@allenai.org` per the README) |
| Weights trained on AI2D | Not addressed; it is unclear whether a model is a "derivation" | Not addressed by the label | Open; outside step 5 |

Per-image rights in the scraped diagrams remain open whichever AI2 licence applies.

**Finding AI2D in the exports.** Index rows (`index/*.parquet`) have `source == "fv_ai2d"` and `terms == "ai2-noncommercial"`, and portable and Miles rows have `metadata.source == "fv_ai2d"`. Full v2 has 4,277 AI2D rows: 4,245 in `default/sft/train` (247 of them in the sweep), 32 in `default/sft/validation`, and none in RL or pending buckets. They use 2,415 images (434.5 MB). No other source shares any of those images, so AI2D's images can be kept apart or left out as a set. In v3, `fv_ai2d` has 4,057 rows (220 left SFT because their image groups now hold RL rows), and AI2D content also reaches RL through ViRL39K: 163 exported rows with `provenance.upstream_source` `ai2d` or `ai2dConverted`, 107 of them on an image byte-identical to an `fv_ai2d` image. VisualWebInstruct-verified includes AI2D images too but marks no row's image source. `images.parquet` has no source column, so select AI2D images through the index:

```python
import pyarrow.parquet as pq, pyarrow.compute as pc, glob
rows = [pq.read_table(f, columns=["source", "image_path"]) for f in glob.glob("exports/full-v3/index/*.parquet")]
ai2d = sorted({p for t in rows for p in pc.filter(t, pc.equal(t["source"], "fv_ai2d"))["image_path"].to_pylist()})
```

### RL sources admitted in v3

ViRL39K, VisualWebInstruct-verified and ThinkLite-VL-70k were pending in v2 because each card's only terms are a `license: mit` metadata tag. Each dataset's creators set that tag on their own repository. TIGER-Lab published the first two, and ThinkLite's paper repository points to the `russwang` account for both the dataset and the models. Under the revised policy that declaration is the basis. It covers what the creators released. It cannot relicense data they took from elsewhere, so each upstream's terms are recorded separately ([terms evidence](../evidence/step05/terms.json)). The explicit upstream restrictions found all allow noncommercial research training. Missing upstream terms are recorded as missing.

| Source | Explicit upstream restrictions | Upstream terms not found | Rows the terms attach to |
|---|---|---|---|
| ViRL39K | ScienceQA CC BY-NC-SA 4.0, "non-commercial research purposes only"; AI2D, AI2's licence (no redistribution); ChartQA GPL-3.0; LLaVA-OneVision-Data "academic research and education purpose" only | MMK12, K12, DVQA, GeoQA+ | Every row names its upstream in `provenance.upstream_source`. Exported default and pending rows: ScienceQA 668, ai2d 163, chartqa 141, MMK12 10,863, K12 499, dvqa 1,089, geoqa_plus 1,586. LLaVA-OneVision is listed on the card but not marked per row |
| VisualWebInstruct-verified | Images from ScienceQA, AI2D and ChartQA carry those terms | Rights in the web images (found through Google Image Search), DocVQA, GeoQA+ | Rows have no image source; only image matching can attach a restriction to a row |
| ThinkLite-VL-70k | ScienceQA, IconQA and TabMWP CC BY-NC-SA 4.0 | GeoQA, FigureQA data, OK-VQA | Rows have no source field; the paper lists the eight upstreams and their sizes |

Redistribution is not cleared for any of the three. Nothing is redistributed.

**Inspection.** I reviewed sampled prompts, images and targets from each source and checked targets against the images. A labelled triangle gives sin A = 7/25, and a fare table gives the 1,044 price difference, both as the targets say. One concrete problem turned up. ViRL39K's 1,302 `DeepScaleR` rows are text-only maths problems attached to 19 blank white images, one reused for 136 unrelated questions. No other RL source has a blank image. Select now holds RL rows whose thumbnail is constant as row-level pending, with `pending_reason: image_blank`, instead of training vision RL on them (1,297 after deduplication). 4,735 exported ThinkLite prompts and 945 ViRL39K prompts are in Chinese, mostly GeoQA-style geometry. The answer instruction stays English, and the targets are short and checkable.

**Scorer.** Admission runs every target through `k2hv.reward` with correct, equivalent, wrong, near-wrong and unclosed-reasoning responses. Testing raw targets from the three sources found correct LaTeX answers the scorer rejected: `\$22.50`, `150\degree`, `1{,}574` and `28 in^2`. `normalize` now accepts these; no check on wrong answers was relaxed. Targets the scorer cannot verify stay out, for example `\dfrac{18}{5}\ \unit{}`, `0<x<5`, `200 mph`, `194.6 N.m`, `C) 64`, `(0, 0)` and sentences. Long decimals with no stated precision go to the existing pending rule (ViRL39K 17, VisualWebInstruct-verified 362).

### RL candidates

| Source @ revision | Reference | Notes |
|---|---|---|
| [TIGER-Lab/ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) `812ec61`, MIT (creators' declaration) | `answer` (boxed) | `category`, `source`, `qid` and both pass-rate fields kept in provenance only. Images from the verified `images.zip`; multi-image rows rejected. A trailing "Let's think step by step and output the final answer within \boxed{}." is replaced by the scorer format. |
| [TIGER-Lab/VisualWebInstruct-verified](https://huggingface.co/datasets/TIGER-Lab/VisualWebInstruct-verified) `0e54d11`, MIT (creators' declaration) | `short_answer` | **97,295 rows in 95 files at this revision**, not the 973K in the card prose. Sub-questions (`idx` 3247.1, 3247.2) share a group. The reasoning `answer` is not read. |
| [opendatalab/ChartVerse-RL-40K](https://huggingface.co/datasets/opendatalab/ChartVerse-RL-40K) `0b28d26`, Apache-2.0 | `reward_model.ground_truth` | The system prompt is dropped (recorded). Long-decimal references without a stated precision are pending. |
| [russwang/ThinkLite-VL-70k](https://huggingface.co/datasets/russwang/ThinkLite-VL-70k) `5c86ea4`, MIT (creators' declaration) | `ground_truth` | No per-row upstream source. Items resemble FigureQA, TabMWP, Geometry3K and others that feed MathVista, so image and question matching is the only provenance control; the pilot found MathVista and MMBench duplicates. |
| [hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k) `fd21e53`, MIT (InterGPS release) | `answer` | `train` only; validation/test preserved. The pilot found two training images pixel-identical to MathVista testmini images (removed). |

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
  "problem": "<image>\nFind x.\n\nSolve the problem. After your reasoning, end your response with a line of the form \"Answer: X\" (without quotes), where X is the final answer as a number or expression, without units.",
  "answer": "17.1",
  "images": ["/data/k2hv/images/…"],
  "metadata": {"data_source": "k2hv_math_v1", "dataset": "hiyouga/geometry3k", "source": "rl_geometry3k", "upstream_id": "train:12", "reference": "17.1", "answer_type": "numeric", "task": "geometry", "record_id": "…", "image_sha256": "…"}
}
```

`metadata.data_source` is only the scorer selector. `dataset`, `source`, `upstream_id` and the raw `reference` (the source label before canonicalization) are separate. `k2hv_decimal_v1` rows also carry `answer_decimals`.

The selectors are implemented in [`src/k2hv/reward.py`](../../src/k2hv/reward.py), which states the response contract. In short:

- Only the text after the closing think tag for the run's reasoning effort is scored (`</ifm|think>`, `</ifm|think_fast>`, `</ifm|think_faster>`). A missing, repeated or mismatched close tag scores 0 as a format failure, so an `Answer:` line inside unfinished reasoning earns nothing.
- The final answer is the last `Answer:` line after that tag, or else the last `\boxed{}`.
- `k2hv_math_v1`: equal after notation normalization (delimiters, `\text{}`, `\dfrac`, degrees, `%`, trailing units on numeric answers), or exactly equal as sympy expressions built by a whitelist parser (never `eval`). `3/2`, `\frac32`, `1.50` and `\boxed{1.5}` match `1.5`; `0.33` does not match `\frac13`.
- `k2hv_choice_v1`: the option letter.
- `k2hv_decimal_v1`: a number that rounds, half up, to the target at the decimal places the question states.
- Aborted or failed rollouts and scorer exceptions return `score: None` (infrastructure), not 0. An unknown selector raises.

These replace RL360 `math_dapo` and `stem__supergpqa` for this data rather than changing what those names mean. The pinned `math_dapo` compares strings: it rejects `3/2`, `1.50`, `20%` and boxed-only answers, and credits an `Answer:` line inside unclosed reasoning (characterized in `tests/test_data_pipeline.py`).

Classification and admission (`k2hv.data.scoring`):

- References are unboxed and stripped of math delimiters (`$…$`, `\(…\)`, `\[…\]`). v1 kept `\( \)` in 1,150 targets, which only the delimited string matched; the raw label stays in `metadata.reference`.
- One letter matching parsed options, or text matching exactly one option, becomes a `k2hv_choice_v1` letter.
- Numbers (separators, currency, `%`, degrees and units removed) become `k2hv_math_v1`. If the question states a precision ("round to four decimal places", "nearest tenth", "nearest 0.01") and the reference is written at exactly that precision, the row uses `k2hv_decimal_v1`.
- Numbers with more than three decimals and no matching stated precision go to the `pending/rl` buckets with `data_source: pending_decimal_precision`. No defensible rule exists for them: a blanket 1% tolerance accepts 689.5 for 683.33. They are not trainable until a source- or question-specific rule exists.
- Expressions with `\frac`, `\sqrt`, π or powers become `k2hv_math_v1` if the parser reads them and they have at most one variable.
- Yes/no, scientific notation, free text, multi-variable formulas, coordinate pairs and unparseable expressions are excluded with a reason.

A row is admitted only if its selector, run through `k2hv.reward.score`:

- accepts the target in all three think-tag forms, boxed, after explanatory text, and in equivalent notations (for example `a/b`, `\frac{a}{b}`, trailing zero, sympy's own text and LaTeX forms);
- rejects wrong and near-wrong values (±1, ×1.5+1, ±one unit in the last written place);
- gives no credit to the target written inside unclosed reasoning.

`validate` re-runs this admission on every exported RL row.

Every RL prompt is also rendered with the image-aware template (`add_generation_prompt=True`) and processed; prompts above 1,536 tokens are rejected. Step 8 must still check Miles's own RL chat-template path.

## Overlap and splits

**Evaluation exclusion.** The `evals` stage indexes the exact [step 7](07-eval.md) splits at pinned revisions (lmms-eval packaging): MMMU `validation` (`lmms-lab/MMMU` `364f2e2`), MathVista `testmini` (`AI4Math/MathVista` `2b6ad69`), ChartQA `test` (`lmms-lab/ChartQA` `9e63b7d`), DocVQA `validation` (`lmms-lab/DocVQA` `539088e`), TextVQA `validation` (`lmms-lab/textvqa` `9c0699c`), MMBench `en` `dev` (`lmms-lab/MMBench` `56ba1af`) and POPE (`lmms-lab/POPE` `4db1276`, `default` config, `test` split: adversarial + popular + random, 9,000 questions over 500 COCO val2014 images, matching lmms-eval's `pope` task). Every candidate image is compared with every evaluation image:

1. identical decoded pixels, or external IDs (COCO image IDs for POPE, Flickr photo and Open Images IDs for TextVQA);
2. 64-bit pHash within Hamming distance 10 and 32×32 thumbnail correlation ≥ 0.6, then a decoded comparison at 256 px: gradient (edge) correlation and the largest mean RGB difference over 16×16-px blocks. **Confirmed**: edge ≥ 0.98 (same structure, possibly recolored), or edge ≥ 0.93 with max block difference ≤ 0.10. **Uncertain**: edge ≥ 0.85. Both are excluded; uncertain matches are listed for inspection.
3. A question identical to an evaluation question after removing format lines is excluded when it is at least 120 characters, or when the two images are even loosely similar; otherwise it is listed but kept, since template questions ("What is the date mentioned in this letter?") recur across unrelated images.

The stage-2 thresholds come from inspecting every pilot pair (sheets via `inspect`). The first single-stage rule flagged 233 pilot images; nearly all were same-template Statista charts, TabMWP tables, US maps with a different highlighted state, or different pages of one document. The two-stage rule kept the true duplicates: POPE images inside COCO train2017, Geometry3K/VisOnlyQA/ThinkLite images inside MathVista testmini and MMBench dev, a recolored table, a VSR image. Crops are not detected. Re-encoded or resized copies are (tested).

**Grouping.** Images are joined into groups by identical pixels, stage-2-confirmed near-duplicates (pHash ≤ 3 via 16-bit band buckets), and shared document/rendering keys (CoSyn `data` hash, VisualWebInstruct base `idx`, VisOnlyQA source image). A group's ID is its smallest image sha256. Splits are drawn per group (`sha256(seed:sft-split:group)`), so every question about an image, its copies in other collections and its alternative renderings land in one split.

**Deduplication.** Two records are duplicates when they share an image group, the normalized question (format lines removed) and the normalized answer. The record from the earlier recipe source wins. The same generic question on different images is not a duplicate.

**RL reservation.** RL rows, including pending RL rows, are selected first; any SFT record whose group contains an RL image is removed, and the report counts the overlap. In v2 no group was shared, so nothing was removed.

**Preserved splits.** CoSyn `validation`, PixMo-Count `validation`/`test` and Geometry3K `validation`/`test` are never read for training. The evaluation repositories of VisOnlyQA are never read.

## Outputs and rematerialization

`data/step05/` after a build:

```text
images/<sha[:2]>/<sha>.<ext>   content-addressed original bytes (P/LA transparency composited on white, recorded)
evals/                         evaluation images (exclusion index only)
exports/<build>/               <build> = <profile>-<recipe version>, e.g. full-v2; v1 builds used <profile>
  portable/<variant>/<split>/part-*.parquet   rows with relative images/… paths
  miles/…                                     same rows with absolute paths (materialize)
  index/*.parquet              per-row provenance: source/revision/file/row/turn, upstream id, terms, group, split, token counts, row hash
  images.parquet               sha256, path, bytes and size of every referenced image
  manifest.json, SHA256SUMS, reports/{quality,verify}.json
work/<build>/                  scan candidates/rejects, selection, validation results (resumable state)
hf/, media/, inventory/        verified source downloads, pinned media (COCO annotations) and listings
```

Buckets: `default/sft/{train,validation,sweep50k}`, `default/rl/{pool,heldout}`, and `pending/sft/{train,validation}`, `pending/rl/{pool,heldout}` for rows inside included sources that are not yet trainable (`metadata.pending_reason`). Pending rows keep the same group-level splits and reserve their image groups from SFT, so admitting them later needs no reselection. They are outside quotas and must not be trained until their reason is resolved. The sweep is the first 50,000 default training rows in `sha256(seed:sweep:record_id)` order: a documented subset of train, never validation.

A recipe version builds into new `work/<build>` and `exports/<build>` directories, so earlier builds stay as they were. Export rewrites its own `portable/`, `index/` and `miles/` from scratch. To use the data on another machine, copy `images/` and `exports/<build>/` (rsync or object storage), then:

```bash
uv run python -m k2hv.data materialize --profile full --data-root /path/to/copy --target-root /data/k2hv
uv run python -m k2hv.data verify --profile full --data-root /path/to/copy --target-root /data/k2hv
```

`verify` checks SHA256SUMS. It requires a materialized file for every portable part and nothing extra, and every index row to appear exactly once in its bucket's Miles files with the validated row hash. It checks each row's image file name against its `image_sha256`, the referenced images against `images.parquet`, and that each image re-hashes and decodes. It recomputes group sharing from the exported indices: any group across train/validation, pool/held-out or SFT/RL fails, and same-split sharing between `default` and `pending` is listed as permitted. The 50k sweep must be a subset of train. Miles's loader does not read `s3://`, so images must be on a local or mounted path at training time.

### Step 6 dependency: lazy image loading

Pinned Miles `Dataset.__init__` calls `process_vision_info` for every row and keeps each decoded image, about 3 MB per row for this data (2.6 TB for the v1 training split). `k2hv.lazy_data.LazyRolloutDataSource` replaces only the dataset and keeps pinned Miles batching, shuffling and checkpointing. See [step 6](06-sft.md#loading-the-step-5-data).

### Step 6 fix: one BOS in the SFT path

The chat template already emits `<|ifm|begin_of_text|>`, and the tokenizer's `TemplateProcessing` post-processor (in `tokenizer.json`) adds another unless the caller passes `add_special_tokens=False`. The step 4 `prepare_sample` used the default, so every v1 training sequence started with two BOS tokens. It now passes `add_special_tokens=False`, which matches SGLang and the step 4 gate tokenization, and the contract rejects any row without exactly one BOS at position 0. On 16 v1 rows (one per source) the fix removes only the duplicate: totals drop by one, while image tokens, grids and supervised answer/end spans are unchanged ([evidence](../evidence/step05/bos.json), `tools/step05/bos_evidence.py`). Row contents are built before `prepare_sample` runs. The 803,646 SFT rows present in both v1 and v2 are identical apart from a new `metadata.pending_reason: null` key; token counts in indices and reports drop by one.

Step 4 certified the old `sft_rollout.py` (sha256 `6d2884ae…`) on GPU. The fixed path has not run on a GPU; step 6 must rerun a short Miles SFT check before a long run.

The pinned Miles multimodal RL rollout (`call_processor` in `miles/utils/processing_utils.py`) tokenizes the rendered prompt the same way and would also double the BOS. See [step 8](08-rl.md).

## Build results

### v3 (current)

Recipe version `v3` on 2026-11-01, built into `data/step05/{work,exports}/full-v3`. `full-v2` is unchanged. The scorer change made every scan fingerprint stale, so all 248 shards were rescanned in 57 minutes, and the final run reused that scan from a stopped attempt. Wall time of the final run: select 84 min, validate 14 min, export 1.5 min, materialize 10 s, verify 9.5 min. Validation reused v2's results for 871,781 SFT rows whose record id and content hash were unchanged (`validate --reuse-validation full-v2`); every RL row was validated fresh. Logs: `outputs/step05/full-v3-build.log` and the stopped attempts `full-v3-build.attempt{1,2}.log`, `full-v3-scan.attempt1.log`. Summaries: [`full-v3.json`](../evidence/step05/full-v3.json) and the per-source accounting [`rl-v3.json`](../evidence/step05/rl-v3.json) from `tools/step05/rl_v3_report.py`.

| Stage | Records |
|---|---|
| Candidates after text and image rules (20 included sources) | 2,299,805 over 718,621 unique images, 591,256 groups |
| Excluded for evaluation overlap | 7,237 (MMBench dev 3,876, MathVista testmini 1,853, ChartQA test 1,259, POPE 227, TextVQA val 19, MMMU val 2, DocVQA val 1) |
| Exact QA duplicates | 44,655 |
| SFT candidates in RL image groups (reserved for RL) | 12,943 |
| RL rows held because the image is blank | 1,297 |
| Selected and validated | 1,019,427; 0 contract failures |
| Exported after quotas | 953,854 rows (1,003,854 index rows with the sweep), 627,987 images (77.13 GB) |

| Bucket | Rows | Images | SFT total tokens or RL prompt tokens, p50 / p99 / max | Supervised p50 / p99 / max |
|---|---|---|---|---|
| `default/sft/train` | 799,209 | 502,121 | 516 / 1,084 / 1,141 | 2 / 14 / 101 |
| `default/sft/validation` | 8,075 | 5,113 | 517 / 1,087 / 1,139 | 2 / 15 / 60 |
| `default/sft/sweep50k` (subset of train) | 50,000 | 48,648 | 516 / 1,087 / 1,139 | 2 / 14 / 44 |
| `default/rl/pool` | 122,896 | 100,064 | 200 / 1,145 / 1,505 | |
| `default/rl/heldout` | 6,650 | 5,451 | 202 / 1,149 / 1,391 | |
| `pending/sft/train` | 238 | 238 | 392 / 550 / 571 | 2 / 2 / 2 |
| `pending/sft/validation` | 3 | 3 | | |
| `pending/rl/pool` | 16,004 | 14,539 | 1,089 / 1,122 / 1,423 | |
| `pending/rl/heldout` | 779 | 776 | | |

Per new source (losses in order; full detail in `rl-v3.json`):

| | ViRL39K | VisualWebInstruct-verified | ThinkLite-VL-70k |
|---|---|---|---|
| Source rows | 38,870 | 97,295 | 69,997 |
| Rejected at scan | 7,317: multi-image 2,289, free-form answer 2,334, options not parsed 1,793, yes/no 424, unparseable expression 380, multivariable 56, number format 24, choice unmatched 6, image too small 4, scorer accepts a wrong answer 7 | 55,225: free-form 28,567, multi-image 17,202, unparseable 4,157, yes/no 1,740, options not parsed 1,255, choice unmatched 1,044, multivariable 981, reserved marker 235, number format 20, scorer accepts wrong 16, scientific 5, scorer rejects reference 3 | 24,225: free-form 13,016, yes/no 11,161, choice unmatched 35, options not parsed 8, empty 3, unparseable 2 |
| Evaluation overlap excluded | 345 (MMBench 233, MathVista 96, ChartQA test 16) | 4 | 3,473 (MMBench 2,789, MathVista 684) |
| Duplicates | 1,278 | 2,373 | 1,182 |
| Held pending | 1,314 (blank image 1,297, no stated precision 17) | 362 (no stated precision) | 0 |
| Exported default pool / held-out | 27,143 / 1,473 | 37,322 / 2,009 | 38,964 / 2,153 |
| Default selectors | math 19,692, choice 8,873, decimal 51 | math 37,981, decimal 1,021, choice 329 | math 22,597, choice 18,199, decimal 321 |

No row failed the length limits. Every exported row passed the processor contract and target re-admission.

- **RL.** 129,546 default rows (122,896 pool, 6,650 held-out), up from 20,998: ThinkLite 41,117, VisualWebInstruct-verified 39,331, ViRL39K 28,616, ChartVerse-RL 18,972, Geometry3K 1,510. Selectors: `k2hv_math_v1` 100,638, `k2hv_choice_v1` 27,401, `k2hv_decimal_v1` 1,507. By task: maths and science reasoning 69,733, web STEM 39,331, charts 18,972, geometry 1,510. ViRL39K categories in the default buckets: grade-school geometry 9,674, other grade-school maths 9,350, tables/diagrams/charts 4,700, grade-school science 3,262, spatial 1,529, broader STEM 802, social science 496, commonsense 117.
- **Geometry3K.** 516 of v2's Geometry3K rows are now ThinkLite rows: ThinkLite contains the same problems, every one has a ThinkLite twin with the same normalized question and answer, and deduplication keeps the source listed first. Another 39 Geometry3K rows moved between pool and held-out, with unchanged content, because merging with ThinkLite's copies of their images changed their group.
- **Pending.** The ChartVerse long-decimal rows are byte-identical to v2 (15,107). New pending rows: 1,297 `image_blank` and 379 `no_stated_precision` from the new sources. Nemotron spatial pending: 241 (one moved with an image group reserved for RL).
- **SFT.** 807,525 rows against v2's 812,260. 8,641 v2 rows left SFT, each because its v3 image group holds a selected RL row. Select reserved 8,595 of them for RL directly. The other 46 are IconQA rows that deduplication dropped first, because another IconQA row with the same question and answer sat in the same merged group, and select then reserved that row. 7,524 of the 8,641 already shared their v2 group with an exported RL row. The other 1,117 joined an RL group in v3 through the new sources' images. For 75 the row's own image carries an RL row, 999 sit one identical-pixel or verified near-duplicate edge from an RL image, and 43 sit 2 to 28 edges away. Every RL row in those groups is exported, from ThinkLite or ViRL39K. No group holds only pending RL rows. See [traced removals](#traced-sft-removals). Quota-limited sources refilled 3,906 rows (IconQA 2,090, VQAv2 1,127, TallyQA 570, CLEVR 106, VisOnlyQA 13). The 803,619 rows in both builds are identical in content and bucket. Losses by source: IconQA 5,821 (ThinkLite carries IconQA images), VQAv2 1,127, ChartQA 727, TallyQA 570, AI2D 220, CLEVR 106, Nemotron 44, VisOnlyQA 13, VSR 12, TextVQA 1.

**Checks.**

| Check | Scope | Result |
|---|---|---|
| Miles normalization, `prepare_sample`, processor contract, one BOS | 1,019,427 selected rows (871,781 SFT results reused from v2) | 0 failures |
| RL target re-admission through `k2hv.reward` | every selected RL row | 0 failures |
| Files against SHA256SUMS; materialized file set against portable parts | 27 files | complete, 0 bad |
| Each index row once in its bucket's Miles files with the validated hash | 1,003,854 rows | 0 missing, 0 mismatches |
| Images re-hashed and decoded; referenced set equals `images.parquet` | 627,987 | 0 bad, equal |
| Group sharing across train/validation, pool/held-out or SFT/RL | all buckets | none |
| Same-split sharing between `default` and `pending` (permitted) | | SFT train 197, validation 3; RL pool 277, held-out 13 |
| Sweep is a subset of train | 50,000 | true |
| Select rerun with `--trace` reproduces the selection | `selected.parquet`, `exclusions.parquet`, `question_text_matches.parquet`, `select.json` | byte-identical |
| Tests | 279 passed, 1 skipped | |

RL difficulty calibration on the GPU still waits for the SFT checkpoint ([below](#rl-prompt-selection-after-sft)).

#### Traced SFT removals

The first v3 write-up explained nine removed rows only by pHash distance. Their images are 4 or 6 from the nearest selected RL image, where the others are within 2. Select did not record why two images share a group, so I added `select --trace`. `select/trace/candidates.parquet` holds the decision for every candidate, one of evaluation exclusion, a limit, duplicate with its winner, selected, RL quota, reserved for RL, per-image cap or SFT quota. `select/trace/edges.parquet` holds every identical-pixel, verified near-duplicate and group-hint edge, with stage-2 metrics. Tracing only adds these two files.

I reran select alone with `--trace` in a scratch root that links to the full-v3 scan, evaluation index and image store, so nothing in `work/full-v3` changed. In 94 minutes it reproduced `selected.parquet`, `exclusions.parquet`, `question_text_matches.parquet` and `select.json` byte for byte. `tools/step05/sft_removals_trace.py` then follows the shortest edge chain from each removed row's image to an image with a selected RL row. The summary and the nine rows are in [`sft-removals-v3.json`](../evidence/step05/sft-removals-v3.json). Detail for all 8,641 rows is machine-local in `outputs/step05/full-v3-sft-removals.parquet`.

All nine are FineVision IconQA rows `fv_iconqa:iconqa/train-00000-of-00001.parquet:<row>:0` at revision `3c380a73…`, in v2's `default/sft/train`. Each reaches an exported RL row in its v3 group through confirmed near-duplicate edges. Every edge in these chains is at pHash distance 0 or 2 and passed stage-2 verification. The distance of 4 or 6 between the end images builds up over several edges.

| IconQA row | v3 decision | Edges to the RL image | RL row reached (pool) | Group images |
|---|---|---|---|---|
| 13559 | reserved for RL | 2 | `rl_thinklite` 63131 | 5 |
| 14322 | reserved for RL | 7 | `rl_thinklite` 62749 | 24 |
| 1646 | duplicate of IconQA 26401 (reserved) | 3 | `rl_thinklite` 59283 | 5 |
| 16806 | reserved for RL | 5 | `rl_thinklite` 60865 | 22 |
| 16978 | reserved for RL | 3 | `rl_thinklite` 56101 | 7 |
| 21067 | reserved for RL | 3 | `rl_thinklite` 62494 | 13 |
| 2367 | duplicate of IconQA 12734 (reserved) | 28 | `rl_virl39k` 1259 | 191 |
| 26337 | duplicate of IconQA 22188 (reserved) | 3 | `rl_thinklite` 55455 | 13 |
| 7594 | reserved for RL | 2 | `rl_thinklite` 60067 | 3 |

The images are IconQA templates, mostly a ruler under a line of varying length, plus a flower beside a cube and a row of tiled squares. Neighbouring images differ by a few pixels, so near-duplicate chains join template images whose questions differ. Row 2367's group has 191 images. The grouping is doing what the recipe specifies. A long chain costs some SFT rows, but it cannot put related images on both sides of a split. No exported row changed, so full-v3 stays as built and needs no new version.

### v2

Recipe version `v2` on 2026-11-01 (n2d-standard-16, 16 vCPU, 62 GiB), seed 20261031, built into `data/step05/{work,exports}/full-v2`. Wall time: scan 56 min, evals 4 s (reused), select 61 min, validate 104 min, export 1 min, materialize 8 s, verify 9 min. Log: `outputs/step05/full-v2-build.log`; progress log: `outputs/step05/progress.log`. Summary: [`full-v2.json`](../evidence/step05/full-v2.json), pilot [`pilot-v2.json`](../evidence/step05/pilot-v2.json).

| Stage | Records |
|---|---|
| Candidates after text and image rules (17 included sources) | 2,180,410 over 634,210 unique images, 517,627 groups |
| Excluded for evaluation overlap | 3,415 |
| Exact QA duplicates | 39,247 |
| SFT records reserved for RL | 0; no group is shared by SFT and RL candidates |
| Selected (1.15× oversampling for admitted rows, all pending rows), all validated | 917,421; 0 contract failures |
| Exported after quotas | 848,365 rows (898,365 index rows with the sweep), 547,348 images (73.07 GB) |

| Bucket | Rows | Images | SFT total tokens or RL prompt tokens, p50 / p99 / max | Supervised p50 / p99 / max |
|---|---|---|---|---|
| `default/sft/train` | 803,904 | 506,660 | 515 / 1,084 / 1,141 | 2 / 14 / 101 |
| `default/sft/validation` | 8,114 | 5,149 | 516 / 1,087 / 1,139 | 2 / 15 / 60 |
| `default/sft/sweep50k` (subset of train) | 50,000 | 48,661 | 515 / 1,087 / 1,139 | 2 / 14 / 44 |
| `default/rl/pool` | 19,961 | 19,477 | 1,090 / 1,125 / 1,161 | |
| `default/rl/heldout` | 1,037 | 1,005 | 1,090 / 1,127 / 1,144 | |
| `pending/sft/train` | 239 | 239 | 392 / 551 / 571 | 2 / 2 / 2 |
| `pending/sft/validation` | 3 | 3 | | |
| `pending/rl/pool` | 14,337 | 14,271 | 1,091 / 1,122 / 1,170 | |
| `pending/rl/heldout` | 770 | 769 | | |

- **SFT.** 812,018 default rows, every one with exactly one BOS. Image tokens never exceed 1,024 and totals never exceed 1,141.
- **RL.** 20,998 admitted rows: `k2hv_math_v1` 20,735 and `k2hv_decimal_v1` 263, from ChartVerse-RL (18,972) and Geometry3K (2,026). No admitted RL source has option-letter targets, so `k2hv_choice_v1` has no rows in v2.
- **Pending (15,349 rows).** 15,107 ChartVerse-RL long decimals with no stated precision, and 242 Nemotron spatial rows whose instance pairs disagree. Examples are in `full-v2.json` under `pending`.

Sources below quota exported every eligible row: AI2D 4,277 of 15,000, VSR 3,075 of 10,000, TextVQA 34,552 of 40,000, ChartQA 26,808 of 40,000, Nemotron 14,255 admitted of 15,000, CoSyn chart 119,051 of 120,000, ChartVerse-RL 18,972 admitted of 40,000, Geometry3K 2,026 of 3,000. Rejection reasons by source are in `per_source.*.scan.rejected` of `full-v2.json`.

**Evaluation exclusion.** 3,415 records:

| Set | Excluded |
|---|---|
| ChartQA test | 1,243 |
| MathVista testmini | 1,070 |
| MMBench dev | 854 |
| POPE | 227 |
| TextVQA val | 19 |
| DocVQA val, MMMU val | 1 each |

By method: identical pixels 177, stage-2 confirmed 1,247, stage-2 uncertain 1,547, question text 444. A further 1,097 template-question matches are listed and kept. The count fell from v1's 8,938 because the pending sources (ThinkLite, DVQA, ViRL39K and others) are no longer selected. Pair sheets: `outputs/step05/full-v2/exclusions/` (32 categories).

**Checks.**

| Check | Scope | Result |
|---|---|---|
| Miles normalization, `prepare_sample`, processor contract, one BOS | every selected row (917,421) | 0 failures |
| RL target re-admission through `k2hv.reward` | every selected admitted RL row (20,998 after quotas) | 0 failures |
| Files against SHA256SUMS; materialized file set against portable parts | 27 files | complete, 0 bad |
| Each index row once in its bucket's Miles files with the validated hash, and its image file named by its `image_sha256` | 898,365 rows | 0 missing, 0 mismatches |
| Images re-hashed and decoded; referenced set equals `images.parquet` | 547,348 | 0 bad, equal |
| Group sharing across train/validation, pool/held-out or SFT/RL | all buckets | none |
| Same-split sharing between `default` and `pending` (permitted) | | SFT train 198, validation 3; RL pool 184, held-out 13 |
| Sweep is a subset of train | 50,000 | true |
| Interrupted and resumed pilot equals a clean pilot | scan, selection, export bytes | identical ([evidence](../evidence/step05/reproducibility.json)) |
| Visual review | sampled: 27 v2 cards plus the v1 reviews | [findings](../evidence/step05/manual-audit.json) |

The processor check and the sampled review do not show that every label is correct or that the corpus is enough. Model quality needs the SFT sweep and held-out evaluation.

### Changes from v1

v1 (`exports/full`, code `061982f`, recipe `c91f84fe…`) is preserved unchanged: its SHA256SUMS still hashes to `6d78bf8b…`.

| | v1 | v2 |
|---|---|---|
| Default SFT train / validation | 839,928 / 8,605 | 803,904 / 8,114 |
| Add-on SFT (GPL/NC sources) | 112,349 | none; ChartQA, IconQA and VisOnlyQA are in default |
| RL pool / held-out | 119,413 / 6,409 (`math_dapo`, `stem__supergpqa`) | 19,961 / 1,037 (`k2hv_*`) |
| Pending | 15,486 long-decimal RL rows | 15,107 RL decimals and 242 Nemotron spatial rows |
| BOS per SFT row | 2 | 1 |
| Images | 709,353 (99.28 GB) | 547,348 (73.07 GB) |

What drives the differences:

- **Terms policy.** Visual7W, ST-VQA, DocVQA, InfographicVQA, DVQA, ViRL39K, VisualWebInstruct and ThinkLite are pending. That removes about 113k SFT rows and the three largest RL sources.
- **Unaffected sources.** Scan outputs of CoSyn, CLEVR, TallyQA, TextVQA and VQAv2 are byte-identical to v1. Other sources differ only in `terms`, `status`, the 243 Nemotron spatial rows, or the RL prompt and target fields ([comparison](../evidence/step05/scan-v1-v2.json)).
- **Unchanged SFT rows.** The 803,646 SFT rows in both builds are identical except for a new `metadata.pending_reason: null` key.
- **RL targets.** The 1,150 delimited RL targets are fixed. Re-admitting every v1 RL row under the new contract is recorded in [`rl-recheck-v1.json`](../evidence/step05/rl-recheck-v1.json).

### Artifacts

| Path (under `data/step05/`) | Size | sha256 |
|---|---|---|
| `exports/full-v3/SHA256SUMS` (27 Parquet files) | 524 MB export tree | `35adeee86e8f6963d6e3af844da71f1de31edf0a45bc16537ac7fa50ebac468c` |
| `exports/full-v3/manifest.json` | | `8c40c9fc79da44cc591c2799bc1967896496cd296337b4be092f00d5217e9a92` |
| `exports/full-v3/images.parquet` | | `97b05072e4e4ab01d791406b96b29817535ae61081b93b704d7ec18b15302ccc` |
| `exports/full-v2/SHA256SUMS` (v2, preserved) | 483 MB export tree | `d12c0a06e5ff2a13fe1458f31b80a21195b3d7c13f248dba014ec60e2cb40a73` |
| `exports/full-v2/manifest.json` | | `e8a6f8a90d498d677cde32e786e1bacd51648fd7a13a43cff5a0e7fb1b9d732d` |
| `exports/full-v2/images.parquet` | | `02538c8fefec0af304f2f176fd86335471091eb87d7c89f563de7565fed0b529` |
| `exports/pilot-v2/SHA256SUMS` | | `6f6cd126950302131510b04df17cdccb82013eefe833712da818aa4a097eb5a6` |
| `exports/full/SHA256SUMS` (v1, preserved) | | `6d78bf8bd6a5b1a720ac9b552c5931094d1e7ea663f09ff829c0c2903531fd73` |
| `images/` | 77.13 GB referenced by v3, 73.07 GB by v2 (one store for all builds) | per image, by name |

These exist only on the CPU instance, and nothing was uploaded. The same holds for:

- `hf/` (verified source downloads), `media/` (pinned COCO annotations, URL cache), `evals/` and `work/{full,full-v2,full-v3,pilot-v2,pilot-v2r}`
- `data/step05-resume/`
- `outputs/step05/`: logs, inspection and exclusion sheets, terms documents (`terms/`, including `terms/ai2d` and `terms/rl-v3`), PixMo checks
- the `third_party/` checkouts

To use the data elsewhere, copy `images/` and `exports/full-v3/` and run `materialize` and `verify` as [above](#outputs-and-rematerialization). To rebuild, follow [Setup](#setup) and run `build --profile full`. A fresh build reproduces these bytes while the pinned Hub revisions, the COCO train2017 zip (ETag-pinned) and the COCO annotations zip (sha256-pinned) stay available.

### Known limitations

- Cropped copies of evaluation images are not detected; re-encoded, resized and recolored copies are.
- RL coverage is maths, science, web STEM, charts and geometry (129,546 default rows). Most targets are numbers, expressions or option letters; RL has no free-form, yes/no or open-text targets because the scorer cannot verify them. Difficulty is not calibrated until the SFT checkpoint exists.
- Upstream terms that could not be found (MMK12, K12, DVQA, GeoQA, FigureQA data, OK-VQA, web images in VisualWebInstruct) are recorded as missing; rows from VisualWebInstruct-verified and ThinkLite carry no upstream source, so upstream restrictions can be traced to their rows only by image matching.
- Per-image rights of scraped media (ChartQA charts) and textbook diagrams (Geometry3K) are not verified; only the dataset release terms are.
- AI2D is admitted under its bundled academic, non-profit licence, which grants no redistribution. AI2 also labelled it CC BY-SA, and which terms govern is unresolved ([AI2D terms](#ai2d-terms)). Its rows are tagged `fv_ai2d` so they can be held back from anything shared.
- The fixed BOS path, the lazy loader and the `k2hv.reward` scorers have CPU evidence only; each needs its first GPU run (steps 6 and 8).

## RL prompt selection after SFT

This GPU-dependent stage waits for the [SFT checkpoint](06-sft.md). Follow IFM's protocol from RL360 (`src/agent360/harbor/miles/calibration.py`):

1. Take a 10–20k-prompt subset of `default/rl/pool` (122,896 prompts in v3).
2. Sample the SFT checkpoint 16 times per prompt, at the RL temperature, image budget and reasoning effort used for training and evaluation.
3. Keep prompts with a pass rate between 20% and 40%.

Use our checkpoint's measured pass rates, not difficulty or pass-rate annotations from other models (ViRL39K's `PassRate_*` fields stay in provenance only). Prompts the model always or never solves give every sample the same reward, so GRPO assigns them zero advantage. This calibration is a dependency for RL, not part of the CPU preparation above.
