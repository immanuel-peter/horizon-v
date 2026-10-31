"""Source-specific readers. Each adapter yields rows of raw items; validation is shared.

A row is ``{"row", "upstream_id", "image": bytes | ("url", url, sha) | ("zip", key, member),
"group_hint", "external_ids", "orig_split", "reject", "turns": [...]}`` and each turn has
``question``/``answer`` (SFT) or ``problem``/``reference`` (RL) plus ``provenance``.
"""

import hashlib
import json
import re
import zipfile

import pyarrow.parquet as pq

from . import hub

FLICKR = re.compile(r"staticflickr\.com/(?:\d+/)+(\d+)_")
COCO_FILE = re.compile(r"(?:train|val)(?:2014|2017)[/_](?:COCO_(?:train|val)2014_)?0*(\d+)\.jpg")


def sha(text):
    return hashlib.sha256((text or "").encode()).hexdigest()


def spaced(items, n):
    items = sorted(items)
    if n is None or n == "all" or n >= len(items):
        return items
    return [items[(i * len(items)) // n] for i in range(n)]


def parquet_files(ctx, ds, prefix, split="train"):
    files = hub.listing(ctx.root, ds["repo"], ds["revision"])["files"]
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet") and f.rsplit("/", 1)[-1].startswith(split))


def other_splits(ctx, ds, prefix):
    files = hub.listing(ctx.root, ds["repo"], ds["revision"])["files"]
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet") and not f.rsplit("/", 1)[-1].startswith("train"))


def parquet_shards(ctx, src, files):
    if ctx.pilot:
        return [{"file": f, "row_groups": list(range(ctx.profile.get("row_groups", 1)))} for f in spaced(files, ctx.profile.get("files", 2))]
    return [{"file": f, "row_groups": None} for f in spaced(files, src.get("files", "all"))]


def read_rows(ctx, ds, shard, columns):
    """Yield (row_index, row) from a whole verified file, or from selected row groups in a pilot."""
    if shard["row_groups"] is None:
        pf = pq.ParquetFile(hub.download(ctx.root, ds["repo"], ds["revision"], shard["file"]))
        index = 0
        for batch in pf.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                yield index, row
                index += 1
        return
    pf = pq.ParquetFile(hub.remote(ds["repo"], ds["revision"], shard["file"]))
    starts, offset = [], 0
    for g in range(pf.metadata.num_row_groups):
        starts.append(offset)
        offset += pf.metadata.row_group(g).num_rows
    for g in shard["row_groups"]:
        if g < pf.metadata.num_row_groups:
            for i, row in enumerate(pf.read_row_group(g, columns=columns).to_pylist()):
                yield starts[g] + i, row


def one_image(images):
    if not images:
        return None, "image_missing"
    if len(images) != 1:
        return None, "multi_image"
    data = images[0]["bytes"] if isinstance(images[0], dict) else images[0]
    return data, None if data else "image_missing"


# ------------------------------------------------------------------- SFT adapters


class FineVision:
    dataset = "finevision"
    columns = ["images", "texts", "visual_dependency_ratings", "image_correspondence_ratings", "relevance_ratings", "formatting_ratings"]

    def shards(self, ctx, src):
        return parquet_shards(ctx, src, parquet_files(ctx, ctx.datasets[self.dataset], src["config"] + "/"))

    def rows(self, ctx, src, shard):
        for i, r in read_rows(ctx, ctx.datasets[self.dataset], shard, self.columns):
            image, reject = one_image(r["images"])
            turns = []
            for t, qa in enumerate(r["texts"]):
                ratings = {k.removesuffix("_ratings"): (r[k][t] if r[k] and t < len(r[k]) else None) for k in self.columns[2:]}
                turns.append({"turn": t, "question": qa["user"], "answer": qa["assistant"], "provenance": {"ratings": ratings, "turns_in_row": len(r["texts"])}})
            yield {"row": i, "upstream_id": None, "image": image, "reject": reject, "turns": turns, "orig_split": "train", "group_hint": None, "external_ids": []}


class CoSyn:
    dataset = "cosyn"

    def shards(self, ctx, src):
        return parquet_shards(ctx, src, parquet_files(ctx, ctx.datasets[self.dataset], src["config"] + "/"))

    def preserved(self, ctx, src):
        return other_splits(ctx, ctx.datasets[self.dataset], src["config"] + "/")

    def rows(self, ctx, src, shard):
        for i, r in read_rows(ctx, ctx.datasets[self.dataset], shard, ["id", "image", "qa_pairs", "metadata", "data", "code"]):
            qa = r["qa_pairs"] or {}
            questions, answers, explanations = qa.get("question") or [], qa.get("answer") or [], qa.get("explanation") or []
            turns = [{"turn": t, "question": q, "answer": a, "provenance": {"explanation_chars": len(explanations[t]) if t < len(explanations) and explanations[t] else 0}} for t, (q, a) in enumerate(zip(questions, answers))]
            reject = None if len(questions) == len(answers) else "qa_lists_misaligned"
            meta = r["metadata"] or {}
            yield {"row": i, "upstream_id": r["id"], "image": (r["image"] or {}).get("bytes"), "reject": reject, "turns": turns, "orig_split": "train", "group_hint": "cosyn-data:" + sha(r["data"])[:24], "external_ids": [], "provenance": {"figure_type": meta.get("figure_type"), "topic": meta.get("topic"), "data_sha256": sha(r["data"]), "code_sha256": sha(r["code"])}}


class ChartVerseSFT:
    dataset = "chartverse_sft"

    def shards(self, ctx, src):
        return parquet_shards(ctx, src, parquet_files(ctx, ctx.datasets[self.dataset], ""))

    def rows(self, ctx, src, shard):
        for i, r in read_rows(ctx, ctx.datasets[self.dataset], shard, ["id", "images", "question", "answer", "code"]):
            image, reject = one_image(r["images"])
            question = r["question"] or ""
            if question.count("<image>") == 1 and question.lstrip().startswith("<image>"):
                question = question.lstrip()[len("<image>") :]
            yield {"row": i, "upstream_id": r["id"], "image": image, "reject": reject, "turns": [{"turn": 0, "question": question, "answer": r["answer"], "provenance": {}}], "orig_split": "train", "group_hint": "chartverse-code:" + sha(r["code"])[:24], "external_ids": [], "provenance": {"code_sha256": sha(r["code"])}}


def flickr_ids(url):
    m = FLICKR.search(url or "")
    return [f"flickr:{m.group(1)}"] if m else []


class PixmoCount:
    dataset = "pixmo_count"

    def shards(self, ctx, src):
        return [{"file": f, "row_groups": None, "stride": 40 if ctx.pilot else 1} for f in parquet_files(ctx, ctx.datasets[self.dataset], "data/")]

    def preserved(self, ctx, src):
        return other_splits(ctx, ctx.datasets[self.dataset], "data/")

    def rows(self, ctx, src, shard):
        for i, r in read_rows(ctx, ctx.datasets[self.dataset], shard, ["image_url", "image_sha256", "count", "points", "label"]):
            if i % shard["stride"]:
                continue
            points = len((r["points"] or {}).get("x") or [])
            reject = None if points == r["count"] else "count_points_mismatch"
            label = (r["label"] or "").strip()
            turns = [{"turn": 0, "question": f"How many {label} are in the image?\nAnswer with a number.", "answer": str(r["count"]), "provenance": {"label": label, "points": points, "question_template": "How many {label} are in the image?"}}]
            yield {"row": i, "upstream_id": r["image_sha256"], "image": ("url", r["image_url"], r["image_sha256"]), "reject": reject, "turns": turns, "orig_split": "train", "group_hint": None, "external_ids": flickr_ids(r["image_url"]), "url": r["image_url"]}


class PixmoAMA:
    dataset = "pixmo_ama"

    def shards(self, ctx, src):
        return [{"file": f, "row_groups": None, "stride": 150 if ctx.pilot else 1} for f in parquet_files(ctx, ctx.datasets[self.dataset], "data/")]

    def rows(self, ctx, src, shard):
        for i, r in read_rows(ctx, ctx.datasets[self.dataset], shard, ["image_url", "image_sha256", "question", "answer"]):
            if i % shard["stride"]:
                continue
            turns = [{"turn": 0, "question": r["question"], "answer": r["answer"], "provenance": {}}]
            yield {"row": i, "upstream_id": r["image_sha256"], "image": ("url", r["image_url"], r["image_sha256"]), "reject": None, "turns": turns, "orig_split": "train", "group_hint": None, "external_ids": flickr_ids(r["image_url"]), "url": r["image_url"]}


class Docmatix:
    dataset = "docmatix"

    def shards(self, ctx, src):
        files = parquet_files(ctx, ctx.datasets[self.dataset], "data/")
        return [{"file": f, "row_groups": [0]} for f in spaced(files, src.get("pilot_files", 3))] if ctx.pilot else [{"file": f, "row_groups": None} for f in spaced(files, src.get("files", "all"))]

    def rows(self, ctx, src, shard):
        for i, r in read_rows(ctx, ctx.datasets[self.dataset], shard, ["images", "texts"]):
            image, reject = one_image(r["images"])
            if reject == "multi_image":
                reject = "multi_page"
            turns = [{"turn": t, "question": qa["user"], "answer": qa["assistant"], "provenance": {"generator": qa.get("source")}} for t, qa in enumerate(r["texts"])]
            yield {"row": i, "upstream_id": None, "image": image, "reject": reject, "turns": turns, "orig_split": "train", "group_hint": None, "external_ids": [], "provenance": {"pages": len(r["images"] or [])}}


class VisOnlyQA:
    dataset = "visonlyqa"

    def shards(self, ctx, src):
        files = sorted(f for f in hub.listing(ctx.root, ctx.datasets[self.dataset]["repo"], ctx.datasets[self.dataset]["revision"])["files"] if f.startswith("data/") and f.endswith(".parquet"))
        if ctx.pilot:
            by_task = {}
            for f in files:
                by_task.setdefault(f.split("-")[0], f)
            return [{"file": f, "row_groups": [0]} for f in sorted(by_task.values())]
        return [{"file": f, "row_groups": None} for f in files]

    def rows(self, ctx, src, shard):
        for i, r in read_rows(ctx, ctx.datasets[self.dataset], shard, ["decoded_image", "prompt_no_reasoning", "answer", "id", "source", "image_path", "task_category", "image_category", "question_type"]):
            reject = None if r["question_type"] == "single_answer" else "multi_answer"
            stem = (r["image_path"] or "").rsplit("/", 1)[-1].rsplit(".", 1)[0]
            yield {"row": i, "upstream_id": r["id"], "image": (r["decoded_image"] or {}).get("bytes"), "reject": reject, "turns": [{"turn": 0, "question": r["prompt_no_reasoning"], "answer": r["answer"], "provenance": {"task": r["task_category"]}}], "orig_split": "train", "group_hint": "visonlyqa-image:" + stem, "external_ids": [f"{r['source'].lower()}:{stem}"], "provenance": {"source": r["source"], "image_category": r["image_category"]}}


class NemotronCoco:
    dataset = "nemotron_coco"
    path = "coco_count_spatial_qa/coco_count_spatial_qa.jsonl"

    def shards(self, ctx, src):
        return [{"file": self.path, "row_groups": None, "stride": 30 if ctx.pilot else 1}]

    def rows(self, ctx, src, shard):
        ds = ctx.datasets[self.dataset]
        path = hub.download(ctx.root, ds["repo"], ds["revision"], self.path)
        with open(path) as f:
            for i, line in enumerate(f):
                if i % shard["stride"]:
                    continue
                r = json.loads(line)
                user, assistant = r["messages"][0]["content"], r["messages"][1]["content"]
                images = [c["image"] for c in user if isinstance(c, dict) and c.get("type") == "image"]
                texts = [c for c in user if isinstance(c, str)]
                answer = "".join(assistant)
                answer = re.sub(r"^<think>\s*</think>", "", answer).strip()
                reject = None if len(images) == 1 and len(texts) == 1 and len(r["messages"]) == 2 else "multi_image_or_turn"
                m = COCO_FILE.search(images[0] if images else "")
                yield {"row": i, "upstream_id": r["id"], "image": ("zip", "coco_train2017", images[0]) if images else None, "reject": reject, "turns": [{"turn": 0, "question": texts[0] if texts else "", "answer": answer, "provenance": {}}], "orig_split": "train", "group_hint": None, "external_ids": [f"coco:{int(m.group(1))}"] if m else []}


# ------------------------------------------------------------------- RL adapters


class ViRL39K:
    dataset = "virl39k"

    def shards(self, ctx, src):
        return [{"file": "39Krelease.parquet", "row_groups": None, "stride": 400 if ctx.pilot else 1}]

    def rows(self, ctx, src, shard):
        ds = ctx.datasets[self.dataset]
        archive = zipfile.ZipFile(hub.download(ctx.root, ds["repo"], ds["revision"], "images.zip"))
        members = set(archive.namelist())
        for i, r in read_rows(ctx, ds, shard, ["question", "answer", "category", "source", "qid", "image", "PassRate_32BTrained", "PassRate_7BBase"]):
            if i % shard["stride"]:
                continue
            paths = list(r["image"] or [])
            image, reject = None, None
            if len(paths) != 1:
                reject = "multi_image" if paths else "image_missing"
            elif paths[0] not in members:
                reject = "image_missing"
            else:
                image = archive.read(paths[0])
            prov = {"category": r["category"], "upstream_source": r["source"], "pass_rate_32b_trained": r["PassRate_32BTrained"], "pass_rate_7b_base": r["PassRate_7BBase"]}
            yield {"row": i, "upstream_id": r["qid"], "image": image, "reject": reject, "turns": [{"turn": 0, "problem": r["question"], "reference": r["answer"], "provenance": prov}], "orig_split": "train", "group_hint": None, "external_ids": [], "category": r["category"]}


class VWIVerified:
    dataset = "vwi_verified"

    def shards(self, ctx, src):
        files = sorted(f for f in hub.listing(ctx.root, ctx.datasets[self.dataset]["repo"], ctx.datasets[self.dataset]["revision"])["files"] if f.endswith(".parquet"))
        if ctx.pilot:
            return [{"file": f, "row_groups": [0]} for f in spaced(files, 3)]
        return [{"file": f, "row_groups": None} for f in files]

    def rows(self, ctx, src, shard):
        for i, r in read_rows(ctx, ctx.datasets[self.dataset], shard, ["images", "idx", "question", "short_answer", "difficulty"]):
            image, reject = one_image(r["images"])
            problem = r["question"] or ""
            if "<image>" not in problem:
                problem = "<image>\n" + problem
            base = str(r["idx"]).split(".")[0]
            yield {"row": i, "upstream_id": str(r["idx"]), "image": image, "reject": reject, "turns": [{"turn": 0, "problem": problem, "reference": r["short_answer"], "provenance": {"difficulty": r["difficulty"]}}], "orig_split": "train", "group_hint": f"vwi:{base}", "external_ids": []}


class ChartVerseRL:
    dataset = "chartverse_rl"

    def shards(self, ctx, src):
        return [{"file": "ChartVerse-RL-40K.parquet", "row_groups": [0, 100, 200, 300] if ctx.pilot else None}]

    def rows(self, ctx, src, shard):
        for i, r in read_rows(ctx, ctx.datasets[self.dataset], shard, ["prompt", "reward_model", "images", "ability", "data_source"]):
            image, reject = one_image(r["images"])
            users = [m["content"] for m in r["prompt"] if m["role"] == "user"]
            system = [m["content"] for m in r["prompt"] if m["role"] == "system"]
            if len(users) != 1:
                reject = reject or "multi_turn"
            yield {"row": i, "upstream_id": f"row{i}", "image": image, "reject": reject, "turns": [{"turn": 0, "problem": users[0] if users else "", "reference": (r["reward_model"] or {}).get("ground_truth"), "provenance": {"system_prompt": system[0] if system else None, "ability": r["ability"]}}], "orig_split": "train", "group_hint": None, "external_ids": []}


class ThinkLite:
    dataset = "thinklite"

    def shards(self, ctx, src):
        return [{"file": "ThinkLite-VL-70k.parquet", "row_groups": None, "stride": 700 if ctx.pilot else 1}]

    def rows(self, ctx, src, shard):
        for i, r in read_rows(ctx, ctx.datasets[self.dataset], shard, ["image", "problem", "answer", "id", "ground_truth"]):
            if i % shard["stride"]:
                continue
            prov = {"answer_field": r["answer"], "answer_matches_ground_truth": r["answer"] == r["ground_truth"]}
            yield {"row": i, "upstream_id": str(r["id"]), "image": r["image"], "reject": None if r["image"] else "image_missing", "turns": [{"turn": 0, "problem": r["problem"], "reference": r["ground_truth"], "provenance": prov}], "orig_split": "train", "group_hint": None, "external_ids": []}


class Geometry3K:
    dataset = "geometry3k"

    def shards(self, ctx, src):
        return [{"file": f, "row_groups": None, "stride": 30 if ctx.pilot else 1} for f in parquet_files(ctx, ctx.datasets[self.dataset], "data/")]

    def preserved(self, ctx, src):
        return other_splits(ctx, ctx.datasets[self.dataset], "data/")

    def rows(self, ctx, src, shard):
        for i, r in read_rows(ctx, ctx.datasets[self.dataset], shard, ["images", "problem", "answer"]):
            if i % shard["stride"]:
                continue
            image, reject = one_image(r["images"])
            yield {"row": i, "upstream_id": f"train:{i}", "image": image, "reject": reject, "turns": [{"turn": 0, "problem": r["problem"], "reference": r["answer"], "provenance": {}}], "orig_split": "train", "group_hint": None, "external_ids": []}


ADAPTERS = {
    "finevision": FineVision(),
    "cosyn": CoSyn(),
    "chartverse_sft": ChartVerseSFT(),
    "pixmo_count": PixmoCount(),
    "pixmo_ama": PixmoAMA(),
    "docmatix": Docmatix(),
    "visonlyqa": VisOnlyQA(),
    "nemotron_coco": NemotronCoco(),
    "virl39k": ViRL39K(),
    "vwi_verified": VWIVerified(),
    "chartverse_rl": ChartVerseRL(),
    "thinklite": ThinkLite(),
    "geometry3k": Geometry3K(),
}


# ------------------------------------------------------------------- evaluation sets


def eval_rows(ctx, name, spec):
    """Yield evaluation items: images (bytes list), question text and identifiers."""
    files = spec.get("files")
    if files is None:
        import fnmatch

        listing = hub.listing(ctx.root, spec["repo"], spec["revision"])["files"]
        files = sorted(f for f in listing if fnmatch.fnmatch(f, spec["glob"]))
    for file in files:
        pf = pq.ParquetFile(hub.download(ctx.root, spec["repo"], spec["revision"], file))
        for batch in pf.iter_batches(batch_size=64):
            for r in batch.to_pylist():
                yield file, _eval_item(name, r)


def _img(v):
    return v.get("bytes") if isinstance(v, dict) else None


def _eval_item(name, r):
    if name == "mmmu_val":
        images = [_img(r.get(f"image_{k}")) for k in range(1, 8)]
        return {"id": r["id"], "question": r["question"], "images": [b for b in images if b], "external_ids": []}
    if name == "mathvista_testmini":
        meta = r.get("metadata") or {}
        return {"id": r["pid"], "question": r["question"], "images": [_img(r["decoded_image"])], "external_ids": [], "source": meta.get("source")}
    if name == "chartqa_test":
        return {"id": None, "question": r["question"], "images": [_img(r["image"])], "external_ids": []}
    if name == "docvqa_val":
        return {"id": r["questionId"], "question": r["question"], "images": [_img(r["image"])], "external_ids": [f"ucsf:{r['ucsf_document_id']}:{r['ucsf_document_page_no']}"]}
    if name == "textvqa_val":
        return {"id": str(r["question_id"]), "question": r["question"], "images": [_img(r["image"])], "external_ids": [f"openimages:{r['image_id']}"] + flickr_ids(r.get("flickr_original_url"))}
    if name == "mmbench_dev":
        return {"id": str(r["index"]), "question": r["question"], "images": [_img(r["image"])], "external_ids": [], "source": r.get("source")}
    if name == "pope_test":
        m = re.search(r"(\d+)$", r["image_source"] or "")
        return {"id": r["question_id"], "question": r["question"], "images": [_img(r["image"])], "external_ids": [f"coco:{int(m.group(1))}"] if m else []}
    raise KeyError(name)


class ZipMedia:
    """Lazily opened remote zip archives for media named by upstream annotations."""

    URLS = {"coco_train2017": "http://images.cocodataset.org/zips/train2017.zip"}
    ETAGS = {"coco_train2017": '"62ff7d7fbcc7e0c0604cbb0f9047ce77-2306"'}

    def __init__(self):
        self.archives = {}

    def read(self, key, member):
        if key not in self.archives:
            f = hub.HttpRangeFile(self.URLS[key])
            if f.etag != self.ETAGS[key]:
                raise hub.UrlError("zip_etag_changed")
            self.archives[key] = zipfile.ZipFile(f)
        try:
            return self.archives[key].read(member)
        except KeyError:
            raise hub.UrlError("zip_member_missing")
        except zipfile.BadZipFile:
            raise hub.UrlError("zip_crc_failed")
