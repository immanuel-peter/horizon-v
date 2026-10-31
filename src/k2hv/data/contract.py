"""Exported-row contracts, checked through the pinned Miles media/message path and the real processor."""

import copy
import hashlib
import json
from pathlib import Path

from . import images, runtime


class ContractError(ValueError):
    pass


class Checker:
    def __init__(self, processor_path, limits):
        runtime.limit_threads(1)
        self.miles = runtime.miles()
        self.processor = runtime.load_processor(processor_path)
        self.limits = limits
        tok = self.processor.tokenizer
        self.ids = {name: tok.convert_tokens_to_ids(token) for name, token in (("start", "<image>"), ("pad", "reserved_special_token_12"), ("end", "</image>"))}
        self.bos = tok.bos_token_id
        self.im_end = tok.convert_tokens_to_ids("<|ifm|im_end|>")
        self.closing = tok.encode("</ifm|think>", add_special_tokens=False)
        self.opening = tok.encode("<ifm|think>", add_special_tokens=False)
        ip = self.processor.image_processor
        self.image_config = {k: getattr(ip, k) for k in ("patch_size", "merge_size", "temporal_patch_size", "min_image_tokens", "max_image_tokens")}
        from k2hv.sft_rollout import prepare_sample

        self.prepare_sample = prepare_sample

    def _image_block(self, ids, grids):
        starts = [i for i, t in enumerate(ids) if t == self.ids["start"]]
        ends = [i for i, t in enumerate(ids) if t == self.ids["end"]]
        if len(starts) != len(grids) or len(ends) != len(grids):
            raise ContractError("image_block_count")
        counts = []
        for s, e, grid in zip(starts, ends, grids):
            block = ids[s + 1 : e]
            expected = int(grid[0] * grid[1] * grid[2]) // self.image_config["merge_size"] ** 2
            if e < s or any(t != self.ids["pad"] for t in block) or len(block) != expected:
                raise ContractError("image_block_incomplete")
            counts.append(expected)
        if sum(1 for t in ids if t == self.ids["pad"]) != sum(counts):
            raise ContractError("image_pad_outside_block")
        return counts

    def _check_files(self, paths, expected_shas):
        for path, digest in zip(paths, expected_shas):
            if digest and runtime.file_sha256(path) != digest:
                raise ContractError("image_hash_mismatch")

    def sft(self, data, expected_sha=None, keep_pixels=False):
        """Validate one raw Miles SFT row; return measured lengths or raise ContractError."""
        msgs = data.get("messages")
        if not isinstance(msgs, list) or len(msgs) != 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
            raise ContractError("row_not_single_pair")
        user, answer = msgs[0].get("content"), msgs[1].get("content")
        if not isinstance(user, str) or not isinstance(answer, str):
            raise ContractError("row_content_not_string")
        if msgs[1].get("think") != "":
            raise ContractError("row_think_not_empty")
        paths = list(data.get("images") or [])
        if user.count("<image>") != len(paths) or len(paths) != 1:
            raise ContractError("row_image_marker_count")
        if any(m in answer for m in ("<image>", "<video>", "<audio>")):
            raise ContractError("row_marker_in_answer")
        if not all(Path(p).is_file() for p in paths):
            raise ContractError("image_file_missing")
        self._check_files(paths, [expected_sha])
        prompt = self.miles.build_messages(copy.deepcopy(data), "messages", True, {"image": "images"})
        try:
            mm = self.miles.process_vision_info(prompt, self.processor)
        except Exception as e:
            raise ContractError(f"miles_media_{type(e).__name__}") from e
        sample = self.miles.Sample(prompt=prompt, multimodal_inputs=mm)
        try:
            self.prepare_sample(self.processor, sample)
        except ValueError as e:
            raise ContractError("prepare_sample_" + ("supervised_span" if "supervised" in str(e) else "rejected")) from e
        ids = sample.tokens
        grids = sample.multimodal_train_inputs["image_grid_thw"].tolist()
        pixels = sample.multimodal_train_inputs["pixel_values"]
        if pixels.shape[0] != sum(int(g[0] * g[1] * g[2]) for g in grids):
            raise ContractError("pixel_rows_mismatch")
        image_tokens = self._image_block(ids, grids)
        start = len(ids) - sample.response_length
        if ids[start - len(self.closing) : start] != self.closing or ids[-1] != self.im_end:
            raise ContractError("mask_boundary")
        supervised = ids[start:]
        if any(t in (self.ids["start"], self.ids["pad"], self.ids["end"]) for t in supervised):
            raise ContractError("mask_covers_image")
        opening = [i for i in range(start) if ids[i : i + len(self.opening)] == self.opening]
        if len(opening) != 1 or opening[0] + len(self.opening) > start - len(self.closing):
            raise ContractError("think_block")
        if sample.loss_mask != [1] * sample.response_length:
            raise ContractError("loss_mask")
        expected_tokens, size = images.image_tokens(*images.decode(Path(paths[0]).read_bytes())[0].size, self.image_config)
        if image_tokens != [expected_tokens]:
            raise ContractError("image_tokens_unexpected")
        result = {"total_tokens": len(ids), "image_tokens": sum(image_tokens), "supervised_tokens": sample.response_length, "grid": [list(map(int, g)) for g in grids], "bos_tokens": sum(1 for t in ids[:3] if t == self.bos), "model_input_size": list(size)}
        if result["image_tokens"] > self.limits["max_image_tokens"]:
            raise ContractError("limit_image_tokens")
        if result["total_tokens"] > self.limits["max_total_tokens"]:
            raise ContractError("limit_total_tokens")
        if result["supervised_tokens"] > self.limits["max_supervised_tokens"]:
            raise ContractError("limit_supervised_tokens")
        if keep_pixels:
            result["pixels"] = (pixels.numpy(), grids[0], ids, start)
        return result

    def rl(self, data, expected_sha=None):
        """Validate one raw Miles RL row: single image block, scorer selector, prompt length."""
        problem, paths, meta = data.get("problem"), list(data.get("images") or []), data.get("metadata") or {}
        if not isinstance(problem, str) or problem.count("<image>") != 1 or len(paths) != 1:
            raise ContractError("row_image_marker_count")
        if meta.get("data_source") not in ("math_dapo", "stem__supergpqa", "pending_numeric_tolerance"):
            raise ContractError("row_selector")
        if not isinstance(data.get("answer"), str) or not data["answer"]:
            raise ContractError("row_answer")
        if not all(Path(p).is_file() for p in paths):
            raise ContractError("image_file_missing")
        self._check_files(paths, [expected_sha])
        prompt = self.miles.build_messages(copy.deepcopy(data), "problem", True, {"image": "images"})
        try:
            mm = self.miles.process_vision_info(prompt, self.processor)
        except Exception as e:
            raise ContractError(f"miles_media_{type(e).__name__}") from e
        text = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        out = self.processor(text=text, images=mm["images"], return_tensors="pt")
        ids = out["input_ids"][0].tolist()
        image_tokens = self._image_block(ids, out["image_grid_thw"].tolist())
        result = {"prompt_tokens": len(ids), "image_tokens": sum(image_tokens)}
        if result["image_tokens"] > self.limits["max_image_tokens"]:
            raise ContractError("limit_image_tokens")
        if result["prompt_tokens"] > self.limits["rl_max_prompt_tokens"]:
            raise ContractError("limit_prompt_tokens")
        return result


def row_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
