"""Before/after evidence for the SFT BOS fix: run the previous and current prepare_sample on the same rows.

usage: python tools/step05/bos_evidence.py OLD_COMMIT EXPORT_DIR OUT_JSON [ROWS_PER_SOURCE]
"""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from k2hv import sft_rollout
from k2hv.data import runtime

old_commit, export_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
per_source = int(sys.argv[4]) if len(sys.argv) > 4 else 1
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "sft_rollout_old.py"
    path.write_text(subprocess.run(["git", "show", f"{old_commit}:src/k2hv/sft_rollout.py"], check=True, capture_output=True, text=True).stdout)
    spec = importlib.util.spec_from_file_location("sft_rollout_old", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)

processor = runtime.load_processor("hf/K2-Horizon-V-init")
miles = runtime.miles()
tok = processor.tokenizer
index = pq.read_table(export_dir / "index" / "default__sft__train.parquet", columns=["record_id", "source", "rank"]).to_pylist()
rows = pq.read_table(sorted((export_dir / "miles" / "default" / "sft" / "train").glob("*.parquet"))[0]).to_pylist()
by_id = {r["metadata"]["record_id"]: r for r in rows}
picked, seen = [], {}
for r in index:
    if r["record_id"] in by_id and seen.get(r["source"], 0) < per_source:
        seen[r["source"]] = seen.get(r["source"], 0) + 1
        picked.append(by_id[r["record_id"]])


def run(module, data):
    prompt = miles.build_messages(copy.deepcopy(data), "messages", True, {"image": "images"})
    sample = miles.Sample(prompt=prompt, multimodal_inputs=miles.process_vision_info(prompt, processor))
    module.prepare_sample(processor, sample)
    ids = sample.tokens
    start = len(ids) - sample.response_length
    pad = tok.convert_tokens_to_ids("reserved_special_token_12")
    return {
        "first_ids": ids[:4], "first_tokens": tok.convert_ids_to_tokens(ids[:4]), "bos_count": ids.count(tok.bos_token_id),
        "total_tokens": len(ids), "image_tokens": ids.count(pad), "grid": sample.multimodal_train_inputs["image_grid_thw"].tolist(),
        "supervised_ids": ids[start:], "supervised_text": tok.decode(ids[start:]), "loss_mask_sum": sum(sample.loss_mask),
    }


cases = []
for data in picked:
    before, after = run(old, data), run(sft_rollout, data)
    cases.append({"record_id": data["metadata"]["record_id"], "before": before, "after": after,
                  "checks": {"bos_before": before["bos_count"], "bos_after": after["bos_count"],
                             "total_delta": after["total_tokens"] - before["total_tokens"],
                             "same_image_tokens_and_grid": before["image_tokens"] == after["image_tokens"] and before["grid"] == after["grid"],
                             "same_supervised_span": before["supervised_ids"] == after["supervised_ids"],
                             "ids_after_equal_before_without_first": after["first_ids"][:3] == before["first_ids"][1:4]}})
summary = {"old_commit": old_commit, "rows": len(cases), "bos_before": sorted({c["checks"]["bos_before"] for c in cases}), "bos_after": sorted({c["checks"]["bos_after"] for c in cases}),
           "total_delta": sorted({c["checks"]["total_delta"] for c in cases}), "all_same_image_and_supervised": all(c["checks"]["same_image_tokens_and_grid"] and c["checks"]["same_supervised_span"] for c in cases)}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"summary": summary, "cases": cases}, indent=1) + "\n")
print(json.dumps(summary))
