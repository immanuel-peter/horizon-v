"""Token-weighted SFT loss of the init or a Miles checkpoint's projector on selected rows.

The frozen weights come from the init checkpoint; ``verify_frozen_master`` separately proves the Miles
checkpoint's frozen FP32 masters equal them. The six projector tensors are read from the checkpoint in
FP32, as the trainer keeps them. Each row runs through ``k2hv.sft_rollout.prepare_sample`` (one BOS,
answer-plus-end mask) and an unpacked HF forward with the step 4 numerics policy. The loss is the
mean negative log-likelihood over every supervised token, matching Miles's per-token SFT loss.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader, load
from transformers import AutoModelForImageTextToText, AutoProcessor

from .gate_provenance import INIT_CHECKPOINT, capture
from .gpu_numerics import enable
from .lazy_data import LazyDataset
from .sft_rollout import prepare_sample


def projector_state(checkpoint):
    model_dir = Path(checkpoint) / "model"
    metadata = FileSystemReader(str(model_dir)).read_metadata().state_dict_metadata
    names = sorted(k for k in metadata if k.startswith("model_state.model.projector."))
    if len(names) != 6:
        raise ValueError(f"Expected six projector tensors in {model_dir}, found {names}")
    tensors = {name.removeprefix("model_state.model."): torch.empty(metadata[name].size, dtype=torch.float32) for name in names}
    load(state_dict={"model_state": {"model": tensors}}, checkpoint_id=str(model_dir))
    return tensors


def digest(tensors):
    h = hashlib.sha256()
    for name in sorted(tensors):
        h.update(name.encode())
        h.update(tensors[name].contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(model, processor, path):
    dataset = LazyDataset(path, processor, prompt_key="messages", multimodal_keys={"image": "images"}, metadata_key="metadata")
    rows = []
    for index in range(len(dataset)):
        sample = prepare_sample(processor, dataset.samples[index])
        ids = torch.tensor([sample.tokens], device="cuda")
        first = ids.shape[1] - sample.response_length
        images = {k: v.cuda() for k, v in sample.multimodal_train_inputs.items()}
        with torch.no_grad():
            logits = model(input_ids=ids, **images, use_cache=False).logits[0, first - 1 : -1].float()
        nll = torch.nn.functional.cross_entropy(logits, ids[0, first:], reduction="sum").item()
        rows.append({"record_id": sample.metadata.get("record_id"), "tokens": ids.shape[1], "supervised_tokens": sample.response_length, "nll_sum": nll})
    supervised = sum(r["supervised_tokens"] for r in rows)
    return {
        "rows": len(rows),
        "supervised_tokens": supervised,
        "token_weighted_loss": sum(r["nll_sum"] for r in rows) / supervised,
        "sample_mean_loss": sum(r["nll_sum"] / r["supervised_tokens"] for r in rows) / len(rows),
        "per_row": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, help="Miles iter_<n> directory; omit for the init projector")
    p.add_argument("--rows", nargs="+", required=True, help="name=parquet pairs")
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    enable()
    processor = AutoProcessor.from_pretrained(str(INIT_CHECKPOINT), trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(INIT_CHECKPOINT), trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    model.projector.float()
    if a.checkpoint:
        with torch.no_grad():
            for name, tensor in projector_state(a.checkpoint).items():
                model.get_parameter(name).copy_(tensor)
    projector = {f"projector.{n}": p.detach().cpu() for n, p in model.projector.named_parameters()}
    report = {
        "provenance": capture(),
        "checkpoint": str(a.checkpoint) if a.checkpoint else "init",
        "projector_sha256": digest(projector),
        "projector_norms": {n: float(t.norm()) for n, t in projector.items()},
        "splits": {},
    }
    for pair in a.rows:
        name, path = pair.split("=", 1)
        report["splits"][name] = evaluate(model, processor, path)
        print(name, json.dumps({k: v for k, v in report["splits"][name].items() if k != "per_row"}), flush=True)
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
