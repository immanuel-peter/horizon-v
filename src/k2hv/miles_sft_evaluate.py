"""Evaluate the real Miles FSDP checkpoint on its captured 64 training examples."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import load as load_checkpoint
from transformers import AutoModelForImageTextToText, AutoProcessor

from .gate_provenance import capture as provenance
from .gate_training import frozen_hashes
from .gpu_numerics import enable


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--fixture", type=Path, default=Path("outputs/step04/miles-sft/fixture"))
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    enable()
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained("hf/K2-Horizon-V-init", trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained("hf/K2-Horizon-V-init", trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    before = frozen_hashes(model)
    initial = {n: p.detach().cpu().clone() for n, p in model.projector.named_parameters()}
    state = {"model_state": {"model": model.state_dict()}}
    load_checkpoint(state_dict=state, checkpoint_id=str(a.checkpoint / "model"))
    model.load_state_dict(state["model_state"]["model"])
    unchanged = before == frozen_hashes(model)
    assert unchanged, "Frozen model or vision weights changed in the real Miles checkpoint"
    files = sorted(a.fixture.glob("*.pt"))
    assert len(files) == 64, len(files)
    end = processor.tokenizer.convert_tokens_to_ids("<|ifm|im_end|>")
    close = processor.tokenizer.encode("</ifm|think>", add_special_tokens=False)
    rows = []
    with torch.no_grad():
        for path in files:
            sample = torch.load(path, weights_only=False, map_location="cpu")
            ids = torch.tensor([sample["tokens"]], device="cuda")
            response = sample["response_length"]
            first = ids.shape[1] - response
            assert ids[0, first - len(close) : first].tolist() == close
            assert ids[0, -1].item() == end and sum(sample["loss_mask"]) == response
            images = {k: sample[k].cuda() for k in ("pixel_values", "image_grid_thw")}
            logits = model(input_ids=ids, **images, use_cache=False).logits[0, first - 1 : -1].float()
            labels = ids[0, first:]
            nll = torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item() / response
            predicted = logits.argmax(-1)
            generated = model.generate(
                input_ids=ids[:, :first], attention_mask=torch.ones_like(ids[:, :first]), **images, max_new_tokens=8, do_sample=False, eos_token_id=end, pad_token_id=0, use_cache=True
            )[0, first:].tolist()
            row = {
                "fixture_sha256": path.stem,
                "supervised_span": processor.tokenizer.decode(labels.tolist(), skip_special_tokens=False),
                "loss": nll,
                "teacher_forced_exact": bool(torch.equal(predicted, labels)),
                "free_generation": processor.tokenizer.decode(generated, skip_special_tokens=False),
                "free_generation_exact": generated == labels.tolist(),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    changed = any(not torch.equal(p.detach().cpu(), initial[n]) for n, p in model.projector.named_parameters())

    report = {
        "provenance": provenance(),
        "checkpoint": str(a.checkpoint),
        "criterion": "all 64 per-example losses < 0.05, teacher-forced and free-generation answers exact",
        "examples": rows,
        "frozen_tensors_checked": len(before),
        "frozen_unchanged": unchanged,
        "projector_changed": changed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "passed": unchanged and changed and all(r["loss"] < 0.05 and r["teacher_forced_exact"] and r["free_generation_exact"] for r in rows),
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    save_file({f"projector.{n}": p.detach().cpu().contiguous() for n, p in model.projector.named_parameters()}, str(a.report.with_suffix(".projector.safetensors")))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
