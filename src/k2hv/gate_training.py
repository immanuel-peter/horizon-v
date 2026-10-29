import hashlib
import json
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

from .sft_rollout import prepare_sample


def fixtures(processor):

    samples = []
    colors = (("red", (220, 30, 30)), ("green", (30, 190, 30)), ("blue", (30, 30, 220)), ("yellow", (220, 220, 30)))
    for color, rgb in colors:
        for shape in ("circle", "square", "triangle", "cross"):
            for offset in (0, 1, 2, 3):
                image = Image.new("RGB", (112, 112), (240, 240, 240))
                draw = ImageDraw.Draw(image)
                x, y = 4 + (offset % 2) * 52, 4 + (offset // 2) * 52
                if shape == "circle":
                    draw.ellipse((x, y, x + 44, y + 44), fill=rgb)
                elif shape == "square":
                    draw.rectangle((x, y, x + 44, y + 44), fill=rgb)
                elif shape == "triangle":
                    draw.polygon(((x + 22, y), (x, y + 44), (x + 44, y + 44)), fill=rgb)
                else:
                    draw.rectangle((x + 16, y, x + 28, y + 44), fill=rgb)
                    draw.rectangle((x, y + 16, x + 44, y + 28), fill=rgb)
                answer = f"{color} {shape}"
                sample = SimpleNamespace(
                    prompt=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Name the color and shape."}]}, {"role": "assistant", "content": answer, "think": ""}],
                    multimodal_inputs={"images": [image]},
                )
                samples.append(prepare_sample(processor, sample))
    return samples


def model_inputs(model, samples):
    from miles.backends.fsdp_utils.actor import FSDPTrainRayActor
    from miles.backends.fsdp_utils.adaptations.specs import resolve_arch_adapter

    lengths = [len(s.tokens) for s in samples]
    bounds = [0]
    for n in lengths:
        bounds.append(bounds[-1] + n)
    batch = {
        "tokens": torch.tensor([[t for s in samples for t in s.tokens]], device="cuda"),
        "position_ids": torch.tensor([[i for n in lengths for i in range(n)]], device="cuda"),
        "cu_seqlens": torch.tensor(bounds, device="cuda", dtype=torch.int32),
        "cu_seqlens_host": tuple(bounds),
        "max_seqlen": max(lengths),
        "multimodal_train_inputs": {k: torch.cat([s.multimodal_train_inputs[k] for s in samples]).cuda() for k in ("pixel_values", "image_grid_thw")},
    }
    actor = SimpleNamespace(arch_adapter=resolve_arch_adapter(model.config))
    inputs = FSDPTrainRayActor._get_model_inputs_args(actor, batch)
    labels = torch.full_like(batch["tokens"], -100)
    for s, end in zip(samples, bounds[1:]):
        labels[0, end - s.response_length : end] = torch.tensor(s.tokens[-s.response_length :], device="cuda")
    return inputs, labels


def loss(model, samples):
    inputs, labels = model_inputs(model, samples)
    logits = model(**inputs).logits
    return torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum"), (labels[:, 1:] != -100).sum()


def packing(model, processor):
    from .gpu_gates import compare

    samples = fixtures(processor)
    pair = [samples[0], samples[-1]]
    model.zero_grad(set_to_none=True)
    total = sum(s.response_length for s in pair)
    separate = 0.0
    packed_inputs, packed_labels = model_inputs(model, pair)
    for index, sample in enumerate(pair):
        inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in packed_inputs.items()}
        labels = torch.full_like(packed_labels, -100)
        first = sum(len(x.tokens) for x in pair[:index])
        end = first + len(sample.tokens)
        inputs["input_ids"][:, :first] = 0
        inputs["input_ids"][:, end:] = 0
        for key in ("pixel_values", "image_grid_thw"):
            inputs[key] = sample.multimodal_train_inputs[key].cuda()
        labels[:, first:end] = packed_labels[:, first:end]
        logits = model(**inputs).logits
        value = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        separate += value.detach().item() / total
        (value / total).backward()
    gradients = {n: p.grad.detach().clone() for n, p in model.projector.named_parameters()}
    model.zero_grad(set_to_none=True)
    value, count = loss(model, pair)
    (value / count).backward()
    metrics = {
        "separate_loss": separate,
        "packed_loss": (value / count).item(),
        "supervised_tokens": total,
        "comparison": "two packed image documents versus individual documents padded to the same tensor shape and offsets; other document tokens replaced by zero",
        "gradient_tolerance": {"atol": 0.005, "rtol": 0.03},
        "loss_atol": 0.02,
    }
    assert abs(metrics["packed_loss"] - separate) <= 0.02, metrics
    metrics["gradients"] = {n: compare(p.grad, gradients[n], atol=0.005, rtol=0.03) for n, p in model.projector.named_parameters()}
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("projector."))
    return metrics


def frozen_hashes(model):

    return {n: hashlib.sha256(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for n, p in model.named_parameters() if not n.startswith("projector.")}


def overfit(model, processor, *, steps=256, lr=0.001, seed=0):
    samples = fixtures(processor)
    print("Masked span:", processor.tokenizer.decode(samples[0].tokens[-samples[0].response_length :], skip_special_tokens=False), flush=True)
    before = frozen_hashes(model)
    projector_before = {n: p.detach().clone() for n, p in model.projector.named_parameters()}
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=lr, weight_decay=0)
    curve = []
    generator = torch.Generator().manual_seed(seed)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            indices = torch.randperm(64, generator=generator)[:4].tolist()
            total = sum(samples[i].response_length for i in indices)
            value_sum = 0.0
            for i in indices:
                value, count = loss(model, [samples[i]])
                value_sum += value.detach().item()
                (value / total).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.projector.parameters(), 1.0)
            assert torch.isfinite(norm) and norm > 0, f"invalid projector gradient: {norm}"
            optimizer.step()
            curve.append({"step": step, "loss": value_sum / total, "grad_norm": norm.item()})
            if step % 16 == 0:
                print(curve[-1], flush=True)
        model.eval()
        per_example = []
        with torch.no_grad():
            for sample in samples:
                value, count = loss(model, [sample])
                inputs, labels = model_inputs(model, [sample])
                predictions = model(**inputs).logits[:, :-1].argmax(-1)
                mask = labels[:, 1:] != -100
                per_example.append({"loss": (value / count).item(), "teacher_forced_exact": bool((predictions[mask] == labels[:, 1:][mask]).all())})
        result = {
            "criterion": "every example loss < 0.05 and teacher-forced answer token agreement 100%",
            "steps": steps,
            "batch": 4,
            "lr": lr,
            "curve": curve,
            "per_example": per_example,
            "frozen_unchanged": before == frozen_hashes(model),
            "projector_changed": any(not torch.equal(p, projector_before[n]) for n, p in model.projector.named_parameters()),
            "masked_span": processor.tokenizer.decode(samples[0].tokens[-samples[0].response_length :], skip_special_tokens=False),
            "integration": "pinned Miles FSDP actor model-input method; not a distributed FSDP launch",
            "criterion_met": all(x["loss"] < 0.05 and x["teacher_forced_exact"] for x in per_example),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        return result
    finally:
        with torch.no_grad():
            for n, p in model.projector.named_parameters():
                p.copy_(projector_before[n])
        model.eval()


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="HF diagnostic only; does not certify Miles distributed SFT")
    parser.add_argument("--checkpoint", default="hf/K2-Horizon-V-init")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    result = overfit(model, processor)
    result["gate_status"] = "blocked: diagnostic does not execute Miles SFT actor training loop"
    args.report.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
