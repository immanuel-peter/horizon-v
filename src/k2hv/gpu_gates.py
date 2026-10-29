"""HF reference component of step 04; assemble all components with k2hv.step04_report."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

GATES = ("cuda", "load_hf", "text_hf", "vision_hf", "packing_controlled")


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def compare(a, b, *, atol=0, rtol=0):
    import torch
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    delta = (a - b).abs()
    metrics = {"shape": list(a.shape), "max_abs": delta.max().item(), "mean_abs": delta.mean().item(), "atol": atol, "rtol": rtol}
    if not torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol):
        raise AssertionError(json.dumps(metrics))
    return metrics


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
    from .gpu_numerics import enable
    enable()
    torch.manual_seed(args.seed)
    report = {"schema": 2, "component": "HF reference and controlled isolation; full completion requires k2hv.step04_report", "command": sys.argv, "seed": args.seed, "starting_sha": args.starting_sha,
              "head_sha": command("git", "rev-parse", "HEAD"),
              "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for root in ("src", "hf/K2-Horizon-V-init", "tools/step04") for p in Path(root).rglob("*.py") if "__pycache__" not in p.parts}, "python": platform.python_version(),
              "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "pillow", "safetensors")},
              "cuda": torch.version.cuda, "hf_backend": "sdpa", "results": []}
    selected = set(args.only or GATES)
    model = processor = None
    cuda_ok = False

    def cuda():
        nonlocal cuda_ok
        assert torch.cuda.is_available(), "CUDA is unavailable; CPU execution cannot satisfy these gates"
        report["device"] = {"name": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()), "arch_list": torch.cuda.get_arch_list()}
        x = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True)
        y.float().square().mean().backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        torch.cuda.synchronize()
        cuda_ok = True
        return {"sdpa_forward_backward": True, "dtype": "bfloat16", "shape": list(x.shape)}

    def load():
        nonlocal model, processor
        processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
        assert processor.image_processor.backend == "pil"
        model = AutoModelForImageTextToText.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable and all(n.startswith("projector.") for n in trainable)
        return {"trainable": trainable, "parameters": sum(p.numel() for p in model.parameters())}

    def text():
        stock = AutoModelForCausalLM.from_pretrained(args.k2, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        results = []
        for prompt in ("The capital of France is Paris. The capital of Japan is", "Explain a sequence of integers. " * 128):
            ids = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
            with torch.no_grad():
                expected = stock(ids, use_cache=False).logits
                actual = model(ids, use_cache=False).logits
            results.append({"length": ids.shape[1], **compare(actual, expected)})
        del stock
        gc.collect()
        torch.cuda.empty_cache()
        return {"prompts": results}

    def vision():
        from PIL import Image
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel
        reference = Glm5NextVisionModel.from_pretrained(args.glm, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        captured = []
        handle = reference.post_layernorm.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
        results = []
        for size in ((448, 448), (560, 112), (112, 560), (1, 1), (1400, 1200)):
            import numpy as np
            rng = np.random.default_rng(args.seed)
            image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
            inputs = processor.image_processor([image], return_tensors="pt")
            pixels = inputs["pixel_values"].to(device="cuda", dtype=torch.bfloat16)
            grid = inputs["image_grid_thw"].cuda()
            with torch.no_grad():
                reference(pixels, grid_thw=grid)
                actual = model.visual(pixels, grid_thw=grid)
            results.append({"size": size, "grid": grid.tolist(), **compare(actual, captured.pop())})
        handle.remove()
        return {"images": results}

    def packing():
        from .gate_training import packing as check
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            return check(model, processor)

    functions = {"cuda": cuda, "load_hf": load, "text_hf": text, "vision_hf": vision, "packing_controlled": packing}
    for name in GATES:
        row = {"gate": name, "status": "blocked"}
        start = time.monotonic()
        try:
            if name not in selected:
                row["reason"] = "not selected"
            elif name not in functions:
                row["reason"] = "runner not implemented; no GPU correctness claim"
            elif name != "cuda" and not cuda_ok:
                row["reason"] = "CUDA prerequisite failed or unrun"
            elif name in ("text_hf", "vision_hf", "packing_controlled") and model is None:
                row["reason"] = "HF load prerequisite failed or unrun"
            else:
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                row["metrics"] = functions[name]()
                row["status"] = "passed"
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        row["seconds"] = time.monotonic() - start
        row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        report["results"].append(row)
        report["complete"] = len(report["results"]) == len(GATES) and all(r["status"] == "passed" for r in report["results"])
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    return 0 if report["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="hf/K2-Horizon-V-init")
    parser.add_argument("--k2", default="models/K2-Horizon-7B")
    parser.add_argument("--glm", default="models/GLM-5.3-Flash-Vision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--starting-sha", default="2b00cf5f32e66741dbe36316c5e785e66ac99ab2")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--only", nargs="+", choices=GATES)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
