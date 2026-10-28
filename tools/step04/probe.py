"""Run real BF16 CUDA matmul and SDPA backward before model installation."""
import importlib.metadata
import json
import platform

import torch

result = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
          "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
          "arch_list": torch.cuda.get_arch_list(), "packages": {}}
for package in ("transformers", "sglang", "flashinfer-python", "flash-attn", "flash-attn-3", "flash-attn-4", "triton"):
    try:
        result["packages"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        result["packages"][package] = None
x = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True)
y.float().square().mean().backward()
assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
assert torch.isfinite(x.detach() @ x.detach().transpose(-1, -2)).all()
torch.cuda.synchronize()
result["bf16_matmul_sdpa_backward"] = "passed"
result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
print(json.dumps(result, indent=2))
