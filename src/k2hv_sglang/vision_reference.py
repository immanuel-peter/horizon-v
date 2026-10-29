"""Preserve the archived HF vision model's BF16 rounding boundaries."""
import torch
from torch import nn
from torch.nn import functional as F


class VisionRMSNorm(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.weight = native.weight
        self.eps = getattr(native, 'variance_epsilon', getattr(native, 'eps', 1e-6))

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
        normalized = x.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(-1,keepdim=True) + self.eps)
        output = self.weight.to(x.dtype) * normalized.to(x.dtype)
        return (output,x) if residual is not None else output


class VisionMLP(nn.Module):
    def __init__(self,native):
        super().__init__()
        self.gate_up_proj = native.gate_up_proj
        self.down_proj = native.down_proj
        self.swiglu_limit = native.swiglu_limit

    def forward(self,x):
        gate_up,_ = self.gate_up_proj(x)
        gate,up=gate_up.chunk(2,dim=-1)
        gate=gate.clamp(max=self.swiglu_limit)
        up=up.clamp(min=-self.swiglu_limit,max=self.swiglu_limit)
        x=F.silu(gate)*up
        return self.down_proj(x)[0]


def preserve_rounding(tower, config):
    for block in tower.blocks:
        if not hasattr(block,'norm1'):
            continue
        block.norm1=VisionRMSNorm(block.norm1)
        block.norm2=VisionRMSNorm(block.norm2)
        block.attn.q_norm=VisionRMSNorm(block.attn.q_norm)
        block.attn.k_norm=VisionRMSNorm(block.attn.k_norm)
        block.attn.q_norm.eps=config.rms_norm_eps
        block.attn.k_norm.eps=config.rms_norm_eps
        block.mlp=VisionMLP(block.mlp)
    tower.post_layernorm=VisionRMSNorm(tower.post_layernorm)
