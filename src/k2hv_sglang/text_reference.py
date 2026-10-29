import torch
from torch import nn
from torch.nn import functional as F


class K2Rotary(nn.Module):
    def __init__(self, config):
        super().__init__()
        parameters = config.rope_parameters
        if parameters.get("rope_type") != "default" or config.rope_head_dim != config.head_dim:
            raise ValueError("The validated K2 gate path requires default full-head RoPE")
        dim = config.head_dim
        self.dim = dim
        inv_freq = 1.0 / (parameters["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions, q, k):
        freqs = positions.float()[:, None] * self.inv_freq.to(positions.device)[None]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().to(q.dtype)[:, None], emb.sin().to(q.dtype)[:, None]

        def rotate(x):
            shape = x.shape
            x = x.reshape(-1, x.shape[-1] // self.dim, self.dim)
            a, b = x.chunk(2, dim=-1)
            return (x * cos + torch.cat([-b, a], dim=-1) * sin).reshape(shape)

        return rotate(q), rotate(k)


class K2SiLU(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


def preserve_rounding(language_model, config):
    for layer in language_model.model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.rotary_emb = K2Rotary(config)
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "act_fn"):
            layer.mlp.act_fn = K2SiLU()
