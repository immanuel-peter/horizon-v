import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def packed_sdpa(module, query, key, value, attention_mask, **kwargs):
    boundaries = kwargs.pop("k2hv_cu_seqlens", None)
    if boundaries is None or not getattr(module, "_k2hv_packed_attention", False):
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    if query.shape[0] != 1 or query.shape[2] != key.shape[2]:
        raise ValueError("K2 packed training requires one uncached concatenated row")
    if boundaries[0] != 0 or boundaries[-1] != query.shape[2]:
        raise ValueError("Miles packing boundaries do not cover the concatenated row")
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start == end:
            continue
        mask = None
        if getattr(module, "sliding_window", None) is not None:
            mask = attention_mask[..., start:end, start:end]
        output, _ = sdpa_attention_forward(module, query[..., start:end, :], key[..., start:end, :], value[..., start:end, :], mask, **kwargs)
        outputs.append(output)
    return torch.cat(outputs, dim=1), None


def install(model):
    ALL_ATTENTION_FUNCTIONS.register("sdpa", packed_sdpa)
    for layer in model.language_model.model.layers:
        layer.self_attn._k2hv_packed_attention = True
