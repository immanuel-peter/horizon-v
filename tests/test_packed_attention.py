from types import SimpleNamespace

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from k2hv.packed_attention import packed_sdpa


def test_packed_attention_keeps_document_gradients_isolated():
    torch.manual_seed(8)
    module = SimpleNamespace(_k2hv_packed_attention=True, num_key_value_groups=1, is_causal=True, sliding_window=None)
    query = torch.randn(1, 2, 7, 8, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    packed, _ = packed_sdpa(module, query, key, value, None, k2hv_cu_seqlens=(0, 4, 7))
    expected = torch.cat([sdpa_attention_forward(module, query[..., a:b, :], key[..., a:b, :], value[..., a:b, :], None)[0] for a, b in ((0, 4), (4, 7))], dim=1)
    torch.testing.assert_close(packed, expected, atol=0, rtol=0)
    packed[:, 4:].sum().backward()
    for tensor in (query, key, value):
        assert torch.count_nonzero(tensor.grad[..., :4, :]) == 0
        assert tensor.grad[..., 4:, :].abs().sum() > 0


def test_packed_attention_rejects_incomplete_boundaries():
    import pytest

    module = SimpleNamespace(_k2hv_packed_attention=True)
    x = torch.randn(1, 2, 7, 8)
    with pytest.raises(ValueError, match="cover"):
        packed_sdpa(module, x, x, x, None, k2hv_cu_seqlens=(0, 4))
