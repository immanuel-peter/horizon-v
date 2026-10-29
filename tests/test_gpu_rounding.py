import torch
from torch import nn

import tiny
from k2hv.hf.configuration_k2_horizon import K2HorizonConfig
from k2hv.hf.modeling_k2_horizon import K2HorizonRotaryEmbedding, apply_rotary_pos_emb
from k2hv_sglang.text_reference import K2Rotary, K2SiLU
from k2hv_sglang.vision_reference import VisionRMSNorm
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextRMSNorm


def test_native_decoder_rope_preserves_pinned_hf_bf16_rounding():
    config=K2HorizonConfig(**tiny.k2_config_dict())
    config.rope_head_dim=config.head_dim
    native=K2Rotary(config)
    reference=K2HorizonRotaryEmbedding(config)
    generator=torch.Generator().manual_seed(4)
    q=torch.randn(9,4,config.head_dim,generator=generator).bfloat16()
    k=torch.randn(9,2,config.head_dim,generator=generator).bfloat16()
    positions=torch.tensor([0,1,2,7,63,255,256,768,1024])
    cos,sin=reference(q,positions[None])
    expected=apply_rotary_pos_emb(q.transpose(0,1)[None],k.transpose(0,1)[None],cos,sin)
    actual=native(positions,q.flatten(1),k.flatten(1))
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b[0].transpose(0,1).flatten(1),atol=0,rtol=0)


def test_native_vision_norm_matches_source_with_bf16_weight_multiplication():
    reference=Glm5NextRMSNorm(64,eps=1e-5).bfloat16()
    generator=torch.Generator().manual_seed(12)
    reference.weight.data.copy_(torch.randn(64,generator=generator).bfloat16())
    source=nn.Module();source.weight=reference.weight;source.variance_epsilon=1e-5
    native=VisionRMSNorm(source)
    x=torch.randn(24,64,generator=generator).bfloat16()
    residual=torch.randn(24,64,generator=generator).bfloat16()
    output,after=native(x,residual)
    torch.testing.assert_close(after,x+residual,atol=0,rtol=0)
    torch.testing.assert_close(output,reference(x+residual),atol=0,rtol=0)


def test_native_silu_keeps_the_hf_activation_materialization():
    x=torch.linspace(-6,6,2048).bfloat16().reshape(8,256)
    gate,up=x.chunk(2,-1)
    expected=torch.nn.functional.silu(gate)*up
    torch.testing.assert_close(K2SiLU()(x),expected,atol=0,rtol=0)
