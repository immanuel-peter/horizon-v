import sys
import types

import pytest
import torch

from k2hv import vision_gemm


class FakeDeepGemm:
    def __init__(self):
        self.deterministic = True
        self.calls = []

    def get_deterministic_algorithms(self):
        return self.deterministic

    def use_deterministic_algorithms(self, value):
        self.deterministic = value

    def bf16_gemm_nn(self, a, b, out):
        self.calls.append(self.deterministic)
        out.copy_((a.float() @ b.float()).to(out.dtype))


@pytest.fixture
def bio(monkeypatch):
    module = types.ModuleType("sglang.srt.batch_invariant_ops.batch_invariant_ops")
    module._ENABLE_MM_DEEPGEMM = True
    module.ENABLE_JIT_DEEPGEMM = True
    module._ENABLE_MM_COMPARISON_TEST = False
    module.deep_gemm = FakeDeepGemm()
    module.pinned = []

    def matmul_persistent(a, b, bias=None, out_dtype=None):
        module.pinned.append(tuple(b.shape))
        return a @ b if bias is None else a @ b + bias

    module.matmul_persistent = matmul_persistent
    for name in ("sglang", "sglang.srt", "sglang.srt.batch_invariant_ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(vision_gemm, "_state", {"depth": 0, "patched": False, "site": None, "towers": 0, "routes": {}, "reported": False})
    return module


class Linear(torch.nn.Linear):
    def __init__(self, bio, k, n):
        super().__init__(k, n, bias=False, dtype=torch.bfloat16)
        self.bio = bio

    def forward(self, x):
        return self.bio.matmul_persistent(x, self.weight.t())


class Block(torch.nn.Module):
    def __init__(self, bio, width=64, inner=128):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.qkv = Linear(bio, width, width)
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = Linear(bio, inner, width)
        self.inner = inner

    def forward(self, x):
        x = self.attn.qkv(x)
        return self.mlp.down_proj(torch.cat([x, x], -1)[:, : self.inner])


class Tower(torch.nn.Module):
    def __init__(self, bio, fail=False):
        super().__init__()
        self.bio, self.fail = bio, fail
        self.blocks = torch.nn.ModuleList([Block(bio)])

    def forward(self, x):
        out = self.blocks[0](x)
        if self.fail:
            raise RuntimeError("tower failed")
        return out


def operands(m=32, k=64, n=48, dtype=torch.bfloat16):
    return torch.randn(m, k, dtype=dtype), torch.randn(n, k, dtype=dtype).t()


def test_padded_runs_down_proj_on_at_least_the_minimum_rows(bio, monkeypatch):
    monkeypatch.setenv(vision_gemm.ENV, "cublaslt-padded")
    monkeypatch.setitem(vision_gemm.MIN_ROWS, "mlp.down_proj", 48)
    shapes = []
    gemm = bio.deep_gemm.bf16_gemm_nn
    monkeypatch.setattr(bio.deep_gemm, "bf16_gemm_nn", lambda a, b, out: (shapes.append(tuple(a.shape)), gemm(a, b, out)))
    tower = Tower(bio)
    record = vision_gemm.install(tower)
    assert record["routing"] == {"attn.qkv": "cublaslt", "mlp.down_proj": "cublaslt-min48rows"}
    block = tower.blocks[0]
    for m in (32, 64):
        x = torch.randn(m, 64, dtype=torch.bfloat16)
        qkv = (x.float() @ block.attn.qkv.weight.float().t()).to(torch.bfloat16)
        inner = torch.cat([qkv, qkv], -1)[:, :128]
        out = tower(x)
        assert out.shape == (m, 64)
        assert torch.equal(out, (inner.float() @ block.mlp.down_proj.weight.float().t()).to(torch.bfloat16))
    assert shapes == [(32, 64), (48, 128), (64, 64), (64, 128)]
    assert bio.deep_gemm.calls == [False] * 4 and bio.deep_gemm.deterministic and bio.pinned == []
    assert vision_gemm.provenance()["calls"] == {"attn.qkv|cublaslt|K=64": 2, "mlp.down_proj|cublaslt-min48rows|K=128": 1, "mlp.down_proj|cublaslt|K=128": 1}


def test_padding_keeps_the_bias(bio, monkeypatch):
    monkeypatch.setenv(vision_gemm.ENV, "cublaslt-padded")
    a, b = operands(m=8)
    bias = torch.randn(48, dtype=torch.bfloat16)
    out = vision_gemm._cublaslt(bio.deep_gemm, a, b, bias, torch.bfloat16, rows=32)
    assert out.shape == (8, 48)
    assert torch.equal(out, (a.float() @ b.float()).to(torch.bfloat16) + bias)


def test_cublaslt_routes_every_eligible_site_inside_the_tower(bio, monkeypatch):
    monkeypatch.setenv(vision_gemm.ENV, "cublaslt")
    tower = Tower(bio)
    assert vision_gemm.install(tower)["patched"]
    x = torch.randn(32, 64, dtype=torch.bfloat16)
    out = tower(x)
    assert bio.deep_gemm.calls == [False, False] and bio.pinned == []
    qkv = (x.float() @ tower.blocks[0].attn.qkv.weight.float().t()).to(torch.bfloat16)
    inner = torch.cat([qkv, qkv], -1)[:, :128]
    assert torch.equal(out, (inner.float() @ tower.blocks[0].mlp.down_proj.weight.float().t()).to(torch.bfloat16))
    bio.matmul_persistent(*operands())
    assert bio.pinned == [(64, 48)]
    assert vision_gemm.provenance()["tower_forwards"] == 1


def test_ineligible_calls_keep_the_pinned_path(bio, monkeypatch):
    monkeypatch.setenv(vision_gemm.ENV, "cublaslt")
    tower = Tower(bio).float()
    vision_gemm.install(tower)
    tower(torch.randn(32, 64))
    assert bio.deep_gemm.calls == [] and len(bio.pinned) == 2
    assert set(vision_gemm.provenance()["calls"]) == {"attn.qkv|pinned-ineligible|K=64", "mlp.down_proj|pinned-ineligible|K=128"}


def test_scope_closes_when_the_tower_raises(bio, monkeypatch):
    monkeypatch.setenv(vision_gemm.ENV, "cublaslt-padded")
    tower = Tower(bio, fail=True)
    vision_gemm.install(tower)
    with pytest.raises(RuntimeError):
        tower(torch.randn(32, 64, dtype=torch.bfloat16))
    a, b = operands()
    bio.matmul_persistent(a, b)
    assert bio.pinned == [(64, 48)] and len(bio.deep_gemm.calls) == 2


def test_deepgemm_leaves_the_pinned_path(bio, monkeypatch):
    monkeypatch.delenv(vision_gemm.ENV, raising=False)
    tower = Tower(bio)
    original = bio.matmul_persistent
    assert vision_gemm.install(tower)["backend"] == "deepgemm"
    assert bio.matmul_persistent is original
    tower(torch.randn(32, 64, dtype=torch.bfloat16))
    assert bio.deep_gemm.calls == [] and len(bio.pinned) == 2


def test_site_names():
    assert vision_gemm.site_name("blocks.23.attn.qkv") == "attn.qkv"
    assert vision_gemm.site_name("blocks.0.mlp.down_proj") == "mlp.down_proj"
    assert vision_gemm.site_name("merger.proj") == "merger.proj"


def test_unknown_backend_is_refused(monkeypatch):
    monkeypatch.setenv(vision_gemm.ENV, "triton")
    with pytest.raises(ValueError):
        vision_gemm.backend()
