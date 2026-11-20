"""Vision-tower BF16 GEMM backend for SGLang's batch-invariant matmul.

Under the pinned batch-invariant mode every eligible BF16 matmul runs DeepGEMM's sm120 kernel. Inside the full Miles forward that kernel intermittently returned different vision features for identical inputs, at K=1024 sites and at `mlp.down_proj` (K=4096), so restarts from one checkpoint did not repeat. cuBLASLt is repeatable, and where it does not split K it matches DeepGEMM bitwise. For the `mlp.down_proj` shape (K=4096, N=1024) it splits K at M <= 800 rows, which moves the summation order away from the reference.

`K2HV_VISION_GEMM` selects the route for vision-tower calls that meet the pinned DeepGEMM predicate. Routed calls go through the same `deep_gemm.bf16_gemm_nn` binding with the library's deterministic flag cleared, which selects cuBLASLt. `cublaslt-padded` routes every tower site and runs `mlp.down_proj` on at least 1024 rows, zero-padding smaller inputs and slicing the result, so cuBLASLt never splits K there. `cublaslt` routes every tower site without padding. `deepgemm` (the default when unset) leaves the pinned path untouched. Calls outside the tower keep the pinned path. Sites are identified by the tower's Linear module names, and every call is counted per site, route and K.
"""

import json
import os
import sys

ENV = "K2HV_VISION_GEMM"
BACKENDS = ("deepgemm", "cublaslt", "cublaslt-padded")
MIN_ROWS = {"mlp.down_proj": 1024}
SCOPE = "vision tower BF16 GEMMs meeting the pinned DeepGEMM predicate"
_state = {"depth": 0, "patched": False, "site": None, "towers": 0, "routes": {}, "reported": False}


def backend():
    value = os.environ.get(ENV, "deepgemm")
    if value not in BACKENDS:
        raise ValueError(f"{ENV}={value!r}; expected one of {BACKENDS}")
    return value


def route(site, selected=None):
    selected = selected or backend()
    if selected == "cublaslt-padded" and site in MIN_ROWS:
        return f"cublaslt-min{MIN_ROWS[site]}rows"
    return "deepgemm" if selected == "deepgemm" else "cublaslt"


def min_rows(site, selected=None):
    return MIN_ROWS.get(site, 0) if (selected or backend()) == "cublaslt-padded" else 0


def site_name(module_name):
    parts = module_name.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 and parts[-2] in ("attn", "mlp") else module_name


def _eligible(bio, a, b):
    import torch

    K, N = b.shape
    size = a.element_size()
    return (
        bool(bio._ENABLE_MM_DEEPGEMM)
        and bool(bio.ENABLE_JIT_DEEPGEMM)
        and not bool(bio._ENABLE_MM_COMPARISON_TEST)
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and a.is_contiguous()
        and b.transpose(0, 1).is_contiguous()
        and N >= 16
        and (N * size) % 16 == 0
        and (K * size) % 16 == 0
    )


def _cublaslt(deep_gemm, a, b, bias, out_dtype, rows=0):
    import torch

    M = a.shape[0]
    if M < rows:
        padded = a.new_zeros((rows, a.shape[1]))
        padded[:M] = a
        a = padded
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype)
    previous = bool(deep_gemm.get_deterministic_algorithms())
    if previous:
        deep_gemm.use_deterministic_algorithms(False)
    try:
        deep_gemm.bf16_gemm_nn(a, b, out)
    finally:
        if previous:
            deep_gemm.use_deterministic_algorithms(True)
    out = out[:M]
    if bias is not None:
        out += bias
    return out


def _count(site, chosen, K):
    key = f"{site}|{chosen}|K={K}"
    _state["routes"][key] = _state["routes"].get(key, 0) + 1


def _patch_matmul():
    if _state["patched"]:
        return
    import sglang.srt.batch_invariant_ops.batch_invariant_ops as bio

    original = bio.matmul_persistent

    def matmul_persistent(a, b, bias=None, out_dtype=None):
        if _state["depth"]:
            site = _state["site"] or "other"
            if not _eligible(bio, a, b):
                _count(site, "pinned-ineligible", b.shape[0])
            elif (chosen := route(site)) != "deepgemm":
                rows = min_rows(site)
                _count(site, chosen if a.shape[0] < rows else "cublaslt", b.shape[0])
                return _cublaslt(bio.deep_gemm, a, b, bias, out_dtype or a.dtype, rows)
            else:
                _count(site, "deepgemm", b.shape[0])
        return original(a, b, bias=bias, out_dtype=out_dtype)

    bio.matmul_persistent = matmul_persistent
    _state["patched"] = True


def _tag_sites(visual):
    import torch

    def enter(site):
        def hook(module, args):
            _state["site"] = site

        return hook

    def leave(module, args, output):
        _state["site"] = None

    sites = {}
    for name, module in visual.named_modules():
        if isinstance(module, torch.nn.Linear):
            site = site_name(name)
            sites[site] = sites.get(site, 0) + 1
            module.register_forward_pre_hook(enter(site))
            module.register_forward_hook(leave)
    return sites


def _log(event, record):
    print(f"k2hv.vision_gemm {event} {json.dumps(record, sort_keys=True)}", file=sys.stderr, flush=True)


def install(visual):
    """Route the eligible GEMMs of `visual`'s forward according to `K2HV_VISION_GEMM` and log the routing."""
    selected = backend()
    if selected != "deepgemm" and not getattr(visual, "_k2hv_vision_gemm", False):
        _patch_matmul()
        _state["modules"] = _tag_sites(visual)
        forward = visual.forward

        def scoped(*args, **kwargs):
            _state["depth"] += 1
            _state["towers"] += 1
            try:
                return forward(*args, **kwargs)
            finally:
                _state["depth"] -= 1
                _state["site"] = None
                if not _state["depth"] and not _state["reported"]:
                    _state["reported"] = True
                    _log("first_forward", provenance())

        visual.forward = scoped
        visual._k2hv_vision_gemm = True
    record = provenance()
    _log("install", record)
    return record


def provenance():
    selected = backend()
    record = {
        "backend": selected,
        "scope": SCOPE,
        "patched": _state["patched"],
        "routing": {site: route(site, selected) for site in sorted(_state.get("modules", {}))},
        "linear_modules": dict(sorted(_state.get("modules", {}).items())),
        "calls": dict(sorted(_state["routes"].items())),
        "tower_forwards": _state["towers"],
    }
    try:
        import sglang.srt.batch_invariant_ops.batch_invariant_ops as bio

        record["deep_gemm_deterministic_default"] = bool(bio.deep_gemm.get_deterministic_algorithms())
    except Exception as error:
        record["deep_gemm_deterministic_default"] = repr(error)
    return record


def wrap_adapter():
    """Install on the trainer model's vision tower right after `K2HorizonVAdapter.patch_model`."""
    from k2hv.miles_gate_adapter import K2HorizonVAdapter

    if getattr(K2HorizonVAdapter.patch_model, "_k2hv_vision_gemm", False):
        return
    original = K2HorizonVAdapter.patch_model

    def patch_model(self, model, args):
        original(self, model, args)
        install(model.visual)

    patch_model._k2hv_vision_gemm = True
    K2HorizonVAdapter.patch_model = patch_model
