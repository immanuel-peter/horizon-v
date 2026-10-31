"""Pinned Miles, qwen-vl-utils and RL360 code used to validate exported rows.

Importing ``miles.utils.data`` pulls in SGLang through the rollout and chat-template
packages, so the data stage loads only the functions it needs from the pinned
checkout and verifies their source hashes. The executed code is the pinned code.
"""

import ast
import hashlib
import importlib.util
import logging
import os
import re
import subprocess
import sys
import types
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

MILES_COMMIT = "5078ff7cd6387f553db010bb38148a59e9e12cb0"
MILES_SOURCES = {
    "miles/utils/data.py": "38e252720b113f5c0458e8ff087624a53ace1c8ec62f1e78cabfa215da2a2ee9",
    "miles/utils/processing_utils.py": "787876a2f5b4ec2b3a80aaba868ca88c6ea5278e7a60f06b86afd223a7475c5f",
    "miles/utils/types.py": "0a60609a1a3407da1161fe5341d9aa3661d0533c47088bffd84673246ce28cd2",
}
RL360_COMMIT = "3e13010e0e47b7d08f19dfb6bdd3ac6d21dc200e"
RL360_SOURCES = {
    "src/miles360/reward/math_dapo.py": "fa80849a7942e781c09f123e4e0670f63d45b80365231df1022914f844c8a3ee",
    "src/miles360/reward/supergpqa.py": "013fa894df95d0bbbae935d3c15641f9a0b866f0ee3b90a02857956f48fbcd7c",
}
QWEN_VL_UTILS = "0.0.14"


def limit_threads(n=1):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(n)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass


def _checkout(env, default, commit, sources):
    root = Path(os.environ.get(env, REPO / default)).resolve()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != commit:
        raise RuntimeError(f"{root} is at {head or 'no checkout'}, expected {commit}; clone it at that commit or set {env}")
    for rel, digest in sources.items():
        actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"{root / rel} has sha256 {actual}, expected {digest}")
    return root


def _function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module([node], []), str(path), "exec"), namespace)
    return namespace[name], ast.get_source_segment(path.read_text(), node)


@cache
def miles():
    root = _checkout("K2HV_MILES_ROOT", "third_party/miles", MILES_COMMIT, MILES_SOURCES)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import transformers  # noqa: F401  (resolve torchvision availability before any stub exists)
    from miles.utils.processing_utils import process_vision_info
    from miles.utils.types import MultimodalTypes, Sample

    _import_qwen_vl_utils()
    ns = {"re": re, "logger": logging.getLogger("miles.utils.data"), "MultimodalTypes": MultimodalTypes, "os": os, "itertools": __import__("itertools"), "json": __import__("json")}
    import pyarrow.parquet as pq

    ns["pq"] = pq
    data_py = root / "miles/utils/data.py"
    build_messages, _ = _function(data_py, "_build_messages", ns)
    _function(data_py, "_parse_generalized_path", ns)
    read_file, _ = _function(data_py, "read_file", ns)
    _, load_source = _function(root / "miles/utils/processing_utils.py", "load_processor", {})
    return types.SimpleNamespace(root=root, build_messages=build_messages, read_file=read_file, process_vision_info=process_vision_info, Sample=Sample, load_processor_source=load_source)


def _import_qwen_vl_utils():
    from importlib.metadata import version

    if version("qwen-vl-utils") != QWEN_VL_UTILS:
        raise RuntimeError(f"qwen-vl-utils {version('qwen-vl-utils')} is installed, expected {QWEN_VL_UTILS}")
    if "qwen_vl_utils" in sys.modules or importlib.util.find_spec("torchvision") is not None:
        import qwen_vl_utils  # noqa: F401

        return
    # qwen_vl_utils imports torchvision at module level for its video readers only.
    # The stub exists only during that import, so transformers still sees no torchvision.
    def missing(*args, **kwargs):
        raise RuntimeError("torchvision is not installed; video inputs are unsupported")

    tv = types.ModuleType("torchvision")
    tv.__version__ = "0+absent"
    tv.io = types.ModuleType("torchvision.io")
    tv.io.read_video = missing
    tv.transforms = types.ModuleType("torchvision.transforms")
    tv.transforms.InterpolationMode = types.SimpleNamespace(BICUBIC=None)
    tv.transforms.functional = types.SimpleNamespace(resize=missing)
    stub = {"torchvision": tv, "torchvision.io": tv.io, "torchvision.transforms": tv.transforms}
    sys.modules.update(stub)
    try:
        import qwen_vl_utils  # noqa: F401
    finally:
        for key in stub:
            sys.modules.pop(key, None)


def load_processor(path):
    """Equivalent of Miles ``load_processor`` for this non-Inkling checkpoint."""
    from transformers import AutoProcessor, PreTrainedTokenizerBase, ProcessorMixin

    proc = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        raise RuntimeError("Expected a multimodal processor")
    return proc


@cache
def rl360():
    root = _checkout("K2HV_RL360_ROOT", "third_party/rl360", RL360_COMMIT, RL360_SOURCES)
    package = types.ModuleType("k2hv_rl360_reward")
    package.__path__ = []
    utils = types.ModuleType("k2hv_rl360_reward.utils")
    # The scorers' process-based timeout wrapper belongs to step 8; admission tests call the scoring logic directly.
    utils.timeout_limit = lambda seconds, use_signals=False: (lambda func: func)
    sys.modules.update({package.__name__: package, utils.__name__: utils})
    modules = {}
    for name in ("math_dapo", "supergpqa"):
        path = root / f"src/miles360/reward/{name}.py"
        module = types.ModuleType(f"{package.__name__}.{name}")
        module.__package__ = package.__name__
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
        modules[name] = module
    return types.SimpleNamespace(root=root, **modules)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def processor_fingerprint(path):
    path = Path(path)
    names = ["chat_template.jinja", "processor_config.json", "preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "processing_k2_horizon_v.py"]
    return {name: file_sha256(path / name) for name in names}
