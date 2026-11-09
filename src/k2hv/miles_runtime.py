"""Pinned Miles code loading for the consumer side of this repo.

Importing ``miles.utils.data`` pulls in SGLang through the rollout and chat-template
packages, so only the functions training and its tests need are loaded from the pinned
checkout, after verifying the checkout commit and the source hashes. The executed code
is the pinned code. The dataset builder keeps its own fuller copy of this machinery.
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

REPO = Path(__file__).resolve().parents[2]

MILES_COMMIT = "5078ff7cd6387f553db010bb38148a59e9e12cb0"
MILES_SOURCES = {
    "miles/utils/data.py": "38e252720b113f5c0458e8ff087624a53ace1c8ec62f1e78cabfa215da2a2ee9",
    "miles/utils/processing_utils.py": "787876a2f5b4ec2b3a80aaba868ca88c6ea5278e7a60f06b86afd223a7475c5f",
    "miles/utils/types.py": "0a60609a1a3407da1161fe5341d9aa3661d0533c47088bffd84673246ce28cd2",
    "miles/rollout/data_source.py": "ab91c7c94088fb099a3f844a5e22016ce86d8db801535bf4e8e7ea709eb57317",
}
QWEN_VL_UTILS = "0.0.14"


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


def _class(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module([node], []), str(path), "exec"), namespace)
    return namespace[name]


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


@cache
def miles_loading():
    """Pinned eager ``Dataset`` and ``RolloutDataSource`` classes, executed without the SGLang imports of their modules.

    Used on CPU to compare ``k2hv.lazy_data`` with the code it replaces. ``Dataset`` runs here without a
    chat template or length filter, the configuration SFT training uses.
    """
    m = miles()
    ns = {"random": __import__("random"), "json": __import__("json"), "np": __import__("numpy"), "logger": logging.getLogger("miles.utils.data"),
          "read_file": m.read_file, "_build_messages": m.build_messages, "Sample": m.Sample, "filter_long_prompt": lambda samples, *a: samples, "chat_template_utils": None}
    dataset = _class(m.root / "miles/utils/data.py", "Dataset", ns)
    from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint

    source_path = m.root / "miles/rollout/data_source.py"
    ns = {"abc": __import__("abc"), "copy": __import__("copy"), "logging": logging, "logger": logging.getLogger("miles.rollout.data_source"), "Path": Path,
          "Dataset": dataset, "Sample": m.Sample, "load_simple_checkpoint": load_simple_checkpoint, "save_simple_checkpoint": save_simple_checkpoint}
    _class(source_path, "DataSource", ns)
    rollout = _class(source_path, "RolloutDataSource", ns)
    return types.SimpleNamespace(Dataset=dataset, RolloutDataSource=rollout)


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


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
