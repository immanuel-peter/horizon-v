"""Opt-in tensor capture for local GPU gates; disabled in ordinary serving."""
import itertools
import os
from pathlib import Path

import torch

_SEQUENCE = itertools.count()


def emit(kind, **values):
    directory = os.environ.get("K2HV_GATE_TRACE")
    if not directory:
        return
    def copy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, (list, tuple)):
            return [copy(x) for x in value]
        if isinstance(value, dict):
            return {k: copy(v) for k, v in value.items()}
        return value
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{os.getpid()}-{next(_SEQUENCE):06d}-{kind}.pt"
    torch.save({"kind": kind, **{k: copy(v) for k, v in values.items()}}, path)


def factory(config):
    def hook(module, inputs, output):
        emit(config["name"], inputs=inputs[:1], output=output if isinstance(output, (torch.Tensor, tuple, list)) else None, weight=getattr(module,"weight",None), eps=getattr(module,"eps",None))
    return hook


def weight_state(model,names):
    if not os.environ.get('K2HV_GATE_WEIGHT_STATE'):
        return
    import hashlib
    digest=hashlib.sha256()
    for name,param in sorted(model.named_parameters()):
        if name.startswith('projector.'):
            continue
        digest.update(name.encode())
        digest.update(param.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    emit('weights',names=names,frozen_sha256=digest.hexdigest(),projector={n:p for n,p in model.projector.named_parameters()})
