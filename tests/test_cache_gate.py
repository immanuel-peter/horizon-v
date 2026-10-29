from types import SimpleNamespace
from pathlib import Path
import os

from k2hv_sglang import cache_gate


def test_weight_state_uses_latest_capture_despite_unordered_directory(monkeypatch,tmp_path):
    first=tmp_path/'120-000000-weights.pt';last=tmp_path/'120-000038-weights.pt'
    first.touch();last.touch()
    os.utime(first,ns=(100,100));os.utime(last,ns=(200,200))
    monkeypatch.setattr(cache_gate,'traces',lambda root:{last:{'kind':'weights','projector':'restored'},first:{'kind':'weights','projector':'updated'}})
    assert cache_gate.state(SimpleNamespace(trace=tmp_path))['projector']=='restored'
