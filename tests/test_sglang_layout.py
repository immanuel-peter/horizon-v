from array import array

import pytest
import torch

from k2hv_sglang.input_layout import expand_image_tokens, map_vision_weight, split_image_features
from k2hv_sglang.runtime import MODEL_ARCHITECTURE, REASONING_PARSER, registration_environment
from k2hv_sglang.check_runtime import check_source


def expand(ids, grids):
    return expand_image_tokens(ids, torch.tensor(grids).reshape(-1, 3), image_start_id=300, image_token_id=302, image_end_id=301)


@pytest.mark.parametrize("container", [list, tuple, lambda x: array("q", x)])
def test_unexpanded_tokens_preserve_text_ids_and_image_order(container):
    original = container([17, 300, 302, 301, 29, 300, 302, 301, 7])
    ids, offsets = expand(original, [(1, 4, 4), (1, 2, 6)])
    assert ids == [17, 300, 302, 302, 302, 302, 301, 29, 300, 302, 302, 302, 301, 7]
    assert offsets == [(2, 5), (9, 11)]
    assert list(original) == [17, 300, 302, 301, 29, 300, 302, 301, 7]


def test_miles_expanded_ids_are_kept_verbatim():
    original = [10, 300] + [302] * 256 + [301, 11, 12]
    result, offsets = expand(original, [(1, 32, 32)])
    assert result == original
    assert offsets == [(2, 257)]


@pytest.mark.parametrize("ids,grids", [
    ([300, 302, 302, 301], [(1, 4, 4)]),
    ([300, 301], [(1, 4, 4)]),
    ([300, 302], [(1, 4, 4)]),
    ([302], []),
    ([301], []),
    ([300, 302, 301], []),
    ([9], [(1, 4, 4)]),
])
def test_malformed_or_mismatched_markers_are_rejected(ids, grids):
    with pytest.raises(ValueError):
        expand(ids, grids)


def test_text_only_ids_and_empty_image_features():
    assert expand([1, 2, 3], []) == ([1, 2, 3], [])
    assert split_image_features(torch.empty(0, 96), torch.empty(0, 3, dtype=torch.long)) == []


def test_split_keeps_one_grid_and_the_correct_patch_rows_per_image():
    pixels = torch.arange(28 * 96).reshape(28, 96)
    grids = torch.tensor([[1, 4, 4], [1, 2, 6]])
    parts = split_image_features(pixels, grids)
    assert len(parts) == 2
    assert torch.equal(parts[0][0], pixels[:16]) and torch.equal(parts[1][0], pixels[16:])
    assert parts[0][0].data_ptr() == pixels.data_ptr()
    assert torch.equal(parts[1][1], grids[1:])


@pytest.mark.parametrize("grid", [[[1, 3, 4]], [[0, 4, 4]], [[1, -2, 4]], [[1, 4]], [1, 4, 4]])
def test_invalid_image_grids_are_rejected(grid):
    with pytest.raises(ValueError):
        split_image_features(torch.empty(16, 96), torch.tensor(grid))


def test_patch_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="patch counts"):
        split_image_features(torch.empty(15, 96), torch.tensor([[1, 4, 4]]))


@pytest.mark.parametrize("name,mapped,shard", [
    ("visual.blocks.0.attn.qkv.weight", "visual.blocks.0.attn.qkv_proj.weight", None),
    ("visual.blocks.0.attn.qkv.bias", "visual.blocks.0.attn.qkv_proj.bias", None),
    ("visual.blocks.2.mlp.gate_proj.weight", "visual.blocks.2.mlp.gate_up_proj.weight", 0),
    ("visual.blocks.2.mlp.up_proj.bias", "visual.blocks.2.mlp.gate_up_proj.bias", 1),
    ("visual.blocks.2.mlp.down_proj.weight", "visual.blocks.2.mlp.down_proj.weight", None),
    ("visual.patch_embed.proj.weight", "visual.patch_embed.proj.weight", None),
])
def test_native_vision_weight_names(name, mapped, shard):
    assert map_vision_weight(name) == (mapped, shard)


def test_all_external_multimodal_registrations_are_declared():
    assert registration_environment() == {
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "k2hv_sglang",
        "SGLANG_EXTERNAL_MM_MODEL_ARCH": MODEL_ARCHITECTURE,
        "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "k2hv_sglang",
    }
    assert REASONING_PARSER == "k2_horizon"


def test_runtime_source_audit_rejects_missing_or_changed_code(tmp_path, monkeypatch):
    import hashlib
    from k2hv_sglang import check_runtime

    expected = b"pinned source\n"
    monkeypatch.setattr(check_runtime, "SOURCE_SHA256", {"model.py": hashlib.sha256(expected).hexdigest()})
    with pytest.raises(RuntimeError, match="model.py"):
        check_source(tmp_path)
    (tmp_path / "model.py").write_bytes(expected)
    assert check_source(tmp_path) == 1
    (tmp_path / "model.py").write_bytes(b"drifted source\n")
    with pytest.raises(RuntimeError, match="model.py"):
        check_source(tmp_path)
