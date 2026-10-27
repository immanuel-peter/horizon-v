import numpy as np
import pytest
import torch

from k2hv.hf.modeling_k2_horizon_v import Projector, build_projector


def _window_ids(grid_h: int, grid_w: int, merge: int = 2):
    rows, cols = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    block = (grid_h // merge, merge, grid_w // merge, merge)
    rows = rows.reshape(block).transpose(0, 2, 1, 3).reshape(-1)
    cols = cols.reshape(block).transpose(0, 2, 1, 3).reshape(-1)
    return np.stack([rows, cols], axis=-1)


def test_merge_groups_follow_the_glm_image_processor_patch_order():
    from transformers.models.glm5_next.image_processing_pil_glm5_next import Glm5NextImageProcessorPil

    patch, grid_h, grid_w = 4, 6, 8
    image = np.zeros((3, grid_h * patch, grid_w * patch), dtype=np.float32)
    for r in range(grid_h):
        for c in range(grid_w):
            image[0, r * patch : (r + 1) * patch, c * patch : (c + 1) * patch] = r
            image[1, r * patch : (r + 1) * patch, c * patch : (c + 1) * patch] = c

    patches, gh, gw = Glm5NextImageProcessorPil().patchify(
        image, patch_size=patch, merge_size=2, temporal_patch_size=2
    )
    assert (gh, gw) == (grid_h, grid_w)
    patches = patches.reshape(gh * gw, 3, 2, patch, patch)
    emitted = np.stack([patches[:, 0, 0, 0, 0], patches[:, 1, 0, 0, 0]], axis=-1).astype(int)

    np.testing.assert_array_equal(emitted, _window_ids(grid_h, grid_w))

    feats = torch.from_numpy(emitted).float()
    merged = Projector(2, 8).merge(feats).reshape(-1, 4, 2)
    for window in range(merged.shape[0]):
        r, c = divmod(window, grid_w // 2)
        expected = [[2 * r, 2 * c], [2 * r, 2 * c + 1], [2 * r + 1, 2 * c], [2 * r + 1, 2 * c + 1]]
        assert merged[window].tolist() == expected


def test_merge_matches_the_view_glm_downsample_uses():
    x = torch.randn(12, 7)
    assert torch.equal(Projector(7, 5).merge(x).reshape(-1, 2, 2, 7), x.view(-1, 2, 2, 7))


def test_real_size_shapes_and_parameter_count():
    projector = Projector(1024, 4096)
    shapes = {name: tuple(p.shape) for name, p in projector.named_parameters()}
    assert shapes == {
        "norm.weight": (1024,),
        "norm.bias": (1024,),
        "fc1.weight": (4096, 4096),
        "fc1.bias": (4096,),
        "fc2.weight": (4096, 4096),
        "fc2.bias": (4096,),
    }
    assert sum(p.numel() for p in projector.parameters()) == 33_564_672
    assert projector(torch.randn(8, 1024)).shape == (2, 4096)


@pytest.mark.parametrize("shape", [(6, 1024), (1024,), (2, 4, 1024)])
def test_rejects_inputs_that_cannot_be_merged(shape):
    with pytest.raises(ValueError, match="divisible by 4"):
        Projector(1024, 64)(torch.randn(*shape))


def _flat(projector):
    return torch.cat([p.detach().float().flatten() for p in projector.parameters()])


def test_build_projector_is_deterministic_and_independent_of_global_rng():
    torch.manual_seed(123)
    a = build_projector(16, 32, seed=0)
    torch.manual_seed(999)
    b = build_projector(16, 32, seed=0)
    assert torch.equal(_flat(a), _flat(b))
    assert not torch.equal(_flat(a), _flat(build_projector(16, 32, seed=1)))


def test_build_projector_does_not_consume_global_rng():
    torch.manual_seed(5)
    before = torch.get_rng_state()
    build_projector(16, 32, seed=0)
    assert torch.equal(before, torch.get_rng_state())


def test_build_projector_bf16_is_the_float32_draw_rounded():
    fp32 = build_projector(16, 32, seed=0, dtype=torch.float32)
    bf16 = build_projector(16, 32, seed=0, dtype=torch.bfloat16)
    for (_, p32), (_, p16) in zip(fp32.named_parameters(), bf16.named_parameters(), strict=True):
        assert p16.dtype == torch.bfloat16
        assert torch.equal(p32.to(torch.bfloat16), p16)


def test_init_scheme_matches_torch_defaults():
    projector = build_projector(64, 128, seed=0)
    assert torch.equal(projector.norm.weight, torch.ones(64))
    assert torch.equal(projector.norm.bias, torch.zeros(64))
    for fc in (projector.fc1, projector.fc2):
        bound = fc.in_features**-0.5
        assert fc.weight.abs().max() <= bound and fc.bias.abs().max() <= bound
        assert fc.weight.std().item() == pytest.approx(bound / 3**0.5, rel=0.05)
