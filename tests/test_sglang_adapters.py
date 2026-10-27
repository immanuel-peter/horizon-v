import asyncio
from array import array
import sys
from types import SimpleNamespace

import pytest
import torch

import sglang_double
import tiny
from test_processor import image, processor, tokenizer


@pytest.fixture
def adapters(monkeypatch):
    modules = sglang_double.install(monkeypatch)
    yield modules
    for name in ("k2hv_sglang.processor", "k2hv_sglang.k2_horizon_v"):
        sys.modules.pop(name, None)


def make_processor(adapters, processor, **settings):
    _, module = adapters
    config = SimpleNamespace(image_token_id=processor.image_token_id, image_start_token_id=processor.image_start_token_id,
                             image_end_token_id=processor.image_end_token_id)
    return module.K2HorizonVImageProcessor(config, SimpleNamespace(**settings), processor, "cpu")


def process(adapter, pictures, text, request=None):
    try:
        return asyncio.run(adapter.process_mm_data_async(pictures, text, request or SimpleNamespace()))
    finally:
        adapter.io_executor.shutdown()


@pytest.mark.parametrize("expanded", [False, True])
def test_raw_and_miles_token_ids_match_hf_without_retokenization(adapters, processor, expanded, monkeypatch):
    pictures = [image(112, 112), image(224, 112)]
    prompt = "question" + processor.image_marker + "question" + processor.image_marker + "answer"
    reference = processor(text=prompt, images=pictures, add_special_tokens=False)
    ids = reference["input_ids"][0].tolist() if expanded else processor.tokenizer.encode(prompt, add_special_tokens=False)
    adapter = make_processor(adapters, processor)
    def no_retokenization(*args, **kwargs):
        raise AssertionError("Token IDs were re-tokenized")
    monkeypatch.setattr(adapter._tokenizer, "encode", no_retokenization)
    monkeypatch.setattr(adapter._tokenizer, "decode", no_retokenization)
    result = process(adapter, pictures, ids)
    assert result.input_ids == reference["input_ids"][0].tolist()
    if expanded:
        assert result.input_ids == ids
    assert result.mrope_positions is None
    assert torch.equal(torch.cat([item.feature for item in result.mm_items]), reference["pixel_values"])
    assert torch.equal(torch.cat([item.image_grid_thw for item in result.mm_items]), reference["image_grid_thw"])
    for item, grid in zip(result.mm_items, reference["image_grid_thw"]):
        first, last = item.offsets[0]
        assert last - first + 1 == int(grid.prod()) // 4
        assert result.input_ids[first - 1] == processor.image_start_token_id
        assert result.input_ids[last + 1] == processor.image_end_token_id


def test_text_prompts_are_encoded_once_and_image_free_requests_preserve_ids(adapters, processor):
    adapter = make_processor(adapters, processor)
    result = process(adapter, [], [7, 11, 13])
    assert result.input_ids == [7, 11, 13] and result.mm_items == []
    adapter = make_processor(adapters, processor)
    result = process(adapter, [image(112, 112)], processor.image_marker + "question")
    reference = processor(text=processor.image_marker + "question", images=image(112, 112), add_special_tokens=False)
    assert result.input_ids == reference["input_ids"][0].tolist()


def test_invalid_expansion_budget_is_rejected(adapters, processor):
    adapter = make_processor(adapters, processor)
    ids = [processor.image_start_token_id] + [processor.image_token_id] * 3 + [processor.image_end_token_id]
    with pytest.raises(ValueError, match="expected 1 or 16"):
        process(adapter, [image(112, 112)], ids)


def test_cache_hash_includes_grid_geometry_for_identical_flat_pixels(adapters, processor):
    from PIL import Image

    portraits = [Image.new("RGB", (112, 224)), Image.new("RGB", (224, 112))]
    adapter = make_processor(adapters, processor)
    result = process(adapter, portraits, processor.image_marker * 2)
    first, second = result.mm_items
    assert torch.equal(first.feature, second.feature)
    assert not torch.equal(first.image_grid_thw, second.image_grid_thw)
    assert first.hash != second.hash
    assert not adapter.gpu_image_decode


def test_media_and_processor_configuration_errors(adapters, processor):
    adapter = make_processor(adapters, processor)
    with pytest.raises(ValueError, match="images and text"):
        process(adapter, [], "question", SimpleNamespace(video_data=["unsupported"]))
    with pytest.raises(ValueError, match="Only min_image_tokens"):
        make_processor(adapters, processor, image_config={"patch_size": 7})


def test_truncated_vision_returns_post_layernorm_without_glm_heads(adapters):
    module, _ = adapters
    config = tiny.composite_config()
    tower = module.TruncatedGlm5NextVision(config.vision_config)
    assert not hasattr(tower, "merger") and not hasattr(tower, "downsample")
    inputs = tiny.image_inputs()
    with torch.no_grad():
        expected = tower.post_layernorm(tower.patch_embed(inputs["pixel_values"]) + 1)
        actual = tower(inputs["pixel_values"], inputs["image_grid_thw"])
    assert actual.shape == (16, config.vision_config.hidden_size)
    torch.testing.assert_close(actual, expected)


def test_wrapper_prepares_multimodal_embeddings_with_ordinary_positions(adapters):
    module, _ = adapters
    model = module.K2HorizonVForConditionalGeneration(tiny.composite_config())
    inputs = tiny.image_inputs()
    item = SimpleNamespace(feature=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"])
    batch = SimpleNamespace(contains_mm_inputs=lambda: True, items=[item])
    ids = inputs["input_ids"].flatten()
    positions = torch.arange(len(ids))
    actual = model(ids, positions, batch)
    features = model.get_image_feature([item])
    assert actual.shape == (len(ids), tiny.VOCAB)
    assert model.language_model.last_positions is positions
    torch.testing.assert_close(model.language_model.last_embeds[ids == tiny.IMAGE_PAD], features)
    assert model.get_input_embeddings() is model.language_model.model.embed_tokens


def test_wrapper_preserves_explicit_text_embeddings(adapters):
    module, _ = adapters
    model = module.K2HorizonVForConditionalGeneration(tiny.composite_config())
    ids, positions = torch.tensor([5, 6]), torch.arange(2)
    embeds = torch.randn(2, 64)
    batch = SimpleNamespace(contains_mm_inputs=lambda: False)
    output = model(ids, positions, batch, input_embeds=embeds)
    assert model.language_model.last_embeds is embeds
    assert output.shape == (2, tiny.VOCAB)


def test_weight_loading_streams_language_and_maps_native_vision_fusions(adapters):
    module, _ = adapters
    model = module.K2HorizonVForConditionalGeneration(tiny.composite_config())
    qkv = torch.randn_like(model.visual.blocks[0].attn.qkv_proj.weight)
    gates = model.visual.blocks[0].mlp.gate_up_proj.weight
    gate, up = [torch.randn_like(part) for part in gates.chunk(2, dim=0)]
    norm = torch.randn_like(model.projector.norm.weight)
    entries = [("language_model.model.layers.0.attn.q_proj.weight", torch.ones(2, 2)),
               ("visual.blocks.0.attn.qkv.weight", qkv), ("visual.blocks.0.mlp.gate_proj.weight", gate),
               ("projector.norm.weight", norm), ("visual.blocks.0.mlp.up_proj.weight", up)]
    model.load_weights(iter(entries))
    assert "model.layers.0.attn.q_proj.weight" in model.language_model.loaded
    assert torch.equal(model.visual.blocks[0].attn.qkv_proj.weight, qkv)
    assert torch.equal(gates, torch.cat([gate, up]))
    assert torch.equal(model.projector.norm.weight, norm)


def test_projector_only_updates_preserve_both_frozen_components(adapters):
    module, _ = adapters
    model = module.K2HorizonVForConditionalGeneration(tiny.composite_config())
    before = {name: tensor.clone() for name, tensor in model.state_dict().items() if not name.startswith("projector.")}
    updates = {name: torch.randn_like(tensor) for name, tensor in model.state_dict().items() if name.startswith("projector.")}
    model.load_weights(iter(updates.items()))
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())
    assert all(torch.equal(model.state_dict()[name], value) for name, value in updates.items())
    assert model.language_model.loaded == {}
    assert module.mm_schedule.embedding_cache.entries == {}
    assert module.mm_schedule.embedding_cache.clears == 1


@pytest.mark.parametrize("name", ["unexpected.weight", "visual.downsample.weight", "projector.nonexistent.weight"])
def test_unknown_or_dropped_weight_names_are_rejected(adapters, name):
    module, _ = adapters
    model = module.K2HorizonVForConditionalGeneration(tiny.composite_config())
    with pytest.raises(ValueError):
        model.load_weights(iter([(name, torch.ones(1))]))


def test_multimodal_padding_uses_each_images_own_hash_value(adapters):
    module, _ = adapters
    model = module.K2HorizonVForConditionalGeneration(tiny.composite_config())
    original = array("q", [7, 300, 302, 302, 301, 8, 300, 302, 301])
    items = [SimpleNamespace(offsets=[(2, 3)], pad_value=1000001), SimpleNamespace(offsets=[(7, 7)], pad_value=1000002)]
    padded = model.pad_input_ids(original, SimpleNamespace(mm_items=items))
    assert padded.tolist() == [7, 300, 1000001, 1000001, 301, 8, 300, 1000002, 301]
    assert original.tolist() == [7, 300, 302, 302, 301, 8, 300, 302, 301]


def test_quantization_fails_explicitly(adapters):
    module, _ = adapters
    with pytest.raises(ValueError, match="unquantized"):
        module.K2HorizonVForConditionalGeneration(tiny.composite_config(), quant_config=object())
