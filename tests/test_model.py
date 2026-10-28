import pytest
import torch
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

import tiny
from k2hv.hf.configuration_k2_horizon import K2HorizonConfig
from k2hv.hf.modeling_k2_horizon import K2HorizonForCausalLM


@pytest.fixture(scope="module")
def model():
    return tiny.tiny_model()


def test_state_dict_layout_matches_the_planned_keys(model):
    keys = list(model.state_dict())
    assert {k.split(".")[0] for k in keys} == {"language_model", "visual", "projector"}
    assert [k for k in keys if k.startswith("projector.")] == [
        "projector.norm.weight",
        "projector.norm.bias",
        "projector.fc1.weight",
        "projector.fc1.bias",
        "projector.fc2.weight",
        "projector.fc2.bias",
    ]
    assert not any("downsample" in k or "merger" in k for k in keys)

    stock_lm = K2HorizonForCausalLM(K2HorizonConfig(**tiny.k2_config_dict()))
    assert {k for k in keys if k.startswith("language_model.")} == {"language_model." + k for k in stock_lm.state_dict()}


def test_visual_keys_are_glm_keys_minus_downsample_and_merger(model):
    native = Glm5NextVisionModel(Glm5NextVisionConfig(**tiny.glm_vision_config_dict()))
    kept = {k for k in native.state_dict() if not k.startswith(("downsample.", "merger."))}
    assert {k[len("visual.") :] for k in model.state_dict() if k.startswith("visual.")} == kept


def test_only_the_projector_is_trainable(model):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {n for n, _ in model.named_parameters() if n.startswith("projector.")}
    assert len(trainable) == 6


def test_tower_output_equals_native_vision_model_at_post_layernorm(model):
    native = Glm5NextVisionModel(Glm5NextVisionConfig(**tiny.glm_vision_config_dict())).eval()
    kept = {k[len("visual.") :]: v for k, v in model.state_dict().items() if k.startswith("visual.")}
    missing, unexpected = native.load_state_dict(kept, strict=False)
    assert not unexpected and {k.split(".")[0] for k in missing} == {"downsample", "merger"}

    captured = {}
    native.post_layernorm.register_forward_hook(lambda m, i, o: captured.setdefault("x", o.detach()))
    inputs = tiny.image_inputs(grids=((1, 4, 4), (1, 2, 6)))
    with torch.no_grad():
        native(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
        ours = model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    assert ours.shape == (28, 32)
    torch.testing.assert_close(ours, captured["x"], rtol=0, atol=0)


def test_text_only_is_exactly_ks_original_path(model):
    ids = torch.randint(0, 290, (2, 9))
    mask = torch.ones_like(ids)
    stock = K2HorizonForCausalLM(K2HorizonConfig(**tiny.k2_config_dict())).eval()
    stock.load_state_dict({k[len("language_model.") :]: v for k, v in model.state_dict().items() if k.startswith("language_model.")})
    with torch.no_grad():
        ours = model(input_ids=ids, attention_mask=mask).logits
        reference = stock(input_ids=ids, attention_mask=mask).logits
    assert torch.equal(ours, reference)


def test_image_features_replace_embeddings_at_pad_positions_only(model):
    inputs = tiny.image_inputs()
    seen = {}
    handle = model.language_model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.update(embeds=kwargs["inputs_embeds"].detach()), with_kwargs=True
    )
    try:
        with torch.no_grad():
            model(**inputs)
            features = model.projector(model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]))
            token_embeds = model.get_input_embeddings()(inputs["input_ids"])
    finally:
        handle.remove()

    is_pad = inputs["input_ids"] == tiny.IMAGE_PAD
    assert is_pad.sum() == 4
    torch.testing.assert_close(seen["embeds"][is_pad], features, rtol=0, atol=0)
    assert torch.equal(seen["embeds"][~is_pad], token_embeds[~is_pad])


def test_two_images_in_one_sequence_match_encoding_each_alone(model):
    both = tiny.image_inputs(grids=((1, 4, 4), (1, 2, 6)))
    first = tiny.image_inputs(grids=((1, 4, 4),))
    second = tiny.image_inputs(grids=((1, 2, 6),))
    n_first = first["pixel_values"].shape[0]
    with torch.no_grad():
        together = model.get_image_features(both["pixel_values"], both["image_grid_thw"])
        alone_1 = model.get_image_features(both["pixel_values"][:n_first], first["image_grid_thw"])
        alone_2 = model.get_image_features(both["pixel_values"][n_first:], second["image_grid_thw"])
    assert together.shape == (7, 64)
    torch.testing.assert_close(together, torch.cat([alone_1, alone_2]), rtol=1e-5, atol=1e-5)
    assert model(**both).logits.shape[:2] == both["input_ids"].shape


def _drop_pad(inputs, n=1):
    ids = inputs["input_ids"]
    pad_positions = (ids[0] == tiny.IMAGE_PAD).nonzero().flatten()[:n]
    keep = torch.ones(ids.shape[1], dtype=torch.bool)
    keep[pad_positions] = False
    return {**inputs, "input_ids": ids[:, keep], "attention_mask": inputs["attention_mask"][:, keep]}


def test_fewer_placeholders_than_features_raises(model):
    with pytest.raises(ValueError, match=r"3 tokens .* 4 features"):
        model(**_drop_pad(tiny.image_inputs()))


def test_more_placeholders_than_features_raises(model):
    inputs = tiny.image_inputs()
    ids = torch.cat([inputs["input_ids"], torch.full((1, 2), tiny.IMAGE_PAD)], dim=1)
    inputs = {**inputs, "input_ids": ids, "attention_mask": torch.ones_like(ids)}
    with pytest.raises(ValueError, match=r"6 tokens .* 4 features"):
        model(**inputs)


def test_images_without_any_placeholder_raise(model):
    inputs = tiny.image_inputs()
    ids = inputs["input_ids"][:, :2]
    with pytest.raises(ValueError, match=r"0 tokens .* 4 features"):
        model(**{**inputs, "input_ids": ids, "attention_mask": torch.ones_like(ids)})


def test_placeholders_without_pixel_values_raise(model):
    inputs = tiny.image_inputs()
    with pytest.raises(ValueError, match="image placeholder tokens .* no pixel_values"):
        model(input_ids=inputs["input_ids"])


def test_bad_grids_are_rejected(model):
    inputs = tiny.image_inputs()
    pixels = inputs["pixel_values"]
    with pytest.raises(ValueError, match="patches but image_grid_thw describes"):
        model.get_image_features(pixels, torch.tensor([[1, 4, 6]]))
    with pytest.raises(ValueError, match="divisible by 2"):
        model.get_image_features(pixels[:12], torch.tensor([[1, 3, 4]]))
    with pytest.raises(ValueError, match=r"\(num_images, 3\)"):
        model.get_image_features(pixels, torch.tensor([1, 4, 4]))
    with pytest.raises(ValueError, match="requires image_grid_thw"):
        model(input_ids=inputs["input_ids"], pixel_values=pixels)
    with pytest.raises(ValueError, match="requires input_ids"):
        model(inputs_embeds=torch.zeros(1, 4, 64), pixel_values=pixels, image_grid_thw=inputs["image_grid_thw"])


def test_text_forward_ignores_image_grid_without_pixel_values(model):
    inputs = tiny.image_inputs()
    text = inputs["input_ids"][:, :2]
    with torch.no_grad():
        expected = model(input_ids=text).logits
        actual = model(input_ids=text, image_grid_thw=inputs["image_grid_thw"]).logits
    assert torch.equal(actual, expected)


def _labels(inputs):
    labels = inputs["input_ids"].clone()
    labels[labels == tiny.IMAGE_PAD] = -100
    return labels


def _projector_grads(model, **flags):
    model.zero_grad(set_to_none=True)
    inputs = tiny.image_inputs()
    out = model(**inputs, labels=_labels(inputs))
    out.loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}, out.loss.item()


def test_gradients_reach_the_projector_through_the_frozen_language_model():
    model = tiny.tiny_model().train()
    seen = {}
    model.projector.register_forward_pre_hook(lambda m, args: seen.update(vit_out_requires_grad=args[0].requires_grad))
    grads, loss = _projector_grads(model)

    assert seen["vit_out_requires_grad"] is False
    assert set(grads) == {n for n, _ in model.named_parameters() if n.startswith("projector.")}
    for name, grad in grads.items():
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("projector."))
    assert loss > 0


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_gradient_checkpointing_gives_the_same_projector_gradients(use_reentrant):
    model = tiny.tiny_model().train()
    reference, _ = _projector_grads(model)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    assert model.language_model.model.gradient_checkpointing
    model.visual.train()
    assert not model.visual.training
    checkpointed, _ = _projector_grads(model)

    assert checkpointed.keys() == reference.keys()
    for name in reference:
        torch.testing.assert_close(checkpointed[name], reference[name], rtol=1e-4, atol=1e-6)


def test_frozen_modules_stay_frozen_after_train_and_eval_toggles(model):
    model.train()
    model.eval()
    assert [n for n, p in model.named_parameters() if p.requires_grad and not n.startswith("projector.")] == []


def test_generate_with_an_image_uses_pixels_only_for_prefill(model):
    inputs = tiny.image_inputs()
    out = model.generate(**inputs, max_new_tokens=3, do_sample=False, pad_token_id=1)
    assert out.shape == (1, inputs["input_ids"].shape[1] + 3)

    text = model.generate(input_ids=inputs["input_ids"][:, :3], max_new_tokens=2, do_sample=False, pad_token_id=1)
    assert text.shape == (1, 5)


def test_from_pretrained_keeps_frozen_modules_frozen(tmp_path):
    from k2hv.hf.modeling_k2_horizon_v import K2HorizonVForConditionalGeneration

    source = tiny.tiny_model()
    source.save_pretrained(tmp_path)
    expected = {n for n, _ in source.named_parameters() if n.startswith("projector.")}
    for kwargs in ({}, {"output_loading_info": True}):
        loaded = K2HorizonVForConditionalGeneration.from_pretrained(tmp_path, **kwargs)
        loaded, info = loaded if kwargs else (loaded, None)
        assert {n for n, p in loaded.named_parameters() if p.requires_grad} == expected
        assert info is None or not any(info.values())
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name]), name


def test_generate_asks_the_language_model_for_only_the_last_logits(model):
    assert model._supports_logits_to_keep()
    seen = []
    handle = model.language_model.register_forward_pre_hook(lambda m, args, kwargs: seen.append(kwargs.get("logits_to_keep")), with_kwargs=True)
    try:
        model.generate(**tiny.image_inputs(), max_new_tokens=2, do_sample=False, pad_token_id=1)
    finally:
        handle.remove()
    assert seen and set(seen) == {1}


def test_logits_to_keep_is_forwarded_to_the_language_model(model):
    inputs = tiny.image_inputs()
    with torch.no_grad():
        full = model(**inputs).logits
        last = model(**inputs, logits_to_keep=1).logits
    assert last.shape[1] == 1
    torch.testing.assert_close(last, full[:, -1:], rtol=1e-5, atol=1e-6)


def _mixed_batch():
    gen = torch.Generator().manual_seed(3)
    samples = [[(1, 4, 4)], [(1, 2, 6), (1, 4, 4)], []]
    rows, per_sample_pixels, per_sample_grids = [], [], []
    for images in samples:
        row = [5]
        for t, h, w in images:
            row += [tiny.IMAGE_START] + [tiny.IMAGE_PAD] * (t * h * w // 4) + [tiny.IMAGE_END]
        rows.append(row + [7])
        per_sample_pixels.append(torch.randn(sum(t * h * w for t, h, w in images), tiny.PATCH_DIM, generator=gen))
        per_sample_grids.append(torch.tensor(images, dtype=torch.long).reshape(-1, 3))
    width = max(map(len, rows))
    input_ids = torch.tensor([row + [1] * (width - len(row)) for row in rows])
    return input_ids, per_sample_pixels, per_sample_grids


def test_expansion_for_beams_repeats_each_samples_whole_images(model):
    input_ids, pixels, grids = _mixed_batch()
    expanded_ids, kwargs = model._expand_inputs_for_generation(
        expand_size=2, input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        pixel_values=torch.cat(pixels), image_grid_thw=torch.cat(grids),
    )
    assert torch.equal(expanded_ids, input_ids.repeat_interleave(2, dim=0))
    assert kwargs["attention_mask"].shape[0] == 6
    assert torch.equal(kwargs["pixel_values"], torch.cat([pixels[0], pixels[0], pixels[1], pixels[1]]))
    assert torch.equal(kwargs["image_grid_thw"], torch.cat([grids[0], grids[0], grids[1], grids[1]]))
    n_features = int(kwargs["image_grid_thw"].prod(-1).sum() // 4)
    assert int((expanded_ids == tiny.IMAGE_PAD).sum()) == n_features


def test_expansion_is_a_no_op_for_a_single_sequence(model):
    inputs = tiny.image_inputs()
    ids, kwargs = model._expand_inputs_for_generation(
        expand_size=1, input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"]
    )
    assert ids is inputs["input_ids"] and kwargs["pixel_values"] is inputs["pixel_values"]


def test_expansion_rejects_grids_that_do_not_match_the_placeholders(model):
    input_ids, pixels, grids = _mixed_batch()
    with pytest.raises(ValueError, match="placeholder"):
        model._expand_inputs_for_generation(
            expand_size=2, input_ids=input_ids[:, :3], pixel_values=torch.cat(pixels), image_grid_thw=torch.cat(grids)
        )


@pytest.mark.parametrize("kwargs", [{"num_beams": 2}, {"num_beams": 2, "num_return_sequences": 2}, {"do_sample": True, "num_return_sequences": 2}])
def test_every_beam_and_returned_sequence_sees_the_original_image(model, kwargs):
    inputs = tiny.image_inputs()
    first_embeds = []

    def capture(module, args, call_kwargs):
        if call_kwargs.get("inputs_embeds") is not None and not first_embeds:
            first_embeds.append(call_kwargs["inputs_embeds"].detach().clone())

    handle = model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        model.generate(**inputs, max_new_tokens=2, pad_token_id=1, **{"do_sample": False, **kwargs})
    finally:
        handle.remove()
    with torch.no_grad():
        features = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
    embeds = first_embeds[0]
    assert embeds.shape[0] == kwargs.get("num_beams", 1) * kwargs.get("num_return_sequences", 1) or embeds.shape[0] >= 2
    is_pad = inputs["input_ids"][0] == tiny.IMAGE_PAD
    for row in embeds:
        torch.testing.assert_close(row[is_pad], features, rtol=1e-5, atol=1e-6)


def test_projected_images_and_gradients_do_not_depend_on_other_images(model):
    inputs = tiny.image_inputs(grids=((1, 4, 4), (1, 2, 6)))
    pixels = inputs["pixel_values"].split([16, 12])
    grids = inputs["image_grid_thw"]
    model.zero_grad(set_to_none=True)
    together = model.get_image_features(inputs["pixel_values"], grids)
    together.square().sum().backward()
    grads = {n: p.grad.clone() for n, p in model.projector.named_parameters()}
    model.zero_grad(set_to_none=True)
    separate = torch.cat([model.get_image_features(p, g[None]) for p, g in zip(pixels, grids)])
    separate.square().sum().backward()
    torch.testing.assert_close(together, separate, atol=0, rtol=0)
    for name, param in model.projector.named_parameters():
        torch.testing.assert_close(param.grad, grads[name], atol=0, rtol=0)
