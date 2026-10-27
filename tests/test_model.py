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


# --- composition -------------------------------------------------------------------------------


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


# --- forward -----------------------------------------------------------------------------------


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
    assert torch.equal(seen["embeds"][~is_pad], token_embeds[~is_pad])  # <image>, </image> and text keep their embeddings


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
    torch.testing.assert_close(together, torch.cat([alone_1, alone_2]), rtol=1e-5, atol=1e-5)  # attention stays per image
    assert model(**both).logits.shape[:2] == both["input_ids"].shape


# --- validation --------------------------------------------------------------------------------


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
    # generate() keeps image_grid_thw after dropping pixel_values for decode steps; it must be ignored
    text = inputs["input_ids"][:, :2]
    assert model(input_ids=text, image_grid_thw=inputs["image_grid_thw"]).logits.shape[:2] == text.shape
    with pytest.raises(ValueError, match="requires input_ids"):
        model(inputs_embeds=torch.zeros(1, 4, 64), pixel_values=pixels, image_grid_thw=inputs["image_grid_thw"])


# --- gradients ---------------------------------------------------------------------------------


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

    assert seen["vit_out_requires_grad"] is False  # nothing is recorded through the vision tower
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
    model.visual.train()  # the tower pins itself to eval, so its blocks never checkpoint
    assert not model.visual.training
    checkpointed, _ = _projector_grads(model)

    assert checkpointed.keys() == reference.keys()
    for name in reference:
        torch.testing.assert_close(checkpointed[name], reference[name], rtol=1e-4, atol=1e-6)


def test_frozen_modules_stay_frozen_after_train_and_eval_toggles(model):
    model.train()
    model.eval()
    assert [n for n, p in model.named_parameters() if p.requires_grad and not n.startswith("projector.")] == []


# --- generation --------------------------------------------------------------------------------


def test_generate_with_an_image_uses_pixels_only_for_prefill(model):
    inputs = tiny.image_inputs()
    out = model.generate(**inputs, max_new_tokens=3, do_sample=False, pad_token_id=1)
    assert out.shape == (1, inputs["input_ids"].shape[1] + 3)

    text = model.generate(input_ids=inputs["input_ids"][:, :3], max_new_tokens=2, do_sample=False, pad_token_id=1)
    assert text.shape == (1, 5)


def test_from_pretrained_keeps_frozen_modules_frozen(tmp_path):
    """transformers' loader resets requires_grad to True on every loaded parameter; the model must undo that."""
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
