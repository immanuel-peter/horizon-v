import copy
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
from PIL import Image
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.glm5_next.image_processing_pil_glm5_next import Glm5NextImageProcessorPil

import tiny
from k2hv.hf.processing_k2_horizon_v import K2HorizonVProcessor
from k2hv.processor_assets import IMAGE_TEMPLATE_BRANCH, patch_chat_template, processor_image_config


@pytest.fixture(scope="module")
def tokenizer():
    tokens = ["[UNK]", "[PAD]", "<image>", "reserved_special_token_12", "</image>",
              "<|ifm|begin_of_text|>", "<|ifm|endoftext|>", "<|ifm|im_start|>", "<|ifm|im_end|>",
              "<ifm|think>", "</ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>",
              "user", "assistant", "system", "Describe", "this", "image", "answer", "question", "."]
    backend = Tokenizer(WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="<|ifm|begin_of_text|>", eos_token="<|ifm|endoftext|>",
        additional_special_tokens=tokens[2:13],
        chat_template=(tiny.HF_DIR / "chat_template.jinja").read_text(),
    )


@pytest.fixture(scope="module")
def processor(tokenizer):
    config = json.loads((tiny.HF_DIR / "preprocessor_config.json").read_text())
    return K2HorizonVProcessor(Glm5NextImageProcessorPil(**config), tokenizer)


def image(width=448, height=448, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize("size", [(448, 448), (224, 448), (31, 47), (280, 1008), (1400, 1200)])
def test_native_glm_pixels_and_per_image_token_counts(processor, size):
    picture = image(*size)
    inputs = processor(text=processor.image_marker + "Describe this image.", images=picture)
    native = Glm5NextImageProcessorPil(max_image_tokens=1024)(picture, return_tensors="pt")
    assert set(inputs) == set(processor.model_input_names)
    assert all(isinstance(v, torch.Tensor) for v in inputs.values())
    assert torch.equal(inputs["pixel_values"], native["pixel_values"])
    assert torch.equal(inputs["image_grid_thw"], native["image_grid_thw"])
    expected = int(inputs["image_grid_thw"].prod(-1).sum()) // 4
    assert int((inputs["input_ids"] == processor.image_token_id).sum()) == expected
    assert expected <= 1024
    assert inputs["pixel_values"].shape == (4 * expected, 1176)
    if size == (448, 448):
        assert expected == 256


def test_multiple_images_and_mixed_batch_preserve_image_order(processor):
    pictures = [image(112, 112, 1), image(224, 112, 2), image(112, 224, 3)]
    texts = ["question", processor.image_marker * 2 + "question", processor.image_marker + "question"]
    original = list(texts)
    inputs = processor(text=texts, images=[[], pictures[:2], pictures[2:]], padding=True)
    assert texts == original
    native = processor.image_processor(pictures, return_tensors="pt")
    assert torch.equal(inputs["pixel_values"], native["pixel_values"])
    counts = inputs["image_grid_thw"].prod(-1) // 4
    actual = (inputs["input_ids"] == processor.image_token_id).sum(-1)
    assert actual.tolist() == [0, int(counts[:2].sum()), int(counts[2])]


def test_nested_image_groups_cannot_cross_batch_boundaries(processor):
    with pytest.raises(ValueError, match="batch group"):
        processor(text=[processor.image_marker, "question"], images=[[], [image()]])


@pytest.mark.parametrize("text,n_images", [("question", 1), ("<image>reserved_special_token_12</image>", 0),
                                         ("<image>reserved_special_token_12</image>", 2)])
def test_image_count_mismatch_fails(processor, text, n_images):
    with pytest.raises(ValueError, match="image markers"):
        processor(text=text, images=[image()] * n_images if n_images else None)


@pytest.mark.parametrize("text", ["literal <image>", "literal </image>", "literal reserved_special_token_12",
                                 "<image>reserved_special_token_12reserved_special_token_12</image>"])
def test_literal_and_preexpanded_markers_fail(processor, text):
    with pytest.raises(ValueError, match="unexpanded and complete"):
        processor(text=text)


def test_truncation_cannot_drop_image_features(processor):
    with pytest.raises(ValueError, match="[Tt]runcat|[Ii]mage tokens"):
        processor(text=processor.image_marker, images=image(), truncation=True, max_length=5)


@pytest.mark.parametrize("images_kwargs", [{"max_image_tokens": 16}, {"max_image_tokens": 64}, {"max_image_tokens": 128}])
def test_per_call_token_cap_uses_glm_token_units(processor, images_kwargs):
    inputs = processor(text=processor.image_marker, images=image(1400, 900), images_kwargs=images_kwargs)
    count = int((inputs["input_ids"] == processor.image_token_id).sum())
    assert 0 < count <= images_kwargs["max_image_tokens"]


@pytest.mark.parametrize("kwargs", [{"patch_size": 7}, {"images_kwargs": {"merge_size": 1}}, {"temporal_patch_size": 1}])
def test_patch_layout_overrides_are_rejected(processor, kwargs):
    with pytest.raises(ValueError, match="must match"):
        processor(text=processor.image_marker, images=image(), **kwargs)


def test_text_only_tokenization_is_unchanged(processor):
    texts = ["question", "Describe this image."]
    actual = processor(text=texts, padding=True)
    expected = processor.tokenizer(texts, padding=True, return_tensors="pt", return_token_type_ids=False)
    assert set(actual) == {"input_ids", "attention_mask"}
    assert all(torch.equal(actual[key], expected[key]) for key in expected)


@pytest.mark.parametrize("images", [None, [], [[], []]])
def test_empty_image_batches_are_text_only(processor, images):
    inputs = processor(text=["question", "answer"], images=images, padding=True)
    assert set(inputs) == {"input_ids", "attention_mask"}


@pytest.mark.parametrize("effort", ["high", "medium", "low"])
@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "question"}],
    [{"role": "system", "content": "question"}, {"role": "user", "content": [{"type": "text", "text": "question"}]}],
    [{"role": "user", "content": "question"}, {"role": "assistant", "reasoning_content": "", "content": "answer"}],
])
def test_text_only_templates_are_byte_identical(processor, effort, messages):
    upstream = processor.chat_template.replace(IMAGE_TEMPLATE_BRANCH, "")
    kwargs = {"tokenize": False, "add_generation_prompt": True, "reasoning_effort": effort}
    expected = processor.tokenizer.apply_chat_template(messages, chat_template=upstream, **kwargs)
    assert processor.apply_chat_template(copy.deepcopy(messages), **kwargs) == expected


def test_tool_template_is_unchanged(processor):
    upstream = processor.chat_template.replace(IMAGE_TEMPLATE_BRANCH, "")
    messages = [{"role": "user", "content": "question"}]
    tools = [{"type": "function", "function": {"name": "lookup", "description": "Look up a value",
              "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}]
    expected = processor.tokenizer.apply_chat_template(messages, chat_template=upstream, tools=tools, tokenize=False)
    assert processor.apply_chat_template(messages, tools=tools, tokenize=False) == expected


@pytest.mark.parametrize("role", ["system", "assistant"])
def test_only_user_images_are_accepted(processor, role):
    messages = [{"role": role, "reasoning_content": "", "content": [{"type": "image"}]}]
    with pytest.raises(Exception, match="Unsupported"):
        processor.apply_chat_template(messages, tokenize=False)


def test_chat_template_returns_model_ready_tensors_and_assistant_mask(processor):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image(112, 112)}, {"type": "text", "text": "question"}]},
        {"role": "assistant", "reasoning_content": "", "content": "answer"},
    ]
    inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", return_assistant_tokens_mask=True)
    assert set(inputs) == {*processor.model_input_names, "assistant_masks"}
    assert inputs["assistant_masks"].shape == inputs["input_ids"].shape
    assert inputs["assistant_masks"].sum() > 0
    assert not inputs["assistant_masks"][inputs["input_ids"] == processor.image_token_id].any()


def test_openai_image_url_content_and_batched_chat(processor):
    import base64
    import io

    buffer = io.BytesIO()
    image(112, 112).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    messages = [[{"role": "user", "content": "question"}],
                [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "question"}]}]]
    result = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", processor_kwargs={"padding": True})
    assert result["input_ids"].shape[0] == 2
    assert (result["input_ids"] == processor.image_token_id).sum(-1).tolist() == [0, 16]


def test_processor_outputs_run_through_the_composite_model(processor):
    model = tiny.tiny_model()
    model.config.image_token_id = processor.image_token_id
    model.config.image_start_token_id = processor.image_start_token_id
    model.config.image_end_token_id = processor.image_end_token_id
    mini = K2HorizonVProcessor(Glm5NextImageProcessorPil(patch_size=tiny.PATCH, min_image_tokens=1, max_image_tokens=16), processor.tokenizer)
    inputs = mini(text=mini.image_marker + "question", images=image(16, 16))
    with torch.no_grad():
        logits = model(**inputs, logits_to_keep=1).logits
    assert logits.shape == (1, 1, tiny.VOCAB)
    assert torch.isfinite(logits).all()


def test_processor_save_load_is_self_contained_and_offline(processor, tmp_path):
    saved = tmp_path / "standalone"
    original_vocab = processor.tokenizer.get_vocab()
    processor.save_pretrained(saved)
    assert processor.tokenizer.get_vocab() == original_vocab
    script = textwrap.dedent('''
        import sys
        sys.modules['k2hv'] = None
        from PIL import Image
        from transformers import AutoProcessor, AutoTokenizer
        p = AutoProcessor.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
        assert p.image_processor.backend == 'pil'
        inputs = p(text=p.image_marker, images=Image.new('RGB', (448, 448)))
        assert int((inputs['input_ids'] == p.image_token_id).sum()) == 256
        tok = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
        assert tok.chat_template == p.chat_template
        assert p.tokenizer.backend == 'tokenizers'
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': Image.new('RGB', (112, 112))}]},
                    {'role': 'assistant', 'reasoning_content': '', 'content': 'answer'}]
        masked = p.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors='pt', return_assistant_tokens_mask=True)
        assert masked['assistant_masks'].sum() > 0
        assert sys.modules['k2hv'] is None
        print('offline AutoProcessor: 256 image tokens, source package unavailable')
    ''')
    env = {key: value for key, value in os.environ.items() if not any(secret in key for secret in ("TOKEN", "HF_", "GH_", "GITHUB"))}
    env.update(HF_HUB_OFFLINE="1", HF_HOME=str(tmp_path / "empty-home"), HF_MODULES_CACHE=str(tmp_path / "empty-modules"))
    result = subprocess.run([sys.executable, "-I", "-c", script, str(saved)], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr


def test_asset_patch_is_exact_and_preserves_preprocessing(processor):
    upstream = processor.chat_template.replace(IMAGE_TEMPLATE_BRANCH, "")
    assert patch_chat_template(upstream) == processor.chat_template
    with pytest.raises(ValueError, match="one K2 text-content branch"):
        patch_chat_template("unexpected template")
    source = json.loads((tiny.HF_DIR / "preprocessor_config.json").read_text())
    source.update(image_processor_type="Glm5NextImageProcessor", processor_class="AutoImageProcessor", max_image_tokens=8000)
    source.pop("auto_map")
    patched = processor_image_config(source)
    assert patched == json.loads((tiny.HF_DIR / "preprocessor_config.json").read_text())
