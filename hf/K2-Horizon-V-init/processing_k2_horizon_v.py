import torch

from transformers import AutoImageProcessor, AutoTokenizer
from transformers.image_processing_utils import BatchFeature
from transformers.models.glm5_next.image_processing_pil_glm5_next import Glm5NextImageProcessorKwargs
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin


class K2HorizonVProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Glm5NextImageProcessorKwargs
    _defaults = {
        "text_kwargs": {"return_attention_mask": True, "return_token_type_ids": False},
        "common_kwargs": {"return_tensors": "pt"},
    }


class K2HorizonVProcessor(ProcessorMixin):
    """Prepare K2 conversations and GLM image patches with matching image-token positions.

    Each image must have one unexpanded ``<image>reserved_special_token_12</image>``
    marker in the rendered text. Images are consumed in conversation and batch order.
    Calls return only PyTorch model-input tensors.
    """

    valid_processor_kwargs = K2HorizonVProcessorKwargs
    image_token = "reserved_special_token_12"
    image_start_token = "<image>"
    image_end_token = "</image>"

    def __init__(self, image_processor, tokenizer, chat_template=None, image_backend=None):
        super().__init__(image_processor, tokenizer, chat_template=chat_template or tokenizer.chat_template)
        self.image_backend = image_processor.backend
        for token in (self.image_token, self.image_start_token, self.image_end_token):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token not in tokenizer.get_vocab() or tokenizer.encode(token, add_special_tokens=False) != [token_id]:
                raise ValueError(f"{token!r} must be an existing single tokenizer token")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_start_token_id = tokenizer.convert_tokens_to_ids(self.image_start_token)
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        backend = kwargs.pop("image_backend", kwargs.pop("backend", (processor_dict or {}).get("image_backend", "pil")))
        image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path, backend=backend, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return [image_processor, tokenizer]

    @property
    def image_marker(self):
        return self.image_start_token + self.image_token + self.image_end_token

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]

    def __call__(self, images=None, text=None, **kwargs):
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, (list, tuple)) and text and all(isinstance(item, str) for item in text):
            texts = list(text)
        else:
            raise ValueError("text must be a string or a nonempty batch of strings")
        if kwargs.get("videos") is not None or kwargs.get("audio") is not None:
            raise ValueError("K2-Horizon-V supports text and images only")
        for key in ("patch_size", "merge_size", "temporal_patch_size"):
            value = kwargs.get("images_kwargs", {}).get(key, kwargs.get(key, getattr(self.image_processor, key)))
            if value != getattr(self.image_processor, key):
                raise ValueError(f"{key} must match the checkpoint image processor")
        if isinstance(images, (list, tuple)) and images and all(isinstance(group, (list, tuple)) for group in images):
            if len(images) != len(texts) or any(len(group) != prompt.count(self.image_marker) for group, prompt in zip(images, texts)):
                raise ValueError("Each image batch group must match its text's image markers")
            if not any(images):
                images = None
        elif isinstance(images, (list, tuple)) and not images:
            images = None

        inputs = super().__call__(images=images, text=texts, **kwargs)
        if not isinstance(inputs["input_ids"], torch.Tensor):
            raise ValueError("K2-Horizon-V requires return_tensors='pt'")
        if len(inputs["input_ids"]) != len(texts):
            raise ValueError("Tokenization must return one sequence per input text; overflowing windows are unsupported")
        grids = inputs.get("image_grid_thw", [])
        merge_length = self.image_processor.merge_size**2
        first_image = 0
        for prompt, ids in zip(texts, inputs["input_ids"]):
            count = prompt.count(self.image_marker)
            selected = grids[first_image : first_image + count]
            expected = sum(int(grid.prod()) // merge_length for grid in selected)
            actual = (int((ids == self.image_token_id).sum()), int((ids == self.image_start_token_id).sum()), int((ids == self.image_end_token_id).sum()))
            if actual != (expected, count, count):
                raise ValueError("Image tokens were truncated or changed; increase max_length or reduce the image token budget")
            first_image += count

        keys = self.model_input_names + ["offset_mapping", "text_replacement_offsets"]
        return BatchFeature({key: value for key, value in inputs.items() if key in keys})

    def get_text_with_replacements(self, text, images_replacements=(), videos_replacements=(), audio_replacements=()):
        count = 0
        for prompt in text:
            count += prompt.count(self.image_marker)
            remainder = prompt.replace(self.image_marker, "")
            if any(token in remainder for token in (self.image_start_token, self.image_token, self.image_end_token)):
                raise ValueError("Image markers must be unexpanded and complete; remove literal image-token strings from text")
        if count != len(images_replacements):
            raise ValueError(f"Text contains {count} image markers, but {len(images_replacements)} images were supplied")
        return super().get_text_with_replacements(text, images_replacements, videos_replacements, audio_replacements)

    def replace_image_token(self, image_inputs, image_idx, **kwargs):
        grid = image_inputs["image_grid_thw"][image_idx]
        return self.image_token * (int(grid.prod()) // self.image_processor.merge_size**2)


K2HorizonVProcessor.register_for_auto_class()

__all__ = ["K2HorizonVProcessor"]
