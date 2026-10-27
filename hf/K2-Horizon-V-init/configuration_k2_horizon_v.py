from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig

from .configuration_k2_horizon import K2HorizonConfig


@strict
class K2HorizonVConfig(PreTrainedConfig):
    r"""
    Configuration of [`K2HorizonVForConditionalGeneration`]: K2-Horizon-7B plus a frozen GLM vision
    tower and a trainable projector.

    text_config (`dict | K2HorizonConfig`):
        K2-Horizon-7B's configuration, verbatim.
    vision_config (`dict | Glm5NextVisionConfig`):
        GLM-5.3-Flash-Vision's configuration. `out_hidden_size` and `projection_intermediate_size` describe
        GLM's own `downsample`/`merger` modules, which this model does not instantiate.
    image_token_id (`int`, *optional*, defaults to 250062):
        The image pad token (`reserved_special_token_12`). Image features replace its embeddings.
    image_start_token_id (`int`, *optional*, defaults to 250031):
        `<image>`.
    image_end_token_id (`int`, *optional*, defaults to 250032):
        `</image>`.
    """

    model_type = "k2_horizon_v"
    sub_configs = {"text_config": K2HorizonConfig, "vision_config": Glm5NextVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 250062
    image_start_token_id: int = 250031
    image_end_token_id: int = 250032
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        for key in ("text_config", "vision_config"):
            value = getattr(self, key)
            if isinstance(value, dict):
                setattr(self, key, self.sub_configs[key](**value))
            elif value is None:
                setattr(self, key, self.sub_configs[key]())

        super().__post_init__(**kwargs)


__all__ = ["K2HorizonVConfig"]
