import torch
import torch.nn.functional as F
from torch import nn

from transformers import initialization as init
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextRMSNorm,
    Glm5NextVisionBlock,
    Glm5NextVisionPatchEmbed,
    Glm5NextVisionRotaryEmbedding,
)
from transformers.vision_utils import get_vision_attention_seqlens, get_vision_position_ids

from .configuration_k2_horizon_v import K2HorizonVConfig
from .modeling_k2_horizon import K2HorizonForCausalLM


class GlmVisionTower(PreTrainedModel):
    """Accept patches in merge-window order and return features in the same order.

    Each window contains ``spatial_merge_size ** 2`` consecutive patches in row-major order.
    """

    config: Glm5NextVisionConfig
    base_model_prefix = ""
    input_modalities = ("image",)
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _no_split_modules = ["Glm5NextVisionBlock"]

    def __init__(self, config: Glm5NextVisionConfig):
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Glm5NextVisionPatchEmbed(config)
        self.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList([Glm5NextVisionBlock(config) for _ in range(config.depth)])
        self.post_layernorm = Glm5NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        pixel_values (`torch.Tensor` of shape `(sum(t*h*w), C*temporal_patch*patch*patch)`):
            Flattened patches from the GLM image processor.
        grid_thw (`torch.Tensor` of shape `(num_images, 3)`):
            Temporal, height and width of each image's patch grid.

        Returns:
            `torch.Tensor` of shape `(sum(t*h*w), hidden_size)`.
        """
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)

        hidden_states = self.patch_embed(pixel_values)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
            )
        return self.post_layernorm(hidden_states)


class Projector(nn.Module):
    def __init__(self, d_vit: int, d_llm: int, merge_size: int = 2):
        super().__init__()
        self.merge_size = merge_size
        self.norm = nn.LayerNorm(d_vit)
        self.fc1 = nn.Linear(merge_size**2 * d_vit, d_llm)
        self.fc2 = nn.Linear(d_llm, d_llm)

    def merge(self, x: torch.Tensor) -> torch.Tensor:
        group = self.merge_size**2
        if x.ndim != 2 or x.shape[0] % group:
            raise ValueError(
                f"Projector expects (num_patches, {self.norm.normalized_shape[0]}) with num_patches divisible "
                f"by {group}, got {tuple(x.shape)}"
            )
        return x.reshape(-1, group * x.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(x, self.norm.normalized_shape, self.norm.weight.to(x.dtype), self.norm.bias.to(x.dtype), self.norm.eps)
        hidden = F.linear(self.merge(normalized), self.fc1.weight.to(x.dtype), self.fc1.bias.to(x.dtype))
        return F.linear(F.gelu(hidden), self.fc2.weight.to(x.dtype), self.fc2.bias.to(x.dtype))

    def init_module(self, module: nn.Module, generator: torch.Generator | None = None) -> None:
        if module is self.norm:
            init.ones_(module.weight)
            init.zeros_(module.bias)
        elif module is self.fc1 or module is self.fc2:
            bound = module.in_features**-0.5
            init.uniform_(module.weight, -bound, bound, generator=generator)
            init.uniform_(module.bias, -bound, bound, generator=generator)

    @torch.no_grad()
    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        for module in (self.norm, self.fc1, self.fc2):
            self.init_module(module, generator)


def build_projector(
    d_vit: int, d_llm: int, merge_size: int = 2, seed: int = 0, dtype: torch.dtype = torch.float32
) -> Projector:
    """Deterministically initialized projector.

    Draw values in float32 from a private generator seeded with ``seed``, then cast to ``dtype``.
    Draws are independent of the global RNG and target dtype. Leave the global RNG state unchanged.
    """
    with torch.random.fork_rng(devices=[]):
        projector = Projector(d_vit, d_llm, merge_size)
    projector.reset_parameters(torch.Generator().manual_seed(seed))
    return projector.to(dtype)


class K2HorizonVForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """K2-Horizon-7B with images.

    * ``language_model``: `K2HorizonForCausalLM`, frozen.
    * ``visual``: GLM vision tower through ``post_layernorm``, frozen, always run under ``torch.no_grad()``.
    * ``projector``: the only trainable module.

    Image features replace the input embeddings at ``image_token_id`` positions. Positions are K2's
    ordinary 1D positions; image tokens are just tokens in the sequence.
    """

    config: K2HorizonVConfig
    base_model_prefix = ""
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _no_split_modules = ["K2HorizonDecoderLayer", "Glm5NextVisionBlock"]
    _skip_keys_device_placement = ["past_key_values"]

    def __init__(self, config: K2HorizonVConfig):
        super().__init__(config)
        self.visual = GlmVisionTower._from_config(config.vision_config)
        self.projector = Projector(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            config.vision_config.spatial_merge_size,
        )
        self.language_model = K2HorizonForCausalLM._from_config(config.text_config)

        self.freeze_pretrained_modules()
        self.post_init()

    def freeze_pretrained_modules(self) -> None:
        self.visual.requires_grad_(False)
        self.language_model.requires_grad_(False)
        self.projector.requires_grad_(True)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        result = super().from_pretrained(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        model.freeze_pretrained_modules()
        return result

    @torch.no_grad()
    def _init_weights(self, module):
        self.projector.init_module(module)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Image patches -> LLM-space features of shape ``(sum(t*h*w) // merge**2, hidden_size)``.

        The vision tower runs under ``torch.no_grad()``; the projector output carries gradients.
        """
        merge = self.config.vision_config.spatial_merge_size
        grid = image_grid_thw.to(device=self.visual.device, dtype=torch.long)
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError(f"image_grid_thw must have shape (num_images, 3), got {tuple(grid.shape)}")
        t, h, w = grid.unbind(-1)
        if bool(((h % merge) | (w % merge)).any()):
            raise ValueError(f"Every image grid height/width must be divisible by {merge}, got {grid.tolist()}")
        num_patches = int((t * h * w).sum())
        if num_patches != pixel_values.shape[0]:
            raise ValueError(
                f"pixel_values has {pixel_values.shape[0]} patches but image_grid_thw describes {num_patches}"
            )

        if len(grid) > 1:
            patches = pixel_values.split(grid.prod(-1).tolist())
            return torch.cat([self.get_image_features(pixels, image_grid[None]) for pixels, image_grid in zip(patches, grid)])

        pixel_values = pixel_values.to(device=self.visual.device, dtype=self.visual.dtype)
        with torch.no_grad():
            patch_features = self.visual(pixel_values, grid_thw=grid)
        return self.projector(patch_features)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        """
        pixel_values (`torch.Tensor` of shape `(sum(t*h*w), patch_dim)`, *optional*):
            Flattened patches from the GLM image processor. Requires `input_ids` and `image_grid_thw`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Patch grid of each image.

        logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
            Compute logits for the last N positions, or the positions selected by a tensor.
            Use 0 for all positions.

        The number of `image_token_id` tokens must equal ``sum(t*h*w) // merge**2``.
        Image placeholders require `pixel_values` during prefill. Without `pixel_values`,
        ignore `image_grid_thw` and use K2's text path.
        """
        if pixel_values is None:
            if input_ids is not None and self._is_prefill(past_key_values):
                num_placeholders = int((input_ids == self.config.image_token_id).sum())
                if num_placeholders:
                    raise ValueError(
                        f"input_ids contain {num_placeholders} image placeholder tokens (id "
                        f"{self.config.image_token_id}) but no pixel_values were passed"
                    )
        else:
            if image_grid_thw is None:
                raise ValueError("pixel_values requires image_grid_thw")
            if input_ids is None or inputs_embeds is not None:
                raise ValueError("pixel_values requires input_ids (and not inputs_embeds) to locate image tokens")
            inputs_embeds = self.get_input_embeddings()(input_ids)
            image_features = self.get_image_features(pixel_values, image_grid_thw)
            mask = (input_ids == self.config.image_token_id).to(inputs_embeds.device)
            num_placeholders = int(mask.sum())
            if num_placeholders != image_features.shape[0]:
                raise ValueError(
                    f"Image features and image tokens do not match: {num_placeholders} tokens "
                    f"(id {self.config.image_token_id}) in input_ids, {image_features.shape[0]} features"
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                mask.unsqueeze(-1).expand_as(inputs_embeds), image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            )
            input_ids = None

        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    def _expand_inputs_for_generation(self, expand_size: int = 1, is_encoder_decoder: bool = False, input_ids=None, **model_kwargs):
        if expand_size > 1 and model_kwargs.get("pixel_values") is not None:
            model_kwargs["pixel_values"], model_kwargs["image_grid_thw"] = self._repeat_images_per_sample(
                input_ids, model_kwargs["pixel_values"], model_kwargs.get("image_grid_thw"), expand_size
            )
            expanded_images = {key: model_kwargs.pop(key) for key in ("pixel_values", "image_grid_thw")}
            input_ids, model_kwargs = super()._expand_inputs_for_generation(
                expand_size=expand_size, is_encoder_decoder=is_encoder_decoder, input_ids=input_ids, **model_kwargs
            )
            return input_ids, {**model_kwargs, **expanded_images}
        return super()._expand_inputs_for_generation(
            expand_size=expand_size, is_encoder_decoder=is_encoder_decoder, input_ids=input_ids, **model_kwargs
        )

    def _repeat_images_per_sample(self, input_ids, pixel_values, image_grid_thw, expand_size):
        if input_ids is None or image_grid_thw is None:
            raise ValueError("expanding pixel_values for generation requires input_ids and image_grid_thw")
        merge = self.config.vision_config.spatial_merge_size
        grid = image_grid_thw.to(torch.long)
        patches_per_image = grid.prod(-1).tolist()
        if sum(patches_per_image) != pixel_values.shape[0]:
            raise ValueError(f"pixel_values has {pixel_values.shape[0]} patches but image_grid_thw describes {sum(patches_per_image)}")
        images = pixel_values.split(patches_per_image)
        tokens_per_image = [n // merge**2 for n in patches_per_image]
        placeholders_per_sample = (input_ids == self.config.image_token_id).sum(-1).tolist()

        expanded_pixels, expanded_grids, first_image = [], [], 0
        for count in placeholders_per_sample:
            last_image, assigned_tokens = first_image, 0
            while assigned_tokens < count and last_image < len(tokens_per_image):
                assigned_tokens += tokens_per_image[last_image]
                last_image += 1
            if assigned_tokens != count:
                raise ValueError(f"a sample has {count} image placeholder tokens that no whole number of images in image_grid_thw accounts for")
            if last_image > first_image:
                expanded_pixels += [torch.cat(images[first_image:last_image])] * expand_size
                expanded_grids += [grid[first_image:last_image]] * expand_size
            first_image = last_image
        if first_image != len(tokens_per_image):
            raise ValueError("image_grid_thw describes more images than the image placeholder tokens in input_ids")
        return torch.cat(expanded_pixels), torch.cat(expanded_grids)

    @staticmethod
    def _is_prefill(past_key_values) -> bool:
        return past_key_values is None or past_key_values.get_seq_length() == 0


__all__ = ["GlmVisionTower", "K2HorizonVForConditionalGeneration", "Projector", "build_projector"]
