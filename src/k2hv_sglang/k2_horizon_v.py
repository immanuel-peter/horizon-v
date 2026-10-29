# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
# Vision forward adapted from sgl-project/sglang at b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf,
# python/sglang/srt/models/glm_ocr.py. See https://www.apache.org/licenses/LICENSE-2.0.

from itertools import groupby
import os

import torch
from torch import nn

from sglang.srt.layers.attention.vision import prepare_vision_attention_metadata
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens, prepare_mm_inputs
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.glm5_next import Glm5NextVisionModel
from sglang.srt.models.xllm import K2HorizonForCausalLM
from sglang.srt.runtime_context import get_mm
from sglang.srt.utils.common import add_prefix

from k2hv.gpu_numerics import enable as enable_batch_invariant
from k2hv.hf.modeling_k2_horizon_v import Projector
from .gate_trace import emit, weight_state
from .vision_reference import preserve_rounding
from .text_reference import preserve_rounding as preserve_text_rounding
from .input_layout import map_vision_weight, validate_image_grids


class TruncatedGlm5NextVision(Glm5NextVisionModel):
    def __init__(self, config, prefix=""):
        super().__init__(config, quant_config=None, prefix=prefix, use_data_parallel=False)
        self.patch_embed.proj.enable_linear = False
        del self.downsample
        del self.merger
        preserve_rounding(self, config)

    def forward(self, pixel_values, grid_thw):
        validate_image_grids(pixel_values, grid_thw, merge_size=self.spatial_merge_size)
        x = self.patch_embed(pixel_values.to(device=self.device, dtype=self.dtype))
        cos, sin, _ = self.rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
        metadata = prepare_vision_attention_metadata(cu_seqlens, device=x.device)
        cos, sin = torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)
        x = x.unsqueeze(1)
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, rotary_pos_emb_cos=cos, rotary_pos_emb_sin=sin, forward_metadata=metadata)
        return self.post_layernorm(x).reshape(-1, self.hidden_size)


class K2HorizonVForConditionalGeneration(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if quant_config is not None:
            raise ValueError("K2-Horizon-V currently loads the unquantized initialization checkpoint")
        if get_mm().mm_enable_dp_encoder:
            raise ValueError("K2-Horizon-V does not yet support the data-parallel encoder path")
        if os.environ.get("K2HV_BATCH_INVARIANT") == "1":
            enable_batch_invariant()
        self.mm_config = config
        self.config = config.text_config
        self.visual = TruncatedGlm5NextVision(config.vision_config, prefix=add_prefix("visual", prefix))
        self.projector = Projector(config.vision_config.hidden_size, config.text_config.hidden_size, config.vision_config.spatial_merge_size)
        self.language_model = K2HorizonForCausalLM(config.text_config, quant_config=None, prefix=add_prefix("language_model", prefix))
        preserve_text_rounding(self.language_model, config.text_config)
        self.pp_group = self.language_model.pp_group
        self.is_mrope_enabled = False

    @property
    def model(self):
        return self.language_model.model

    @property
    def lm_head(self):
        return self.language_model.lm_head

    @property
    def start_layer(self):
        return self.language_model.start_layer

    @property
    def end_layer(self):
        return self.language_model.end_layer

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def pad_input_ids(self, input_ids, mm_inputs):
        original = list(input_ids)
        padded = MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(input_ids, mm_inputs)
        emit("padding", original=original, padded=padded, offsets=[item.offsets for item in mm_inputs.mm_items])
        return padded

    def get_image_feature(self, items):
        features = []
        for item in items:
            grid = item.image_grid_thw.to(device="cpu", dtype=torch.long)
            patches = self.visual(item.feature, grid_thw=grid)
            projected = self.projector(patches.to(self.projector.fc1.weight))
            emit("vision", pixels=item.feature, grid=grid, tower=patches, projected=projected)
            features.append(projected)
        return torch.cat(features)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
        emit("forward", ids=input_ids, positions=positions)
        if input_embeds is None or forward_batch.contains_mm_inputs():
            input_embeds = prepare_mm_inputs(input_ids, forward_batch, language_model=self, kwargs={}, multimodal_model=self)
        emit("inserted", ids=input_ids, positions=positions, embeddings=input_embeds)
        output = self.language_model(input_ids, positions, forward_batch, input_embeds=input_embeds, pp_proxy_tensors=pp_proxy_tensors)
        emit("forward_end", peak_allocated_bytes=torch.cuda.max_memory_allocated())
        return output

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded_names = []

        def observed():
            for name, tensor in weights:
                loaded_names.append(name)
                yield name,tensor

        for component, group in groupby(observed(), key=lambda entry: entry[0].split(".", 1)[0]):
            if component == "language_model":
                language_weights = ((name.removeprefix("language_model."), weight) for name, weight in group)
                self.language_model.load_weights(language_weights)
                if next(language_weights, None) is not None:
                    raise RuntimeError("The native K2 loader did not consume every language weight")
            elif component in ("visual", "projector"):
                for name, weight in group:
                    shard = None
                    if component == "visual":
                        name, shard = map_vision_weight(name)
                    if name not in params:
                        raise ValueError(f"Checkpoint weight does not map to a model parameter: {name}")
                    param = params[name]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    if shard is None:
                        loader(param, weight)
                    else:
                        loader(param, weight, shard)
            else:
                name, _ = next(group)
                raise ValueError(f"Unknown checkpoint weight: {name}")
        if mm_schedule.embedding_cache is not None:
            mm_schedule.embedding_cache.clear()
        weight_state(self,loaded_names)


EntryClass = K2HorizonVForConditionalGeneration
