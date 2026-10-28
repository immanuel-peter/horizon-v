from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import importlib
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = nn.Linear(hidden, 3 * hidden)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = nn.Linear(hidden, hidden * 2)
        for param in self.mlp.gate_up_proj.parameters():
            def load(target, weight, shard):
                target.data.chunk(2, dim=0)[shard].copy_(weight)
            param.weight_loader = load

    def forward(self, x, *, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, forward_metadata):
        return x + 1


class Patch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(3 * config.temporal_patch_size * config.patch_size**2, config.hidden_size)

    def forward(self, x):
        return self.proj(x)


class Vision(nn.Module):
    def __init__(self, config, quant_config=None, prefix="", use_data_parallel=False):
        super().__init__()
        assert quant_config is None and not use_data_parallel
        self.hidden_size = config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Patch(config)
        self.blocks = nn.ModuleList([Block(config.hidden_size)])
        self.post_layernorm = nn.LayerNorm(config.hidden_size)
        self.downsample = nn.Linear(config.hidden_size, 1)
        self.merger = nn.Linear(1, 1)

    @property
    def device(self):
        return self.patch_embed.proj.weight.device

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def rot_pos_emb(self, grid):
        return torch.ones(int(grid.prod(-1).sum()), 2), torch.zeros(int(grid.prod(-1).sum()), 2), None


class LanguageModel(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True, world_size=1)
        self.start_layer, self.end_layer = 0, config.num_hidden_layers
        self.loaded = {}
        self.last_positions = None
        self.last_embeds = None

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
        self.last_positions = positions
        if input_embeds is None:
            input_embeds = self.model.embed_tokens(input_ids)
        self.last_embeds = input_embeds
        return self.lm_head(input_embeds)

    def load_weights(self, weights):
        for name, value in weights:
            self.loaded[name] = value.clone()

def default_weight_loader(param, weight):
    if param.shape != weight.shape:
        raise ValueError("Shape mismatch")
    param.data.copy_(weight)


def prepare_mm_inputs(input_ids, forward_batch, language_model, kwargs, multimodal_model):
    embeds = language_model.get_input_embeddings()(input_ids)
    if forward_batch.contains_mm_inputs():
        features = multimodal_model.get_image_feature(forward_batch.items)
        mask = input_ids == multimodal_model.mm_config.image_token_id
        embeds = embeds.masked_scatter(mask[:, None], features)
    return embeds


class PaddingPattern:
    def pad_input_tokens(self, input_ids, mm_inputs):
        result = input_ids[:]
        for item in mm_inputs.mm_items:
            for first, last in item.offsets:
                result[first : last + 1] = type(result)(result.typecode, [item.pad_value] * (last - first + 1))
        return result


@dataclass
class Item:
    modality: object
    feature: torch.Tensor
    offsets: list
    model_specific_data: dict
    pad_value: int = 1000000

    @property
    def image_grid_thw(self):
        return self.model_specific_data["image_grid_thw"]

    def set_hash(self, value):
        self.hash = value
        self.pad_value = 1000000 + value % (1 << 30)


@dataclass
class Output:
    mm_items: list
    input_ids: list
    im_token_id: int
    im_start_id: int
    im_end_id: int
    mrope_positions: object = None


class BaseProcessor:
    def __init__(self, hf_config, server_args, _processor, transport_mode, *args, **kwargs):
        self.hf_config, self._processor, self._tokenizer = hf_config, _processor, _processor.tokenizer
        self.image_config = getattr(server_args, "image_config", {})
        self.io_executor = ThreadPoolExecutor(2)

    async def fast_load_mm_data(self, prompt, multimodal_tokens, image_data=None, **kwargs):
        assert multimodal_tokens.image_token_id == self.hf_config.image_token_id
        import base64
        import io

        pictures = []
        for item in image_data:
            if isinstance(item, str):
                item = Image.open(io.BytesIO(base64.b64decode(item.split(",")[-1]))).convert("RGB")
            pictures.append(item)
        return SimpleNamespace(images=pictures)

    def _finalize_mm_items(self, items, *, images):
        assert len(items) == len(images)
        return items


class EmbeddingCache:
    def __init__(self):
        self.entries = {"old_projected_image": torch.ones(1)}
        self.clears = 0

    def clear(self):
        self.entries.clear()
        self.clears += 1


def install(monkeypatch, *, load_adapters=True):
    names = {
        "sglang.srt.layers.attention.vision": {"prepare_vision_attention_metadata": lambda x, device: x},
        "sglang.srt.managers.mm_utils": {"MultiModalityDataPaddingPatternMultimodalTokens": PaddingPattern, "prepare_mm_inputs": prepare_mm_inputs},
        "sglang.srt.managers.mm_schedule": {"embedding_cache": EmbeddingCache()},
        "sglang.srt.model_loader.weight_utils": {"default_weight_loader": default_weight_loader},
        "sglang.srt.models.glm5_next": {"Glm5NextVisionModel": Vision},
        "sglang.srt.models.xllm": {"K2HorizonForCausalLM": LanguageModel},
        "sglang.srt.runtime_context": {"get_mm": lambda: SimpleNamespace(mm_enable_dp_encoder=False)},
        "sglang.srt.utils.common": {"add_prefix": lambda name, prefix: f"{prefix}.{name}" if prefix else name},
        "sglang.srt.managers.schedule_batch": {"Modality": SimpleNamespace(IMAGE="image"), "MultimodalDataItem": Item, "MultimodalProcessorOutput": Output},
        "sglang.srt.multimodal.processors.base_processor": {"BaseMultimodalProcessor": BaseProcessor, "MultimodalSpecialTokens": SimpleNamespace},
    }
    for name, attributes in names.items():
        pieces = name.split(".")
        for end in range(1, len(pieces) + 1):
            module_name = ".".join(pieces[:end])
            if module_name not in sys.modules:
                module = ModuleType(module_name)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, module_name, module)
        for key, value in attributes.items():
            monkeypatch.setattr(sys.modules[name], key, value, raising=False)
    for module in ("k2hv_sglang.k2_horizon_v", "k2hv_sglang.processor"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    if load_adapters:
        return importlib.import_module("k2hv_sglang.k2_horizon_v"), importlib.import_module("k2hv_sglang.processor")
