"""Tiny stand-ins for K2-Horizon-7B and GLM-5.3-Flash-Vision.

The configs mirror the structure of the real ``config.json`` files (same keys, including K2's
``auto_map``) at a size that builds in milliseconds. Token ids are small so the vocab stays tiny;
the real ids (250031/250032/250062) are only used by the real build.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

VOCAB = 320
IMAGE_START, IMAGE_END, IMAGE_PAD = 300, 301, 302
PATCH = 4
PATCH_DIM = 3 * 2 * PATCH * PATCH  # channels * temporal_patch_size * patch^2
HF_DIR = Path(__file__).resolve().parents[1] / "hf" / "K2-Horizon-V-init"  # the tracked HF repo tree


def k2_config_dict(dtype: str = "float32", **overrides) -> dict:
    cfg = {
        "architectures": ["K2HorizonForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_gate_func": None,
        "auto_map": {
            "AutoConfig": "configuration_k2_horizon.K2HorizonConfig",
            "AutoModel": "modeling_k2_horizon.K2HorizonModel",
            "AutoModelForCausalLM": "modeling_k2_horizon.K2HorizonForCausalLM",
        },
        "bos_token_id": 0,
        "decoder_sparse_step": 1,
        "dtype": dtype,
        "eos_token_id": 1,
        "head_dim": 16,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 96,
        "layernorm_num_groups": 4,
        "max_position_embeddings": 512,
        "mlp_only_layers": [0, 1],
        "model_type": "k2_horizon",
        "moe_gate_bias": False,
        "moe_intermediate_size": 0,
        "mova_num_experts": 0,
        "mova_num_experts_per_tok": 0,
        "norm_topk_prob": True,
        "num_attention_heads": 4,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_shared_experts": 0,
        "pad_token_id": None,
        "query_key_norm": False,
        "rms_norm_eps": 1e-6,
        "rope_head_dim": 16,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "router_aux_loss_coef": 0.001,
        "router_scaling_factor": 1.0,
        "router_score_func": "sigmoid",
        "sliding_window": None,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": VOCAB,
    }
    cfg.update(overrides)
    return cfg


def glm_vision_config_dict(dtype: str = "float32", **overrides) -> dict:
    cfg = {
        "architectures": ["Glm5NextVisionModel"],
        "attention_bias": True,
        "attention_dropout": 0.0,
        "depth": 2,
        "dtype": dtype,
        "hidden_act": "silu",
        "hidden_size": 32,
        "image_size": 56,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 48,
        "model_type": "glm5_next_vision",
        "num_heads": 2,
        "out_hidden_size": 64,
        "patch_size": PATCH,
        "projection_intermediate_size": 80,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"},
        "spatial_merge_size": 2,
        "swiglu_limit": 10.0,
        "temporal_patch_size": 2,
    }
    cfg.update(overrides)
    return cfg


def composite_config(dtype: str = "float32"):
    from k2hv.hf.configuration_k2_horizon_v import K2HorizonVConfig

    return K2HorizonVConfig(
        text_config=k2_config_dict(dtype),
        vision_config=glm_vision_config_dict(dtype),
        image_token_id=IMAGE_PAD,
        image_start_token_id=IMAGE_START,
        image_end_token_id=IMAGE_END,
        dtype=dtype,
    )


def tiny_model(dtype: str = "float32", seed: int = 0):
    """Randomly initialized tiny composite model (projector has non-trivial values)."""
    from k2hv.hf.modeling_k2_horizon_v import K2HorizonVForConditionalGeneration

    torch.manual_seed(seed)
    model = K2HorizonVForConditionalGeneration(composite_config(dtype))
    return model.eval()


def image_inputs(grids=((1, 4, 4),), seed: int = 1, batch_tokens_prefix=(5, 6)):
    """Hand-built multimodal batch: ``prefix + <image> + pads + </image> + suffix`` per image."""
    gen = torch.Generator().manual_seed(seed)
    grid = torch.tensor(grids, dtype=torch.long)
    num_patches = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    pixel_values = torch.randn(num_patches, PATCH_DIM, generator=gen)
    ids: list[int] = list(batch_tokens_prefix)
    for t, h, w in grids:
        ids += [IMAGE_START] + [IMAGE_PAD] * (t * h * w // 4) + [IMAGE_END]
    ids += [7, 8, 9]
    return {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        "pixel_values": pixel_values,
        "image_grid_thw": grid,
    }


def write_source_checkpoints(root: Path, *, k2_shards: int = 2, glm_sharded: bool = False, dtype: str = "bfloat16"):
    """Write tiny K2/GLM source dirs shaped like the real downloads, with random bf16 weights.

    Returns ``(k2_dir, glm_dir)``. K2 is always sharded with an index (like the real repo); GLM is a
    single ``model.safetensors`` unless ``glm_sharded``. Also writes every non-weight file the builder
    needs, with the tokenizer reduced to the three image tokens.
    """
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    torch_dtype = getattr(torch, dtype)
    k2_dir, glm_dir = root / "K2-Horizon-7B", root / "GLM-5.3-Flash-Vision"
    k2_dir.mkdir(parents=True)
    glm_dir.mkdir(parents=True)

    # K2 weights: the stock K2 model's own state dict.
    from k2hv.hf.configuration_k2_horizon import K2HorizonConfig
    from k2hv.hf.modeling_k2_horizon import K2HorizonForCausalLM

    torch.manual_seed(100)
    k2_cfg = k2_config_dict(dtype)
    k2_state = {
        k: v.detach().clone().to(torch_dtype).contiguous()
        for k, v in K2HorizonForCausalLM(K2HorizonConfig(**k2_cfg)).state_dict().items()
    }
    _write_sharded(k2_dir, k2_state, k2_shards)
    (k2_dir / "config.json").write_text(json.dumps(k2_cfg, indent=2))
    for name in ("configuration_k2_horizon.py", "modeling_k2_horizon.py"):
        (k2_dir / name).write_bytes((HF_DIR / name).read_bytes())
    (k2_dir / "generation_config.json").write_text(json.dumps({"bos_token_id": 0, "eos_token_id": [1], "pad_token_id": 1}))
    (k2_dir / "special_tokens_map.json").write_text("{}")
    (k2_dir / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))
    (k2_dir / "chat_template.jinja").write_text("{{ messages }}")
    (k2_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": IMAGE_START, "content": "<image>", "special": True},
                    {"id": IMAGE_END, "content": "</image>", "special": True},
                    {"id": IMAGE_PAD, "content": "reserved_special_token_12", "special": True},
                ],
                "model": {"type": "BPE", "vocab": {}, "merges": []},
            }
        )
    )

    # GLM weights: the full native vision model (including downsample + merger, as in the real repo).
    torch.manual_seed(200)
    glm_cfg = glm_vision_config_dict(dtype)
    glm_state = {
        k: v.detach().clone().to(torch_dtype).contiguous()
        for k, v in Glm5NextVisionModel(Glm5NextVisionConfig(**glm_cfg)).state_dict().items()
    }
    if glm_sharded:
        _write_sharded(glm_dir, glm_state, 3)
    else:
        save_file(glm_state, glm_dir / "model.safetensors", metadata={"format": "pt"})
    (glm_dir / "config.json").write_text(json.dumps(glm_cfg, indent=2))
    (glm_dir / "preprocessor_config.json").write_text(json.dumps({"image_processor_type": "Glm5NextImageProcessor"}))
    (glm_dir / "LICENSE").write_text("MIT License\n\nCopyright (c) 2026 Z.AI Co., Ltd\n")
    return k2_dir, glm_dir


def _write_sharded(directory: Path, state: dict, num_shards: int) -> None:
    names = sorted(state)
    chunks = [names[i::num_shards] for i in range(num_shards)]
    weight_map, total = {}, 0
    for i, chunk in enumerate(chunks, 1):
        fname = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        save_file({k: state[k] for k in chunk}, directory / fname, metadata={"format": "pt"})
        weight_map.update({k: fname for k in chunk})
        total += sum(state[k].numel() * state[k].element_size() for k in chunk)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )


# Files build_init expects to find in the output directory, split by where they come from.
AUTHORED = ("configuration_k2_horizon_v.py", "modeling_k2_horizon_v.py", "configuration_k2_horizon.py", "modeling_k2_horizon.py",
            "README.md", "NOTICE", "licenses/Apache-2.0.txt")
FROM_K2 = ("tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "chat_template.jinja")


def write_recipe(out: Path, k2_dir: Path, glm_dir: Path) -> Path:
    """The tracked part of an HF repo tree, as build_init expects it: our files from hf/K2-Horizon-V-init,
    mirrored upstream files from the (tiny) sources."""
    for name in AUTHORED:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes((HF_DIR / name).read_bytes())
    for name in FROM_K2:
        (out / name).write_bytes((k2_dir / name).read_bytes())
    (out / "preprocessor_config.json").write_bytes((glm_dir / "preprocessor_config.json").read_bytes())
    (out / "licenses" / "MIT-GLM-5.3-Flash-Vision.txt").write_bytes((glm_dir / "LICENSE").read_bytes())
    return out
