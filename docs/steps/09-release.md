# Step 9: Release

## Model repo

Push the RL checkpoint with the same layout as [the init checkpoint](01-init-checkpoint.md):

- weights;
- remote code: `configuration_k2_horizon_v.py`, `modeling_k2_horizon_v.py`, `processing_k2_horizon_v.py`, plus K2's modeling files;
- tokenizer files and the patched `chat_template.jinja`;
- processor configs.

Before going public, check that the K2-Horizon-7B and GLM-5.3-Flash-Vision licenses allow redistributing derived weights.

## SGLang package

Publish `k2hv_sglang` ([step 3](03-sglang-model.md)) with install and launch instructions:

```bash
uv pip install k2hv-sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=k2hv_sglang
export SGLANG_EXTERNAL_MM_MODEL_ARCH=K2HorizonVForConditionalGeneration
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=k2hv_sglang
python -m sglang.launch_server \
  --model-path <you>/K2-Horizon-V --trust-remote-code \
  --enable-multimodal --image-processor-backend pil --reasoning-parser k2_horizon
```

## Model card

- Architecture: frozen K2-Horizon-7B, frozen GLM-5.3-Flash ViT, trained ~33.6M-param projector.
- Training: SFT data, RL data, step counts, hyperparameters, H100-hours.
- Eval tables: SFT vs RL vs stock K2 (text-only), with the [step 7](07-eval.md) settings.
- Text-only behavior: identical to K2-Horizon-7B by construction.
- Credits: Baseten's recipe, IFM's K2-Horizon and RL360, GLM.

## Later

- vLLM support. vLLM already has `k2_horizon.py`.
- Video input. GLM's processor supports temporal patches; neither stage trained on video.
