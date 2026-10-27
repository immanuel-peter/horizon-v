# K2-Horizon-V

A multimodal K2-Horizon-7B.

Following Baseten's post-training strategy to add vision to base LLMs, we use a GLM-5.3-Flash vision head and train a new projector.

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --locked
uv run python -c "import k2hv, k2hv_sglang, torch, transformers; print(transformers.__version__)"
```

The project uses Python 3.12. PyPI supplies the CUDA build of `torch` on Linux
and the CPU/MPS build on macOS. The Linux build also runs on CPU machines;
the venv uses about 5 GB. GPU nodes reuse the Miles image's torch. Follow the
[environment guide](docs/steps/00-environment.md) for that setup.

Setup needs no GPU or model downloads. The composite model, checkpoint builder,
verifier, processor and image chat template are implemented. SGLang integration
and training remain planned. See the [checkpoint handoff](docs/handoff/01-init-checkpoint.md)
and [processor handoff](docs/handoff/02-processor-and-template.md).

## Layout

```text
hf/K2-Horizon-V-init/   Hugging Face checkpoint directory
src/k2hv/hf             symlink to hf/K2-Horizon-V-init, importable as k2hv.hf.*
src/k2hv/               checkpoint builder, verifier and file helpers
src/k2hv_sglang/        SGLang package, planned in step 3
tests/                 CPU tests with tiny configs
models/                downloaded source checkpoints, git-ignored
docs/                  plan and handoff notes
```

Git tracks the checkpoint's Python code, docs, licenses, `config.json` and small
upstream assets. Weights, the index, `tokenizer.json`, `build_info.json` and
`SHA256SUMS` are git-ignored. Tests need no downloaded weights.

To fetch the tokenizer without the weights:

```bash
uv run hf download IFM/K2-Horizon-7B --revision 4622f94a6632384784471fd674d5b78f8d527798 \
  --include tokenizer.json --local-dir hf/K2-Horizon-V-init
```

Then prepare an image conversation on CPU:

```python
from PIL import Image
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "hf/K2-Horizon-V-init", trust_remote_code=True, local_files_only=True
)
messages = [{"role": "user", "content": [
    {"type": "image", "image": Image.new("RGB", (448, 448))},
    {"type": "text", "text": "Describe this image."},
]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt",
    add_generation_prompt=True,
)
```

This requires the tokenizer, not the model weights. Image processing defaults
to GLM's PIL backend with a cap of 1,024 image tokens. Run `uv run pytest`
for CPU tests with a small synthetic tokenizer and model.

Read the [post-training plan](docs/README.md) for implementation steps and
the [environment guide](docs/steps/00-environment.md) for the GPU setup.

## References

- [K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B)
- [Baseten post-training post](https://www.baseten.co/blog/glm-52-with-vision)
- [GLM-5.3-Flash Vision Head](https://huggingface.co/immanuelpeter/GLM-5.3-Flash-Vision)
