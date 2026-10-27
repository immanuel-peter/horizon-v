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

Setup needs no GPU or model downloads. The composite model, checkpoint builder
and verifier are implemented. The processor, image chat template, SGLang model
and training code are planned. See the [step 1 handoff](docs/handoff/01-init-checkpoint.md).

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

Read the [post-training plan](docs/README.md) for implementation steps and
the [environment guide](docs/steps/00-environment.md) for the GPU setup.

## References

- [K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B)
- [Baseten post-training post](https://www.baseten.co/blog/glm-52-with-vision)
- [GLM-5.3-Flash Vision Head](https://huggingface.co/immanuelpeter/GLM-5.3-Flash-Vision)
