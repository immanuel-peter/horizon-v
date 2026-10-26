# K2-Horizon-V

A multimodal K2-Horizon-7B.

Following Baseten's post-training strategy to add vision to base LLMs, we use a GLM-5.3-Flash vision head and train a new projector.

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --locked
uv run python -c "import k2hv, k2hv_sglang, torch, transformers; print(transformers.__version__)"
```

The project uses Python 3.12. This installs the CPU development environment;
no GPU or model downloads are needed for setup. The package directories are
in place; the model, processor, and training code still need implementing.

Read the [post-training plan](docs/README.md) for implementation steps and
the [environment guide](docs/steps/00-environment.md) for the GPU setup.

## References

- [K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B)
- [Baseten post-training post](https://www.baseten.co/blog/glm-52-with-vision)
- [GLM-5.3-Flash Vision Head](https://huggingface.co/immanuelpeter/GLM-5.3-Flash-Vision)
