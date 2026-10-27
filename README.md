# K2-Horizon-V

A multimodal K2-Horizon-7B.

Following Baseten's post-training strategy to add vision to base LLMs, we use a GLM-5.3-Flash vision head and train a new projector.

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --locked
uv run python -c "import k2hv, k2hv_sglang, torch, transformers; print(transformers.__version__)"
```

The project uses Python 3.12. `torch` comes from PyPI: the CUDA build on Linux
(it also runs on CPU-only machines, at about 5 GB for the venv) and the CPU/MPS
build on macOS. GPU nodes reuse the Miles image's torch and do not sync this
lock ([environment guide](docs/steps/00-environment.md)). No GPU or model
downloads are needed for setup. The init-checkpoint builder (`k2hv.build_init`),
its verifier (`k2hv.verify_init`) and the composite model are implemented. The
processor, chat template, SGLang model and training code still need
implementing; see the [step 1 handoff](docs/handoff/01-init-checkpoint.md).

## Layout

```text
hf/K2-Horizon-V-init/   the Hugging Face repo working tree (what `hf upload` would push)
                        tracked: model + config code, README, NOTICE, licenses, config.json, mirrored upstream assets
                        git-ignored: weights, index, tokenizer.json, build_info.json, SHA256SUMS
src/k2hv/hf             symlink to hf/K2-Horizon-V-init, so its code imports as `k2hv.hf.*`
src/k2hv/               build_init.py, verify_init.py, checkpoint_io.py (training glue lands here later)
src/k2hv_sglang/        SGLang model package (plan step 3)
tests/                  CPU tests on tiny configs; no weights needed
models/                 git-ignored source checkpoints (K2-Horizon-7B, GLM-5.3-Flash-Vision)
docs/                   the plan, and handoff notes
```

A file is tracked if we wrote or changed it. Weights and untouched large upstream files are not.
On a machine without the weights, fetch the one large tokenizer file the code needs:
`uv run hf download IFM/K2-Horizon-7B --revision 4622f94a6632384784471fd674d5b78f8d527798 --include tokenizer.json --local-dir hf/K2-Horizon-V-init`.

Read the [post-training plan](docs/README.md) for implementation steps and
the [environment guide](docs/steps/00-environment.md) for the GPU setup.

## References

- [K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B)
- [Baseten post-training post](https://www.baseten.co/blog/glm-52-with-vision)
- [GLM-5.3-Flash Vision Head](https://huggingface.co/immanuelpeter/GLM-5.3-Flash-Vision)
