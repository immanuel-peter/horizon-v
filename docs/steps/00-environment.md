# Step 0: Environment (uv)

The project uses uv for its code, its dependencies and every command. There are two environments:

The CPU environment, composite model, checkpoint tools and processor are
implemented. The training modules shown below are planned. Add the Miles
checkout when setting up a GPU node.

| Where | CUDA stack (torch, SGLang, flash-attn) | Project code and light deps |
|---|---|---|
| Laptop / CPU box | Not needed | `uv sync` from `uv.lock` |
| GPU nodes | Miles's Docker image | uv venv on top of the image |

## Layout

```text
horizon-v/
├── pyproject.toml
├── uv.lock
├── .python-version          3.12
├── third_party/miles/       radixark/miles @ 5078ff7, editable (patched in step 8)
├── hf/K2-Horizon-V-init/    the HF repo tree: tracked code, docs, configs; weights are git-ignored
├── src/k2hv/                build_init.py, verify_init.py, sft_rollout.py, reward.py, rl360_reward/
│   └── hf -> ../../hf/K2-Horizon-V-init     symlink: imports as k2hv.hf.*
├── src/k2hv_sglang/         SGLang model package (step 3)
└── docs/                    this plan
```

## `pyproject.toml`

```toml
[project]
name = "k2-horizon-v"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "huggingface_hub",
    "pillow",
    "safetensors",
    "torch",
    "transformers==5.19.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
exclude = ["*.safetensors", "tokenizer.json", "model.safetensors.index.json", "build_info.json", "SHA256SUMS", ".cache", ".building"]

[tool.hatch.build.targets.wheel]
only-include = ["src/k2hv", "src/k2hv_sglang", "hf/K2-Horizon-V-init"]
sources = {"src" = "", "hf/K2-Horizon-V-init" = "k2hv/hf"}
dev-mode-dirs = ["src"]
```

- `transformers==5.19.0` matches Miles's pin.
- SGLang, flash-attn and Miles are not listed. On GPU nodes they come from the image ([below](#gpu-nodes)); the laptop does not need them.

## Laptop / CPU box

```bash
uv sync
```

Use this environment to [build the checkpoint](01-init-checkpoint.md) and test
the [template and processor](02-processor-and-template.md). The checkpoint build
peaked at about 18 GiB RSS.

PyPI supplies the CUDA build of `torch` on Linux and the CPU/MPS build on macOS.
The Linux build runs on CPU machines and uses about 5 GB for the venv.
`uv sync` installs `pytest` from the `dev` dependency group.

## GPU nodes

Miles uses the `sglang-miles` branch and prebuilt flash-attn / FA3 wheels from its Dockerfile. That branch moves unless the build sets `SGLANG_COMMIT`. The adapter targets the revision in the [SGLang handoff](../handoff/03-sglang-model.md), so pin or audit the installed source before serving. A PyPI SGLang resolved by uv may not match that runtime. Start from the Miles image and add the project on top:

```bash
git clone https://github.com/radixark/miles third_party/miles
git -C third_party/miles checkout 5078ff7

uv venv --system-site-packages --python "$(which python3)"
uv pip install --no-deps -e third_party/miles -e .

export UV_NO_SYNC=1
source .venv/bin/activate
```

- `--system-site-packages` keeps the image's torch, SGLang fork and flash-attn visible.
- `--no-deps` stops uv from replacing them with PyPI builds.
- The editable `third_party/miles` checkout shadows the image's copy, so the weight-sync patch in [step 8](08-rl.md) applies.
- `UV_NO_SYNC=1` stops `uv run` from syncing `uv.lock` into this venv, which would install a second torch.
- Activate the venv before `ray start`, so Ray and its workers inherit it. Later steps assume an activated venv on GPU nodes.
