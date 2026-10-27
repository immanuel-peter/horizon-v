import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import tiny
from k2hv import build_init, verify_init
from k2hv.checkpoint_io import hash_tree, read_tensor_infos, split_glm_keys

TOKENS = {"<image>": tiny.IMAGE_START, "</image>": tiny.IMAGE_END, "reserved_special_token_12": tiny.IMAGE_PAD}
SHARD_LIMIT = 100_000  # bytes; the tiny model is ~330 KB, so this forces several shards
SRC = Path(__file__).resolve().parents[1] / "src" / "k2hv"


def build_tiny(tmp_path, *, glm_sharded=False, name="out", sources=None, **kwargs):
    k2, glm = sources or tiny.write_source_checkpoints(tmp_path / "src", glm_sharded=glm_sharded)
    out = tmp_path / name
    info = build_init.build(k2, glm, out, max_shard_size=SHARD_LIMIT, token_ids=TOKENS, **kwargs)
    return k2, glm, out, info


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_tiny(tmp_path_factory.mktemp("built"))


def test_vendored_k2_code_is_unmodified():
    """Pinned to the hashes IFM recorded in K2-Horizon-7B's migration_manifest.json (target_hashes)."""
    expected = {
        "configuration_k2_horizon.py": "5c2f993c1053d9462ebea6dea416c897fddfbb4a5edd904e486936b20d4badc5",
        "modeling_k2_horizon.py": "fb09e010956bd51cfa7d4055b4381cff34c9e06164066b49e3546f38b2e6242f",
    }
    assert {n: hashlib.sha256((SRC / n).read_bytes()).hexdigest() for n in expected} == expected


# --- successful builds ---------------------------------------------------------------------------


@pytest.mark.parametrize("glm_sharded", [False, True], ids=["glm-single-file", "glm-sharded"])
def test_build_covers_every_key_and_preserves_frozen_bytes(tmp_path, glm_sharded):
    k2, glm, out, info = build_tiny(tmp_path, glm_sharded=glm_sharded)
    k2_infos, glm_infos, out_infos = read_tensor_infos(k2), read_tensor_infos(glm), read_tensor_infos(out)
    kept, excluded = split_glm_keys(glm_infos)

    assert len(excluded) == 8 and all(k.startswith(("downsample.", "merger.")) for k in excluded)
    assert set(out_infos) == (
        {"language_model." + k for k in k2_infos}
        | {"visual." + k for k in kept}
        | {f"projector.{n}" for n in ("norm.weight", "norm.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")}
    )
    assert info["tensors"] == len(out_infos) and info["dtype_conversions"] == 0
    assert sorted(info["sources"]["glm"]["excluded_tensors"]) == sorted(excluded)

    shards = sorted(out.glob("model-*.safetensors"))
    assert len(shards) > 1
    assert all(s.stat().st_size <= SHARD_LIMIT for s in shards)
    assert all(s.stat().st_mode & 0o044 for s in shards)  # group/other-readable like every other file
    assert {i.dtype for i in out_infos.values()} == {"BF16"}
    for key, src in [("language_model." + k, k2_infos[k]) for k in k2_infos] + [("visual." + k, glm_infos[k]) for k in kept]:
        assert verify_init._compare_ranges(out_infos[key], src), key  # raw bytes identical, not just numerically equal
    assert not (tmp_path / "out.building").exists()


def test_static_verifier_passes_on_a_good_build(built):
    k2, glm, out, _ = built
    report = verify_init.Report()
    config = verify_init.check_layout(out, k2, glm, report, token_ids=TOKENS)
    infos = verify_init.check_shards(out, report)
    k2_infos, glm_infos = verify_init.check_key_coverage(infos, k2, glm, config, report)
    verify_init.check_frozen_bytes(infos, k2_infos, glm_infos, report)
    verify_init.check_projector(out, infos, config, report)
    verify_init.check_manifest(out, report)
    failed = [c for c in report.checks if not c["ok"]]
    assert not failed, failed
    assert len(report.checks) >= 15


def test_verifier_catches_corruption(built, tmp_path):
    k2, glm, out, _ = built
    # flip one byte inside a frozen tensor of a copy
    copy = tmp_path / "copy"
    copy.mkdir()
    for p in out.rglob("*"):
        if p.is_file():
            (copy / p.relative_to(out)).parent.mkdir(parents=True, exist_ok=True)
            (copy / p.relative_to(out)).write_bytes(p.read_bytes())
    infos = read_tensor_infos(copy)
    victim = infos["language_model.model.layers.0.mlp.up_proj.weight"]
    data = bytearray(victim.file.read_bytes())
    data[victim.start + 3] ^= 0x01
    victim.file.write_bytes(bytes(data))

    report = verify_init.Report()
    verify_init.check_frozen_bytes(infos, read_tensor_infos(k2), read_tensor_infos(glm), report)
    verify_init.check_manifest(copy, report)
    assert [c["ok"] for c in report.checks] == [False, False]

    # shipped model code drifting from the package is flagged ("rebuild")
    (copy / "modeling_k2_horizon_v.py").write_text((copy / "modeling_k2_horizon_v.py").read_text() + "\n# drift\n")
    report = verify_init.Report()
    verify_init.check_layout(copy, k2, glm, report, token_ids=TOKENS)
    stale = [c for c in report.checks if c["name"].startswith("layout: shipped code")]
    assert stale and not stale[0]["ok"] and "modeling_k2_horizon_v.py" in stale[0]["detail"]


def test_build_is_deterministic_across_runs(tmp_path):
    sources = tiny.write_source_checkpoints(tmp_path / "src")
    _, _, a, _ = build_tiny(tmp_path, name="a", sources=sources)
    _, _, b, _ = build_tiny(tmp_path, name="b", sources=sources)
    ha, hb = hash_tree(a), hash_tree(b)
    differing = {k for k in ha if ha[k] != hb[k]}
    assert differing <= {"build_info.json", "SHA256SUMS"}  # only the timing/memory record differs


def test_different_seed_changes_only_the_projector(tmp_path):
    sources = tiny.write_source_checkpoints(tmp_path / "src")
    _, _, a, _ = build_tiny(tmp_path, name="a", sources=sources, seed=0)
    _, _, b, _ = build_tiny(tmp_path, name="b", sources=sources, seed=1)
    infos_a, infos_b = read_tensor_infos(a), read_tensor_infos(b)
    for key in infos_a:
        same = verify_init._compare_ranges(infos_a[key], infos_b[key])
        assert same == (not key.startswith("projector.") or key.startswith("projector.norm.")), key  # LayerNorm init is seed-free


def test_offline_load_in_a_fresh_process_matches_stock_k2(built, tmp_path):
    k2, glm, out, _ = built
    script = textwrap.dedent(
        """
        import json, sys
        sys.modules["k2hv"] = None  # the checkpoint must not need this repository
        out, k2 = sys.argv[1:3]
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
        cfg = AutoConfig.from_pretrained(out, trust_remote_code=True, local_files_only=True)
        model, info = AutoModelForImageTextToText.from_pretrained(
            out, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, output_loading_info=True)
        stock = AutoModelForCausalLM.from_pretrained(k2, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16)
        ids = torch.randint(2, 290, (2, 11), generator=torch.Generator().manual_seed(0))
        with torch.inference_mode():
            same = torch.equal(model(input_ids=ids).logits, stock(input_ids=ids).logits)
        print(json.dumps({
            "config": f"{type(cfg).__module__}.{type(cfg).__name__}", "model": f"{type(model).__module__}.{type(model).__name__}",
            "clean": all(not v for v in info.values()), "info": {k: sorted(v) for k, v in info.items()},
            "k2hv_imported": sys.modules.get("k2hv") is not None, "logits_equal": same,
            "trainable": sorted(n for n, p in model.named_parameters() if p.requires_grad),
        }))
        """
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "HUGGING"))}
    env.update(HF_HUB_OFFLINE="1", HF_HOME=str(tmp_path / "hf"), HF_MODULES_CACHE=str(tmp_path / "modules"))
    proc = subprocess.run([sys.executable, "-I", "-c", script, str(out), str(k2)], env=env, capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    assert result["config"].startswith("transformers_modules.") and result["config"].endswith("K2HorizonVConfig")
    assert result["model"].endswith("K2HorizonVForConditionalGeneration")
    assert result["clean"], result["info"]
    assert not result["k2hv_imported"] and result["logits_equal"]
    assert result["trainable"] == sorted(f"projector.{n}" for n in ("norm.weight", "norm.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"))
    cached = {p.name for p in (tmp_path / "modules").rglob("*.py")}
    assert {"configuration_k2_horizon_v.py", "modeling_k2_horizon_v.py", "configuration_k2_horizon.py", "modeling_k2_horizon.py"} <= cached


# --- refusals and failures -----------------------------------------------------------------------


def test_refuses_to_overwrite_an_existing_output(tmp_path):
    k2, glm = tiny.write_source_checkpoints(tmp_path / "src")
    out = tmp_path / "out"
    out.mkdir()
    (out / "precious.txt").write_text("keep me")
    with pytest.raises(build_init.BuildError, match="refusing to overwrite"):
        build_init.build(k2, glm, out, token_ids=TOKENS)
    assert (out / "precious.txt").read_text() == "keep me"
    assert not (tmp_path / "out.building").exists()


def test_refuses_when_a_previous_partial_build_is_present(tmp_path):
    k2, glm = tiny.write_source_checkpoints(tmp_path / "src")
    (tmp_path / "out.building").mkdir()
    with pytest.raises(build_init.BuildError, match="earlier failed build"):
        build_init.build(k2, glm, tmp_path / "out", token_ids=TOKENS)


def _rewrite_shard(directory, transform):
    for shard in directory.glob("*.safetensors"):
        with safe_open(shard, "pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        save_file(transform(tensors), shard, metadata={"format": "pt"})


def test_dtype_mismatch_is_an_error_not_a_silent_cast(tmp_path):
    k2, glm = tiny.write_source_checkpoints(tmp_path / "src")
    key = "model.layers.1.input_layernorm.weight"
    _rewrite_shard(k2, lambda t: {k: v.float() if k == key else v for k, v in t.items()})
    with pytest.raises(build_init.BuildError, match=rf"{key}: source dtype F32, checkpoint dtype BF16"):
        build_init.build(k2, glm, tmp_path / "out", token_ids=TOKENS)
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.building").exists()


def test_missing_and_unexpected_tensors_are_errors(tmp_path):
    k2, glm = tiny.write_source_checkpoints(tmp_path / "src", glm_sharded=False)
    save_file({k: v for k, v in _read_all(glm).items() if k != "blocks.1.attn.qkv.bias"}, glm / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(build_init.BuildError, match="expected tensors missing from source"):
        build_init.build(k2, glm, tmp_path / "out", token_ids=TOKENS)

    k2b, glmb = tiny.write_source_checkpoints(tmp_path / "src2")
    save_file({**_read_all(glmb), "blocks.0.stray.weight": torch.zeros(2, dtype=torch.bfloat16)}, glmb / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(build_init.BuildError, match="source tensors the model does not use"):
        build_init.build(k2b, glmb, tmp_path / "out2", token_ids=TOKENS)


def _read_all(directory):
    with safe_open(directory / "model.safetensors", "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def test_wrong_token_id_is_an_error(tmp_path):
    k2, glm = tiny.write_source_checkpoints(tmp_path / "src")
    wrong = {**TOKENS, "<image>": tiny.IMAGE_START + 1}
    with pytest.raises(build_init.BuildError, match=r"maps '<image>' to 300, the plan expects 301"):
        build_init.build(k2, glm, tmp_path / "out", token_ids=wrong)


def test_modified_k2_code_is_an_error(tmp_path):
    k2, glm = tiny.write_source_checkpoints(tmp_path / "src")
    (k2 / "modeling_k2_horizon.py").write_text((k2 / "modeling_k2_horizon.py").read_text() + "\n# local edit\n")
    with pytest.raises(build_init.BuildError, match="differs from the copy vendored"):
        build_init.build(k2, glm, tmp_path / "out", token_ids=TOKENS)


def test_unsharded_source_without_weights_is_an_error(tmp_path):
    k2, glm = tiny.write_source_checkpoints(tmp_path / "src")
    (glm / "model.safetensors").unlink()
    with pytest.raises(Exception, match="neither"):
        build_init.build(k2, glm, tmp_path / "out", token_ids=TOKENS)


def test_cli_reports_errors_with_a_nonzero_exit(tmp_path, capsys):
    k2, glm = tiny.write_source_checkpoints(tmp_path / "src")
    (tmp_path / "out").mkdir()
    assert build_init.main(["--k2", str(k2), "--glm", str(glm), "--out", str(tmp_path / "out")]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err


# --- shard planner -------------------------------------------------------------------------------


def test_plan_shards_packs_in_order_within_the_limit():
    entries = [(f"t{i}", 400, 2) for i in range(10)]
    shards = build_init.plan_shards(entries, limit=4096 + 3 * (400 + build_init.header_bound("t0", 2)))
    assert [len(s) for s in shards] == [3, 3, 3, 1]
    assert [k for s in shards for k in s] == [e[0] for e in entries]


def test_plan_shards_rejects_a_tensor_larger_than_the_limit():
    with pytest.raises(build_init.BuildError, match="cannot fit"):
        build_init.plan_shards([("huge", 10_000, 2)], limit=5_000)


def test_real_scale_plan_fits_five_gigabyte_shards():
    sizes = [(f"layer{i}", 385_893_544, 2) for i in range(36)] + [("embed", 2_051_080_192, 2), ("lm_head", 2_051_080_192, 2)]
    shards = build_init.plan_shards(sizes, limit=5_000_000_000)
    assert len(shards) == 4
    for shard in shards:
        assert sum(dict((k, n) for k, n, _ in sizes)[k] for k in shard) < 5_000_000_000
