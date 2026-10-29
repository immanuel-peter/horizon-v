import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def archive_verifier():
    spec = importlib.util.spec_from_file_location("verify_archive", Path(__file__).parents[1] / "tools/step04/verify_archive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify


def test_archive_rejects_missing_coverage(tmp_path, archive_verifier):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"tensor": "model.safetensors"}}))
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text("0" * 64 + "  tokenizer.json\n")
    with pytest.raises(ValueError, match="coverage"):
        archive_verifier(tmp_path, manifest)


def test_archive_verifies_only_generated_assets_and_rejects_corruption(tmp_path, archive_verifier):
    files = {"build_info.json": b"{}", "tokenizer.json": b"{}", "model.safetensors": b"weights", "model.safetensors.index.json": json.dumps({"weight_map": {"tensor": "model.safetensors"}}).encode()}
    lines = []
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
        lines.append(f"{hashlib.sha256(content).hexdigest()}  {name}")
    lines.append("0" * 64 + "  config.json")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text("\n".join(lines))
    assert set(archive_verifier(tmp_path, manifest)) == set(files)
    (tmp_path / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        archive_verifier(tmp_path, manifest)
