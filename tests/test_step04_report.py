import json

from k2hv.gate_provenance import capture
from k2hv.step04_report import assemble


def test_aggregate_rejects_unrun_gates_and_changed_model_sources(tmp_path):
    hashes = capture()["source_sha256"]
    hashes["src/k2hv_sglang/cache_gate.py"] = "unused historical evidence CLI"
    hashes["src/k2hv/step04_report.py"] = "unused historical assembler"
    report = {"code_sha256": hashes, "results": [{"gate": name, "status": "passed"} for name in ("cuda", "load_hf", "text_hf", "vision_hf")]}
    path = tmp_path / "hf.json"
    path.write_text(json.dumps(report))
    assembled = assemble(tmp_path)
    assert not assembled["step04_complete"]
    assert assembled["results"][0]["status"] == "passed"
    assert any(row["status"] == "blocked" for row in assembled["results"])
    hashes["hf/K2-Horizon-V-init/modeling_k2_horizon_v.py"] = "changed model computation"
    path.write_text(json.dumps(report))
    row = assemble(tmp_path)["results"][0]
    assert row["status"] == "failed" and "Stale executed source" in row["error"]
