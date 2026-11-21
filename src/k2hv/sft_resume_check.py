"""Compare an uninterrupted Miles SFT run with a saved-and-restarted run of the same settings.

Inputs are two checkpoint roots (``--save`` directories), their logs and monitor directories, the
restart iteration and the final iteration. The restarted run's first leg (``--first``) must have
saved iteration ``--restart``; the second leg (``--resumed``) loaded it and continued to ``--final``.
Missing resume state, a failed or skipped load, or a mismatch outside the tolerances file fails.
"""

import argparse
import json
import math
import tempfile
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from .gate_provenance import capture
from .sft_eval import projector_state
from .sft_monitor import jsonl, parse_log

REQUIRED = ("model", "optimizer", "lr_scheduler", "rng.pt", "meta.json")


def iteration_dir(root, iteration):
    return Path(root) / f"iter_{iteration:07d}"


def dcp_dict(directory):
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "state.pt"
        dcp_to_torch_save(str(directory), str(target))
        return torch.load(target, weights_only=False)


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}{key}/"))
        return out
    if isinstance(value, (list, tuple)):
        out = {}
        for i, item in enumerate(value):
            out.update(flatten(item, f"{prefix}{i}/"))
        return out
    return {prefix.rstrip("/"): value}


def equal(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    return a == b


def tensor_diff(a, b):
    a, b = a.double(), b.double()
    return {"max_abs": float((a - b).abs().max()) if a.numel() else 0.0, "rel_frobenius": float((a - b).norm() / max(float(a.norm()), 1e-30))}


def compare_state(a, b, moment_tolerance):
    """Exact for non-tensors and step counters; tensors report exact equality and bounded differences."""
    fa, fb = flatten(a), flatten(b)
    if set(fa) != set(fb):
        return {"passed": False, "key_difference": sorted(set(fa) ^ set(fb))[:10]}
    exact = all(equal(fa[k], fb[k]) for k in fa)
    worst = {"max_abs": 0.0, "rel_frobenius": 0.0}
    bad = []
    for key in fa:
        if isinstance(fa[key], torch.Tensor) and fa[key].is_floating_point() and fa[key].numel() > 1:
            d = tensor_diff(fa[key], fb[key])
            worst = {k: max(worst[k], d[k]) for k in worst}
            if moment_tolerance is None or d["rel_frobenius"] > moment_tolerance:
                if not torch.equal(fa[key], fb[key]):
                    bad.append(key)
        elif not equal(fa[key], fb[key]):
            bad.append(key)
    return {"passed": not bad, "exact": exact, "entries": len(fa), "worst": worst, "mismatched": bad[:20]}


def compare_iteration(root_a, root_b, iteration, tolerances):
    a, b = iteration_dir(root_a, iteration), iteration_dir(root_b, iteration)
    missing = [str(d / name) for d in (a, b) for name in REQUIRED if not (d / name).exists()]
    if missing:
        return {"passed": False, "missing": missing}
    pa, pb = projector_state(a), projector_state(b)
    projector = {name: {"exact": torch.equal(pa[name], pb[name]), **tensor_diff(pa[name], pb[name])} for name in pa}
    projector_ok = all(v["max_abs"] <= tolerances["projector_max_abs"] and v["rel_frobenius"] <= tolerances["projector_rel_frobenius"] for v in projector.values())
    meta_a, meta_b = (json.loads((d / "meta.json").read_text()) for d in (a, b))
    meta_keys = ("iteration", "rollout_id", "next_rollout_id", "global_step", "micro_step", "world_size")
    rng_a, rng_b = (torch.load(d / "rng.pt", weights_only=False) for d in (a, b))
    rng_exact = torch.equal(rng_a["torch"], rng_b["torch"]) and len(rng_a["cuda"]) == len(rng_b["cuda"]) and all(torch.equal(x, y) for x, y in zip(rng_a["cuda"], rng_b["cuda"]))
    data_a, data_b = (torch.load(Path(r) / "rollout" / str(iteration - 1) / "data_source" / "state.pt", weights_only=False) for r in (root_a, root_b))
    result = {
        "iteration": iteration,
        "projector": projector,
        "projector_exact": all(v["exact"] for v in projector.values()),
        "projector_within_tolerance": projector_ok,
        "optimizer": compare_state(dcp_dict(a / "optimizer"), dcp_dict(b / "optimizer"), tolerances["adam_moment_rel_frobenius"]),
        "lr_scheduler": compare_state(dcp_dict(a / "lr_scheduler"), dcp_dict(b / "lr_scheduler"), None),
        "meta_equal": {k: meta_a.get(k) == meta_b.get(k) for k in meta_keys},
        "meta": {k: [meta_a.get(k), meta_b.get(k)] for k in meta_keys},
        "rng_exact": rng_exact,
        "data_source": {"a": data_a, "b": data_b, "equal": data_a == data_b},
    }
    result["passed"] = (
        projector_ok and result["optimizer"]["passed"] and result["lr_scheduler"]["passed"] and all(result["meta_equal"].values()) and rng_exact and result["data_source"]["equal"]
    )
    return result


def compare_logs(control_log, resumed_log, control_monitor, resumed_monitor, restart, final, tolerances):
    control, _, _ = parse_log(control_log)
    resumed, _, events = parse_log(resumed_log)
    steps = list(range(restart, final))
    rows = []
    for step in steps:
        c, r = control.get(step), resumed.get(step)
        if c is None or r is None:
            rows.append({"step": step, "present": False})
            continue
        rows.append({
            "step": step,
            "loss": [c["train/loss"], r["train/loss"]],
            "grad_norm": [c["train/grad_norm"], r["train/grad_norm"]],
            "lr": [c.get("train/lr-pg_0"), r.get("train/lr-pg_0")],
            "loss_ok": abs(c["train/loss"] - r["train/loss"]) <= tolerances["loss_abs"],
            "grad_norm_ok": abs(c["train/grad_norm"] - r["train/grad_norm"]) <= tolerances["grad_norm_rel"] * abs(c["train/grad_norm"]),
            "lr_exact": c.get("train/lr-pg_0") == r.get("train/lr-pg_0"),
            "exact": c["train/loss"] == r["train/loss"] and c["train/grad_norm"] == r["train/grad_norm"],
        })
    ids = {}
    for name, directory in (("control", control_monitor), ("resumed", resumed_monitor)):
        ids[name] = {rec["rollout_id"]: [s["record_id"] for s in rec["samples"]] for rec in jsonl(Path(directory) / "rollout.jsonl")}
    # Miles prefetches rollout <restart> before saving <restart>, so the restarted run replays that batch from the
    # executor snapshot instead of calling the rollout function; it has no rollout record of its own.
    replayed = restart not in ids["resumed"]
    batches = {step: ids["control"].get(step) == ids["resumed"].get(step) and ids["control"].get(step) is not None for step in steps if not (replayed and step == restart)}
    forwards = {}
    for name, directory in (("control", control_monitor), ("resumed", resumed_monitor)):
        forwards[name] = [r for r in jsonl(Path(directory) / "trainer-rank0.jsonl") if r["kind"] == "forward"]
    per_step = len(forwards["control"]) // max(1, len(control))
    projector = {}
    for step in range(restart, final + 1):
        c_index, r_index = step * per_step, (step - restart) * per_step
        if c_index < len(forwards["control"]) and r_index < len(forwards["resumed"]):
            c, r = forwards["control"][c_index], forwards["resumed"][r_index]
            projector[step] = {"tokens_equal": c["packed_tokens"] == r["packed_tokens"] and c["image_tokens"] == r["image_tokens"], "sumsq_equal": c["projector_local_sumsq"] == r["projector_local_sumsq"]}
    first = rows[0] if rows else {}
    exactness = {
        "replayed_restart_batch": replayed,
        "restored_projector_equals_control_at_restart": projector.get(restart, {}).get("sumsq_equal"),
        "first_resumed_update_loss_and_grad_norm_bitwise": first.get("exact"),
        "projector_after_first_resumed_update_equal": projector.get(restart + 1, {}).get("sumsq_equal"),
        "microbatch_tokens_equal_all_steps": all(v["tokens_equal"] for v in projector.values()),
        "projector_sumsq_equal_by_step": {k: v["sumsq_equal"] for k, v in projector.items()},
        "first_bitwise_divergent_step": next((r["step"] for r in rows if not r.get("exact")), None),
    }
    resumed_steps_before = sorted(s for s in resumed if s < restart)
    restart_dir = f"iter_{restart:07d}"
    loads = {kind: any(f"Loaded {kind} from" in e and restart_dir in e for e in events) for kind in ("model", "optimizer", "LR scheduler")}
    problems = [e for e in events if "Failed" in e or "not found" in e or "skipping" in e]
    return {
        "steps": rows,
        "all_steps_present": all(r.get("present", True) for r in rows),
        "loss_within_tolerance": all(r.get("loss_ok") for r in rows),
        "grad_norm_within_tolerance": all(r.get("grad_norm_ok") for r in rows),
        "lr_exact": all(r.get("lr_exact") for r in rows),
        "losses_bitwise_equal": all(r.get("exact") for r in rows),
        "first_resumed_step": rows[0] if rows else None,
        "batch_ids_equal": batches,
        "exactness": exactness,
        "resumed_trained_before_restart": resumed_steps_before,
        "resume_loads": loads,
        "load_problems": problems,
        "passed": all(r.get("present", True) for r in rows)
        and all(r.get("loss_ok") and r.get("grad_norm_ok") and r.get("lr_exact") for r in rows)
        and all(batches.values())
        and not resumed_steps_before
        and all(loads.values())
        and not problems,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True, help="uninterrupted run's checkpoint root")
    p.add_argument("--first", type=Path, required=True, help="first leg's checkpoint root (saved --restart)")
    p.add_argument("--resumed", type=Path, required=True, help="second leg's checkpoint root")
    p.add_argument("--control-log", type=Path, required=True)
    p.add_argument("--resumed-log", type=Path, required=True)
    p.add_argument("--control-monitor", type=Path, required=True)
    p.add_argument("--resumed-monitor", type=Path, required=True)
    p.add_argument("--restart", type=int, required=True)
    p.add_argument("--final", type=int, required=True)
    p.add_argument("--tolerances", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    tolerances = json.loads(a.tolerances.read_text())["bounded_resume_vs_control"]
    report = {
        "provenance": capture(),
        "tolerances": tolerances,
        "restart_state_vs_control": compare_iteration(a.control, a.first, a.restart, tolerances),
        "final_state_vs_control": compare_iteration(a.control, a.resumed, a.final, tolerances),
        "logs": compare_logs(a.control_log, a.resumed_log, a.control_monitor, a.resumed_monitor, a.restart, a.final, tolerances),
    }
    report["passed"] = all(report[k]["passed"] for k in ("restart_state_vs_control", "final_state_vs_control", "logs"))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=1, default=str) + "\n")
    summary = {k: report[k]["passed"] for k in ("restart_state_vs_control", "final_state_vs_control", "logs")}
    print(json.dumps({"passed": report["passed"], **summary}, indent=1))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
