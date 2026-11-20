"""Real-data SFT instrumentation, enabled by ``K2HV_SFT_MONITOR=<directory>``.

Nothing here changes the training computation. ``generate_rollout`` wraps
``k2hv.sft_rollout.generate_rollout``: it times lazy decoding (``get_samples``) separately from
processing, checks every prepared sample (one leading BOS, image-pad count equal to the grid's merged
patch count, one start/end marker pair per image, a supervised span of answer plus end only), and
records rollout-process RSS and live image/tensor objects at each call. ``install_trainer_hooks``
records per-rank GPU and host memory, packed token mix and projector gradient norms on training
forwards. ``sample`` polls NVML and process RSS; ``summarize`` turns the logs into one report.

Records are JSON lines under the monitor directory: ``rollout.jsonl``, ``trainer-rank<r>.jsonl``,
``trainer-rank<r>-setup.json`` and ``sampler.jsonl``.
"""

import argparse
import ast
import gc
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

ENV = "K2HV_SFT_MONITOR"
CLOSE_THINK = "</ifm|think>"
END = "<|ifm|im_end|>"


def directory():
    path = Path(os.environ[ENV])
    path.mkdir(parents=True, exist_ok=True)
    return path


def append(name, record):
    with (directory() / name).open("a") as file:
        file.write(json.dumps(record) + "\n")


def rss_bytes(pid=None):
    import psutil

    return psutil.Process(pid).memory_info().rss


def live_objects():
    from PIL import Image
    import torch

    start = time.perf_counter()
    images = tensors = tensor_bytes = 0
    for obj in gc.get_objects():
        if isinstance(obj, Image.Image):
            images += 1
        elif isinstance(obj, torch.Tensor):
            tensors += 1
            if obj.device.type == "cpu":
                tensor_bytes += obj.untyped_storage().nbytes()
    return {"pil_images": images, "cpu_tensors": tensors, "cpu_tensor_bytes": tensor_bytes, "scan_s": time.perf_counter() - start}


class TimedSource:
    """Proxy for the data source that times ``get_samples``, where the lazy source decodes images."""

    def __init__(self, source):
        self._source = source
        self.seconds = 0.0

    def get_samples(self, num_samples):
        start = time.perf_counter()
        try:
            return self._source.get_samples(num_samples)
        finally:
            self.seconds += time.perf_counter() - start

    def __getattr__(self, name):
        return getattr(self._source, name)


def token_ids(processor, checkpoint):
    tokenizer = processor.tokenizer
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    return {
        "bos": tokenizer.bos_token_id,
        "pad": config["image_token_id"],
        "start": config["image_start_token_id"],
        "end": config["image_end_token_id"],
        "close": tokenizer.encode(CLOSE_THINK, add_special_tokens=False),
        "im_end": tokenizer.convert_tokens_to_ids(END),
    }


def inspect_sample(processor, ids, sample):
    """Check one prepared sample; raise on any violation and return its token counts."""
    tokens = sample.tokens
    grid = sample.multimodal_train_inputs["image_grid_thw"]
    images = len(sample.multimodal_inputs["images"])
    merged = int(grid.prod(-1).sum()) // 4
    response = sample.response_length
    supervised = tokens[-response:]
    record_id = (sample.metadata or {}).get("record_id")
    checks = {
        "one_leading_bos": tokens[0] == ids["bos"] and tokens.count(ids["bos"]) == 1,
        "image_pads_match_grid": tokens.count(ids["pad"]) == merged,
        "image_markers_per_image": tokens.count(ids["start"]) == images == tokens.count(ids["end"]) == len(grid),
        "pixel_rows_match_grid": sample.multimodal_train_inputs["pixel_values"].shape[0] == int(grid.prod(-1).sum()),
        "mask_is_response": list(sample.loss_mask) == [1] * response and 0 < response < len(tokens),
        "span_follows_close_think": tokens[-response - len(ids["close"]) : -response] == ids["close"],
        "span_ends_with_im_end": supervised[-1] == ids["im_end"] and supervised.count(ids["im_end"]) == 1,
        "span_has_no_image_or_bos": not ({ids["pad"], ids["start"], ids["end"], ids["bos"]} & set(supervised)),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"SFT sample {record_id} failed {failed}")
    return {"record_id": record_id, "tokens": len(tokens), "image_tokens": merged, "supervised_tokens": response, "images": images}


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    from k2hv import sft_rollout

    before_rss = rss_bytes()
    before_live = live_objects()
    source = TimedSource(data_buffer)
    start = time.perf_counter()
    samples = sft_rollout.generate_rollout(args, rollout_id, source, evaluation)
    total = time.perf_counter() - start
    ids = token_ids(sft_rollout.PROCESSOR, args.hf_checkpoint)
    rows = [inspect_sample(sft_rollout.PROCESSOR, ids, sample) for (sample,) in samples]
    dataset = getattr(data_buffer, "dataset", None)
    append("rollout.jsonl", {
        "rollout_id": rollout_id,
        "pid": os.getpid(),
        "time": time.time(),
        "get_samples_s": source.seconds,
        "process_s": total - source.seconds,
        "rss_before": before_rss,
        "rss_after": rss_bytes(),
        "live_before": before_live,
        "decoded_total": getattr(dataset, "decoded", None),
        "dataset_rows": len(dataset) if dataset is not None else None,
        "epoch_id": getattr(data_buffer, "epoch_id", None),
        "sample_offset": getattr(data_buffer, "sample_offset", None),
        "samples": rows,
    })
    return samples


def install_trainer_hooks(model, args):
    import torch
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    mode = os.environ.get("K2HV_SFT_DETERMINISTIC")
    if mode:
        # Diagnostic: report (warn) or refuse (strict) nondeterministic PyTorch kernels in the trainer.
        torch.use_deterministic_algorithms(True, warn_only=mode == "warn")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names or not all(n.startswith("projector.") for n in names):
        raise AssertionError(f"Unexpected trainable parameters: {names}")
    tokenizer_ids = None
    state = {"setup_written": False, "forwards": 0}

    def local(tensor):
        return tensor.to_local() if hasattr(tensor, "to_local") else tensor

    def setup():
        dtypes = {}
        for name, parameter in model.named_parameters():
            group = name.split(".")[0]
            dtypes.setdefault(group, {}).setdefault(str(parameter.dtype), 0)
            dtypes[group][str(parameter.dtype)] += 1
        trainable = {n: {"shape": list(p.shape), "dtype": str(p.dtype), "requires_grad": p.requires_grad} for n, p in model.named_parameters() if p.requires_grad}
        (directory() / f"trainer-rank{rank}-setup.json").write_text(json.dumps({
            "rank": rank,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "trainable": trainable,
            "parameter_dtypes_by_module": dtypes,
            "projector_local_numel": sum(local(p).numel() for p in model.projector.parameters()),
            "gradient_checkpointing": getattr(model, "is_gradient_checkpointing", None),
        }, indent=1))

    def forward(module, inputs, kwargs, output):
        nonlocal tokenizer_ids
        if not module.training or not torch.is_grad_enabled():
            return
        if not state["setup_written"]:
            setup()
            state["setup_written"] = True
        if tokenizer_ids is None:
            config = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())
            tokenizer_ids = config["image_token_id"]
        ids = kwargs["input_ids"]
        boundaries = kwargs.get("k2hv_cu_seqlens") or (0, ids.shape[-1])
        projector = [local(p).detach().float() for p in model.projector.parameters()]
        state["forwards"] += 1
        append(f"trainer-rank{rank}.jsonl", {
            "kind": "forward",
            "rank": rank,
            "forward": state["forwards"],
            "time": time.time(),
            "packed_tokens": ids.shape[-1],
            "documents": len(boundaries) - 1,
            "image_tokens": int((ids == tokenizer_ids).sum()),
            "has_pixels": kwargs.get("pixel_values") is not None,
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
            "rss": rss_bytes(),
            "projector_local_sumsq": sum(float(p.square().sum()) for p in projector),
            "projector_local_dtypes": sorted({str(local(p).dtype) for p in model.projector.parameters()}),
        })

    def backward(module, grad_input, grad_output):
        gradients = [g for g in grad_output if g is not None]
        norm = math.sqrt(sum(float(g.float().square().sum()) for g in gradients)) if gradients else 0.0
        append(f"trainer-rank{rank}.jsonl", {"kind": "projector_output_grad", "rank": rank, "time": time.time(), "norm": norm, "finite": math.isfinite(norm)})
        if not (math.isfinite(norm) and norm > 0):
            raise AssertionError(f"Projector output gradient is {norm}; the frozen LM did not pass gradients")

    model.register_forward_hook(forward, with_kwargs=True)
    model.projector.register_full_backward_hook(backward)
    if os.environ.get("K2HV_SFT_TRACE"):
        install_trace(model, rank)


def install_trace(model, rank):
    """Diagnostic: checksums of vision, projector and decoder outputs on training forwards."""
    import hashlib

    import torch

    def checksum(name):
        def hook(module, inputs, output):
            if not torch.is_grad_enabled() and name != "visual":
                return
            tensor = output.logits if hasattr(output, "logits") else output
            tensor = tensor[0] if isinstance(tensor, tuple) else tensor
            data = tensor.detach().float().cpu().numpy().tobytes() if tensor.numel() < 1 << 24 else repr(float(tensor.detach().double().sum())).encode()
            append(f"trace-rank{rank}.jsonl", {"module": name, "grad": torch.is_grad_enabled(), "shape": list(tensor.shape), "sha": hashlib.sha256(data).hexdigest()[:16]})

        return hook

    model.visual.register_forward_hook(checksum("visual"))
    model.projector.register_forward_hook(checksum("projector"))
    for index, layer in enumerate(model.language_model.model.layers):
        layer.register_forward_hook(checksum(f"layer{index}"))
    model.register_forward_hook(checksum("logits"))


def wrap_adapter():
    """Install the trainer hooks after ``K2HorizonVAdapter.patch_model`` when the monitor is enabled."""
    from k2hv.miles_gate_adapter import K2HorizonVAdapter

    if getattr(K2HorizonVAdapter.patch_model, "_k2hv_monitored", False):
        return
    original = K2HorizonVAdapter.patch_model

    def patch_model(self, model, args):
        original(self, model, args)
        install_trainer_hooks(model, args)

    patch_model._k2hv_monitored = True
    K2HorizonVAdapter.patch_model = patch_model


def sample(out, interval):
    import psutil

    with Path(out).open("a") as file:
        while True:
            gpus = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True
            ).stdout.strip().splitlines()
            processes = []
            for process in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
                info = process.info
                cmd = " ".join(info["cmdline"] or [])
                if not info["memory_info"] or not ("python" in cmd or cmd.startswith("ray::") or "raylet" in cmd or "gcs_server" in cmd):
                    continue
                processes.append({"pid": info["pid"], "name": (cmd.split()[0] if cmd else info["name"])[:80], "rss": info["memory_info"].rss})
            memory = psutil.virtual_memory()
            file.write(json.dumps({
                "time": time.time(),
                "gpus": [dict(zip(("index", "used_mib", "util"), map(int, line.split(", ")))) for line in gpus],
                "host_used": memory.total - memory.available,
                "processes": processes,
            }) + "\n")
            file.flush()
            time.sleep(interval)


def parse_log(path):
    steps, perf, lines = {}, {}, Path(path).read_text(errors="replace").splitlines()
    for line in lines:
        if match := re.search(r"\bstep (\d+): (\{.*\})", line):
            values = ast.literal_eval(match[2])
            if "train/loss" in values:
                steps[int(match[1])] = values
        elif match := re.search(r"\bperf (\d+): (\{.*\})", line):
            # The rollout executor and the trainer both log "perf <id>"; keep both sets of keys.
            perf.setdefault(int(match[1]), {}).update(ast.literal_eval(match[2]))
    events = [line for line in lines if re.search(r"\[FSDP\] (Loaded|Failed|Saved|.*not found|.*skipping)", line)]
    return steps, perf, events


def stats(values):
    values = list(values)
    if not values:
        return None
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "mean": statistics.fmean(values), "max": max(values)}


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summarize(run, log, warmup):
    steps, perf, events = parse_log(log)
    rollouts = jsonl(run / "rollout.jsonl")
    measured = sorted(perf)[warmup:]
    tokens_by_rollout = {r["rollout_id"]: sum(s["tokens"] for s in r["samples"]) for r in rollouts}
    supervised_by_rollout = {r["rollout_id"]: sum(s["supervised_tokens"] for s in r["samples"]) for r in rollouts}
    step_time = {i: perf[i]["perf/step_time"] for i in measured if "perf/step_time" in perf[i]}
    trainer = {}
    for path in sorted(run.glob("trainer-rank*.jsonl")):
        records = jsonl(path)
        forwards = [r for r in records if r["kind"] == "forward"]
        grads = [r for r in records if r["kind"] == "projector_output_grad"]
        trainer[path.stem] = {
            "training_forwards": len(forwards),
            "packed_tokens": stats(r["packed_tokens"] for r in forwards),
            "documents_per_microbatch": stats(r["documents"] for r in forwards),
            "peak_allocated_gib": max((r["max_allocated"] for r in forwards), default=0) / 2**30,
            "peak_reserved_gib": max((r["max_reserved"] for r in forwards), default=0) / 2**30,
            "allocated_gib_first_last": [forwards[0]["allocated"] / 2**30, forwards[-1]["allocated"] / 2**30] if forwards else None,
            "rss_gib_first_last_max": [forwards[0]["rss"] / 2**30, forwards[-1]["rss"] / 2**30, max(r["rss"] for r in forwards) / 2**30] if forwards else None,
            "projector_output_grad_norm": stats(r["norm"] for r in grads),
            "projector_output_grads_finite_nonzero": all(r["finite"] and r["norm"] > 0 for r in grads) and bool(grads),
            "projector_local_dtypes": sorted({d for r in forwards for d in r["projector_local_dtypes"]}),
            "projector_sumsq_changes": sum(a["projector_local_sumsq"] != b["projector_local_sumsq"] for a, b in zip(forwards, forwards[1:])),
        }
    samples = jsonl(run / "sampler.jsonl")
    gpu_peak = {}
    for record in samples:
        for gpu in record["gpus"]:
            gpu_peak[gpu["index"]] = max(gpu_peak.get(gpu["index"], 0), gpu["used_mib"])
    losses = {i: v["train/loss"] for i, v in sorted(steps.items())}
    report = {
        "optimizer_updates": len(steps),
        "loss_first_last": [losses[min(losses)], losses[max(losses)]] if losses else None,
        "loss_finite": all(math.isfinite(v) for v in losses.values()),
        "grad_norm": stats(v["train/grad_norm"] for v in steps.values()),
        "grad_norm_finite_nonzero": all(math.isfinite(v["train/grad_norm"]) and v["train/grad_norm"] > 0 for v in steps.values()),
        "lr": {i: v.get("train/lr-pg_0") for i, v in sorted(steps.items())},
        "loss": losses,
        "grad_norms": {i: v["train/grad_norm"] for i, v in sorted(steps.items())},
        "warmup_rollouts_excluded": warmup,
        "step_time_s": stats(step_time.values()),
        "train_wait_s": stats(perf[i].get("perf/train_wait_time", 0) for i in measured),
        "actor_train_s": stats(perf[i].get("perf/actor_train_time", 0) for i in measured),
        "log_probs_s": stats(perf[i].get("perf/log_probs_time", 0) for i in measured),
        "tokens_per_s": stats(tokens_by_rollout[i] / step_time[i] for i in step_time if i in tokens_by_rollout),
        "supervised_tokens_per_s": stats(supervised_by_rollout[i] / step_time[i] for i in step_time if i in supervised_by_rollout),
        "samples_per_s": stats(len(next(r["samples"] for r in rollouts if r["rollout_id"] == i)) / step_time[i] for i in step_time if i in tokens_by_rollout),
        "rollout": {
            "batches": len(rollouts),
            "samples_checked": sum(len(r["samples"]) for r in rollouts),
            "tokens_per_batch": stats(tokens_by_rollout.values()),
            "image_token_fraction": sum(s["image_tokens"] for r in rollouts for s in r["samples"]) / max(1, sum(tokens_by_rollout.values())),
            "supervised_token_fraction": sum(supervised_by_rollout.values()) / max(1, sum(tokens_by_rollout.values())),
            "get_samples_s": stats(r["get_samples_s"] for r in rollouts),
            "process_s": stats(r["process_s"] for r in rollouts),
            "decode_plus_process_rows_per_s": stats(len(r["samples"]) / (r["get_samples_s"] + r["process_s"]) for r in rollouts),
            "rss_gib_before": [r["rss_before"] / 2**30 for r in rollouts],
            "live_pil_images_before": [r["live_before"]["pil_images"] for r in rollouts],
            "live_cpu_tensor_gib_before": [r["live_before"]["cpu_tensor_bytes"] / 2**30 for r in rollouts],
            "decoded_total_last": rollouts[-1]["decoded_total"] if rollouts else None,
            "dataset_rows": rollouts[-1]["dataset_rows"] if rollouts else None,
            "sample_ids": {r["rollout_id"]: [s["record_id"] for s in r["samples"]] for r in rollouts},
        },
        "trainer": trainer,
        "nvml_peak_used_mib": gpu_peak,
        "host_used_gib_max": max((r["host_used"] for r in samples), default=0) / 2**30,
        "checkpoint_events": events,
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--interval", type=float, default=2.0)
    r = sub.add_parser("summarize")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--log", type=Path, required=True)
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    if a.command == "sample":
        sample(a.out, a.interval)
    else:
        report = summarize(a.run, a.log, a.warmup)
        a.report.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in ("lr", "loss", "grad_norms", "rollout", "checkpoint_events")}, indent=1))


if __name__ == "__main__":
    main()
