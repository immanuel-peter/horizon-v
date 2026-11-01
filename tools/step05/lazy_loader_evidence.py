"""Bounded CPU memory measurement for k2hv.lazy_data on an exported Miles SFT split.

usage: python tools/step05/lazy_loader_evidence.py MILES_SPLIT_DIR OUT_JSON [BATCH] [BATCHES]
"""

import ctypes
import gc
import json
import resource
import sys
import time
import types
from pathlib import Path


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024


split, out = Path(sys.argv[1]), Path(sys.argv[2])
batch, batches = (int(sys.argv[3]) if len(sys.argv) > 3 else 256), (int(sys.argv[4]) if len(sys.argv) > 4 else 4)
from k2hv import lazy_data  # noqa: E402
from k2hv.data import runtime  # noqa: E402

runtime.limit_threads(1)
m = runtime.miles()
proc = runtime.load_processor("hf/K2-Horizon-V-init")
loading = runtime.miles_loading()
fns = types.SimpleNamespace(build_messages=m.build_messages, process_vision_info=m.process_vision_info, Sample=m.Sample)
cls = type("Lazy", (lazy_data.LazyInit, loading.RolloutDataSource), {"miles": fns, "load_processor": staticmethod(lambda a: proc)})
args = types.SimpleNamespace(rollout_global_dataset=True, prompt_data=str(split), input_key="messages", multimodal_keys={"image": "images"}, label_key=None,
                             metadata_key="metadata", tool_key=None, apply_chat_template=False, rollout_max_prompt_len=None, rollout_seed=0,
                             rollout_shuffle=True, n_samples_per_prompt=1)
base = rss_mb()
t0 = time.time()
src = cls(args)
construct = {"rows": len(src.dataset), "seconds": round(time.time() - t0, 1), "rss_growth_mb": round(rss_mb() - base, 1), "decoded": src.dataset.decoded,
             "arrow_bytes_mb": round(src.dataset.table.nbytes / 2**20, 1)}
steps = []
for _ in range(batches):
    before, t0 = rss_mb(), time.time()
    groups = src.get_samples(batch)
    held = sum(im.width * im.height * 3 for (s,) in groups for im in s.multimodal_inputs["images"])
    steps.append({"decoded_total": src.dataset.decoded, "seconds": round(time.time() - t0, 1), "rss_growth_mb": round(rss_mb() - before, 1),
                  "held_rgb_mb": round(held / 2**20, 1), "rows": len(groups)})
    del groups
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    steps[-1]["rss_after_release_mb"] = round(rss_mb() - base, 1)
result = {"split": str(split), "files": len(lazy_data.parquet_files(split)), "batch": batch, "construct": construct, "batches": steps,
          "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
          "note": "CPU only. rss_after_release_mb is RSS above the pre-construction baseline after gc and malloc_trim. Eager pinned Dataset would decode every row at construction."}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps(result))
