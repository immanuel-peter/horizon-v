"""Write only the deterministic 64-example gate fixture, never a bulk dataset."""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoProcessor

from k2hv.gate_training import fixtures

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--directory", type=Path, required=True)
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
p = AutoProcessor.from_pretrained("hf/K2-Horizon-V-init", trust_remote_code=True, local_files_only=True)
rows = []
for i, s in enumerate(fixtures(p)):
    image = args.directory / f"{i:02}.png"
    s.multimodal_inputs["images"][0].save(image)
    rows.append(
        {
            "messages": [{"role": "user", "content": "<image>\nName the color and shape.", "think": None}, {"role": "assistant", "content": s.prompt[1]["content"], "think": ""}],
            "images": [str(image.resolve())],
        }
    )
pq.write_table(pa.Table.from_pylist(rows), args.directory / "fixture.parquet")
print(len(rows), args.directory / "fixture.parquet")
