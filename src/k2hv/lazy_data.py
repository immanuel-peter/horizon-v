"""Miles data source that decodes images only for the samples a rollout requests.

Pinned Miles ``Dataset.__init__`` runs ``process_vision_info`` on every row and keeps the decoded
images, which needs terabytes for the full SFT export. ``LazyRolloutDataSource`` keeps the
pinned ``RolloutDataSource`` batching, epoch, shuffle and checkpoint code and replaces only the
dataset: rows stay as Arrow text and image paths, and a sample's images are decoded when
``get_samples`` slices it. Row order and the per-epoch permutation match the pinned ``Dataset``.

Use with ``--data-source-path k2hv.lazy_data.LazyRolloutDataSource``. ``--prompt-data`` may be a
Parquet file, a directory of Parquet parts (read in sorted order) or a comma-separated list.
Supported configuration: ``--multimodal-keys`` without ``--apply-chat-template``, ``--tool-key``
or ``--rollout-max-prompt-len`` (the data pipeline validates every row's length).
"""

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parquet_files(path):
    files = []
    for part in str(path).split(","):
        p = Path(part)
        files.extend(sorted(p.glob("*.parquet")) if p.is_dir() else [p])
    if not files or not all(f.is_file() for f in files):
        raise FileNotFoundError(f"No Parquet files at {path}")
    return files


class LazyDataset:
    """Rows as Arrow text; ``samples[i]`` builds the Miles message list and decodes its images."""

    def __init__(self, path, processor, *, prompt_key, multimodal_keys, label_key=None, metadata_key="metadata", seed=42, miles=None):
        if not multimodal_keys:
            raise ValueError("LazyDataset is for multimodal rows; use the Miles Dataset for text-only data")
        self.path = str(path)
        self.processor = processor
        self.prompt_key, self.multimodal_keys, self.label_key, self.metadata_key = prompt_key, multimodal_keys, label_key, metadata_key
        self.seed = seed
        self._miles = miles
        self.table = pa.concat_tables([pq.read_table(f) for f in parquet_files(self.path)])
        self.origin_indices = list(range(self.table.num_rows))
        self.indices = self.origin_indices
        self.epoch_id = -1
        self.decoded = 0

    @property
    def miles(self):
        if self._miles is None:
            self._miles = miles_functions()
        return self._miles

    def __getstate__(self):
        state = self.__dict__.copy()
        state["table"] = None
        state["_miles"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.table = pa.concat_tables([pq.read_table(f) for f in parquet_files(self.path)])

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return
        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.indices)))
        random.shuffle(permutation)
        self.indices = [self.origin_indices[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def row(self, index):
        return self.table.slice(index, 1).to_pylist()[0]

    def build(self, index):
        m = self.miles
        data = self.row(index)
        prompt = m.build_messages(data, self.prompt_key, True, self.multimodal_keys)
        multimodal_inputs = m.process_vision_info(prompt, self.processor)
        self.decoded += 1
        label = data[self.label_key] if self.label_key is not None else None
        return m.Sample(prompt=prompt, label=label, metadata=data.get(self.metadata_key) or {}, multimodal_inputs=multimodal_inputs)

    @property
    def samples(self):
        return _Samples(self)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.indices)


class _Samples:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.indices)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.dataset.build(i) for i in self.dataset.indices[idx]]
        return self.dataset.build(self.dataset.indices[idx])


def miles_functions():
    from types import SimpleNamespace

    from miles.utils.data import _build_messages
    from miles.utils.processing_utils import process_vision_info
    from miles.utils.types import Sample

    return SimpleNamespace(build_messages=_build_messages, process_vision_info=process_vision_info, Sample=Sample)


def _load_processor(args):
    from miles.utils.processing_utils import load_processor

    return load_processor(args.hf_checkpoint, trust_remote_code=True)


class LazyInit:
    """``__init__`` for a ``RolloutDataSource`` subclass; ``get_samples``, ``save`` and ``load`` stay pinned Miles code."""

    load_processor = staticmethod(_load_processor)
    miles = None

    def __init__(self, args):
        self.args = args
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}
        if not args.rollout_global_dataset:
            self.dataset = None
            return
        unsupported = {"apply_chat_template": args.apply_chat_template, "tool_key": args.tool_key, "rollout_max_prompt_len": args.rollout_max_prompt_len}
        if any(v for v in unsupported.values()):
            raise ValueError(f"LazyRolloutDataSource does not support {[k for k, v in unsupported.items() if v]}")
        self.dataset = LazyDataset(
            args.prompt_data, self.load_processor(args), prompt_key=args.input_key, multimodal_keys=args.multimodal_keys,
            label_key=args.label_key, metadata_key=args.metadata_key, seed=args.rollout_seed, miles=self.miles,
        )
        if args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)


def __getattr__(name):
    if name == "LazyRolloutDataSource":
        from miles.rollout.data_source import RolloutDataSource

        cls = type("LazyRolloutDataSource", (LazyInit, RolloutDataSource), {"__module__": __name__})
        globals()[name] = cls
        return cls
    raise AttributeError(name)
