"""Check external registration in the installed GPU-side SGLang environment."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

from .runtime import MODEL_ARCHITECTURE, REASONING_PARSER, SGLANG_REVISION, SOURCE_SHA256, registration_environment


def check_source(directory):
    mismatches = [name for name, digest in SOURCE_SHA256.items()
                  if not (directory / name).is_file() or hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError(f"SGLang sources differ from {SGLANG_REVISION}: {', '.join(mismatches)}")
    return len(SOURCE_SHA256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="audit an sglang/srt source directory without importing the runtime")
    args = parser.parse_args()
    if args.source_dir is not None:
        print(json.dumps({"revision": SGLANG_REVISION, "verified_source_files": check_source(args.source_dir)}))
        return
    spec = importlib.util.find_spec("sglang")
    if spec is None:
        parser.error("SGLang is not installed; run this in the pinned Miles GPU environment")
    verified = check_source(Path(next(iter(spec.submodule_search_locations))) / "srt")
    if importlib.metadata.version("transformers") != "5.19.0":
        parser.error("This integration requires transformers==5.19.0")
    for name, value in registration_environment().items():
        os.environ[name] = value

    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.managers.multimodal_processor import PROCESSOR_MAPPING, import_processors
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    import_processors("k2hv_sglang", overwrite=True)
    model_class = ModelRegistry.models[MODEL_ARCHITECTURE]
    processor_class = PROCESSOR_MAPPING[model_class]
    assert model_class.__module__ == "k2hv_sglang.k2_horizon_v"
    assert processor_class.__module__ == "k2hv_sglang.processor"
    assert processor_class.__name__ == "K2HorizonVImageProcessor"
    assert REASONING_PARSER in ReasoningParser.DetectorMap
    print(json.dumps({"sglang_version": importlib.metadata.version("sglang"), "reference_revision": SGLANG_REVISION, "verified_source_files": verified,
                      "architecture": MODEL_ARCHITECTURE, "processor": processor_class.__name__, "reasoning_parser": REASONING_PARSER}, indent=2))


if __name__ == "__main__":
    main()
