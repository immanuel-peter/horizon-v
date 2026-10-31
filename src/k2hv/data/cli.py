"""Step 5 data preparation CLI.

Stages (each resumable; rerunning skips work whose inputs are unchanged):
  inventory   pin listings, sizes and schemas of every recipe source and evaluation set
  scan        read sources, apply text/image rules, store images, write candidates
  evals       index evaluation images/questions for exclusion
  select      exclusion, grouping, deduplication, splits and quotas
  validate    run every selected row through the pinned Miles path and processor contract
  export      write portable Parquet, manifests, checksums and reports
  materialize write Miles Parquet with absolute image paths under a data root
  verify      re-hash/decode every exported image and check exported rows match validated rows
  build       scan, evals, select, validate, export, materialize and verify in order
  inspect     HTML sheet of exact model inputs and decoded supervision; exclusion pair sheets
  evidence    compact path-free summary for docs/evidence/step05
"""

import argparse
import os
import sys
from pathlib import Path

DEFAULT_RECIPE = "configs/data/step05.toml"
DEFAULT_ROOT = "data/step05"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m k2hv.data", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["inventory", "scan", "evals", "select", "validate", "export", "materialize", "verify", "build", "inspect", "evidence"])
    parser.add_argument("--profile", default="pilot", help="recipe profile (pilot or full)")
    parser.add_argument("--recipe", default=DEFAULT_RECIPE)
    parser.add_argument("--data-root", default=os.environ.get("K2HV_DATA_ROOT", DEFAULT_ROOT), help="build/data root (sources, images, work, exports)")
    parser.add_argument("--target-root", help="materialize: absolute root that will hold images/ on the training machine")
    parser.add_argument("--out", default="docs/evidence/step05", help="evidence: output directory for the compact summary")
    parser.add_argument("--max-shards", type=int, help="scan: stop after this many shards (interruption testing)")
    args = parser.parse_args(argv)

    from . import build, pipeline

    ctx = build.Context(args.recipe, args.data_root, args.profile)
    stages = ["scan", "evals", "select", "validate", "export", "materialize", "verify"] if args.stage == "build" else [args.stage]
    for stage in stages:
        build.log(f"== {stage} ({args.profile})")
        if stage == "inventory":
            pipeline.inventory(ctx)
        elif stage == "scan":
            if args.max_shards is not None:
                jobs = build.plan_scan(ctx)[: args.max_shards]
                for marker in build._imap(build.scan_shard, jobs, ctx.recipe["workers"]["scan"]):
                    build.log(f"  {marker['source']} {marker['shard']['file']} done")
            else:
                build.scan(ctx)
        elif stage == "evals":
            build.eval_index(ctx)
        elif stage == "select":
            pipeline.select(ctx)
        elif stage == "validate":
            pipeline.validate(ctx)
        elif stage == "export":
            pipeline.export(ctx)
        elif stage == "materialize":
            pipeline.materialize(ctx, Path(args.target_root) if args.target_root else ctx.root)
        elif stage == "verify":
            if not pipeline.verify(ctx, Path(args.target_root) if args.target_root else ctx.root):
                sys.exit(1)
        elif stage == "evidence":
            pipeline.evidence(ctx, args.out)
        elif stage == "inspect":
            pipeline.inspect(ctx)
            pipeline.exclusion_sheets(ctx)


if __name__ == "__main__":
    main()
