"""K2-Horizon-V model and training utilities."""

import os

if os.environ.get("K2HV_SFT_MONITOR"):
    # Real-data SFT instrumentation (k2hv.sft_monitor). The trainer imports the adapter through this
    # package, so wrapping here records memory and gradients without editing the certified adapter.
    try:
        from .sft_monitor import wrap_adapter

        wrap_adapter()
    except ImportError:
        pass

if os.environ.get("K2HV_VISION_GEMM", "deepgemm") != "deepgemm":
    try:
        from .vision_gemm import wrap_adapter as wrap_vision_gemm

        wrap_vision_gemm()
    except ImportError:
        pass
