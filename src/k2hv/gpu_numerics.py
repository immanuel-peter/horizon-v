"""The pinned Miles true-on-policy numerical configuration for the SM120 gates."""


def enable(reset=False):
    import torch
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode, disable_batch_invariant_mode, is_batch_invariant_mode_enabled
    if reset and is_batch_invariant_mode_enabled():
        disable_batch_invariant_mode()
    enable_batch_invariant_mode(enable_bmm=False)
