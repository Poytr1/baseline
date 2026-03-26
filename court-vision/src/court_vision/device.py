"""Device selection for PyTorch inference."""

import torch


def get_device(override: str | None = None) -> torch.device:
    """Select the best available compute device.

    Priority: MPS (Apple Silicon) > CUDA (NVIDIA GPU) > CPU.

    Args:
        override: Force a specific device ("mps", "cuda", "cpu").
                  If None or "auto", auto-detect.
    """
    if override and override != "auto":
        return torch.device(override)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
