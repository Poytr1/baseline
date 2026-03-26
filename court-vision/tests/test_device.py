"""Tests for device selection logic."""

from unittest.mock import patch

from court_vision.device import get_device


def test_get_device_returns_torch_device():
    """get_device returns a torch.device object."""
    import torch

    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_cpu_fallback():
    """Falls back to CPU when MPS and CUDA are unavailable."""
    import torch

    with patch.object(torch.backends.mps, "is_available", return_value=False):
        with patch("torch.cuda.is_available", return_value=False):
            device = get_device()
            assert device.type == "cpu"


def test_get_device_cuda_when_available():
    """Selects CUDA when available and MPS is not."""
    import torch

    with patch.object(torch.backends.mps, "is_available", return_value=False):
        with patch("torch.cuda.is_available", return_value=True):
            device = get_device()
            assert device.type == "cuda"


def test_get_device_mps_preferred():
    """Selects MPS when available (preferred over CUDA)."""
    import torch

    with patch.object(torch.backends.mps, "is_available", return_value=True):
        device = get_device()
        assert device.type == "mps"


def test_get_device_override():
    """Explicit device override bypasses auto-detection."""
    import torch

    device = get_device(override="cpu")
    assert device.type == "cpu"
