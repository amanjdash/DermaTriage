"""Shared accelerator selection for training and offline analysis."""

from __future__ import annotations

import torch


def select_device() -> torch.device:
    """Prefer Intel XPU, then CUDA, and fall back to CPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
