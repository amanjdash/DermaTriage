"""Safe helpers for locally generated model checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint(path: str | Path) -> dict:
    """Load a state-dict checkpoint, allowing the scalar used by early run metadata."""
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)
