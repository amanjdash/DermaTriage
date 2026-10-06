"""Small, explicit helpers for repeatable experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic kernels improve repeatability; some CUDA operations may be slower.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

