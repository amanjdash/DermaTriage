"""Environment-based API configuration."""

from __future__ import annotations

import os


def _get_int(new_name: str, old_name: str, default: int, minimum: int) -> int:
    val_str = os.getenv(new_name, os.getenv(old_name, str(default)))
    try:
        val = int(val_str)
    except ValueError as exc:
        raise RuntimeError(f"{new_name} must be an integer") from exc
    if val < minimum:
        raise RuntimeError(f"{new_name} must be at least {minimum}")
    return val


def _get_str(new_name: str, old_name: str, default: str) -> str:
    return os.getenv(new_name, os.getenv(old_name, default))


MAX_UPLOAD_BYTES = _get_int("DERMATRIAGE_MAX_UPLOAD_MB", "MIRRORMED_MAX_UPLOAD_MB", 10, 1) * 1024 * 1024
MC_PASSES = _get_int("DERMATRIAGE_MC_PASSES", "MIRRORMED_MC_PASSES", 30, 1)
MAX_IMAGE_PIXELS = _get_int("DERMATRIAGE_MAX_IMAGE_PIXELS", "MIRRORMED_MAX_IMAGE_PIXELS", 40_000_000, 1)
CHECKPOINT_PATH = _get_str("DERMATRIAGE_CHECKPOINT", "MIRRORMED_CHECKPOINT", "ml/models/best_model.pth")
CORS_ORIGINS = [
    item.strip()
    for item in _get_str("DERMATRIAGE_CORS_ORIGINS", "MIRRORMED_CORS_ORIGINS", "*").split(",")
    if item.strip()
]

