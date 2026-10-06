"""Monte Carlo Dropout helpers. Entropy is reported in natural-log units."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def predictive_entropy(probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return H(mean p) for probabilities shaped [passes, batch, classes]."""
    if probabilities.ndim != 3 or probabilities.shape[0] < 1:
        raise ValueError("probabilities must have shape [passes>=1, batch, classes]")
    mean_probability = probabilities.mean(dim=0).clamp_min(eps)
    return -(mean_probability * mean_probability.log()).sum(dim=-1)


def mc_dropout_probabilities(model: nn.Module, inputs: torch.Tensor, passes: int = 30) -> torch.Tensor:
    """Enable Dropout while preserving eval-mode BatchNorm and prior module states."""
    if passes < 1:
        raise ValueError("passes must be at least 1")
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.modules.dropout._DropoutNd):
                module.train(True)
        with torch.inference_mode():
            outputs = [torch.softmax(model(inputs), dim=1) for _ in range(passes)]
        return torch.stack(outputs, dim=0)
    finally:
        for module, was_training in states:
            module.training = was_training


def summarize_mc(probabilities: torch.Tensor) -> dict[str, Any]:
    mean_probability = probabilities.mean(dim=0)
    entropy = predictive_entropy(probabilities)
    variance = probabilities.var(dim=0, unbiased=False)
    return {
        "mean_probabilities": mean_probability,
        "predictive_entropy": entropy,
        "normalized_entropy": entropy / math.log(probabilities.shape[-1]),
        "class_variance": variance,
    }

