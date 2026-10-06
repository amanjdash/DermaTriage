"""Evaluation metrics including calibration and a light bootstrap interval."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if mask.any():
            ece += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())
    return float(ece)


def bootstrap_macro_f1(
    targets: np.ndarray, predictions: np.ndarray, class_count: int = 7, seed: int = 42, repeats: int = 1000
) -> list[float]:
    rng = np.random.default_rng(seed)
    scores = []
    labels = np.arange(class_count)
    class_indices = [np.flatnonzero(targets == label) for label in np.unique(targets)]
    for _ in range(repeats):
        # Stratify by true class so each resample includes minority classes and
        # the macro-F1 interval does not collapse when a class is omitted.
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        scores.append(
            f1_score(targets[sample], predictions[sample], labels=labels, average="macro", zero_division=0)
        )
    return [float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))]


def summary_metrics(targets: np.ndarray, predictions: np.ndarray, class_count: int = 7) -> dict[str, float]:
    labels = np.arange(class_count)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, labels=labels, average="weighted", zero_division=0)),
    }

