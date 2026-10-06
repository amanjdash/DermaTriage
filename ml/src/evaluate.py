"""Evaluate a saved checkpoint once on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ml.src.checkpoints import load_checkpoint
from ml.src.dataset import HAMDataset, verify_no_lesion_leakage
from ml.src.device import select_device
from ml.src.labels import CLASS_NAMES
from ml.src.metrics import bootstrap_macro_f1, expected_calibration_error, summary_metrics
from ml.src.model import build_model, build_transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="ml/models/best_model.pth")
    parser.add_argument("--split-dir", default="ml/data/processed/splits")
    parser.add_argument("--output-dir", default="ml/experiments/evaluation")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}; train the model first")
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.get("config", {})
    image_size = int(config.get("model", {}).get("image_size", 300))
    model = build_model(dropout=float(config.get("model", {}).get("dropout", 0.30)))
    model.load_state_dict(checkpoint["state_dict"])
    device = select_device()
    model.to(device).eval()
    _, evaluation_transform = build_transforms(image_size)

    frames = {name: pd.read_csv(Path(args.split_dir) / f"{name}.csv") for name in ("train", "validation", "test")}
    verify_no_lesion_leakage(frames)
    dataset = HAMDataset(frames[args.split], evaluation_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    all_targets, all_probabilities = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type in {"cuda", "xpu"})
            all_probabilities.append(torch.softmax(model(images), dim=1).cpu().numpy())
            all_targets.extend(labels.numpy().tolist())
    targets = np.asarray(all_targets, dtype=int)
    probabilities = np.concatenate(all_probabilities)
    predictions = probabilities.argmax(axis=1)

    report = classification_report(
        targets,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(targets, predictions, labels=np.arange(len(CLASS_NAMES)))
    metrics = summary_metrics(targets, predictions, len(CLASS_NAMES))
    metrics["expected_calibration_error_10_bins"] = expected_calibration_error(probabilities, targets, bins=10)
    metrics["macro_f1_bootstrap_95_percent_ci"] = bootstrap_macro_f1(targets, predictions, len(CLASS_NAMES))
    try:
        auc_value = float(roc_auc_score(targets, probabilities, multi_class="ovr", average="macro"))
        metrics["roc_auc_ovr_macro"] = auc_value if np.isfinite(auc_value) else None
    except ValueError:
        metrics["roc_auc_ovr_macro"] = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "images": len(targets),
                "metrics": metrics,
                "per_class": report,
                "model_version": checkpoint.get("model_version", "unknown"),
                "device": str(device),
            },
            indent=2,
            allow_nan=False,
            default=lambda value: None if isinstance(value, float) and np.isnan(value) else value,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output_dir / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted class", ylabel="True class", title=f"Confusion matrix — {args.split} split")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    confidence = probabilities.max(axis=1)
    correct = (predictions == targets).astype(float)
    edges = np.linspace(0.0, 1.0, 11)
    bin_centers, bin_accuracies = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            bin_centers.append(float(confidence[mask].mean()))
            bin_accuracies.append(float(correct[mask].mean()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(bin_centers, bin_accuracies, marker="o", label="Model")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy", title="Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reliability_diagram.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

