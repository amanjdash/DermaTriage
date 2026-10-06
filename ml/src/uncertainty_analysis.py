"""Compare deterministic and MC predictions and examine entropy/error association."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ml.src.checkpoints import load_checkpoint
from ml.src.dataset import HAMDataset
from ml.src.device import select_device
from ml.src.labels import CLASS_NAMES, IDX_TO_CLASS
from ml.src.model import build_model, build_transforms
from ml.src.uncertainty import mc_dropout_probabilities, summarize_mc
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def _correlation(function, entropy: np.ndarray, errors: np.ndarray) -> float | None:
    if len(np.unique(errors)) < 2 or len(np.unique(entropy)) < 2:
        return None
    value = float(function(entropy, errors).statistic)
    return value if np.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="ml/models/best_model.pth")
    parser.add_argument("--split-dir", default="ml/data/processed/splits")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="ml/experiments/uncertainty-analysis")
    args = parser.parse_args()
    if args.mc_passes < 2 or args.batch_size < 1:
        parser.error("MC Dropout analysis needs at least two passes and a positive batch size")
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
    frame = pd.read_csv(Path(args.split_dir) / f"{args.split}.csv")
    loader = DataLoader(
        HAMDataset(frame, evaluation_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    targets: list[int] = []
    deterministic_predictions: list[int] = []
    mc_predictions: list[int] = []
    entropies: list[float] = []
    cursor = 0
    sample_rows = []
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type in {"cuda", "xpu"})
        with torch.inference_mode():
            deterministic = torch.softmax(model(images), dim=1)
        samples = mc_dropout_probabilities(model, images, args.mc_passes)
        summary = summarize_mc(samples)
        deterministic_batch = deterministic.argmax(dim=1).cpu().tolist()
        mc_batch = summary["mean_probabilities"].argmax(dim=1).cpu().tolist()
        batch_targets = labels.tolist()
        batch_entropy = summary["predictive_entropy"].cpu().tolist()
        batch_normalized = summary["normalized_entropy"].cpu().tolist()
        for offset, (target, one_pass, mean_pass, entropy, normalized) in enumerate(
            zip(batch_targets, deterministic_batch, mc_batch, batch_entropy, batch_normalized)
        ):
            original_index = cursor + offset
            sample_rows.append(
                {
                    "image_id": str(frame.iloc[original_index]["image_id"]),
                    "true_class": IDX_TO_CLASS[target],
                    "deterministic_prediction": IDX_TO_CLASS[one_pass],
                    "mc_mean_prediction": IDX_TO_CLASS[mean_pass],
                    "predictive_entropy_nats": entropy,
                    "predictive_entropy_normalized": normalized,
                    "deterministic_error": one_pass != target,
                    "mc_mean_error": mean_pass != target,
                }
            )
        cursor += len(batch_targets)
        targets.extend(batch_targets)
        deterministic_predictions.extend(deterministic_batch)
        mc_predictions.extend(mc_batch)
        entropies.extend(batch_entropy)

    y = np.asarray(targets, dtype=int)
    deterministic = np.asarray(deterministic_predictions, dtype=int)
    mc_mean = np.asarray(mc_predictions, dtype=int)
    entropy = np.asarray(entropies, dtype=float)
    mc_errors = (mc_mean != y).astype(int)
    entropy_correct = entropy[mc_errors == 0]
    entropy_incorrect = entropy[mc_errors == 1]
    report = {
        "split": args.split,
        "samples": len(y),
        "mc_passes": args.mc_passes,
        "device": str(device),
        "deterministic_accuracy": float(accuracy_score(y, deterministic)),
        "mc_mean_accuracy": float(accuracy_score(y, mc_mean)),
        "deterministic_macro_f1": float(f1_score(y, deterministic, labels=np.arange(len(CLASS_NAMES)), average="macro", zero_division=0)),
        "mc_mean_macro_f1": float(f1_score(y, mc_mean, labels=np.arange(len(CLASS_NAMES)), average="macro", zero_division=0)),
        "mean_entropy_correct_mc_predictions": float(entropy_correct.mean()) if len(entropy_correct) else None,
        "mean_entropy_incorrect_mc_predictions": float(entropy_incorrect.mean()) if len(entropy_incorrect) else None,
        "entropy_error_pearson_r": _correlation(pearsonr, entropy, mc_errors),
        "entropy_error_spearman_r": _correlation(spearmanr, entropy, mc_errors),
        "interpretation": "Exploratory model-uncertainty analysis only; entropy is not clinical severity or risk.",
        "model_version": checkpoint.get("model_version", "unknown"),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.split}_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(sample_rows).to_csv(output_dir / f"{args.split}_samples.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

