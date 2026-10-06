"""Batch-size-1 inference benchmark: five warmups, fifty measured runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from ml.src.inference import load_predictor, predict_image
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="ml/models/best_model.pth")
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output", default="ml/experiments/benchmark.json")
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0 or args.mc_passes < 1:
        parser.error("runs and mc-passes must be positive; warmups cannot be negative")
    with Image.open(args.image) as source:
        image = source.copy()
    predictor = load_predictor(args.checkpoint)
    for _ in range(args.warmups):
        predict_image(predictor, image, args.mc_passes)
    observations = []
    for _ in range(args.runs):
        result = predict_image(predictor, image, args.mc_passes)
        observations.append(result)
    total = [row["inference_time_ms"] for row in observations]
    report = {
        "batch_size": 1,
        "device": str(predictor["device"]),
        "torch_version": torch.__version__,
        "model_name": "EfficientNet-B3",
        "model_version": predictor["checkpoint"].get("model_version", "unknown"),
        "input_resolution": [predictor["image_size"], predictor["image_size"]],
        "preprocessing_included": True,
        "warmup_runs": args.warmups,
        "timed_runs": args.runs,
        "mc_passes": args.mc_passes,
        "mean_latency_ms": statistics.mean(total),
        "median_latency_ms": statistics.median(total),
        "std_latency_ms": statistics.pstdev(total),
        "p95_latency_ms": float(np.percentile(total, 95)),
        "mean_preprocessing_ms": statistics.mean(row["preprocessing_time_ms"] for row in observations),
        "mean_model_inference_ms": statistics.mean(row["model_inference_time_ms"] for row in observations),
        "mean_postprocessing_ms": statistics.mean(row["postprocessing_time_ms"] for row in observations),
        "note": "Repeated inference on one local image; excludes HTTP transport and upload decoding.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

