"""Standalone CPU/CUDA inference, with deterministic and MC Dropout modes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from ml.src.checkpoints import load_checkpoint
from ml.src.labels import CLASS_NAMES
from ml.src.model import build_model, build_transforms
from ml.src.uncertainty import mc_dropout_probabilities, summarize_mc
from PIL import Image, ImageOps, UnidentifiedImageError


def load_predictor(checkpoint_path: str | Path, device: str | None = None):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint is missing: {path}")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = load_checkpoint(path)
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    model = build_model(dropout=float(model_cfg.get("dropout", 0.30)))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(selected_device).eval()
    _, evaluation_transform = build_transforms(int(model_cfg.get("image_size", 300)))
    return {
        "model": model,
        "transform": evaluation_transform,
        "device": selected_device,
        "checkpoint": checkpoint,
        "image_size": int(model_cfg.get("image_size", 300)),
    }


def predict_image(predictor, image: Image.Image, mc_passes: int = 30, stochastic: bool = True) -> dict:
    start = time.perf_counter()
    normalized_image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = predictor["transform"](normalized_image).unsqueeze(0).to(predictor["device"])
    preprocess_ms = (time.perf_counter() - start) * 1000
    if predictor["device"].type == "cuda":
        torch.cuda.synchronize(predictor["device"])
    inference_start = time.perf_counter()
    with torch.inference_mode():
        if stochastic:
            predictor["model"].eval()
            deterministic_probabilities = torch.softmax(predictor["model"](tensor), dim=1)[0]
            samples = mc_dropout_probabilities(predictor["model"], tensor, mc_passes)
        else:
            predictor["model"].eval()
            samples = torch.softmax(predictor["model"](tensor), dim=1).unsqueeze(0)
            deterministic_probabilities = samples[0, 0]
    if predictor["device"].type == "cuda":
        torch.cuda.synchronize(predictor["device"])
    inference_ms = (time.perf_counter() - inference_start) * 1000
    postprocess_start = time.perf_counter()
    summary = summarize_mc(samples)
    probabilities = summary["mean_probabilities"][0].detach().cpu().tolist()
    predicted_index = int(torch.tensor(probabilities).argmax().item())
    deterministic_index = int(deterministic_probabilities.argmax().item())
    uncertainty = float(summary["predictive_entropy"][0].item())
    normalized_entropy = float(summary["normalized_entropy"][0].item())
    class_variance = summary["class_variance"][0].detach().cpu().tolist()
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000
    return {
        "prediction": CLASS_NAMES[predicted_index],
        "deterministic_prediction": CLASS_NAMES[deterministic_index],
        "deterministic_confidence": float(deterministic_probabilities[deterministic_index].item()),
        "confidence": probabilities[predicted_index],
        "uncertainty": uncertainty,
        "uncertainty_normalized": normalized_entropy,
        "uncertainty_method": "predictive_entropy",
        "mc_passes": mc_passes if stochastic else 1,
        "probabilities": dict(zip(CLASS_NAMES, probabilities)),
        "class_probability_variance": dict(zip(CLASS_NAMES, class_variance)),
        "preprocessing_time_ms": preprocess_ms,
        "model_inference_time_ms": inference_ms,
        "postprocessing_time_ms": postprocess_ms,
        "inference_time_ms": preprocess_ms + inference_ms + postprocess_ms,
        "model_name": "EfficientNet-B3",
        "model_version": predictor["checkpoint"].get("model_version", "unknown"),
        "dataset_reference": predictor["checkpoint"].get("dataset_reference", "HAM10000"),
        "device": str(predictor["device"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="ml/models/best_model.pth")
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--deterministic", action="store_true", help="Run one evaluation-mode pass")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.mc_passes < 1:
        parser.error("--mc-passes must be >= 1")
    try:
        with Image.open(args.image) as source:
            source.verify()
        with Image.open(args.image) as source:
            image = source.copy()
    except (OSError, UnidentifiedImageError) as exc:
        parser.error(f"Invalid image: {exc}")
    predictor = load_predictor(args.checkpoint, args.device)
    print(json.dumps(predict_image(predictor, image, args.mc_passes, not args.deterministic), indent=2))


if __name__ == "__main__":
    main()

