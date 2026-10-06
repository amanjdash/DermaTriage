"""Train EfficientNet-B3 in two stages and save state_dict checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from ml.src.checkpoints import load_checkpoint
from ml.src.dataset import HAMDataset
from ml.src.device import select_device
from ml.src.model import build_model, build_transforms
from ml.src.reproducibility import seed_everything
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Keep checkpoints portable so CPU inference can load XPU-trained weights."""
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _save_checkpoint(payload: dict, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
    }


def _run_epoch(model, loader, criterion, optimizer, device, scaler, amp: bool, training: bool):
    model.train(training)
    all_targets: list[int] = []
    all_predictions: list[int] = []
    total_loss = 0.0
    total_examples = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=device.type in {"cuda", "xpu"})
            labels = labels.to(device, non_blocking=device.type in {"cuda", "xpu"})
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            count = labels.shape[0]
            total_loss += float(loss.detach().item()) * count
            total_examples += count
            all_targets.extend(labels.detach().cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            if training and batch_index % 25 == 0:
                print(f"  {batch_index}/{len(loader)} batches", flush=True)
    return total_loss / max(total_examples, 1), _metrics(all_targets, all_predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume-from", help="Optional state_dict checkpoint to fine-tune")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    dataset_cfg, model_cfg, training_cfg = config["dataset"], config["model"], config["training"]
    split_dir = Path(dataset_cfg["split_dir"])
    train_frame = pd.read_csv(split_dir / "train.csv")
    validation_frame = pd.read_csv(split_dir / "validation.csv")
    test_frame = pd.read_csv(split_dir / "test.csv")
    split_frames = {"train": train_frame, "validation": validation_frame, "test": test_frame}
    from ml.src.dataset import verify_no_lesion_leakage

    verify_no_lesion_leakage(split_frames)
    train_transform, eval_transform = build_transforms(int(model_cfg["image_size"]))
    batch_size, workers = int(training_cfg["batch_size"]), int(training_cfg["num_workers"])
    device = select_device()
    use_amp = bool(training_cfg["amp"] and device.type in {"cuda", "xpu"})
    scaler_device = device.type if device.type in {"cuda", "xpu"} else "cuda"
    scaler = torch.amp.GradScaler(scaler_device, enabled=use_amp)
    loader_options = {"batch_size": batch_size, "num_workers": workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(HAMDataset(train_frame, train_transform), shuffle=True, **loader_options)
    validation_loader = DataLoader(HAMDataset(validation_frame, eval_transform), shuffle=False, **loader_options)

    model = build_model(
        dropout=float(model_cfg["dropout"]),
        pretrained=bool(model_cfg["pretrained"] and not args.resume_from),
    )
    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = load_checkpoint(args.resume_from)
        model.load_state_dict(resume_checkpoint["state_dict"])
    resume_state = (resume_checkpoint or {}).get("training_state")
    exact_resume = bool(resume_state and not resume_state.get("complete", False))
    model.to(device)

    # Use the canonical class order and only the training split to avoid validation/test information.
    from ml.src.labels import CLASS_NAMES, CLASS_TO_IDX

    counts = np.bincount(train_frame["dx"].map(CLASS_TO_IDX).to_numpy(), minlength=len(CLASS_NAMES)).astype(float)
    class_weights = np.power(counts.max() / np.maximum(counts, 1.0), float(training_cfg["class_weight_power"]))
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    run_id = (
        str(resume_checkpoint["model_version"])
        if exact_resume
        else datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    if int(training_cfg["head_epochs"]) < 1 or int(training_cfg["finetune_epochs"]) < 0:
        raise ValueError("head_epochs must be >= 1 and finetune_epochs must be >= 0")
    experiment_dir = Path("ml/experiments") / run_id
    experiment_dir.mkdir(parents=True, exist_ok=exact_resume)
    model_dir = Path("ml/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    run_config = (
        resume_checkpoint["config"]
        if exact_resume
        else {
            **config,
            "run_id": run_id,
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__),
                "device": str(device),
            },
            "resume_from": str(Path(args.resume_from).resolve()) if args.resume_from else None,
            "class_counts_train": dict(zip(CLASS_NAMES, counts.astype(int).tolist())),
            "class_weights": dict(zip(CLASS_NAMES, class_weights.tolist())),
            "split_report": str(split_dir / "split_report.json"),
        }
    )
    (experiment_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Training on {device}; batch size {batch_size}; AMP {'on' if use_amp else 'off'}.", flush=True)
    history_path = experiment_dir / "history.csv"
    if exact_resume and not history_path.is_file():
        raise FileNotFoundError(f"Cannot resume run {run_id}: its epoch history is missing")
    best_macro_f1 = (
        float(resume_state["best_validation_macro_f1"])
        if exact_resume
        else float(resume_checkpoint.get("best_validation_macro_f1", -1.0)) if resume_checkpoint else -1.0
    )
    best_epoch = (
        int(resume_state["best_epoch"])
        if exact_resume
        else int(resume_checkpoint.get("best_epoch", 0)) if resume_checkpoint else 0
    )
    global_epoch = int(resume_state["global_epoch"]) if exact_resume else 0

    phases = (
        ("head", int(training_cfg["head_epochs"]), float(training_cfg["head_learning_rate"])),
        ("finetune", int(training_cfg["finetune_epochs"]), float(training_cfg["finetune_learning_rate"])),
    )
    phase_order = {phase: index for index, (phase, _, _) in enumerate(phases)}
    resume_phase_index = phase_order.get(resume_state.get("phase"), 0) if exact_resume else 0
    resume_phase_complete = bool(resume_state.get("phase_complete", False)) if exact_resume else False

    for phase_index, (phase, epochs, learning_rate) in enumerate(phases):
        if exact_resume and (
            phase_index < resume_phase_index
            or (phase_index == resume_phase_index and resume_phase_complete)
        ):
            if phase == "head" and resume_phase_index == phase_index and resume_phase_complete:
                best_head = load_checkpoint(model_dir / "best_model.pth")
                model.load_state_dict(best_head["state_dict"])
            continue
        if phase == "head":
            for parameter in model.features.parameters():
                parameter.requires_grad = False
        else:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=float(training_cfg["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        continuing_phase = exact_resume and phase_index == resume_phase_index and not resume_phase_complete
        phase_best = float(resume_state["phase_best"]) if continuing_phase else -1.0
        no_improvement = int(resume_state["no_improvement"]) if continuing_phase else 0
        if continuing_phase:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            if resume_state.get("scaler_state_dict"):
                scaler.load_state_dict(resume_state["scaler_state_dict"])
        first_phase_epoch = int(resume_state["phase_epoch"]) + 1 if continuing_phase else 1
        for phase_epoch in range(first_phase_epoch, epochs + 1):
            global_epoch += 1
            print(f"Starting {phase} epoch {phase_epoch}/{epochs} (overall epoch {global_epoch}).", flush=True)
            train_loss, train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, True)
            validation_loss, validation_metrics = _run_epoch(
                model, validation_loader, criterion, optimizer, device, scaler, False, False
            )
            scheduler.step(validation_metrics["macro_f1"])
            record = {
                "epoch": global_epoch,
                "phase": phase,
                "phase_epoch": phase_epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            write_header = not history_path.exists()
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=record.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(record)
            print(json.dumps(record))
            score = validation_metrics["macro_f1"]
            if score > best_macro_f1:
                best_macro_f1, best_epoch = score, global_epoch
                payload = {
                    "state_dict": _cpu_state_dict(model),
                    "class_to_idx": CLASS_TO_IDX,
                    "model_name": "EfficientNet-B3",
                    "model_version": run_id,
                    "dataset_reference": "HAM10000, Harvard Dataverse DOI:10.7910/DVN/DBW86T",
                    "config": run_config,
                    "best_validation_macro_f1": best_macro_f1,
                    "best_epoch": best_epoch,
                }
                _save_checkpoint(payload, model_dir / "best_model.pth")
            if score > phase_best:
                phase_best = score
                no_improvement = 0
            else:
                no_improvement += 1
            stop_phase = no_improvement >= int(training_cfg["patience"])
            phase_complete = phase_epoch >= epochs or stop_phase
            _save_checkpoint(
                {
                    "state_dict": _cpu_state_dict(model),
                    "config": run_config,
                    "model_version": run_id,
                    "training_state": {
                        "phase": phase,
                        "phase_epoch": phase_epoch,
                        "global_epoch": global_epoch,
                        "best_validation_macro_f1": best_macro_f1,
                        "best_epoch": best_epoch,
                        "phase_best": phase_best,
                        "no_improvement": no_improvement,
                        "phase_complete": phase_complete,
                        "complete": False,
                        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                    },
                },
                model_dir / "last_model.pth",
            )
            if stop_phase:
                print(f"Early stopping after {global_epoch} epochs; best validation macro F1={best_macro_f1:.4f} at epoch {best_epoch}.")
                break
        # Early stopping is tracked separately for each training phase.
        if phase == "head":
            best_head = load_checkpoint(model_dir / "best_model.pth")
            model.load_state_dict(best_head["state_dict"])

    _save_checkpoint(
        {
            "state_dict": _cpu_state_dict(model),
            "config": run_config,
            "model_version": run_id,
            "training_state": {"global_epoch": global_epoch, "complete": True},
        },
        model_dir / "last_model.pth",
    )
    (experiment_dir / "summary.json").write_text(
        json.dumps({"best_validation_macro_f1": best_macro_f1, "best_epoch": best_epoch, "device": str(device)}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved best checkpoint to {model_dir / 'best_model.pth'}; evaluate the untouched test split separately.")


if __name__ == "__main__":
    main()

