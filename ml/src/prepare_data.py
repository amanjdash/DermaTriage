"""Scan HAM10000 images and write reproducible lesion-level split CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from ml.src.dataset import load_metadata, make_lesion_splits
from ml.src.labels import CLASS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    settings = config["dataset"]
    metadata, excluded = load_metadata(settings["metadata_csv"], settings["image_dirs"])
    splits = make_lesion_splits(
        metadata,
        seed=config["seed"],
        test_fraction=settings["test_fraction"],
        validation_fraction_of_remainder=settings["validation_fraction_of_remainder"],
    )
    output_dir = Path(settings["split_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"seed": config["seed"], "classes": list(CLASS_NAMES), "excluded_image_ids": excluded, "splits": {}}
    for name, frame in splits.items():
        csv_path = output_dir / f"{name}.csv"
        frame.to_csv(csv_path, index=False)
        report["splits"][name] = {
            "images": len(frame),
            "lesions": int(frame["lesion_id"].nunique()),
            "class_counts": frame["dx"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict(),
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        }
    (output_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

