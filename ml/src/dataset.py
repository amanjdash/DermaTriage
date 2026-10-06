"""HAM10000 indexing, integrity checks, lesion-level split, and PyTorch dataset."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from ml.src.labels import CLASS_NAMES, CLASS_TO_IDX
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REQUIRED_COLUMNS = {"image_id", "lesion_id", "dx"}


def index_images(image_dirs: Iterable[str | Path]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for directory in image_dirs:
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {root}")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                if path.stem in paths:
                    raise ValueError(f"Duplicate image id {path.stem!r}: {paths[path.stem]} and {path}")
                paths[path.stem] = path
    return paths


def load_metadata(metadata_csv: str | Path, image_dirs: Iterable[str | Path]) -> tuple[pd.DataFrame, list[str]]:
    metadata = pd.read_csv(metadata_csv)
    missing = REQUIRED_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    if metadata[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("image_id, lesion_id, and dx must not contain missing values")
    metadata["image_id"] = metadata["image_id"].astype(str)
    metadata["lesion_id"] = metadata["lesion_id"].astype(str)
    metadata["dx"] = metadata["dx"].astype(str)
    if metadata["image_id"].duplicated().any():
        duplicates = metadata.loc[metadata["image_id"].duplicated(), "image_id"].tolist()
        raise ValueError(f"Duplicate image_id values in metadata: {duplicates[:5]}")
    unknown = sorted(set(metadata["dx"]) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown diagnosis labels: {unknown}")

    image_index = index_images(image_dirs)
    metadata["path"] = metadata["image_id"].map(image_index)
    missing_images = metadata.loc[metadata["path"].isna(), "image_id"].tolist()
    metadata = metadata.dropna(subset=["path"]).copy()
    corrupt: list[str] = []
    for row in metadata.itertuples():
        try:
            with Image.open(row.path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError, ValueError):
            corrupt.append(row.image_id)
    metadata = metadata.loc[~metadata["image_id"].isin(corrupt)].reset_index(drop=True)
    if metadata.empty:
        raise ValueError("No valid metadata rows with readable images were found")
    return metadata, missing_images + corrupt


def _choose_fold(
    splitter: StratifiedGroupKFold,
    frame: pd.DataFrame,
    y,
    groups,
    target_fraction: float,
) -> tuple[list[int], list[int]]:
    """Choose the fold whose class counts most closely match the requested fraction."""
    total = frame["dx"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_numpy(dtype=float)
    best: tuple[float, list[int], list[int]] | None = None
    for left, right in splitter.split(frame, y, groups):
        observed = frame.iloc[right]["dx"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_numpy(dtype=float)
        expected = total * target_fraction
        class_error = (((observed - expected) ** 2) / (expected + 1.0)).sum()
        size_error = ((len(right) - len(frame) * target_fraction) / max(len(frame) * target_fraction, 1)) ** 2
        score = float(class_error + size_error)
        if best is None or score < best[0]:
            best = score, left.tolist(), right.tolist()
    if best is None:
        raise ValueError("Could not construct a grouped split")
    return best[1], best[2]


def make_lesion_splits(
    metadata: pd.DataFrame,
    seed: int = 42,
    test_fraction: float = 0.20,
    validation_fraction_of_remainder: float = 0.125,
) -> dict[str, pd.DataFrame]:
    """Create approximate 70/10/20 splits without sharing any lesion across splits."""
    if not 0 < test_fraction < 1 or not 0 < validation_fraction_of_remainder < 1:
        raise ValueError("Split fractions must be between zero and one")
    frame = metadata.reset_index(drop=True).copy()
    if not REQUIRED_COLUMNS.issubset(frame.columns):
        raise ValueError(f"Metadata must include {sorted(REQUIRED_COLUMNS)}")
    y = frame["dx"].map(CLASS_TO_IDX)
    if y.isna().any():
        raise ValueError("Unknown diagnosis label in split input")
    groups = frame["lesion_id"].astype(str)
    _, test_indices = _choose_fold(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed), frame, y, groups, test_fraction
    )
    test_index_set = set(test_indices)
    remaining_indices = [index for index in range(len(frame)) if index not in test_index_set]
    development = frame.iloc[remaining_indices].reset_index(drop=True)
    development_y = development["dx"].map(CLASS_TO_IDX)
    development_groups = development["lesion_id"].astype(str)
    train_local, validation_local = _choose_fold(
        StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + 1),
        development,
        development_y,
        development_groups,
        validation_fraction_of_remainder,
    )
    splits = {
        "train": development.iloc[train_local].reset_index(drop=True),
        "validation": development.iloc[validation_local].reset_index(drop=True),
        "test": frame.iloc[test_indices].reset_index(drop=True),
    }
    verify_no_lesion_leakage(splits)
    for split_name, split_frame in splits.items():
        absent = sorted(set(CLASS_NAMES) - set(split_frame["dx"]))
        if absent:
            raise ValueError(f"{split_name} split lacks classes {absent}; adjust the split seed/fractions")
    return splits


def verify_no_lesion_leakage(splits: dict[str, pd.DataFrame]) -> None:
    names = ("train", "validation", "test")
    for i, left_name in enumerate(names):
        left = set(splits[left_name]["lesion_id"].astype(str))
        for right_name in names[i + 1 :]:
            overlap = left & set(splits[right_name]["lesion_id"].astype(str))
            if overlap:
                raise ValueError(f"Lesion leakage between {left_name} and {right_name}: {sorted(overlap)[:5]}")


class HAMDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, CLASS_TO_IDX[row["dx"]]

