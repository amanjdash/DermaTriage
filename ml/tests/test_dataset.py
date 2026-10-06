import pandas as pd
import pytest
from ml.src.dataset import load_metadata, make_lesion_splits, verify_no_lesion_leakage
from ml.src.labels import CLASS_NAMES
from PIL import Image


def _synthetic_metadata(root):
    image_dir = root / "images"
    image_dir.mkdir()
    rows = []
    for label in CLASS_NAMES:
        for group_index in range(10):
            image_id = f"{label}_{group_index:02d}"
            Image.new("RGB", (20, 20), color=(group_index * 13, 50, 100)).save(image_dir / f"{image_id}.jpg")
            rows.append({"image_id": image_id, "lesion_id": f"lesion-{label}-{group_index}", "dx": label})
    csv_path = root / "metadata.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, image_dir


def test_grouped_splits_have_no_lesion_overlap_and_all_classes(tmp_path):
    csv_path, image_dir = _synthetic_metadata(tmp_path)
    metadata, excluded = load_metadata(csv_path, [image_dir])
    splits = make_lesion_splits(metadata, seed=42)
    assert excluded == []
    verify_no_lesion_leakage(splits)
    for frame in splits.values():
        assert set(frame.dx) == set(CLASS_NAMES)


def test_corrupt_and_missing_images_are_reported(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (20, 20), "white").save(image_dir / "good.jpg")
    (image_dir / "bad.jpg").write_bytes(b"not an image")
    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {"image_id": "bad", "lesion_id": "lesion-bad", "dx": "nv"},
            {"image_id": "missing", "lesion_id": "lesion-missing", "dx": "mel"},
            {"image_id": "good", "lesion_id": "lesion-good", "dx": "bkl"},
        ]
    ).to_csv(csv_path, index=False)
    valid, excluded = load_metadata(csv_path, [image_dir])
    assert valid.image_id.tolist() == ["good"]
    assert excluded == ["missing", "bad"]


def test_leakage_guard_detects_shared_lesion():
    left = pd.DataFrame({"lesion_id": ["shared"]})
    right = pd.DataFrame({"lesion_id": ["shared"]})
    with pytest.raises(ValueError, match="Lesion leakage"):
        verify_no_lesion_leakage({"train": left, "validation": right, "test": pd.DataFrame({"lesion_id": []})})

