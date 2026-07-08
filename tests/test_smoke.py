from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from photo_validator.calibration import Thresholds, load_thresholds, save_thresholds
from photo_validator.dataset import load_train_val_folder_dataset, write_split_dataset
from photo_validator.features import extract_features
from photo_validator.model import PhotoValidator


def make_image(path: Path, color: tuple[int, int, int], add_subject: bool) -> None:
    image = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(image)
    if add_subject:
        draw.ellipse((88, 40, 168, 120), fill=(220, 180, 160))
        draw.rectangle((72, 120, 184, 230), fill=(40, 60, 100))
    image.save(path)


def test_feature_vector_length(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    make_image(path, (240, 240, 240), True)
    image = Image.open(path)
    features = extract_features(image)
    assert len(features.values) > 0


def test_training_and_prediction(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    make_image(path, (220, 210, 200), True)

    model = PhotoValidator()
    result = model.predict_path(path)
    assert result.label in {"acceptable", "manual_verification", "rejected"}
    assert np.isfinite(result.confidence)


def test_split_dataset_loader_and_threshold_roundtrip(tmp_path: Path) -> None:
    for split in ["train", "val"]:
        for label in ["acceptable", "manual_verification", "rejected"]:
            folder = tmp_path / split / label
            folder.mkdir(parents=True, exist_ok=True)
            make_image(folder / f"{split}_{label}.png", (200, 200, 200), label != "rejected")

    train_samples, val_samples = load_train_val_folder_dataset(tmp_path)
    assert len(train_samples) == 3
    assert len(val_samples) == 3

    thresholds = Thresholds(blur_reject=12.0, blur_manual=34.0)
    path = tmp_path / "thresholds.json"
    save_thresholds(path, thresholds)
    loaded = load_thresholds(path)
    assert loaded.blur_reject == 12.0
    assert loaded.blur_manual == 34.0


def test_flat_split_dataset_writer(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for label in ["acceptable", "manual_verification", "rejected"]:
        folder = source / label
        folder.mkdir(parents=True, exist_ok=True)
        for idx in range(4):
            make_image(folder / f"{label}_{idx}.png", (190, 190, 190), label != "rejected")

    samples = [
        (str(path), label)
        for label in ["acceptable", "manual_verification", "rejected"]
        for path in (source / label).glob("*.png")
    ]

    train_samples, val_samples = write_split_dataset(samples, tmp_path / "split", val_ratio=0.25, seed=7)
    assert train_samples
    assert val_samples
    assert (tmp_path / "split" / "train" / "acceptable").exists()
    assert (tmp_path / "split" / "val" / "rejected").exists()


def test_dataset_bootstrap_structure(tmp_path: Path) -> None:
    from init_dataset import main as init_main

    import sys

    original_argv = sys.argv
    try:
        sys.argv = ["init_dataset.py", "--output", str(tmp_path / "dataset"), "--csv-template"]
        init_main()
    finally:
        sys.argv = original_argv

    assert (tmp_path / "dataset" / "train" / "acceptable").exists()
    assert (tmp_path / "dataset" / "val" / "rejected").exists()
    assert (tmp_path / "dataset" / "labels.csv").exists()
