from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from photo_validator.ai_detector import AiGeneratedDetector
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


def make_noisy_image(path: Path) -> None:
    rng = np.random.default_rng(7)
    noise = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    Image.fromarray(noise, mode="RGB").save(path)


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
    assert isinstance(result.ai_generated, bool)
    assert 0.0 <= result.ai_generated_score <= 1.0


def test_ai_detector_training_and_prediction(tmp_path: Path) -> None:
    samples: list[tuple[str, str]] = []
    for idx in range(3):
        ai_path = tmp_path / f"ai_{idx}.png"
        real_path = tmp_path / f"real_{idx}.png"
        make_image(ai_path, (220, 210, 200), False)
        make_noisy_image(real_path)
        samples.append((str(ai_path), "ai_generated"))
        samples.append((str(real_path), "real"))

    detector = AiGeneratedDetector.train(samples)
    ai_result = detector.predict_path(tmp_path / "ai_0.png")
    real_result = detector.predict_path(tmp_path / "real_0.png")

    assert 0.0 <= ai_result.score <= 1.0
    assert 0.0 <= real_result.score <= 1.0
    assert ai_result.score >= real_result.score
    assert ai_result.is_ai_generated in {True, False}


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


def test_ai_dataset_bootstrap_structure(tmp_path: Path) -> None:
    from init_ai_dataset import main as init_ai_main

    import sys

    original_argv = sys.argv
    try:
        sys.argv = ["init_ai_dataset.py", "--output", str(tmp_path / "ai_dataset"), "--csv-template"]
        init_ai_main()
    finally:
        sys.argv = original_argv

    assert (tmp_path / "ai_dataset" / "ai_generated").exists()
    assert (tmp_path / "ai_dataset" / "real").exists()
    assert (tmp_path / "ai_dataset" / "ai_labels.csv").exists()


def test_prepare_ai_dataset_copies_files(tmp_path: Path) -> None:
    from prepare_ai_dataset import main as prepare_main

    import sys

    source_root = tmp_path / "images"
    source_root.mkdir(parents=True, exist_ok=True)
    ai_image = source_root / "ai.png"
    real_image = source_root / "real.png"
    make_image(ai_image, (220, 210, 200), False)
    make_noisy_image(real_image)

    labels_csv = tmp_path / "ai_labels.csv"
    labels_csv.write_text(
        "path,label\nai.png,ai_generated\nreal.png,real\n",
        encoding="utf-8",
    )

    original_argv = sys.argv
    try:
        sys.argv = [
            "prepare_ai_dataset.py",
            "--labels",
            str(labels_csv),
            "--source-root",
            str(source_root),
            "--output",
            str(tmp_path / "prepared_ai"),
        ]
        prepare_main()
    finally:
        sys.argv = original_argv

    assert list((tmp_path / "prepared_ai" / "ai_generated").glob("*.png"))
    assert list((tmp_path / "prepared_ai" / "real").glob("*.png"))


def test_prepare_ai_dataset_expands_directory_rows(tmp_path: Path) -> None:
    from prepare_ai_dataset import main as prepare_main

    import sys

    source_root = tmp_path / "images"
    real_folder = source_root / "Real Images"
    ai_folder = source_root / "AI Images"
    real_folder.mkdir(parents=True, exist_ok=True)
    ai_folder.mkdir(parents=True, exist_ok=True)
    make_noisy_image(real_folder / "real_0.png")
    make_noisy_image(ai_folder / "ai_0.png")

    labels_csv = tmp_path / "ai_labels.csv"
    labels_csv.write_text(
        "path,label\nReal Images,real\nAI Images,ai_generated\n",
        encoding="utf-8",
    )

    original_argv = sys.argv
    try:
        sys.argv = [
            "prepare_ai_dataset.py",
            "--labels",
            str(labels_csv),
            "--source-root",
            str(source_root),
            "--output",
            str(tmp_path / "prepared_ai_dirs"),
        ]
        prepare_main()
    finally:
        sys.argv = original_argv

    assert list((tmp_path / "prepared_ai_dirs" / "real").glob("*.png"))
    assert list((tmp_path / "prepared_ai_dirs" / "ai_generated").glob("*.png"))
