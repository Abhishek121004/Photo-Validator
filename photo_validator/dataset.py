from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path
from typing import Iterable

from .ai_detector import AiLabel
from .model import Label


LABEL_ALIASES: dict[str, Label] = {
    "acceptable": "acceptable",
    "accept": "acceptable",
    "manual": "manual_verification",
    "manual_verification": "manual_verification",
    "manual verification": "manual_verification",
    "rejected": "rejected",
    "reject": "rejected",
}

LABEL_FOLDERS: dict[str, Label] = {
    "acceptable": "acceptable",
    "manual_verification": "manual_verification",
    "rejected": "rejected",
}

AI_LABEL_ALIASES: dict[str, AiLabel] = {
    "ai_generated": "ai_generated",
    "ai": "ai_generated",
    "generated": "ai_generated",
    "real": "real",
    "photo": "real",
    "genuine": "real",
}


def normalize_label(value: str) -> Label:
    key = value.strip().lower().replace("-", "_")
    if key not in LABEL_ALIASES:
        raise ValueError(f"Unknown label: {value!r}")
    return LABEL_ALIASES[key]


def normalize_ai_label(value: str) -> AiLabel:
    key = value.strip().lower().replace("-", "_")
    if key not in AI_LABEL_ALIASES:
        raise ValueError(f"Unknown AI label: {value!r}")
    return AI_LABEL_ALIASES[key]


def load_labeled_csv(path: str | Path) -> list[tuple[str, Label]]:
    rows: list[tuple[str, Label]] = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain 'path' and 'label' columns.")
        for row in reader:
            rows.append((row["path"], normalize_label(row["label"])))
    return rows


def load_folder_dataset(root: str | Path) -> list[tuple[str, Label]]:
    root_path = Path(root)
    samples: list[tuple[str, Label]] = []
    for folder_name, label in LABEL_FOLDERS.items():
        folder = root_path / folder_name
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                samples.append((str(path), label))
    if not samples:
        raise ValueError(
            "No labeled images found. Use either a CSV with path,label columns or "
            "a folder tree containing acceptable/, manual_verification/, and rejected/."
        )
    return samples


def load_split_folder_dataset(root: str | Path, split: str) -> list[tuple[str, Label]]:
    root_path = Path(root)
    split_root = root_path / split
    if not split_root.exists():
        raise ValueError(f"Split folder not found: {split_root}")

    samples: list[tuple[str, Label]] = []
    for folder_name, label in LABEL_FOLDERS.items():
        folder = split_root / folder_name
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                samples.append((str(path), label))
    if not samples:
        raise ValueError(
            f"No labeled images found in {split_root}. Expected acceptable/, manual_verification/, and rejected/."
        )
    return samples


def load_train_val_folder_dataset(root: str | Path) -> tuple[list[tuple[str, Label]], list[tuple[str, Label]]]:
    root_path = Path(root)
    train_root = root_path / "train"
    val_root = root_path / "val"
    if train_root.exists() and val_root.exists():
        train_samples = load_split_folder_dataset(root_path, "train")
        val_samples = load_split_folder_dataset(root_path, "val")
        if train_samples or val_samples:
            return train_samples, val_samples

    samples = load_folder_dataset(root_path)
    return samples, []


def load_ai_labeled_csv(path: str | Path) -> list[tuple[str, AiLabel]]:
    rows: list[tuple[str, AiLabel]] = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain 'path' and 'label' columns.")
        for row in reader:
            rows.append((row["path"], normalize_ai_label(row["label"])))
    return rows


def load_ai_folder_dataset(root: str | Path) -> list[tuple[str, AiLabel]]:
    root_path = Path(root)
    samples: list[tuple[str, AiLabel]] = []
    for folder_name, label in {"ai_generated": "ai_generated", "real": "real"}.items():
        folder = root_path / folder_name
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                samples.append((str(path), label))
    if not samples:
        raise ValueError(
            "No AI labeled images found. Use either a CSV with path,label columns or "
            "a folder tree containing ai_generated/ and real/."
        )
    return samples


def iter_image_paths(root: str | Path) -> Iterable[str]:
    for path in Path(root).rglob("*"):
        if path.is_file():
            yield str(path)


def split_labeled_samples(
    samples: list[tuple[str, Label]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[str, Label]], list[tuple[str, Label]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    grouped: dict[Label, list[str]] = {label: [] for label in LABEL_FOLDERS.values()}
    for path, label in samples:
        grouped[label].append(path)

    rng = random.Random(seed)
    train_samples: list[tuple[str, Label]] = []
    val_samples: list[tuple[str, Label]] = []

    for label, paths in grouped.items():
        paths = list(paths)
        rng.shuffle(paths)
        if not paths:
            continue

        if len(paths) == 1:
            train_paths = paths
            val_paths: list[str] = []
        else:
            val_count = int(round(len(paths) * val_ratio))
            val_count = max(1, min(len(paths) - 1, val_count))
            val_paths = paths[:val_count]
            train_paths = paths[val_count:]

        train_samples.extend((path, label) for path in train_paths)
        val_samples.extend((path, label) for path in val_paths)

    return train_samples, val_samples


def write_split_dataset(
    samples: list[tuple[str, Label]],
    output_root: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    move: bool = False,
) -> tuple[list[tuple[str, Label]], list[tuple[str, Label]]]:
    output_root = Path(output_root)
    train_samples, val_samples = split_labeled_samples(samples, val_ratio=val_ratio, seed=seed)

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        for label in LABEL_FOLDERS.values():
            (output_root / split_name / label).mkdir(parents=True, exist_ok=True)

        for index, (source, label) in enumerate(split_samples):
            source_path = Path(source)
            destination = output_root / split_name / label / f"{index:05d}_{source_path.name}"
            if move:
                shutil.move(str(source_path), str(destination))
            else:
                shutil.copy2(str(source_path), str(destination))

    return train_samples, val_samples
