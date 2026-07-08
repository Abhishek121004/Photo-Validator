from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from photo_validator.dataset import normalize_ai_label


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy labeled AI image examples into ai_generated/ and real/ folders."
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="CSV file with path,label columns using ai_generated and real labels.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination directory that will receive ai_generated/ and real/ subfolders.",
    )
    parser.add_argument(
        "--source-root",
        help="Optional root directory for resolving relative image paths from the CSV.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels)
    output_root = Path(args.output)
    source_root = Path(args.source_root) if args.source_root else None

    (output_root / "ai_generated").mkdir(parents=True, exist_ok=True)
    (output_root / "real").mkdir(parents=True, exist_ok=True)

    with open(labels_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain 'path' and 'label' columns.")

        copied = 0
        for index, row in enumerate(reader):
            label = normalize_ai_label(row["label"])
            source_path = Path(row["path"])
            if not source_path.is_absolute() and source_root is not None:
                source_path = source_root / source_path
            if not source_path.exists():
                raise FileNotFoundError(f"Source image not found: {source_path}")

            source_files = [source_path]
            if source_path.is_dir():
                source_files = [
                    path
                    for path in source_path.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ]
                if not source_files:
                    raise ValueError(f"No image files found inside directory: {source_path}")

            for inner_index, file_path in enumerate(source_files):
                destination = output_root / label / f"{index:05d}_{inner_index:03d}_{file_path.name}"
                if args.move:
                    shutil.move(str(file_path), str(destination))
                else:
                    shutil.copy2(str(file_path), str(destination))
                copied += 1

    print(f"Prepared {copied} labeled images in {output_root}")


if __name__ == "__main__":
    main()
