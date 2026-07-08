from __future__ import annotations

import argparse
import csv
from pathlib import Path

from photo_validator.dataset import LABEL_FOLDERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the expected dataset folder structure and an optional CSV template."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where the dataset structure will be created.",
    )
    parser.add_argument(
        "--csv-template",
        action="store_true",
        help="Also create an empty labels.csv template in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    for split in ["train", "val"]:
        for folder_name in LABEL_FOLDERS:
            (output_root / split / folder_name).mkdir(parents=True, exist_ok=True)

    if args.csv_template:
        csv_path = output_root / "labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "label"])
        print(f"Created CSV template at {csv_path}")

    print(f"Created dataset structure at {output_root}")


if __name__ == "__main__":
    main()
