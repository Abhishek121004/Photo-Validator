from __future__ import annotations

import argparse
import csv
from pathlib import Path


AI_FOLDERS = {
    "ai_generated": "ai_generated",
    "real": "real",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the expected AI-generated detector dataset folder structure and an optional CSV template."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where the AI dataset structure will be created.",
    )
    parser.add_argument(
        "--csv-template",
        action="store_true",
        help="Also create an empty ai_labels.csv template in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    for folder_name in AI_FOLDERS:
        (output_root / folder_name).mkdir(parents=True, exist_ok=True)

    if args.csv_template:
        csv_path = output_root / "ai_labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "label"])
            writer.writerow(["", "ai_generated"])
            writer.writerow(["", "real"])
        print(f"Created CSV template at {csv_path}")

    print(f"Created AI dataset structure at {output_root}")


if __name__ == "__main__":
    main()
