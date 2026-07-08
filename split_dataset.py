from __future__ import annotations

import argparse
from pathlib import Path

from photo_validator.dataset import load_folder_dataset, write_split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a flat labeled dataset into train/val folders.")
    parser.add_argument(
        "--source",
        required=True,
        help="Root folder containing acceptable/, manual_verification/, and rejected/ subfolders.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination folder where train/ and val/ will be created.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation fraction per class. Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    samples = load_folder_dataset(source)
    train_samples, val_samples = write_split_dataset(
        samples,
        args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        move=args.move,
    )
    print(
        f"Created split dataset at {args.output} "
        f"with {len(train_samples)} train and {len(val_samples)} val images."
    )


if __name__ == "__main__":
    main()
