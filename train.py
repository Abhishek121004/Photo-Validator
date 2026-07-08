from __future__ import annotations

import argparse
from pathlib import Path

from photo_validator.dataset import load_folder_dataset, load_labeled_csv, load_train_val_folder_dataset
from photo_validator.model import PhotoValidator
from photo_validator.tuning import calibrate_thresholds, evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a lightweight photo validator.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a labeled CSV file or a folder containing labeled subfolders.",
    )
    parser.add_argument(
        "--output",
        default="thresholds.json",
        help="Where to write the saved validator config.",
    )
    parser.add_argument(
        "--thresholds-output",
        help="Optional path to save calibrated thresholds as JSON when a validation split is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if data_path.is_file() and data_path.suffix.lower() == ".csv":
        samples = load_labeled_csv(data_path)
        val_samples = []
    else:
        samples, val_samples = load_train_val_folder_dataset(data_path)
        if not samples:
            samples = load_folder_dataset(data_path)

    model = PhotoValidator.train(samples)

    if val_samples:
        thresholds = calibrate_thresholds(model, val_samples)
        model = model.with_thresholds(thresholds)
        metrics = evaluate_model(model, val_samples)
        print(f"Validation metrics: {metrics}")
        if args.thresholds_output:
            from photo_validator.calibration import save_thresholds

            save_thresholds(args.thresholds_output, thresholds)
            print(f"Saved thresholds to {args.thresholds_output}")

    model.save(args.output)
    print(f"Saved validator config to {args.output}")


if __name__ == "__main__":
    main()
