from __future__ import annotations

import argparse
from pathlib import Path

from photo_validator.calibration import save_thresholds
from photo_validator.dataset import load_folder_dataset, load_labeled_csv, load_train_val_folder_dataset
from photo_validator.model import PhotoValidator
from photo_validator.tuning import calibrate_thresholds, evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate photo validator thresholds on a validation set.")
    parser.add_argument("--model", help="Optional saved validator config path.")
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset root, a validation folder, or a CSV with path,label columns.",
    )
    parser.add_argument(
        "--output",
        default="photo_validator_thresholds.json",
        help="Where to write the calibrated thresholds JSON.",
    )
    parser.add_argument(
        "--write-model",
        help="Optional path to write the updated model with calibrated thresholds embedded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = PhotoValidator.load(args.model) if args.model else PhotoValidator()

    data_path = Path(args.data)
    if data_path.is_file() and data_path.suffix.lower() == ".csv":
        samples = load_labeled_csv(data_path)
    elif (data_path / "train").exists() and (data_path / "val").exists():
        _, samples = load_train_val_folder_dataset(data_path)
    else:
        samples = load_folder_dataset(data_path)

    thresholds = calibrate_thresholds(model, samples)
    save_thresholds(args.output, thresholds)
    print(f"Saved thresholds to {args.output}")

    tuned = model.with_thresholds(thresholds)
    metrics = evaluate_model(tuned, samples)
    print(f"Validation metrics: {metrics}")

    if args.write_model:
        tuned.save(args.write_model)
        print(f"Saved calibrated model to {args.write_model}")


if __name__ == "__main__":
    main()
