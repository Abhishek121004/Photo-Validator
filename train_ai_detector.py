from __future__ import annotations

import argparse
from pathlib import Path

from photo_validator.ai_detector import AiGeneratedDetector
from photo_validator.dataset import load_ai_folder_dataset, load_ai_labeled_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight AI-generated image detector.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a labeled CSV file or a folder containing ai_generated/ and real/ subfolders.",
    )
    parser.add_argument(
        "--output",
        default="ai_generated_detector.joblib",
        help="Where to write the trained detector.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if data_path.is_file() and data_path.suffix.lower() == ".csv":
        samples = load_ai_labeled_csv(data_path)
    else:
        samples = load_ai_folder_dataset(data_path)

    detector = AiGeneratedDetector.train(samples)
    detector.save(args.output)
    print(f"Saved AI-generated detector to {args.output}")


if __name__ == "__main__":
    main()
