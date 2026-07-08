from __future__ import annotations

import argparse
import json

from photo_validator.calibration import load_thresholds
from photo_validator.model import PhotoValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a single image.")
    parser.add_argument("--model", help="Optional saved validator config path.")
    parser.add_argument("--image", required=True, help="Image to validate.")
    parser.add_argument(
        "--thresholds",
        help="Optional thresholds JSON to override the thresholds embedded in the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = PhotoValidator.load(args.model) if args.model else PhotoValidator()
    if args.thresholds:
        model = model.with_thresholds(load_thresholds(args.thresholds))
    result = model.predict_path(args.image)
    print(
        json.dumps(
            {
                "label": result.label,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
