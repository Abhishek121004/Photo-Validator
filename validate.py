from __future__ import annotations

import argparse
import json
from pathlib import Path

from photo_validator.ai_detector import AiGeneratedDetector
from photo_validator.calibration import load_thresholds
from photo_validator.model import PhotoValidator


AI_DETECTOR_PATH = Path("ai_generated_detector.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a single image.")
    parser.add_argument("--model", help="Optional saved validator config path.")
    parser.add_argument("--image", required=True, help="Image to validate.")
    parser.add_argument(
        "--ai-detector",
        help="Optional trained AI-generated detector joblib path.",
    )
    parser.add_argument(
        "--thresholds",
        help="Optional thresholds JSON to override the thresholds embedded in the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector_path = Path(args.ai_detector) if args.ai_detector else AI_DETECTOR_PATH
    ai_detector = AiGeneratedDetector.load(detector_path)
    model = PhotoValidator.load(args.model) if args.model else PhotoValidator(ai_detector=ai_detector)
    model.ai_detector = ai_detector
    model.ai_detector.threshold = model.thresholds.ai_generated_flag
    if args.thresholds:
        model = model.with_thresholds(load_thresholds(args.thresholds))
    result = model.predict_path(args.image)
    print(
        json.dumps(
            {
                "label": result.label,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "ai_generated": result.ai_generated,
                "ai_generated_score": result.ai_generated_score,
                "ai_generated_reason": result.ai_generated_reason,
                "ai_generated_backend": result.ai_generated_backend,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
