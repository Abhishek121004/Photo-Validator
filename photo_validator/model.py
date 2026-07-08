from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from PIL import Image

from .calibration import Thresholds
from .features import VisionSignals, extract_signals
from .image_io import load_image

Label = Literal["rejected", "manual_verification", "acceptable"]


@dataclass(frozen=True)
class ValidationResult:
    label: Label
    confidence: float
    probabilities: dict[str, float]


class PhotoValidator:
    def __init__(self, thresholds: Thresholds | None = None):
        self.thresholds = thresholds or Thresholds()

    @classmethod
    def train(cls, images: Iterable[tuple[str | Path, Label]] | None = None) -> "PhotoValidator":
        """Return a validator using pretrained detectors.

        Training is optional and only used to tune threshold values when labeled
        samples are available.
        """

        model = cls()
        samples = list(images or [])
        if samples:
            try:
                from .tuning import calibrate_thresholds

                model = model.with_thresholds(calibrate_thresholds(model, samples))
            except Exception:
                pass
        return model

    def _rule_based_decision(self, signals: VisionSignals) -> ValidationResult:
        thresholds = self.thresholds

        if min(signals.width, signals.height) < thresholds.min_resolution_side:
            return ValidationResult(
                label="manual_verification",
                confidence=0.35,
                probabilities={"acceptable": 0.0, "manual_verification": 1.0, "rejected": 0.0},
            )

        if signals.blur_variance < thresholds.blur_reject:
            return ValidationResult(
                label="rejected",
                confidence=0.95,
                probabilities={"acceptable": 0.0, "manual_verification": 0.05, "rejected": 0.95},
            )
        if signals.blur_variance < thresholds.blur_manual:
            return ValidationResult(
                label="manual_verification",
                confidence=0.70,
                probabilities={"acceptable": 0.0, "manual_verification": 1.0, "rejected": 0.0},
            )

        if signals.person_count >= 2:
            return ValidationResult(
                label="manual_verification",
                confidence=0.85,
                probabilities={"acceptable": 0.0, "manual_verification": 1.0, "rejected": 0.0},
            )

        if signals.person_count >= 1:
            if signals.person_area_max > 0.28 or signals.lower_body_visible >= 1.0:
                return ValidationResult(
                    label="rejected",
                    confidence=0.92,
                    probabilities={"acceptable": 0.0, "manual_verification": 0.08, "rejected": 0.92},
                )
            if signals.face_count >= 1 and signals.center_skin_ratio >= 0.06:
                return ValidationResult(
                    label="acceptable",
                    confidence=0.88,
                    probabilities={"acceptable": 0.88, "manual_verification": 0.10, "rejected": 0.02},
                )
            return ValidationResult(
                label="manual_verification",
                confidence=0.72,
                probabilities={"acceptable": 0.0, "manual_verification": 1.0, "rejected": 0.0},
            )

        if signals.face_count >= 1 and signals.skin_ratio >= 0.02:
            if signals.center_skin_ratio >= 0.05 and signals.person_area_sum <= 0.20:
                return ValidationResult(
                    label="acceptable",
                    confidence=0.84,
                    probabilities={"acceptable": 0.84, "manual_verification": 0.12, "rejected": 0.04},
                )
            return ValidationResult(
                label="manual_verification",
                confidence=0.68,
                probabilities={"acceptable": 0.0, "manual_verification": 1.0, "rejected": 0.0},
            )

        if signals.skin_ratio < 0.01:
            return ValidationResult(
                label="rejected",
                confidence=0.90,
                probabilities={"acceptable": 0.02, "manual_verification": 0.08, "rejected": 0.90},
            )

        return ValidationResult(
            label="manual_verification",
            confidence=0.60,
            probabilities={"acceptable": 0.0, "manual_verification": 1.0, "rejected": 0.0},
        )

    def predict_image(self, image: Image.Image) -> ValidationResult:
        signals = extract_signals(image)
        return self._rule_based_decision(signals)

    def predict_path(self, path: str | Path) -> ValidationResult:
        return self.predict_image(load_image(path))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps({"thresholds": self.thresholds.to_dict()}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PhotoValidator":
        path = Path(path)
        if not path.exists():
            return cls()

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            thresholds = Thresholds.from_dict(payload.get("thresholds", payload))
            return cls(thresholds=thresholds)
        except Exception:
            return cls()

    def with_thresholds(self, thresholds: Thresholds) -> "PhotoValidator":
        return PhotoValidator(thresholds=thresholds)
