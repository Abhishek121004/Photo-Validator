from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score

from .calibration import Thresholds
from .model import Label, PhotoValidator


def evaluate_model(model: PhotoValidator, samples: Iterable[tuple[str | Path, Label]]) -> dict[str, float]:
    paths: list[str | Path] = []
    labels: list[Label] = []
    for path, label in samples:
        paths.append(path)
        labels.append(label)

    predictions = [model.predict_path(path).label for path in paths]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def calibrate_thresholds(model: PhotoValidator, samples: Iterable[tuple[str | Path, Label]]) -> Thresholds:
    paths: list[str | Path] = []
    labels: list[Label] = []
    for path, label in samples:
        paths.append(path)
        labels.append(label)

    if not paths:
        return model.thresholds

    current = model.thresholds
    blur_reject_candidates = [15.0, 20.0, 30.0, 40.0, 50.0]
    blur_manual_candidates = [35.0, 45.0, 55.0, 65.0, 75.0]
    confidence_candidates = [0.45, 0.5, 0.55, 0.6, 0.65]
    margin_candidates = [0.05, 0.1, 0.12, 0.15, 0.2]

    best_thresholds = current
    best_score = (-1.0, -1.0)

    for blur_reject, blur_manual, low_confidence, low_margin in product(
        blur_reject_candidates,
        blur_manual_candidates,
        confidence_candidates,
        margin_candidates,
    ):
        if blur_manual <= blur_reject:
            continue
        candidate = replace(
            current,
            blur_reject=blur_reject,
            blur_manual=blur_manual,
            low_confidence=low_confidence,
            low_margin=low_margin,
        )
        tuned = model.with_thresholds(candidate)
        predictions = [tuned.predict_path(path).label for path in paths]
        macro_f1 = float(f1_score(labels, predictions, average="macro"))
        accuracy = float(accuracy_score(labels, predictions))
        score = (macro_f1, accuracy)
        if score > best_score:
            best_score = score
            best_thresholds = candidate

    return best_thresholds
