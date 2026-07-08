from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    MobileNet_V3_Small_Weights = None
    mobilenet_v3_small = None

from .features import VisionSignals, extract_signals
from .image_io import load_image

AiLabel = Literal["ai_generated", "real"]

HANDCRAFTED_FEATURE_NAMES = [
    "blur_variance",
    "edge_density",
    "entropy",
    "brightness_std",
    "saturation_std",
    "face_count",
    "face_area_sum",
    "person_count",
    "person_area_sum",
    "skin_ratio",
    "center_skin_ratio",
    "ai_noise_std",
    "ai_frequency_ratio",
    "ai_patch_std",
    "ai_flat_patch_ratio",
    "ai_patch_var_std",
]

EMBEDDING_DIM = 576
AI_FEATURE_NAMES = [f"embedding_{index:03d}" for index in range(EMBEDDING_DIM)] + HANDCRAFTED_FEATURE_NAMES


@dataclass(frozen=True)
class AiDetectionResult:
    score: float
    is_ai_generated: bool
    probabilities: dict[str, float]
    backend: str
    reason: str


def _heuristic_score(signals: VisionSignals) -> tuple[float, str]:
    noise = float(signals.ai_noise_std)
    frequency = float(signals.ai_frequency_ratio)
    patch_std = float(signals.ai_patch_std)
    flat_patch_ratio = float(signals.ai_flat_patch_ratio)
    patch_var_std = float(signals.ai_patch_var_std)

    smoothness = np.clip((0.030 - noise) / 0.030, 0.0, 1.0)
    frequency_score = np.clip((0.34 - frequency) / 0.34, 0.0, 1.0)
    patch_score = np.clip((0.14 - patch_std) / 0.14, 0.0, 1.0)
    flat_patch_score = np.clip((flat_patch_ratio - 0.22) / 0.45, 0.0, 1.0)
    patch_var_score = np.clip((0.015 - patch_var_std) / 0.015, 0.0, 1.0)
    score = float(
        0.30 * smoothness
        + 0.20 * frequency_score
        + 0.20 * patch_score
        + 0.20 * flat_patch_score
        + 0.10 * patch_var_score
    )

    if noise < 0.014:
        reason = "low_sensor_noise"
    elif frequency < 0.12:
        reason = "low_high_frequency_energy"
    elif patch_std < 0.04:
        reason = "uniform_patch_texture"
    elif flat_patch_ratio > 0.40:
        reason = "large_flat_regions"
    elif patch_var_std < 0.006:
        reason = "low_patch_variation"
    else:
        reason = "heuristic_forensic_score"
    return score, reason


@lru_cache(maxsize=1)
def _mobilenet_bundle() -> tuple[object, object] | None:
    if torch is None or MobileNet_V3_Small_Weights is None or mobilenet_v3_small is None:
        return None

    try:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval()
        return model, weights.transforms()
    except Exception:
        return None


def _pretrained_embedding(image: Image.Image) -> np.ndarray | None:
    bundle = _mobilenet_bundle()
    if bundle is None:
        return None

    model, preprocess = bundle
    if torch is None:
        return None

    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        embedding = model(tensor).squeeze(0)
    return embedding.detach().cpu().numpy().astype(np.float32)


def _handcrafted_features(signals: VisionSignals) -> np.ndarray:
    return np.array(
        [
            signals.blur_variance,
            signals.edge_density,
            signals.entropy,
            signals.brightness_std,
            signals.saturation_std,
            signals.face_count,
            signals.face_area_sum,
            signals.person_count,
            signals.person_area_sum,
            signals.skin_ratio,
            signals.center_skin_ratio,
            signals.ai_noise_std,
            signals.ai_frequency_ratio,
            signals.ai_patch_std,
            signals.ai_flat_patch_ratio,
            signals.ai_patch_var_std,
        ],
        dtype=np.float32,
    )


def _feature_vector(signals: VisionSignals, image: Image.Image | None = None, use_pretrained: bool = True) -> np.ndarray:
    pieces: list[np.ndarray] = []
    if use_pretrained:
        embedding = _pretrained_embedding(image) if image is not None else None
        if embedding is None:
            return _handcrafted_features(signals)
        pieces.append(embedding)
    pieces.append(_handcrafted_features(signals))
    return np.concatenate(pieces).astype(np.float32)


@dataclass
class AiGeneratedDetector:
    model: Pipeline | None = None
    threshold: float = 0.5
    use_pretrained_backbone: bool = True

    @classmethod
    def train(cls, images: Iterable[tuple[str | Path, AiLabel]], threshold: float = 0.5) -> "AiGeneratedDetector":
        samples = list(images)
        if not samples:
            return cls()

        use_pretrained = _mobilenet_bundle() is not None
        features: list[np.ndarray] = []
        labels: list[int] = []
        for path, label in samples:
            image = load_image(path)
            signals = extract_signals(image)
            features.append(_feature_vector(signals, image=image, use_pretrained=use_pretrained))
            labels.append(1 if label == "ai_generated" else 0)

        if len(set(labels)) < 2:
            return cls(threshold=threshold, use_pretrained_backbone=use_pretrained)

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=1000,
                    ),
                ),
            ]
        )
        model.fit(np.vstack(features), np.array(labels, dtype=np.int32))
        return cls(model=model, threshold=threshold, use_pretrained_backbone=use_pretrained)

    def predict_signals(self, signals: VisionSignals, image: Image.Image | None = None) -> AiDetectionResult:
        if self.model is None:
            score, reason = _heuristic_score(signals)
            return AiDetectionResult(
                score=score,
                is_ai_generated=score >= self.threshold,
                probabilities={"real": 1.0 - score, "ai_generated": score},
                backend="heuristic",
                reason=reason,
            )

        if self.use_pretrained_backbone and _mobilenet_bundle() is None:
            score, reason = _heuristic_score(signals)
            return AiDetectionResult(
                score=score,
                is_ai_generated=score >= self.threshold,
                probabilities={"real": 1.0 - score, "ai_generated": score},
                backend="heuristic_fallback",
                reason=reason,
            )

        if self.use_pretrained_backbone and image is None:
            score, reason = _heuristic_score(signals)
            return AiDetectionResult(
                score=score,
                is_ai_generated=score >= self.threshold,
                probabilities={"real": 1.0 - score, "ai_generated": score},
                backend="heuristic_fallback_no_image",
                reason=reason,
            )

        feature_vector = _feature_vector(
            signals,
            image=image,
            use_pretrained=self.use_pretrained_backbone,
        ).reshape(1, -1)
        score = float(self.model.predict_proba(feature_vector)[0, 1])
        if score >= self.threshold:
            reason = "classifier_probability_high"
        elif score >= 0.35:
            reason = "classifier_uncertain"
        else:
            reason = "classifier_probability_low"
        return AiDetectionResult(
            score=score,
            is_ai_generated=score >= self.threshold,
            probabilities={"real": 1.0 - score, "ai_generated": score},
            backend="mobilenet_v3_small+logreg" if self.use_pretrained_backbone else "logreg",
            reason=reason,
        )

    def predict_image(self, image: Image.Image) -> AiDetectionResult:
        signals = extract_signals(image)
        return self.predict_signals(signals, image=image)

    def predict_path(self, path: str | Path) -> AiDetectionResult:
        return self.predict_image(load_image(path))

    def save(self, path: str | Path) -> None:
        payload = {
            "model": self.model,
            "threshold": self.threshold,
            "use_pretrained_backbone": self.use_pretrained_backbone,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "AiGeneratedDetector":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            payload = joblib.load(path)
        except Exception:
            return cls()
        if isinstance(payload, dict) and "model" in payload:
            return cls(
                model=payload.get("model"),
                threshold=float(payload.get("threshold", 0.5)),
                use_pretrained_backbone=bool(payload.get("use_pretrained_backbone", True)),
            )
        return cls(model=payload)
