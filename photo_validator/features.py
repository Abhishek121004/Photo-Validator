from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover - optional dependency
    _cv2 = None


THUMBNAIL_SIZE = 4
THUMBNAIL_AREA = THUMBNAIL_SIZE * THUMBNAIL_SIZE
MAX_ANALYSIS_SIDE = 640

BASE_FEATURE_NAMES = [
    "width",
    "height",
    "megapixels",
    "aspect_ratio",
    "brightness_mean",
    "brightness_std",
    "contrast",
    "saturation_mean",
    "saturation_std",
    "blur_variance",
    "edge_density",
    "entropy",
    "face_count",
    "face_confidence_max",
    "face_confidence_mean",
    "face_area_max",
    "face_area_sum",
    "face_center_offset",
    "pose_present",
    "pose_visible_landmarks",
    "pose_bbox_width",
    "pose_bbox_height",
    "pose_bbox_area",
    "shoulders_visible",
    "hips_visible",
    "knees_visible",
    "ankles_visible",
    "lower_body_visible",
    "upper_body_visible",
    "face_to_pose_height_ratio",
    "skin_ratio",
    "center_skin_ratio",
    "person_count",
    "person_confidence_max",
    "person_area_max",
    "person_area_sum",
    "person_center_offset",
    "ai_generated_score",
    "ai_noise_std",
    "ai_frequency_ratio",
    "ai_patch_std",
    "ai_flat_patch_ratio",
    "ai_patch_var_std",
]

FEATURE_NAMES = (
    BASE_FEATURE_NAMES
    + [f"gray_thumb_{idx:02d}" for idx in range(THUMBNAIL_AREA)]
    + [f"sat_thumb_{idx:02d}" for idx in range(THUMBNAIL_AREA)]
)


@dataclass(frozen=True)
class FeatureVector:
    values: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(FEATURE_NAMES, self.values)}


@dataclass(frozen=True)
class VisionSignals:
    backend: str
    width: int
    height: int
    megapixels: float
    aspect_ratio: float
    brightness_mean: float
    brightness_std: float
    contrast: float
    saturation_mean: float
    saturation_std: float
    blur_variance: float
    edge_density: float
    entropy: float
    face_count: float
    face_confidence_max: float
    face_confidence_mean: float
    face_area_max: float
    face_area_sum: float
    face_center_offset: float
    pose_present: float
    pose_visible_landmarks: float
    pose_bbox_width: float
    pose_bbox_height: float
    pose_bbox_area: float
    shoulders_visible: float
    hips_visible: float
    knees_visible: float
    ankles_visible: float
    lower_body_visible: float
    upper_body_visible: float
    face_to_pose_height_ratio: float
    skin_ratio: float
    center_skin_ratio: float
    person_count: float
    person_confidence_max: float
    person_area_max: float
    person_area_sum: float
    person_center_offset: float
    ai_generated_score: float
    ai_noise_std: float
    ai_frequency_ratio: float
    ai_patch_std: float
    ai_flat_patch_ratio: float
    ai_patch_var_std: float
    gray_thumbnail: np.ndarray
    sat_thumbnail: np.ndarray


def _grayscale(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    if _cv2 is not None:
        gray_u8 = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
        return float(_cv2.Laplacian(gray_u8, _cv2.CV_64F).var())

    center = gray[1:-1, 1:-1]
    lap = (
        -4.0 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def _edge_density(gray: np.ndarray) -> float:
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0

    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    magnitude = np.zeros_like(gray)
    magnitude[:, :-1] += gx
    magnitude[:-1, :] += gy
    threshold = np.percentile(magnitude, 90)
    if threshold <= 0:
        return 0.0
    return float((magnitude >= threshold).mean())


def _entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    hist /= total
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    return float(-(hist * np.log2(hist)).sum())


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    return (
        (y > 40.0)
        & (cb >= 77.0)
        & (cb <= 127.0)
        & (cr >= 133.0)
        & (cr <= 173.0)
    )


def _thumbnail(image: Image.Image, mode: str) -> np.ndarray:
    thumb = ImageOps.contain(
        image,
        (THUMBNAIL_SIZE, THUMBNAIL_SIZE),
        method=Image.Resampling.BILINEAR,
    )
    if thumb.size != (THUMBNAIL_SIZE, THUMBNAIL_SIZE):
        thumb = ImageOps.pad(
            image,
            (THUMBNAIL_SIZE, THUMBNAIL_SIZE),
            method=Image.Resampling.BILINEAR,
            color=(0, 0, 0),
            centering=(0.5, 0.5),
        )
    return np.asarray(thumb.convert(mode), dtype=np.float32) / 255.0


def _analysis_image(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    return ImageOps.contain(rgb, (MAX_ANALYSIS_SIDE, MAX_ANALYSIS_SIDE), method=Image.Resampling.BILINEAR)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _average_filter(gray: np.ndarray, size: int = 5) -> np.ndarray:
    pad = size // 2
    padded = np.pad(gray, pad, mode="reflect")
    out = np.zeros_like(gray)
    for dy in range(size):
        for dx in range(size):
            out += padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]]
    return out / float(size * size)


def _ai_generation_metrics(gray: np.ndarray) -> tuple[float, float, float, float, float, float]:
    if gray.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if _cv2 is not None:
        gray_u8 = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
        blurred = _cv2.GaussianBlur(gray_u8, (5, 5), 0).astype(np.float32) / 255.0
    else:
        blurred = _average_filter(gray)

    residual = gray - blurred
    noise_std = float(residual.std())

    small = gray
    if min(gray.shape) > 128:
        step = max(1, min(gray.shape) // 128)
        small = gray[::step, ::step]
    if small.shape[0] >= 8 and small.shape[1] >= 8:
        centered = small - float(small.mean())
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered)))
        yy, xx = np.ogrid[:spectrum.shape[0], :spectrum.shape[1]]
        cy = (spectrum.shape[0] - 1) / 2.0
        cx = (spectrum.shape[1] - 1) / 2.0
        radius = np.hypot(yy - cy, xx - cx)
        max_radius = float(radius.max()) or 1.0
        high_frequency_ratio = float(spectrum[radius >= max_radius * 0.45].sum() / max(float(spectrum.sum()), 1e-6))
    else:
        high_frequency_ratio = 0.0

    patch_h = max(8, gray.shape[0] // 8)
    patch_w = max(8, gray.shape[1] // 8)
    patch_means: list[float] = []
    patch_vars: list[float] = []
    flat_patch_count = 0
    patch_count = 0
    for y in range(0, gray.shape[0], patch_h):
        for x in range(0, gray.shape[1], patch_w):
            patch = gray[y : min(gray.shape[0], y + patch_h), x : min(gray.shape[1], x + patch_w)]
            if patch.size:
                patch_count += 1
                patch_mean = float(patch.mean())
                patch_var = float(patch.var())
                patch_means.append(patch_mean)
                patch_vars.append(patch_var)
                if patch_var < 0.0025:
                    flat_patch_count += 1
    patch_std = float(np.std(patch_means)) if patch_means else 0.0
    patch_var_std = float(np.std(patch_vars)) if patch_vars else 0.0
    flat_patch_ratio = float(flat_patch_count / max(1, patch_count))

    smoothness_score = _clamp01((0.030 - noise_std) / 0.030)
    frequency_score = _clamp01((0.34 - high_frequency_ratio) / 0.34)
    patch_score = _clamp01((0.14 - patch_std) / 0.14)
    flat_patch_score = _clamp01((flat_patch_ratio - 0.22) / 0.45)
    patch_var_score = _clamp01((0.015 - patch_var_std) / 0.015)
    ai_score = float(
        0.30 * smoothness_score
        + 0.20 * frequency_score
        + 0.20 * patch_score
        + 0.20 * flat_patch_score
        + 0.10 * patch_var_score
    )
    return ai_score, noise_std, high_frequency_ratio, patch_std, flat_patch_ratio, patch_var_std


@lru_cache(maxsize=1)
def _person_detector():
    if _cv2 is None:
        return None
    hog = _cv2.HOGDescriptor()
    hog.setSVMDetector(_cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def _fallback_face_metrics(rgb: np.ndarray) -> tuple[float, float, float, float, float, float]:
    skin = _skin_mask(rgb)
    h, w = skin.shape
    center = skin[h // 4 : max(h // 4 + 1, h * 3 // 4), w // 4 : max(w // 4 + 1, w * 3 // 4)]
    face_like = float(center.mean()) if center.size else float(skin.mean())
    face_count = 1.0 if face_like > 0.03 else 0.0
    confidence = min(0.99, face_like * 8.0)
    area = float(face_like)
    center_offset = 0.0 if face_count == 0 else 0.2
    return face_count, confidence, confidence, area, area, center_offset


def _fallback_pose_metrics(rgb: np.ndarray) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    skin = _skin_mask(rgb)
    h, w = skin.shape
    upper = skin[: max(1, h // 2), :]
    lower = skin[h // 2 :, :]
    upper_body_visible = 1.0 if upper.mean() > 0.02 else 0.0
    lower_body_visible = 1.0 if lower.mean() > 0.05 else 0.0
    shoulders_visible = 1.0 if upper.mean() > 0.015 else 0.0
    hips_visible = 1.0 if lower.mean() > 0.04 else 0.0
    knees_visible = 1.0 if lower.mean() > 0.08 else 0.0
    ankles_visible = 1.0 if lower.mean() > 0.12 else 0.0
    pose_present = 1.0 if upper_body_visible or lower_body_visible else 0.0
    pose_visible_landmarks = 6.0 if pose_present else 0.0
    pose_bbox_width = 0.5 if pose_present else 0.0
    pose_bbox_height = 0.7 if pose_present else 0.0
    pose_bbox_area = pose_bbox_width * pose_bbox_height
    face_to_pose_height_ratio = 0.25 if pose_present else 0.0
    return (
        pose_present,
        pose_visible_landmarks,
        pose_bbox_width,
        pose_bbox_height,
        pose_bbox_area,
        shoulders_visible,
        hips_visible,
        knees_visible,
        ankles_visible,
        lower_body_visible,
        upper_body_visible,
        face_to_pose_height_ratio,
    )


def _detect_people(rgb: np.ndarray) -> tuple[float, float, float, float, float]:
    detector = _person_detector()
    if detector is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    gray = _grayscale(rgb.astype(np.float32) / 255.0)
    gray_u8 = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
    rects, weights = detector.detectMultiScale(
        gray_u8,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    if len(rects) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    areas = []
    confidences = []
    offsets = []
    for (x, y, w, h), weight in zip(rects, weights if len(weights) else [0.0] * len(rects)):
        area = float((w * h) / max(1.0, gray_u8.shape[0] * gray_u8.shape[1]))
        center_x = float((x + w / 2.0) / max(1.0, gray_u8.shape[1]))
        center_y = float((y + h / 2.0) / max(1.0, gray_u8.shape[0]))
        offsets.append(float(np.hypot(center_x - 0.5, center_y - 0.5)))
        areas.append(area)
        confidences.append(float(weight))

    return float(len(rects)), float(max(confidences)), float(max(areas)), float(sum(areas)), float(min(offsets))


def extract_signals(image: Image.Image) -> VisionSignals:
    original = image.convert("RGB")
    width, height = original.size
    processed = _analysis_image(original)
    rgb = np.asarray(processed, dtype=np.uint8)
    rgb_float = rgb.astype(np.float32) / 255.0
    gray = _grayscale(rgb_float)
    saturation = rgb_float.max(axis=2) - rgb_float.min(axis=2)
    skin = _skin_mask(rgb.astype(np.float32))
    center_y0 = rgb.shape[0] // 5
    center_y1 = max(center_y0 + 1, rgb.shape[0] * 4 // 5)
    center_x0 = rgb.shape[1] // 5
    center_x1 = max(center_x0 + 1, rgb.shape[1] * 4 // 5)
    center_skin = skin[center_y0:center_y1, center_x0:center_x1]
    center_skin_ratio = float(center_skin.mean()) if center_skin.size else 0.0

    person_count, person_confidence_max, person_area_max, person_area_sum, person_center_offset = _detect_people(rgb)

    face_like_top = skin[: max(1, rgb.shape[0] // 3), rgb.shape[1] // 4 : max(rgb.shape[1] // 4 + 1, rgb.shape[1] * 3 // 4)]
    face_count = 1.0 if face_like_top.size and float(face_like_top.mean()) > 0.035 else 0.0
    face_conf_max = min(0.99, float(face_like_top.mean()) * 10.0) if face_like_top.size else 0.0
    face_conf_mean = face_conf_max
    face_area_max = float(face_like_top.mean()) if face_like_top.size else 0.0
    face_area_sum = face_area_max
    face_center_offset = 0.0 if face_count == 0 else float(abs(center_skin_ratio - 0.08))
    pose_present = 1.0 if person_count > 0 else 0.0
    pose_visible_landmarks = 8.0 if person_count > 0 else 0.0
    pose_bbox_width = float(min(1.0, person_area_max * 1.6))
    pose_bbox_height = float(min(1.0, person_area_max * 2.0))
    pose_bbox_area = pose_bbox_width * pose_bbox_height
    shoulders_visible = 1.0 if face_count and center_skin_ratio > 0.04 else 0.0
    hips_visible = 1.0 if person_count > 0 and person_area_max > 0.15 else 0.0
    knees_visible = 1.0 if person_count > 0 and person_area_max > 0.25 else 0.0
    ankles_visible = 1.0 if person_count > 0 and person_area_max > 0.35 else 0.0
    lower_body_visible = 1.0 if ankles_visible or knees_visible else 0.0
    upper_body_visible = 1.0 if face_count or shoulders_visible else 0.0
    face_to_pose_height_ratio = float(face_area_max / pose_bbox_height) if pose_bbox_height > 0 else 0.0
    ai_generated_score, ai_noise_std, ai_frequency_ratio, ai_patch_std, ai_flat_patch_ratio, ai_patch_var_std = _ai_generation_metrics(gray)

    gray_thumb_image = _thumbnail(processed, "L")
    color_thumb = _thumbnail(processed, "RGB")
    gray_thumb = gray_thumb_image.reshape(-1)
    sat_thumb = (color_thumb.max(axis=2) - color_thumb.min(axis=2)).reshape(-1)

    return VisionSignals(
        backend="opencv_hog" if _cv2 is not None else "fallback",
        width=width,
        height=height,
        megapixels=float(width * height / 1_000_000.0),
        aspect_ratio=float(width / max(1.0, height)),
        brightness_mean=float(gray.mean()),
        brightness_std=float(gray.std()),
        contrast=float(np.percentile(gray, 90) - np.percentile(gray, 10)),
        saturation_mean=float(saturation.mean()),
        saturation_std=float(saturation.std()),
        blur_variance=_laplacian_variance(gray),
        edge_density=_edge_density(gray),
        entropy=_entropy(gray),
        face_count=face_count,
        face_confidence_max=face_conf_max,
        face_confidence_mean=face_conf_mean,
        face_area_max=face_area_max,
        face_area_sum=face_area_sum,
        face_center_offset=face_center_offset,
        pose_present=pose_present,
        pose_visible_landmarks=pose_visible_landmarks,
        pose_bbox_width=pose_bbox_width,
        pose_bbox_height=pose_bbox_height,
        pose_bbox_area=pose_bbox_area,
        shoulders_visible=shoulders_visible,
        hips_visible=hips_visible,
        knees_visible=knees_visible,
        ankles_visible=ankles_visible,
        lower_body_visible=lower_body_visible,
        upper_body_visible=upper_body_visible,
        face_to_pose_height_ratio=face_to_pose_height_ratio,
        skin_ratio=float(skin.mean()),
        center_skin_ratio=center_skin_ratio,
        person_count=person_count,
        person_confidence_max=person_confidence_max,
        person_area_max=person_area_max,
        person_area_sum=person_area_sum,
        person_center_offset=person_center_offset,
        ai_generated_score=ai_generated_score,
        ai_noise_std=ai_noise_std,
        ai_frequency_ratio=ai_frequency_ratio,
        ai_patch_std=ai_patch_std,
        ai_flat_patch_ratio=ai_flat_patch_ratio,
        ai_patch_var_std=ai_patch_var_std,
        gray_thumbnail=gray_thumb.astype(np.float32),
        sat_thumbnail=sat_thumb.astype(np.float32),
    )


def extract_features(image: Image.Image) -> FeatureVector:
    signals = extract_signals(image)
    values = np.array(
        [
            signals.width,
            signals.height,
            signals.megapixels,
            signals.aspect_ratio,
            signals.brightness_mean,
            signals.brightness_std,
            signals.contrast,
            signals.saturation_mean,
            signals.saturation_std,
            signals.blur_variance,
            signals.edge_density,
            signals.entropy,
            signals.face_count,
            signals.face_confidence_max,
            signals.face_confidence_mean,
            signals.face_area_max,
            signals.face_area_sum,
            signals.face_center_offset,
            signals.pose_present,
            signals.pose_visible_landmarks,
            signals.pose_bbox_width,
            signals.pose_bbox_height,
            signals.pose_bbox_area,
            signals.shoulders_visible,
            signals.hips_visible,
            signals.knees_visible,
            signals.ankles_visible,
            signals.lower_body_visible,
            signals.upper_body_visible,
            signals.face_to_pose_height_ratio,
            signals.skin_ratio,
            signals.center_skin_ratio,
            signals.person_count,
            signals.person_confidence_max,
            signals.person_area_max,
            signals.person_area_sum,
            signals.person_center_offset,
            signals.ai_generated_score,
            signals.ai_noise_std,
            signals.ai_frequency_ratio,
            signals.ai_patch_std,
            signals.ai_flat_patch_ratio,
            signals.ai_patch_var_std,
            *signals.gray_thumbnail.tolist(),
            *signals.sat_thumbnail.tolist(),
        ],
        dtype=np.float32,
    )
    return FeatureVector(values=values)
