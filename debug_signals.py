from __future__ import annotations

import argparse
import json

from photo_validator.features import extract_signals
from photo_validator.image_io import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print extracted face, pose, and blur signals.")
    parser.add_argument("--image", required=True, help="Image to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signals = extract_signals(load_image(args.image))
    payload = {
        "backend": signals.backend,
        "size": {"width": signals.width, "height": signals.height, "megapixels": signals.megapixels},
        "quality": {
            "brightness_mean": signals.brightness_mean,
            "brightness_std": signals.brightness_std,
            "contrast": signals.contrast,
            "saturation_mean": signals.saturation_mean,
            "saturation_std": signals.saturation_std,
            "blur_variance": signals.blur_variance,
            "edge_density": signals.edge_density,
            "entropy": signals.entropy,
        },
        "face": {
            "count": signals.face_count,
            "confidence_max": signals.face_confidence_max,
            "confidence_mean": signals.face_confidence_mean,
            "area_max": signals.face_area_max,
            "area_sum": signals.face_area_sum,
            "center_offset": signals.face_center_offset,
            "skin_ratio": signals.skin_ratio,
            "center_skin_ratio": signals.center_skin_ratio,
        },
        "pose": {
            "present": signals.pose_present,
            "visible_landmarks": signals.pose_visible_landmarks,
            "bbox_width": signals.pose_bbox_width,
            "bbox_height": signals.pose_bbox_height,
            "bbox_area": signals.pose_bbox_area,
            "shoulders_visible": signals.shoulders_visible,
            "hips_visible": signals.hips_visible,
            "knees_visible": signals.knees_visible,
            "ankles_visible": signals.ankles_visible,
            "lower_body_visible": signals.lower_body_visible,
            "upper_body_visible": signals.upper_body_visible,
        },
        "person": {
            "count": signals.person_count,
            "confidence_max": signals.person_confidence_max,
            "area_max": signals.person_area_max,
            "area_sum": signals.person_area_sum,
            "center_offset": signals.person_center_offset,
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
