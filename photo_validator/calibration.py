from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json


@dataclass(frozen=True)
class Thresholds:
    min_resolution_side: int = 160
    blur_reject: float = 30.0
    blur_manual: float = 60.0
    low_confidence: float = 0.55
    low_margin: float = 0.12
    max_face_count_for_accept: float = 1.0
    max_face_count_for_manual: float = 2.0
    min_pose_visible_landmarks_for_body_reject: float = 12.0
    body_reject_pose_bbox_height: float = 0.78
    ai_generated_flag: float = 0.45

    @classmethod
    def from_dict(cls, data: dict) -> "Thresholds":
        return cls(
            min_resolution_side=int(data.get("min_resolution_side", cls.min_resolution_side)),
            blur_reject=float(data.get("blur_reject", cls.blur_reject)),
            blur_manual=float(data.get("blur_manual", cls.blur_manual)),
            low_confidence=float(data.get("low_confidence", cls.low_confidence)),
            low_margin=float(data.get("low_margin", cls.low_margin)),
            max_face_count_for_accept=float(data.get("max_face_count_for_accept", cls.max_face_count_for_accept)),
            max_face_count_for_manual=float(data.get("max_face_count_for_manual", cls.max_face_count_for_manual)),
            min_pose_visible_landmarks_for_body_reject=float(
                data.get(
                    "min_pose_visible_landmarks_for_body_reject",
                    cls.min_pose_visible_landmarks_for_body_reject,
                )
            ),
            body_reject_pose_bbox_height=float(
                data.get("body_reject_pose_bbox_height", cls.body_reject_pose_bbox_height)
            ),
            ai_generated_flag=float(data.get("ai_generated_flag", cls.ai_generated_flag)),
        )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def load_thresholds(path: str | Path) -> Thresholds:
    with open(path, "r", encoding="utf-8") as handle:
        return Thresholds.from_dict(json.load(handle))


def save_thresholds(path: str | Path, thresholds: Thresholds) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(thresholds.to_dict(), handle, indent=2)
