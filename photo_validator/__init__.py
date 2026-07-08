"""Lightweight photo validation package."""

from .ai_detector import AiDetectionResult, AiGeneratedDetector
from .model import PhotoValidator, ValidationResult

__all__ = ["AiDetectionResult", "AiGeneratedDetector", "PhotoValidator", "ValidationResult"]
