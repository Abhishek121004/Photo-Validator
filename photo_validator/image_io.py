from __future__ import annotations

from pathlib import Path

from PIL import Image


class UnsupportedImageFormat(ValueError):
    pass


def _register_heic_support() -> bool:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return False

    register_heif_opener()
    return True


_HEIC_READY = False


def load_image(path: str | Path) -> Image.Image:
    """Load JPEG, PNG, and HEIC/HEIF images as RGB PIL images."""

    global _HEIC_READY

    image_path = Path(path)
    suffix = image_path.suffix.lower()

    if suffix in {".heic", ".heif"} and not _HEIC_READY:
        _HEIC_READY = _register_heic_support()

    try:
        with Image.open(image_path) as img:
            return img.convert("RGB")
    except Exception as exc:
        if suffix in {".heic", ".heif"} and not _HEIC_READY:
            raise UnsupportedImageFormat(
                "HEIC/HEIF support requires the optional 'pillow-heif' package."
            ) from exc
        raise
