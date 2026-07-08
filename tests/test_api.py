from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


def make_image(path: Path) -> None:
    image = Image.new("RGB", (256, 256), (220, 210, 200))
    draw = ImageDraw.Draw(image)
    draw.ellipse((88, 40, 168, 120), fill=(220, 180, 160))
    draw.rectangle((72, 120, 184, 230), fill=(40, 60, 100))
    image.save(path)


def test_predict_api(tmp_path: Path) -> None:
    image_path = tmp_path / "upload.png"
    make_image(image_path)
    content_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    client = TestClient(app)
    response = client.post(
        "/api/predict",
        json={"filename": "upload.png", "content_base64": content_base64},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "upload.png"
    assert payload["label"] in {"acceptable", "manual_verification", "rejected"}
