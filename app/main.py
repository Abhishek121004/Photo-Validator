from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image

from photo_validator.ai_detector import AiGeneratedDetector
from photo_validator.image_io import UnsupportedImageFormat
from photo_validator.model import PhotoValidator


app = FastAPI(title="Photo Validator")

THRESHOLDS_PATH = Path("thresholds.json")
AI_DETECTOR_PATH = Path("ai_generated_detector.joblib")


class PhotoUpload(BaseModel):
    filename: str = Field(..., description="Original filename of the uploaded image.")
    content_base64: str = Field(..., description="Base64 encoded file bytes.")
    thresholds_path: str | None = Field(default=None, description="Optional path to a thresholds JSON file.")


def _load_model(thresholds_path: Path | None = None) -> PhotoValidator:
    ai_detector = AiGeneratedDetector.load(AI_DETECTOR_PATH)
    model = PhotoValidator(ai_detector=ai_detector)
    active_thresholds = thresholds_path or THRESHOLDS_PATH
    if active_thresholds.exists():
        from photo_validator.calibration import load_thresholds

        model = model.with_thresholds(load_thresholds(active_thresholds))
    return model


def _decode_image(payload: PhotoUpload) -> Image.Image:
    try:
        raw = base64.b64decode(payload.content_base64, validate=True)
    except Exception as exc:  # pragma: no cover - simple input validation
        raise HTTPException(status_code=400, detail="Invalid base64 image content.") from exc

    try:
        with Image.open(BytesIO(raw)) as image:
            return image.convert("RGB")
    except UnsupportedImageFormat as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read image: {exc}") from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Photo Validator</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; background: #f7f7f7; color: #111; }
      .card { max-width: 720px; background: white; padding: 1.5rem; border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,.08); }
      input, button { font: inherit; }
      button { margin-top: 1rem; padding: .75rem 1rem; border: 0; border-radius: 10px; background: #111; color: white; cursor: pointer; }
      pre { margin-top: 1rem; padding: 1rem; background: #0b1020; color: #dbe7ff; border-radius: 12px; overflow: auto; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Photo Validator</h1>
      <p>Upload a JPEG, PNG, or HEIC image and get acceptable, manual verification, or rejected.</p>
      <input id="file" type="file" accept=".jpg,.jpeg,.png,.heic,.heif,image/jpeg,image/png,image/heic,image/heif" />
      <button id="submit">Validate</button>
      <pre id="output">Select an image to begin.</pre>
    </div>
    <script>
      const fileInput = document.getElementById("file");
      const output = document.getElementById("output");
      document.getElementById("submit").addEventListener("click", async () => {
        const file = fileInput.files && fileInput.files[0];
        if (!file) {
          output.textContent = "Please choose a file first.";
          return;
        }
        const arrayBuffer = await file.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);
        let binary = "";
        const chunkSize = 0x8000;
        for (let i = 0; i < bytes.length; i += chunkSize) {
          binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
        }
        const contentBase64 = btoa(binary);
        output.textContent = "Uploading...";
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filename: file.name,
            content_base64: contentBase64
          })
        });
        const data = await response.json();
        output.textContent = JSON.stringify(data, null, 2);
      });
    </script>
  </body>
</html>
"""


@app.post("/api/predict")
def predict(payload: PhotoUpload) -> dict:
    thresholds_path = Path(payload.thresholds_path) if payload.thresholds_path else None
    model = _load_model(thresholds_path)
    image = _decode_image(payload)
    result = model.predict_image(image)
    return {
        "filename": payload.filename,
        "label": result.label,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "ai_generated": result.ai_generated,
        "ai_generated_score": result.ai_generated_score,
        "ai_generated_reason": result.ai_generated_reason,
        "ai_generated_backend": result.ai_generated_backend,
    }
