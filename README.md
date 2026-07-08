# Photo Validator

This project checks a single uploaded image and returns one of:

- `acceptable`
- `manual_verification`
- `rejected`

It also exposes an additional `ai_generated` signal so you can see when an image looks synthetic without removing the existing validation flow.

It is designed for JPEG, PNG, and HEIC/HEIF input.

## How it works

The validator is intentionally small, but it uses stronger vision signals than a flat baseline:

- a pretrained MobileNetV3-Small backbone for AI-generated detection
- OpenCV blur detection
- coarse spatial thumbnails for fallback and extra context
- skin and framing heuristics for head-and-shoulders validation
- patch-based texture checks as fallback signals

That keeps inference CPU-only, fast, and practical to run.

## Why this approach

For this kind of validation, a lightweight pretrained detector plus rules is often a better fit than a heavyweight neural network if you want:

- CPU-only inference
- small dependency footprint at runtime
- good accuracy on a small model
- easy threshold tuning on your own examples

The `manual_verification` label is used whenever the image is ambiguous or only partially matches the target framing.

## Dependency Note

`opencv-python` and `torch`/`torchvision` are used for the strongest path. If they are not installed, the code falls back to dependency-free heuristics so the project can still run.

## Training Data

You can train the photo validation thresholds from either:

- a CSV file with `path,label`
- a folder tree with subfolders named:
  - `acceptable/`
  - `manual_verification/`
  - `rejected/`
- a split dataset with:
  - `train/acceptable/`
  - `train/manual_verification/`
  - `train/rejected/`
  - `val/acceptable/`
  - `val/manual_verification/`
  - `val/rejected/`

Label aliases like `manual` and `accept` are also accepted.

For the AI-generated classifier, use:

- CSV labels: `ai_generated` and `real`
- folder names:
  - `ai_generated/`
  - `real/`

## Start From Scratch

If you do not have the folders yet, create them with:

```bash
python init_dataset.py --output data --csv-template
```

That creates:

```text
data/
  train/
    acceptable/
    manual_verification/
    rejected/
  val/
    acceptable/
    manual_verification/
    rejected/
  labels.csv
```

You can either:

- move your images into the class folders, or
- fill out `labels.csv` with image paths and labels

## Train or Tune

```bash
python train.py --data path/to/dataset --output thresholds.json
```

If `path/to/dataset` contains both `train/` and `val/`, the script uses `train/` and automatically calibrates thresholds on `val/`.

You can also save the calibrated thresholds separately:

```bash
python train.py --data path/to/dataset --output thresholds.json --thresholds-output thresholds.json
```

## Train AI Detection

Train the lightweight pretrained AI-generated detector with:

```bash
python train_ai_detector.py --data path/to/ai_dataset --output ai_generated_detector.joblib
```

The app and CLI automatically look for `ai_generated_detector.joblib` in the project root.

## Create AI Dataset

If you want a clean folder structure for labeling AI-generated images, create it with:

```bash
python init_ai_dataset.py --output ai_data --csv-template
```

That creates:

```text
ai_data/
  ai_generated/
  real/
  ai_labels.csv
```

If you already have labeled paths in a CSV, copy them into the training folders with:

```bash
python prepare_ai_dataset.py --labels ai_data/ai_labels.csv --source-root path/to/images --output ai_data
```

## Split Dataset

If you have a flat labeled folder, you can split it into train/val folders with:

```bash
python split_dataset.py --source path/to/flat_dataset --output path/to/output_dataset
```

Expected source layout:

```text
flat_dataset/
  acceptable/
  manual_verification/
  rejected/
```

The splitter creates:

```text
output_dataset/
  train/
    acceptable/
    manual_verification/
    rejected/
  val/
    acceptable/
    manual_verification/
    rejected/
```

## Calibrate

```bash
python calibrate.py --data path/to/dataset --output thresholds.json
```

If the dataset root contains `train/` and `val/`, calibration will use `val/`. You can still pass `--model` if you have a saved threshold config to start from.

## Validate

```bash
python validate.py --image some_photo.jpg
```

You can override the embedded thresholds with a calibration file:

```bash
python validate.py --image some_photo.jpg --thresholds thresholds.json
```

You can also point it at a trained AI detector:

```bash
python validate.py --image some_photo.jpg --ai-detector ai_generated_detector.joblib
```

## Debug Signals

```bash
python debug_signals.py --image some_photo.jpg
```

## Upload API

Start the API server:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000/` for the upload page
- `POST /api/predict` for JSON uploads

Example request body:

```json
{
  "filename": "photo.jpg",
  "content_base64": "..."
}
```

## HEIC Support

HEIC/HEIF loading is enabled automatically when `pillow-heif` is installed.
