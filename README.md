# 📸 Photo Validator

A lightweight, CPU-optimized AI-powered photo validation system that analyzes uploaded images and classifies them into one of three categories:

- ✅ Acceptable
- 🟡 Manual Verification
- ❌ Rejected

The system supports **JPEG**, **PNG**, and **HEIC** images and provides a confidence score along with class probabilities for each prediction.

---

## ✨ Features

- Supports JPEG, PNG, and HEIC image formats
- CPU-only inference
- Lightweight and fast prediction
- Returns confidence score and class probabilities
- REST API built with FastAPI
- Simple web interface for uploading images

---

## 🛠️ Tech Stack

- Python
- FastAPI
- OpenCV
- MediaPipe
- Pillow
- NumPy

---
## 📂 Project Structure

```text
PHOTO VALIDATOR/
│
├── app/                        # FastAPI application
│
├── data/                       # Dataset
│   ├── acceptable/
│   ├── manual_verification/
│   ├── rejected/
│   ├── train/
│   ├── val/
│   └── labels.csv
│
├── photo_validator/            # Core validation package
│
├── tests/                      # Test cases
│
├── calibrate.py                # Confidence threshold calibration
├── debug_signals.py            # Debug and feature analysis
├── init_dataset.py             # Dataset initialization
├── split_dataset.py            # Train-validation dataset split
├── train.py                    # Model training
├── validate.py                 # Image validation script
│
├── photo_validator.joblib      # Trained model
├── requirements.txt            # Project dependencies
└── README.md
```
```

---

## 🚀 Installation

Clone the repository

```bash
git clone <repository-url>
cd photo-validator
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
uvicorn app.main:app --reload
```

Open your browser:

```
http://127.0.0.1:8000
```

Swagger API Documentation:

```
http://127.0.0.1:8000/docs
```

---

## 🔍 Validation Pipeline

```
Image Upload
      │
      ▼
Image Preprocessing
      │
      ▼
Feature Extraction
      │
      ▼
Classification Model
      │
      ▼
Acceptable
Manual Verification
Rejected
```

---

## 📄 API Response

Example response:

```json
{
  "filename": "profile_photo.png",
  "label": "acceptable",
  "confidence": 0.88,
  "probabilities": {
    "acceptable": 0.88,
    "manual_verification": 0.10,
    "rejected": 0.02
  }
}
```

---

## 📌 Classification Labels

### ✅ Acceptable

- Single person
- Good quality image
- Suitable profile photo

### 🟡 Manual Verification

- Borderline cases
- Low confidence prediction
- Requires human review

### ❌ Rejected

- Invalid or unsuitable image
- Non-person images
- Poor quality images

---

## 🎯 Future Improvements

- Face detection and pose validation
- Blur detection
- Head-and-shoulders verification
- Explainable validation reasons
- ONNX Runtime optimization
- Docker deployment
- CI/CD pipeline

---

## 📜 License

This project is built using open-source technologies including **MediaPipe** and **OpenCV**, both licensed under the **Apache 2.0 License**.
