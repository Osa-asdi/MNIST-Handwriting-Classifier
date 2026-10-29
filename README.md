# 🧠 MNIST Handwritten Digit Classifier — MLOps Flask API & UI

A production-ready Flask web application and REST API for real-time handwritten digit recognition. Built on top of the MNIST benchmark dataset using an **XGBoost Classifier** achieving **97.87% test accuracy**.

---

## 🌟 Key Features

- **Standard Flask Architecture**: Structured with the application factory pattern (`create_app`), modular blueprints, static assets, and templates.
- **Best Model by Default**: Loads `models/xgboost_MNIST_model.pkl` (97.87% accuracy), with support for dynamic switching to LightGBM (97.59%) or Random Forest (96.75%).
- **Interactive Web Interface**:
  - **Drawing Canvas**: Responsive touch/mouse canvas with live auto-prediction as you draw.
  - **Image Upload**: Drag-and-drop support for PNG/JPG/WEBP with automatic grayscale conversion and bounding-box centering.
  - **Image URL Prediction**: Fetch, preprocess, and classify images directly from any public web URL with sample shortcut chips.
  - **Light & Dark Mode**: Instant, persistent theme switching (Dark Glassmorphism 🌙 / Clean Slate Light ☀️).
  - **Pre-loaded Digits**: Instant 1-click test cards for digits 0–9.
  - **Live Probabilities**: Real-time softmax probability bars and latency benchmarking (`< 15ms`).
- **RESTful API**: Endpoints for Base64 canvas images, multipart file uploads, image web URLs, raw 784-feature vectors, and health checks.

---

## 📁 Project Directory Structure

```text
Project_1/
├── app/
│   ├── __init__.py          # Flask application factory (create_app)
│   ├── routes.py            # API routes and web controllers
│   ├── predictor.py         # Model loader, image preprocessor, and inference engine
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css    # Sleek dark mode & glassmorphism stylesheet
│   │   └── js/
│   │       └── main.js      # Canvas drawing, presets, and API integration
│   └── templates/
│       ├── base.html        # HTML5 layout shell
│       └── index.html       # Primary dashboard and interactive workspace
├── models/
│   ├── xgboost_MNIST_model.pkl      # Best Model (97.87% accuracy, 9.05 MB)
│   ├── lightgbm_MNIST_model.pkl     # Lightweight Model (97.59% accuracy, 3.56 MB)
│   └── random_forest_MNIST_model.pkl# Ensemble Baseline (96.75% accuracy, 203.6 MB)
├── config.py                # Environment configurations and paths
├── run.py                   # Application entry point
├── requirements.txt         # Python dependencies
├── ML_Assignment_2.ipynb    # Jupyter Notebook with exploratory analysis and training
└── README.md                # Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python run.py
```
Or using Flask CLI:
```bash
flask --app run:app run --debug
```

Visit **`http://127.0.0.1:5000`** in your browser.

---

## 🔌 REST API Endpoints

### 1. Predict from Canvas (Base64)
- **Endpoint**: `POST /api/predict/canvas`
- **Request**:
  ```json
  {
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "prediction": 7,
    "confidence": 99.42,
    "probabilities": { "0": 0.01, "1": 0.05, "7": 99.42, ... },
    "latency_ms": 11.2,
    "model_info": {
      "name": "XGBoost (Best Model)",
      "test_accuracy": "97.87%"
    }
  }
  ```

### 2. Predict from File Upload
- **Endpoint**: `POST /api/predict/upload`
- **Content-Type**: `multipart/form-data`
- **Field**: `file=@digit_sample.png`

### 3. Predict from Image URL
- **Endpoint**: `POST /api/predict/url`
- **Request**:
  ```json
  {
    "url": "https://example.com/digit.png"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "prediction": 8,
    "confidence": 98.7,
    "probabilities": { "0": 0.02, "8": 98.7, ... },
    "image_preview": "data:image/png;base64,...",
    "latency_ms": 14.5
  }
  ```

### 4. Predict from Raw 784 Features
- **Endpoint**: `POST /api/predict/features`
- **Request**:
  ```json
  {
    "features": [0.0, 0.0, ..., 0.85, ... 784 values]
  }
  ```

### 4. Health Check
- **Endpoint**: `GET /api/health`
- **Response**:
  ```json
  {
    "status": "healthy",
    "active_model": "xgboost",
    "model_name": "XGBoost (Best Model)",
    "version": "1.0.0"
  }
  ```
