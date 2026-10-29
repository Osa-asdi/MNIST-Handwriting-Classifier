# MNIST Handwritten Digit Classifier

A Flask web application and JSON API for recognizing handwritten digits (0–9). The app preprocesses drawings and images into 28 × 28 pixel inputs, then runs one of three bundled classifiers. The default model is XGBoost.

## Features

- Draw a digit on the web canvas and get a prediction.
- Upload an image, or submit an image URL.
- View the predicted digit, confidence, class probabilities, and inference latency.
- Switch between XGBoost, LightGBM, and Random Forest models.
- Send a 784-value pixel vector directly to the prediction API.
- Inspect image preprocessing output and check application health.

## Project layout

```text
Project_1/
├── app/
│   ├── __init__.py          # Flask application factory
│   ├── routes.py            # Web page and API routes
│   ├── predictor.py         # Model loading, image preprocessing, inference
│   ├── static/
│   │   ├── css/style.css    # Web interface styles
│   │   └── js/main.js       # Canvas, uploads, and API interactions
│   └── templates/
│       ├── base.html
│       └── index.html
├── models/                  # Bundled, pretrained model files
├── config.py                # Flask settings and model paths
├── run.py                   # Application entry point
├── requirements.txt
└── ML_Assignment_2.ipynb    # MNIST analysis and model-training notebook
```

## Requirements and setup

Use Python 3.9 or newer. From the project root, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install the requirements and start the application:

```bash
python -m pip install -r requirements.txt
python run.py
```

Open <http://127.0.0.1:5000> in a browser. `run.py` starts Flask in debug mode and binds to `0.0.0.0`; use it for local development only and do not expose it to an untrusted network. Alternatively, run the Flask development server on its default local address:

```bash
flask --app run:app run --debug
```

The application loads `models/xgboost_MNIST_model.pkl` at startup. Keep the model files in `models/`; the app also needs the LightGBM and Random Forest files when switching to those models. Model files are application assets and are intentionally not excluded by `.gitignore`.

## Web interface

The home page provides a drawing canvas, image upload, generated sample digits, and image URL input. Images are converted to grayscale-style digit features, cropped and centered, then resized and aligned to the MNIST 28 × 28 format. The interface displays the prediction, confidence, class probabilities, and a processed preview. A model selector changes the active classifier.

## API

All routes are served by the Flask app at its base URL (for example, `http://127.0.0.1:5000`). JSON prediction requests return `success`, `prediction`, `confidence`, `probabilities`, `latency_ms`, `model_info`, and, for image inputs, `processed_preview`. Probability and confidence values are percentages.

### Predict from a canvas image

`POST /api/predict/canvas` — JSON body with a Base64-encoded image data URL:

```json
{
  "image": "data:image/png;base64,<encoded-image>"
}
```

### Predict from an uploaded image

`POST /api/predict/upload` — `multipart/form-data` with a `file` field. Supported file extensions are PNG, JPG/JPEG, BMP, and WEBP. The configured maximum request size is 16 MB.

### Predict from an image URL

`POST /api/predict/url` — JSON body containing an HTTP or HTTPS URL:

```json
{
  "url": "https://example.com/digit.png"
}
```

The response includes `image_preview` in addition to the standard image-prediction fields. The server downloads the image, so submit only URLs that the server is allowed to access.

### Predict from 784 pixel features

`POST /api/predict/features` — JSON body containing exactly 784 values in row-major 28 × 28 order. Values can be supplied on a 0–1 scale or a 0–255 scale; values are divided by 255 when the maximum is greater than 1.

```python
payload = {"features": [0.0] * 784}
```

### List models and select the active model

- `GET /api/models` returns the active model key and metadata for available models.
- `POST /api/models/select` accepts a JSON body such as `{"model_key": "lightgbm"}`. Supported keys: `xgboost`, `lightgbm`, and `random_forest`.

### Health check

`GET /api/health` returns the current model and a `healthy` status when the endpoint responds.

### Example response

```json
{
  "success": true,
  "prediction": 7,
  "confidence": 99.42,
  "probabilities": {
    "0": 0.01,
    "1": 0.05,
    "7": 99.42
  },
  "latency_ms": 11.2,
  "model_info": {
    "key": "xgboost",
    "name": "XGBoost (Best Model)",
    "test_accuracy": "97.87%"
  },
  "processed_preview": "data:image/png;base64,<encoded-image>"
}
```

The probabilities object contains an entry for each digit class (0–9); only a few are shown above.

## Model metadata

The accuracy figures below are the values recorded in the application metadata:

| Key | Classifier | Recorded test accuracy |
|---|---|---:|
| `xgboost` | XGBoost (default) | 97.87% |
| `lightgbm` | LightGBM | 97.59% |
| `random_forest` | Random Forest | 96.75% |
