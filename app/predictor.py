import os
import io
import time
import base64
import joblib
import numpy as np
import scipy.ndimage
from PIL import Image, ImageOps


class MNISTPredictor:
    """Handles loading trained models and adaptive preprocessing for MNIST classification."""

    AVAILABLE_MODELS = {
        'xgboost': {
            'filename': 'xgboost_MNIST_model.pkl',
            'name': 'XGBoost (Best Model)',
            'test_accuracy': '97.87%',
            'type': 'Gradient Boosted Trees',
            'file_size': '9.05 MB'
        },
        'lightgbm': {
            'filename': 'lightgbm_MNIST_model.pkl',
            'name': 'LightGBM',
            'test_accuracy': '97.59%',
            'type': 'Light Gradient Boosting',
            'file_size': '3.56 MB'
        },
        'random_forest': {
            'filename': 'random_forest_MNIST_model.pkl',
            'name': 'Random Forest',
            'test_accuracy': '96.75%',
            'type': 'Ensemble Bagging',
            'file_size': '203.6 MB'
        }
    }

    def __init__(self, models_dir: str, default_model_key: str = 'xgboost'):
        self.models_dir = models_dir
        self.current_model_key = default_model_key
        self.models_cache = {}
        self.load_model(default_model_key)

    def load_model(self, model_key: str):
        """Loads a model into cache if not already loaded."""
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(self.AVAILABLE_MODELS.keys())}")

        if model_key in self.models_cache:
            self.current_model_key = model_key
            return self.models_cache[model_key]

        meta = self.AVAILABLE_MODELS[model_key]
        filepath = os.path.join(self.models_dir, meta['filename'])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")

        model = joblib.load(filepath)
        self.models_cache[model_key] = model
        self.current_model_key = model_key
        return model

    @property
    def current_model(self):
        return self.models_cache.get(self.current_model_key)

    def get_model_info(self):
        info = dict(self.AVAILABLE_MODELS.get(self.current_model_key, {}))
        info['key'] = self.current_model_key
        info['is_best'] = (self.current_model_key == 'xgboost')
        return info

    def preprocess_image(self, pil_image: Image.Image) -> tuple[np.ndarray, str]:
        """
        Adaptive MNIST Preprocessor:
        1. Fully adaptive to any background color and number color.
        2. Estimates dominant background color using perimeter/border sampling.
        3. Computes Euclidean color distance to separate the digit from background.
        4. Filters noise, shadows, paper grain, and compression artifacts.
        5. Normalizes and crops bounding box with padding.
        6. Preserves aspect ratio while fitting into a standard 20x20 box.
        7. Aligns digit onto a 28x28 grid using Center of Mass (COM),
           matching the exact normalization algorithm of the original MNIST dataset.
        8. Returns (features, processed_28x28_base64_data_url).
        """
        # Step 1: Handle color mode and transparency
        if pil_image.mode == 'RGBA':
            alpha = np.array(pil_image.split()[-1])
            if np.mean(alpha < 128) > 0.25:
                # Transparent background with drawn strokes
                bg = Image.new('RGB', pil_image.size, (0, 0, 0))
                bg.paste(pil_image, mask=pil_image.split()[-1])
                pil_image = bg
            else:
                pil_image = pil_image.convert('RGB')
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        arr = np.array(pil_image, dtype=np.float32)
        h, w, _ = arr.shape

        # Step 2: Background estimation via perimeter sampling
        top = arr[0, :, :]
        bottom = arr[-1, :, :]
        left = arr[:, 0, :]
        right = arr[:, -1, :]
        borders = np.vstack([top, bottom, left, right])
        bg_color = np.median(borders, axis=0)

        # Step 3: Color difference map (Euclidean distance in RGB space)
        diff = np.linalg.norm(arr - bg_color, axis=2)
        max_diff = np.max(diff)

        if max_diff < 12.0:
            # Blank or uniform image
            blank_feat = np.zeros((1, 784), dtype=np.float32)
            blank_preview = self._matrix_to_b64(np.zeros((28, 28), dtype=np.uint8))
            return blank_feat, blank_preview

        # Step 4: Adaptive noise filtering & contrast gating
        border_diff = np.linalg.norm(borders - bg_color, axis=1)
        noise_floor = np.percentile(border_diff, 95)
        threshold = max(noise_floor * 1.4, max_diff * 0.16)

        diff_clean = np.where(diff > threshold, diff - threshold, 0.0)
        cmax = np.max(diff_clean)
        if cmax <= 0.0:
            blank_feat = np.zeros((1, 784), dtype=np.float32)
            blank_preview = self._matrix_to_b64(np.zeros((28, 28), dtype=np.uint8))
            return blank_feat, blank_preview

        diff_clean = (diff_clean / cmax) * 255.0

        # Step 5: Bounding box detection with margin
        active_y, active_x = np.nonzero(diff_clean > 10.0)
        if len(active_y) < 8:
            blank_feat = np.zeros((1, 784), dtype=np.float32)
            blank_preview = self._matrix_to_b64(np.zeros((28, 28), dtype=np.uint8))
            return blank_feat, blank_preview

        pad = 2
        ymin = max(0, int(np.min(active_y)) - pad)
        ymax = min(h - 1, int(np.max(active_y)) + pad)
        xmin = max(0, int(np.min(active_x)) - pad)
        xmax = min(w - 1, int(np.max(active_x)) + pad)

        cropped = diff_clean[ymin:ymax + 1, xmin:xmax + 1]
        ch, cw = cropped.shape

        # Step 6: Scale to fit inside 20x20 preserving aspect ratio
        if ch > cw:
            nh = 20
            nw = max(1, int(round((cw * 20.0) / ch)))
        else:
            nw = 20
            nh = max(1, int(round((ch * 20.0) / cw)))

        crop_img = Image.fromarray(cropped.astype(np.uint8))
        resized_img = crop_img.resize((nw, nh), Image.Resampling.LANCZOS)
        res_arr = np.array(resized_img, dtype=np.float32)

        # Step 7: Place onto 28x28 canvas centered geometrically first
        canvas = np.zeros((28, 28), dtype=np.float32)
        py = (28 - nh) // 2
        px = (28 - nw) // 2
        canvas[py:py + nh, px:px + nw] = res_arr

        # Step 8: Center of Mass (COM) Alignment (Standard MNIST centering)
        total_mass = np.sum(canvas)
        if total_mass > 0:
            cy, cx = scipy.ndimage.center_of_mass(canvas)
            shift_y = float(np.clip(13.5 - cy, -4.5, 4.5))
            shift_x = float(np.clip(13.5 - cx, -4.5, 4.5))
            canvas = scipy.ndimage.shift(canvas, (shift_y, shift_x), order=1, mode='constant', cval=0.0)

        canvas_uint8 = np.clip(canvas, 0.0, 255.0).astype(np.uint8)
        canvas_norm = canvas_uint8.astype(np.float32) / 255.0

        preview_b64 = self._matrix_to_b64(canvas_uint8)
        return canvas_norm.reshape(1, 784), preview_b64

    def _matrix_to_b64(self, matrix_28: np.ndarray) -> str:
        """Converts a (28, 28) uint8 array to a base64 PNG data URL."""
        img = Image.fromarray(matrix_28, mode='L')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')

    def process_base64(self, b64_string: str) -> tuple[np.ndarray, str]:
        """Decodes base64 data URL and returns (preprocessed_features, 28x28_preview)."""
        if ',' in b64_string:
            b64_string = b64_string.split(',', 1)[1]
        image_bytes = base64.b64decode(b64_string)
        pil_img = Image.open(io.BytesIO(image_bytes))
        return self.preprocess_image(pil_img)

    def process_file(self, file_storage) -> tuple[np.ndarray, str]:
        """Processes an uploaded file storage object and returns (features, 28x28_preview)."""
        pil_img = Image.open(file_storage.stream)
        return self.preprocess_image(pil_img)

    def process_bytes(self, image_bytes: bytes) -> tuple[np.ndarray, str, str]:
        """Processes raw image bytes and returns (features, 28x28_preview, orig_preview_b64)."""
        pil_img = Image.open(io.BytesIO(image_bytes))
        features, preview_28 = self.preprocess_image(pil_img)

        # Also encode original image for UI display
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        orig_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
        return features, preview_28, orig_b64

    def predict(self, feature_array: np.ndarray) -> dict:
        """
        Runs inference on a (1, 784) feature array and returns predictions,
        probabilities for all classes (0-9), and latency.
        """
        if feature_array.shape != (1, 784):
            feature_array = feature_array.reshape(1, 784)

        model = self.current_model
        start_time = time.perf_counter()

        # Predict class
        prediction = int(model.predict(feature_array)[0])

        # Predict probabilities
        probabilities = {}
        top_confidence = 0.0

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(feature_array)[0]
            probabilities = {int(i): float(round(probs[i] * 100, 2)) for i in range(len(probs))}
            top_confidence = float(round(probs[prediction] * 100, 2))
        else:
            probabilities = {int(i): (100.0 if i == prediction else 0.0) for i in range(10)}
            top_confidence = 100.0

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            'prediction': prediction,
            'confidence': top_confidence,
            'probabilities': probabilities,
            'latency_ms': elapsed_ms,
            'model_info': self.get_model_info()
        }
