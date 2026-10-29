import urllib.request
import urllib.error
import urllib.parse
import numpy as np
from flask import Blueprint, render_template, request, jsonify, current_app
from app import predictor

main_bp = Blueprint('main', __name__)

@main_bp.route('/', methods=['GET'])
def index():
    """Renders the main interactive web UI."""
    model_info = predictor.get_model_info()
    available_models = predictor.AVAILABLE_MODELS
    return render_template('index.html', model_info=model_info, available_models=available_models)


@main_bp.route('/api/predict/canvas', methods=['POST'])
def predict_canvas():
    """Predicts a digit from a base64 encoded image drawn on canvas."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing base64 image data in request body'}), 400

    try:
        features, preview_28 = predictor.process_base64(data['image'])
        result = predictor.predict(features)
        result['processed_preview'] = preview_28
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': f"Inference failed: {str(e)}"}), 500


@main_bp.route('/api/predict/upload', methods=['POST'])
def predict_upload():
    """Predicts a digit from an uploaded image file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part found in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in current_app.config['ALLOWED_EXTENSIONS']:
        return jsonify({'error': f'Unsupported file extension. Allowed: {list(current_app.config["ALLOWED_EXTENSIONS"])}'}), 400

    try:
        features, preview_28 = predictor.process_file(file)
        result = predictor.predict(features)
        result['processed_preview'] = preview_28
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': f"Inference failed: {str(e)}"}), 500


@main_bp.route('/api/predict/url', methods=['POST'])
def predict_url():
    """Fetches an image from a URL and runs prediction."""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing "url" in request body'}), 400

    img_url = data['url'].strip()
    parsed = urllib.parse.urlparse(img_url)
    if parsed.scheme not in ('http', 'https'):
        return jsonify({'error': 'Invalid URL scheme. Must be http or https.'}), 400

    try:
        req = urllib.request.Request(
            img_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            image_bytes = response.read()

        if len(image_bytes) > current_app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'Image exceeds maximum allowed size (16MB).'}), 400

        features, preview_28, orig_preview = predictor.process_bytes(image_bytes)
        result = predictor.predict(features)
        result['processed_preview'] = preview_28
        result['image_preview'] = orig_preview
        return jsonify({'success': True, **result})
    except urllib.error.URLError as e:
        return jsonify({'error': f'Failed to download image from URL: {str(e.reason)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500


@main_bp.route('/api/predict/features', methods=['POST'])
def predict_features():
    """Predicts a digit from raw 784 normalized pixel features."""
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing features array in request body'}), 400

    features_raw = data['features']
    if len(features_raw) != 784:
        return jsonify({'error': f'Features array must contain exactly 784 values, got {len(features_raw)}'}), 400

    try:
        features = np.array(features_raw, dtype=np.float32).reshape(1, 784)
        if np.max(features) > 1.0:
            features = features / 255.0
        result = predictor.predict(features)
        preview_28 = predictor._matrix_to_b64((features.reshape(28, 28) * 255.0).astype(np.uint8))
        result['processed_preview'] = preview_28
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': f"Inference failed: {str(e)}"}), 500


@main_bp.route('/api/models', methods=['GET'])
def get_models():
    """Returns available models metadata and active model."""
    return jsonify({
        'current_model': predictor.current_model_key,
        'models': predictor.AVAILABLE_MODELS
    })


@main_bp.route('/api/models/select', methods=['POST'])
def select_model():
    """Switches the active inference model."""
    data = request.get_json()
    if not data or 'model_key' not in data:
        return jsonify({'error': 'Missing model_key in request body'}), 400

    model_key = data['model_key']
    try:
        predictor.load_model(model_key)
        return jsonify({
            'success': True,
            'message': f"Switched to model: {predictor.AVAILABLE_MODELS[model_key]['name']}",
            'model_info': predictor.get_model_info()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@main_bp.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint for containerization and monitoring."""
    return jsonify({
        'status': 'healthy',
        'active_model': predictor.current_model_key,
        'model_name': predictor.AVAILABLE_MODELS[predictor.current_model_key]['name'],
        'version': '1.0.0'
    })
