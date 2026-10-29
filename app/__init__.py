# pyrefly: ignore[missing-import]
from flask import Flask
from config import Config
from app.predictor import MNISTPredictor

predictor = None

def create_app(config_class=Config):
    """Application factory for Flask MNIST classifier."""
    global predictor
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize the predictor with the best model by default
    predictor = MNISTPredictor(
        models_dir=app.config['MODELS_DIR'],
        default_model_key='xgboost'
    )

    # Register blueprints / routes
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
