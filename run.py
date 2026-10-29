from app import create_app

app = create_app()

if __name__ == '__main__':
    # Standard Flask runner
    print("=" * 60)
    print("🚀 Starting MNIST Digit Classifier API & Web Interface")
    print("🌐 Dashboard URL: http://127.0.0.1:5000")
    print("⚡ Default Model: XGBoost (Accuracy: 97.87%)")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
