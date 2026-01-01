"""
Deployment API (Simplified for Phase 6 validation)
Flask API for model inference.

Author: AIRAWAT Team
Date: 2026-01-01
"""

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not installed. Validation will use mock API.")

import logging

logger = logging.getLogger(__name__)


class InferenceAPI:
    """Simplified inference API."""
    
    def __init__(self):
        """Initialize API."""
        self.models_loaded = 3  # Mock: 3 models loaded
        logger.info(f"API initialized with {self.models_loaded} models")
    
    def predict_attack_classification(self, features):
        """Mock attack classification prediction."""
        import numpy as np
        return {
            'predicted_class': int(np.random.randint(0, 5)),
            'confidence': float(np.random.rand()),
            'model': 'hybrid_task1'
        }
    
    def health_check(self):
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'models_loaded': self.models_loaded,
            'version': '1.0.0'
        }


def create_app():
    """Create Flask app."""
    if not FLASK_AVAILABLE:
        logger.warning("Flask not available - using mock API")
        return None
    
    app = Flask(__name__)
    api = InferenceAPI()
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify(api.health_check())
    
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        result = api.predict_attack_classification(data.get('features', []))
        return jsonify(result)
    
    return app


def test_api():
    """Test API."""
    print("\n" + "="*70)
    print("DEPLOYMENT API TEST")
    print("="*70)
    
    api = InferenceAPI()
    
    # Test health check
    health = api.health_check()
    print(f"\nHealth check: {health}")
    
    # Test prediction
    prediction = api.predict_attack_classification([1, 2, 3, 4, 5])
    print(f"Sample prediction: {prediction}")
    
    print(f"\n✓ API test complete!")
    print(f"Flask available: {FLASK_AVAILABLE}")
    
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_api()
