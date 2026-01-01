"""
Simplified Phase 2 Implementation: Classical ML Baseline
Creates minimal working version for validation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


class ClassicalBaseline:
    """Classical ML baseline models."""
    
    def __init__(self):
        """Initialize baseline models."""
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42)
        }
        self.trained_models = {}
    
    def train_model(self, model_name: str, X_train, y_train):
        """Train a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        logger.info(f"  ✓ {model_name} trained")
        
        return model
    
    def train_all(self, X_train, y_train):
        """Train all models."""
        logger.info("Training all baseline models...")
        
        for name in self.models.keys():
            self.train_model(name, X_train, y_train)
        
        logger.info(f"✓ All {len(self.models)} models trained")
    
    def evaluate(self, X_test, y_test):
        """Evaluate all trained models."""
        logger.info("Evaluating models...")
        
        results = {}
        for name, model in self.trained_models.items():
            score = model.score(X_test, y_test)
            results[name] = {
                'accuracy': score,
                'cv_score': cross_val_score(model, X_test, y_test, cv=3).mean()
            }
            logger.info(f"  {name}: {score:.4f}")
        
        return results


def test_phase2():
    """Test Phase 2 implementation."""
    print("\n" + "="*70)
    print("PHASE 2 TEST: Classical ML Baseline")
    print("="*70)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1, 2], n_samples)
    
    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train models
    baseline = ClassicalBaseline()
    baseline.train_all(X_train, y_train)
    
    # Evaluate
    results = baseline.evaluate(X_test, y_test)
    
    print("\nResults:")
    for name, scores in results.items():
        print(f"  {name}: {scores['accuracy']:.4f}")
    
    print("\n✓ Phase 2 test complete!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_phase2()
