"""
Model Saving and Inference Guide
Save trained models and use them for predictions.

Author: AIRAWAT Team
Date: 2026-01-01
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class ModelManager:
    """Manage model training, saving, and loading."""
    
    def __init__(self, model_dir='models/saved'):
        """Initialize model manager."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def train_and_save_models(self, X_train, y_train, X_test, y_test, task_name='attack_success'):
        """Train models and save to disk."""
        print(f"\nTraining models for {task_name}...")
        
        # Train models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            results[name] = {
                'train_accuracy': float(train_acc),
                'test_accuracy': float(test_acc)
            }
            
            # Save model
            model_path = self.model_dir / f'{task_name}_{name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"    Test accuracy: {test_acc:.4f}")
            print(f"    Saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'task_name': task_name,
            'n_features': X_train.shape[1],
            'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else None,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'results': results
        }
        
        metadata_path = self.model_dir / f'{task_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ All models saved to {self.model_dir}")
        return results
    
    def load_model(self, task_name, model_name='random_forest'):
        """Load a saved model."""
        model_path = self.model_dir / f'{task_name}_{model_name}.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Loaded model: {model_path}")
        return model
    
    def load_metadata(self, task_name):
        """Load model metadata."""
        metadata_path = self.model_dir / f'{task_name}_metadata.json'
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def predict(self, task_name, X, model_name='random_forest'):
        """Make predictions with a saved model."""
        # Load model
        model = self.load_model(task_name, model_name)
        
        # Load metadata to check features
        metadata = self.load_metadata(task_name)
        expected_features = metadata['n_features']
        
        if X.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")
        
        # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        return predictions, probabilities


def example_train_and_save():
    """Example: Train models and save to disk."""
    print("\n" + "="*80)
    print("EXAMPLE: TRAIN AND SAVE MODELS")
    print("="*80)
    
    # Load data
    df = pd.read_csv('../dataset_generation/attack_dataset.csv')
    
    # Prepare features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['attack_execution_id', 'encryption_row_id', 'attack_id']
    numeric_features = [col for col in numeric_features if col not in exclude]
    
    X = df[numeric_features].fillna(0)
    y = df['attack_success'].astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train and save
    manager = ModelManager()
    results = manager.train_and_save_models(X_train, y_train, X_test, y_test)
    
    return manager


def example_inference():
    """Example: Load model and make predictions on new data."""
    print("\n" + "="*80)
    print("EXAMPLE: INFERENCE ON NEW DATA")
    print("="*80)
    
    # Create random test data (same shape as training data)
    manager = ModelManager()
    
    # Load metadata to get expected features
    metadata = manager.load_metadata('attack_success')
    n_features = metadata['n_features']
    
    print(f"\nExpected features: {n_features}")
    
    # Create random samples
    n_samples = 10
    X_random = np.random.randn(n_samples, n_features)
    
    print(f"Random test data shape: {X_random.shape}")
    
    # Predict
    predictions, probabilities = manager.predict('attack_success', X_random)
    
    print("\nPredictions:")
    for i in range(n_samples):
        pred = "SUCCESS" if predictions[i] else "FAILURE"
        prob = probabilities[i][1] if probabilities is not None else 0.0
        print(f"  Sample {i+1}: {pred} (confidence: {prob:.4f})")
    
    return predictions


def get_required_features():
    """Show what features are needed for inference."""
    print("\n" + "="*80)
    print("REQUIRED INPUT FEATURES")
    print("="*80)
    
    # Load metadata
    manager = ModelManager()
    try:
        metadata = manager.load_metadata('attack_success')
        
        print(f"\nTask: {metadata['task_name']}")
        print(f"Number of features: {metadata['n_features']}")
        
        if metadata['feature_names']:
            print(f"\nFeature names:")
            for i, name in enumerate(metadata['feature_names'][:20], 1):
                print(f"  {i}. {name}")
            if len(metadata['feature_names']) > 20:
                print(f"  ... and {len(metadata['feature_names']) - 20} more")
        
        print(f"\nTo make predictions, provide a numpy array or DataFrame with {metadata['n_features']} features")
        
    except FileNotFoundError:
        print("No models found. Run training first!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        # Train and save models
        example_train_and_save()
    elif len(sys.argv) > 1 and sys.argv[1] == '--predict':
        # Run inference
        example_inference()
    elif len(sys.argv) > 1 and sys.argv[1] == '--features':
        # Show required features
        get_required_features()
    else:
        print("\nUsage:")
        print("  python model_inference.py --train      # Train and save models")
        print("  python model_inference.py --predict    # Run inference on random data")
        print("  python model_inference.py --features   # Show required features")
