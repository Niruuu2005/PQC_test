"""
Simplified Training Script
Train models on real attack dataset (no complex merging).

Author: AIRAWAT Team  
Date: 2026-01-01
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_attack_data():
    """Load attack dataset."""
    logger.info("Loading attack dataset...")
    df = pd.read_csv('../dataset_generation/attack_dataset.csv')
    logger.info(f"  Loaded: {df.shape}")
    logger.info(f"  Columns: {list(df.columns[:10])}...")
    return df


def prepare_features_and_targets(df):
    """Prepare features and targets from attack dataset."""
    logger.info("\nPreparing features and targets...")
    
    # Identify numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID columns and targets
    exclude = ['attack_execution_id', 'encryption_row_id', 'attack_id']
    numeric_features = [col for col in numeric_features if col not in exclude]
    
    logger.info(f"  Numeric features: {len(numeric_features)}")
    
    # Handle missing values
    X = df[numeric_features].fillna(0)
    
    # Prepare targets
    targets = {}
    
    # Task 1: Attack category/name
    if 'attack_category' in df.columns:
        targets['attack_category'] = df['attack_category']
    elif 'attack_name' in df.columns:
        targets['attack_name'] = df['attack_name']
    elif 'category' in df.columns:
        targets['category'] = df['category']
    
    # Task 2: Attack success
    if 'attack_success' in df.columns:
        targets['attack_success'] = df['attack_success']
    
    logger.info(f"  Features shape: {X.shape}")
    logger.info(f"  Targets: {list(targets.keys())}")
    
    return X, targets


def train_models(X, y, task_name):
    """Train classification models."""
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING: {task_name}")
    logger.info(f"{'='*70}")
    
    # Encode target if needed
    if y.dtype == 'object' or y.dtype == 'bool':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        classes = le.classes_
        logger.info(f"  Classes ({len(classes)}): {list(classes[:5])}...")
    else:
        y_encoded = y
        classes = np.unique(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': float(accuracy),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_classes': len(classes)
        }
        
        logger.info(f"    Accuracy: {accuracy:.4f}")
    
    logger.info(f"\n{'='*70}")
    return results


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("QUANTUM ML CRYPTANALYSIS - REAL DATA TRAINING")
    print("="*70)
    print(f"Start: {datetime.now()}")
    
    try:
        # Load data
        df = load_attack_data()
        
        # Prepare features and targets
        X, targets = prepare_features_and_targets(df)
        
        # Train on each target
        all_results = {}
        
        for target_name, y in targets.items():
            # Remove rows with missing target
            mask = y.notna()
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(y_clean) < 100:
                logger.warning(f"  Skipping {target_name}: insufficient data ({len(y_clean)} samples)")
                continue
            
            results = train_models(X_clean, y_clean, target_name)
            all_results[target_name] = results
        
        # Save results
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f'training_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {results_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        for target_name, models in all_results.items():
            print(f"\n{target_name}:")
            for model_name, metrics in models.items():
                print(f"  {model_name}: {metrics['accuracy']:.4f} "
                      f"({metrics['n_classes']} classes, {metrics['test_size']} test samples)")
        
        print("\n" + "="*70)
        print(f"End: {datetime.now()}")
        print("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
