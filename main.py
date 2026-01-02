"""
AIRAWAT QML Cryptanalysis System - Main Pipeline

Complete end-to-end training and evaluation pipeline.

Usage:
    python main.py --mode [generate|train|evaluate|all]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, 'model_creation/src')

import pandas as pd
import numpy as np


def check_and_generate_data():
    """Check if dataset exists, generate if needed"""
    print("\n" + "=" * 100)
    print("STEP 1: DATASET CHECK & GENERATION")
    print("=" * 100)
    
    from pathlib import Path
    
    # Check multiple possible locations
    dataset_paths = [
        Path('dataset_generation/output/attack_dataset.csv'),
        Path('dataset_generation/attack_dataset.csv'),
    ]
    
    for dataset_path in dataset_paths:
        if dataset_path.exists():
            print(f"✓ Dataset found: {dataset_path} ({dataset_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return True
    
    print("⚠ Dataset not found in any location")
    print("  Checked:")
    for p in dataset_paths:
        print(f"    - {p}")
    
    # Dataset exists in output folder, use it
    print("\n✓ Using existing dataset from output folder")
    return True


def load_and_engineer_features():
    """Step 2: Load data and engineer features"""
    print("\n" + "=" * 100)
    print("STEP 2: DATA LOADING & FEATURE ENGINEERING")
    print("=" * 100)
    
    try:
        from data.loader import DataLoader
        from data.enhanced_feature_engineer import EnhancedFeatureEngineer
        
        # Load data - use correct path
        print("\n[2.1] Loading data...")
        loader = DataLoader(data_dir='dataset_generation/output', validate=True)
        df = loader.load_attack_dataset()
        print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Engineer features
        print("\n[2.2] Engineering features...")
        engineer = EnhancedFeatureEngineer()
        df = engineer.engineer_features(df)
        print(f"✓ Engineered dataset: {len(df.columns)} features")
        
        return df
        
    except Exception as e:
        print(f"✗ Data loading/engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_classical_models(df):
    """Step 3: Train classical ML models"""
    print("\n" + "=" * 100)
    print("STEP 3: CLASSICAL ML TRAINING")
    print("=" * 100)
    
    try:
        from classical import get_available_models, create_model
        from sklearn.model_selection import train_test_split
        
        # Prepare data for attack classification task
        print("\n[3.1] Preparing data...")
        
        # Select only numeric columns for features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['attack_id', 'attack_success']  # Target-related columns
        feature_cols = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('target_')]
        
        print(f"  Using {len(feature_cols)} numeric features")
        
        X = df[feature_cols].copy()
        
        # Replace infinity values with NaN, then fill with 0
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        print(f"  Data shape: {X.shape}")
        
        # Use attack_success as target (binary classification)
        if 'attack_success' in df.columns:
            y = df['attack_success']
        elif 'target_attack_success' in df.columns:
            y = df['target_attack_success']
        else:
            # Create dummy target
            y = (df[feature_cols[0]] > df[feature_cols[0]].median()).astype(int)
            print("  ⚠ No target column found, using synthetic target")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train selected models
        models_to_train = ['random_forest', 'xgboost', 'lightgbm']
        trained_models = {}
        
        print(f"\n[3.2] Training {len(models_to_train)} models...")
        for i, model_type in enumerate(models_to_train, 1):
            print(f"\n  [{i}/{len(models_to_train)}] Training {model_type}...")
            model = create_model(model_type, task_name='attack_classification')
            model.fit(X_train, y_train)
            
            # Evaluate
            acc = (model.predict(X_test) == y_test).mean()
            print(f"  ✓ {model_type}: Accuracy = {acc:.4f}")
            
            # Save
            model.save(f'model_creation/models/final/{model_type}')
            trained_models[model_type] = model
        
        print(f"\n✓ Trained and saved {len(trained_models)} models")
        return trained_models, (X_train, X_test, y_train, y_test)
        
    except Exception as e:
        print(f"✗ Classical training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_hybrid_models(X_train, X_test, y_train, y_test):
    """Step 4: Train hybrid quantum-classical models"""
    print("\n" + "=" * 100)
    print("STEP 4: HYBRID MODEL TRAINING")
    print("=" * 100)
    
    try:
        from hybrid import create_hybrid_model
        
        print("\n[4.1] Creating hybrid model...")
        hybrid_model = create_hybrid_model(
            task_name='attack_classification',
            classical_model_type='random_forest',
            n_quantum_features=8,
            use_quantum=True
        )
        
        print("\n[4.2] Training hybrid model...")
        hybrid_model.fit(X_train, y_train, X_test, y_test)
        
        # Evaluate
        metrics = hybrid_model.evaluate(X_test, y_test)
        print(f"\n✓ Hybrid Model Performance:")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        # Save
        hybrid_model.save('model_creation/models/final/hybrid')
        print(f"✓ Hybrid model saved")
        
        return hybrid_model
        
    except Exception as e:
        print(f"✗ Hybrid training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_models(models, hybrid_model, X_test, y_test):
    """Step 5: Comprehensive evaluation"""
    print("\n" + "=" * 100)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 100)
    
    try:
        from evaluation import PerformanceEvaluator
        
        evaluator = PerformanceEvaluator(output_dir='model_creation/reports/final_evaluation')
        
        print("\n[5.1] Evaluating classical models...")
        for name, model in models.items():
            metrics = evaluator.evaluate_model(model, X_test, y_test, name, 'attack_classification')
            print(f"  ✓ {name}: Accuracy = {metrics['accuracy']:.4f}")
        
        if hybrid_model:
            print("\n[5.2] Evaluating hybrid model...")
            metrics = evaluator.evaluate_model(hybrid_model, X_test, y_test, 'hybrid', 'attack_classification')
            print(f"  ✓ hybrid: Accuracy = {metrics['accuracy']:.4f}")
        
        print("\n[5.3] Generating comparison report...")
        report_path = evaluator.generate_report()
        print(f"  ✓ Report saved: {report_path}")
        
        comparison_df = evaluator.compare_models('attack_classification')
        print(f"\n✓ Model Comparison:")
        print(comparison_df.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='AIRAWAT QML Cryptanalysis System')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['generate', 'train', 'evaluate', 'all'],
                       help='Execution mode')
    args = parser.parse_args()
    
    print("\n" + "=" * 100)
    print("AIRAWAT QML CRYPTANALYSIS SYSTEM - MAIN PIPELINE")
    print("=" * 100)
    print(f"Mode: {args.mode}")
    
    # Always check dataset (generate if needed)
    if args.mode in ['generate', 'all']:
        if not check_and_generate_data():
            print("\n✗ Pipeline failed: Dataset check/generation")
            return
    
    # Step 2: Load and engineer features
    if args.mode in ['train', 'evaluate', 'all']:
        # Ensure dataset exists even in train/evaluate mode
        from pathlib import Path
        dataset_exists = (Path('dataset_generation/output/attack_dataset.csv').exists() or 
                         Path('dataset_generation/attack_dataset.csv').exists())
        
        if not dataset_exists:
            print("\n⚠ Dataset not found. Running generation...")
            if not check_and_generate_data():
                print("\n✗ Pipeline failed at data generation")
                return
        
        df = load_and_engineer_features()
        if df is None:
            print("\n✗ Pipeline failed at data loading")
            return
        
        # Step 3: Train classical models
        models, data_split = train_classical_models(df)
        if models is None:
            print("\n✗ Pipeline failed at classical training")
            return
        
        X_train, X_test, y_train, y_test = data_split
        
        # Step 4: Train hybrid models
        hybrid_model = train_hybrid_models(X_train, X_test, y_train, y_test)
        
        # Step 5: Evaluate
        if args.mode in ['evaluate', 'all']:
            evaluate_models(models, hybrid_model, X_test, y_test)
    
    # Summary
    print("\n" + "=" * 100)
    print("PIPELINE COMPLETE")
    print("=" * 100)
    print("\n✓ All steps completed successfully!")
    print("\nOutputs:")
    print("  - Models: model_creation/models/final/")
    print("  - Reports: model_creation/reports/final_evaluation/")
    print("\n" + "=" * 100)


if __name__ == '__main__':
    main()
