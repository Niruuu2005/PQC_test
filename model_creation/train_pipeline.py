"""
Complete Training Pipeline
End-to-end training on real cryptanalysis datasets.

Author: AIRAWAT Team
Date: 2026-01-01
"""

import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.feature_engineer import FeatureEngineer
from src.data.splitter import DataSplitter
from src.classical.base_learners import ClassicalBaseline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline."""
    
    def __init__(self, data_dir: str = "../dataset_generation"):
        """
        Initialize pipeline.
        
        Args:
            data_dir: Directory containing CSV datasets
        """
        self.data_dir = data_dir
        self.loader = DataLoader(data_dir)
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.splitter = DataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        
        self.results = {}
        
    def load_data(self):
        """Load all datasets."""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*80)
        
        # Load individual datasets
        datasets = self.loader.load_all()
        
        # Check what we have
        logger.info("\nDataset Overview:")
        for name, df in datasets.items():
            logger.info(f"  {name}: {df.shape}")
        
        # Merge datasets
        merged_df = self.loader.merge_datasets()
        
        logger.info(f"\n✓ Merged dataset: {merged_df.shape}")
        return merged_df
    
    def clean_data(self, df):
        """Clean data."""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: CLEANING DATA")
        logger.info("="*80)
        
        # Identify columns for cleaning
        hex_cols = [col for col in df.columns if 'hex' in col.lower()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to likely outlier columns
        outlier_cols = [col for col in numeric_cols if any(x in col.lower() 
                       for x in ['time', 'memory', 'iterations', 'cpu'])]
        
        logger.info(f"Hex columns: {hex_cols}")
        logger.info(f"Outlier columns: {outlier_cols[:5]}...")
        
        # Clean
        cleaned_df = self.cleaner.clean_dataset(
            df,
            outlier_cols=outlier_cols,
            hex_cols=hex_cols
        )
        
        logger.info(f"\n✓ Cleaned dataset: {cleaned_df.shape}")
        return cleaned_df
    
    def engineer_features(self, df):
        """Engineer features."""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("="*80)
        
        features_df = self.engineer.engineer_all_features(df)
        
        logger.info(f"\n✓ Engineered features: {features_df.shape}")
        logger.info(f"Feature count: {len(features_df.columns)}")
        
        return features_df
    
    def prepare_targets(self, df):
        """Prepare target variables for different tasks."""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: PREPARING TARGETS")
        logger.info("="*80)
        
        targets = {}
        
        # Task 1: Attack Classification (if attack_category exists)
        if 'attack_category' in df.columns:
            targets['attack_category'] = df['attack_category']
            logger.info(f"  Task 1 (Attack Category): {df['attack_category'].nunique()} classes")
        elif 'category' in df.columns:
            targets['attack_category'] = df['category']
            logger.info(f"  Task 1 (Category): {df['category'].nunique()} classes")
        
        # Task 2: Attack Success (if exists)
        if 'attack_success' in df.columns:
            targets['attack_success'] = df['attack_success'].astype(int)
            logger.info(f"  Task 2 (Attack Success): {targets['attack_success'].value_counts().to_dict()}")
        
        # Task 3: Algorithm Identification
        if 'algorithm_name' in df.columns:
            targets['algorithm_name'] = df['algorithm_name']
            logger.info(f"  Task 3 (Algorithm): {df['algorithm_name'].nunique()} algorithms")
        
        logger.info(f"\n✓ Prepared {len(targets)} target variables")
        return targets
    
    def split_data(self, features_df, targets):
        """Split data into train/val/test."""
        logger.info("\n" + "="*80)
        logger.info("STEP 5: SPLITTING DATA")
        logger.info("="*80)
        
        # Combine features and targets temporarily
        combined_df = features_df.copy()
        for target_name, target_series in targets.items():
            combined_df[target_name] = target_series
        
        # Time-based split if timestamp exists
        if 'timestamp' in combined_df.columns:
            logger.info("Using time-based split...")
            train_df, val_df, test_df = self.splitter.time_based_split(combined_df)
        else:
            logger.info("Using random split...")
            train_df, val_df, test_df = self.splitter.random_split(combined_df)
        
        # Separate features from targets
        feature_cols = features_df.columns.tolist()
        target_cols = list(targets.keys())
        
        splits = {
            'X_train': train_df[feature_cols],
            'y_train': train_df[target_cols] if target_cols else None,
            'X_val': val_df[feature_cols],
            'y_val': val_df[target_cols] if target_cols else None,
            'X_test': test_df[feature_cols],
            'y_test': test_df[target_cols] if target_cols else None
        }
        
        logger.info(f"\n✓ Split complete:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Val: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        return splits
    
    def train_models(self, splits, task_name='attack_category'):
        """Train classical models."""
        logger.info("\n" + "="*80)
        logger.info("STEP 6: TRAINING MODELS")
        logger.info("="*80)
        
        X_train = splits['X_train']
        y_train = splits['y_train']
        X_test = splits['X_test']
        y_test = splits['y_test']
        
        # Select target column
        if y_train is not None and task_name in y_train.columns:
            y_train_task = y_train[task_name]
            y_test_task = y_test[task_name]
        else:
            logger.warning(f"Target '{task_name}' not found. Skipping training.")
            return {}
        
        # Handle categorical targets
        if y_train_task.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_task = le.fit_transform(y_train_task)
            y_test_task = le.transform(y_test_task)
            logger.info(f"Encoded {len(le.classes_)} classes: {list(le.classes_[:5])}...")
        
        logger.info(f"\nTraining on task: {task_name}")
        logger.info(f"Training data: {X_train.shape}")
        logger.info(f"Target classes: {np.unique(y_train_task)}")
        
        # Train baseline models
        baseline = ClassicalBaseline()
        baseline.train_all(X_train, y_train_task)
        
        # Evaluate
        results = baseline.evaluate(X_test, y_test_task)
        
        logger.info("\n✓ Training complete!")
        logger.info("\nResults:")
        for model_name, scores in results.items():
            logger.info(f"  {model_name}: {scores['accuracy']:.4f}")
        
        return results
    
    def save_results(self, results, output_dir='results'):
        """Save training results."""
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"{output_dir}/training_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n✓ Results saved to {results_file}")
    
    def run_full_pipeline(self):
        """Run complete pipeline."""
        logger.info("\n" + "="*80)
        logger.info("QUANTUM ML CRYPTANALYSIS - FULL TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Start time: {datetime.now()}")
        
        try:
            # Step 1: Load
            merged_df = self.load_data()
            
            # Step 2: Clean
            cleaned_df = self.clean_data(merged_df)
            
            # Step 3: Engineer features
            features_df = self.engineer_features(cleaned_df)
            
            # Step 4: Prepare targets
            targets = self.prepare_targets(cleaned_df)
            
            # Step 5: Split
            splits = self.split_data(features_df, targets)
            
            # Step 6: Train (try first available target)
            if targets:
                first_target = list(targets.keys())[0]
                results = self.train_models(splits, task_name=first_target)
                
                # Save results
                self.save_results(results)
            else:
                logger.warning("No targets found for training")
                results = {}
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETE!")
            logger.info("="*80)
            logger.info(f"End time: {datetime.now()}")
            
            return results
            
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    pipeline = TrainingPipeline(data_dir="../dataset_generation")
    results = pipeline.run_full_pipeline()
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    if results:
        for model, scores in results.items():
            print(f"{model}: {scores['accuracy']:.4f}")
    else:
        print("No results generated")
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
