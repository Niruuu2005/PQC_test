"""
Data Splitter Module
Creates train/validation/test splits.

Author: AIRAWAT Team
Date: 2026-01-01
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split data into train/validation/test sets."""
    
    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        """
        Initialize DataSplitter.
        
        Args:
            train_ratio: Training set ratio (default: 0.6)
            val_ratio: Validation set ratio (default: 0.2)
            test_ratio: Test set ratio (default: 0.2)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def time_based_split(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data based on temporal order.
        
        Args:
            df: Input DataFrame
            timestamp_col: Timestamp column for sorting
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Performing time-based split...")
        
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        
        n = len(df_sorted)
        train_end = int(self.train_ratio * n)
        val_end = int((self.train_ratio + self.val_ratio) * n)
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        logger.info(f"  Train: {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
        logger.info(f"  Val:   {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
        logger.info(f"  Test:  {len(test_df):,} ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def random_split(self, df: pd.DataFrame, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Random split (shuffled).
        
        Args:
            df: Input DataFrame
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Performing random split...")
        
        # Shuffle
        df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        
        n = len(df_shuffled)
        train_end = int(self.train_ratio * n)
        val_end = int((self.train_ratio + self.val_ratio) * n)
        
        train_df = df_shuffled.iloc[:train_end]
        val_df = df_shuffled.iloc[train_end:val_end]
        test_df = df_shuffled.iloc[val_end:]
        
        logger.info(f"  Train: {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
        logger.info(f"  Val:   {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
        logger.info(f"  Test:  {len(test_df):,} ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_splits(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_val: pd.DataFrame, y_val: pd.DataFrame,
                   X_test: pd.DataFrame, y_test: pd.DataFrame,
                   output_dir: str = 'data/processed') -> None:
        """
        Save train/val/test splits to CSV.
        
        Args:
            X_train, y_train: Training features and targets
            X_val, y_val: Validation features and targets
            X_test, y_test: Test features and targets
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving splits to {output_dir}...")
        
        X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
        y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
        
        X_val.to_csv(f'{output_dir}/X_val.csv', index=False)
        y_val.to_csv(f'{output_dir}/y_val.csv', index=False)
        
        X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
        y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
        
        logger.info("  ✓ All splits saved")
    
    def verify_split_balance(self, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
                            target_col: str = 'target') -> Dict:
        """
        Verify class balance across splits.
        
        Args:
            y_train, y_val, y_test: Target series
            target_col: Target column name
            
        Returns:
            Dictionary with balance statistics
        """
        logger.info(f"Verifying class balance for {target_col}...")
        
        if target_col in y_train.columns:
            train_dist = y_train[target_col].value_counts(normalize=True)
            val_dist = y_val[target_col].value_counts(normalize=True)
            test_dist = y_test[target_col].value_counts(normalize=True)
            
            logger.info("  Class distribution:")
            for cls in train_dist.index:
                logger.info(f"    {cls}: Train={train_dist.get(cls, 0)*100:.1f}% "
                          f"Val={val_dist.get(cls, 0)*100:.1f}% "
                          f"Test={test_dist.get(cls, 0)*100:.1f}%")
            
            return {
                'train': train_dist.to_dict(),
                'val': val_dist.to_dict(),
                'test': test_dist.to_dict()
            }
        else:
            logger.warning(f"  Column {target_col} not found")
            return {}


def main():
    """Test data splitter."""
    print("\n" + "=" * 60)
    print("DATA SPLITTER TEST")
    print("=" * 60 + "\n")
    
    # Create test data
    test_data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'timestamp': pd.date_range('2026-01-01', periods=1000, freq='H'),
        'target': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print(f"Test data: {test_data.shape}")
    
    # Split
    splitter = DataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    # Time-based split
    train, val, test = splitter.time_based_split(test_data)
    
    # Separate features and targets
    feature_cols = ['feature1', 'feature2']
    target_cols = ['target']
    
    X_train, y_train = train[feature_cols], train[target_cols]
    X_val, y_val = val[feature_cols], val[target_cols]
    X_test, y_test = test[feature_cols], test[target_cols]
    
    # Verify balance
    splitter.verify_split_balance(y_train, y_val, y_test, 'target')
    
    print("\n✓ Data splitter test complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
