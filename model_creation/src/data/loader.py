"""
Data Loader Module
Loads and merges cryptanalysis datasets.

Author: AIRAWAT Team
Date: 2026-01-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and merge cryptanalysis datasets."""
    
    def __init__(self, data_dir: str = "../dataset_generation"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load_attack_dataset(self) -> pd.DataFrame:
        """
        Load attack execution dataset.
        
        Returns:
            DataFrame with attack execution data
        """
        file_path = self.data_dir / "attack_dataset.csv"
        logger.info(f"Loading attack dataset from {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        self.datasets['attack'] = df
        return df
    
    def load_crypto_dataset(self) -> pd.DataFrame:
        """
        Load cryptographic encryption dataset.
        
        Returns:
            DataFrame with encryption results
        """
        file_path = self.data_dir / "crypto_dataset.csv"
        logger.info(f"Loading crypto dataset from {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        self.datasets['crypto'] = df
        return df
    
    def load_attack_metadata(self) -> pd.DataFrame:
        """
        Load attack metadata/definitions.
        
        Returns:
            DataFrame with attack metadata
        """
        file_path = self.data_dir / "attack_metadata.csv"
        logger.info(f"Loading attack metadata from {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        self.datasets['metadata'] = df
        return df
    
    def load_algorithm_summary(self) -> pd.DataFrame:
        """
        Load cryptographic algorithm summary.
        
        Returns:
            DataFrame with algorithm vulnerability assessment
        """
        file_path = self.data_dir / "Cryptographic_Algorithm_Summary.csv"
        logger.info(f"Loading algorithm summary from {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        self.datasets['summary'] = df
        return df
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets.
        
        Returns:
            Dictionary of all loaded datasets
        """
        logger.info("=" * 60)
        logger.info("LOADING ALL DATASETS")
        logger.info("=" * 60)
        
        self.load_attack_dataset()
        self.load_crypto_dataset()
        self.load_attack_metadata()
        self.load_algorithm_summary()
        
        logger.info("=" * 60)
        logger.info(f"✓ All {len(self.datasets)} datasets loaded successfully")
        logger.info("=" * 60)
        
        return self.datasets
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge all datasets into single DataFrame.
        
        Merging strategy:
        1. Start with attack_dataset (primary)
        2. Merge crypto statistics via encryption_row_id
        3. Merge attack metadata via attack_id
        4. Merge algorithm summary via algorithm_name
        
        Returns:
            Merged DataFrame
        """
        logger.info("\nMERGING DATASETS...")
        
        # Ensure all datasets loaded
        if not self.datasets:
            self.load_all()
        
        # Start with attack dataset
        merged = self.datasets['attack'].copy()
        initial_rows = len(merged)
        logger.info(f"  Starting with attack_dataset: {initial_rows:,} rows")
        
        # Merge crypto statistics
        crypto_cols = ['row_id', 'shannon_entropy', 'chi_square_statistic', 
                      'avalanche_effect', 'encryption_time_ms', 
                      'encryption_successful', 'decryption_successful']
        
        merged = merged.merge(
            self.datasets['crypto'][crypto_cols],
            left_on='encryption_row_id',
            right_on='row_id',
            how='left',
            suffixes=('', '_crypto')
        )
        logger.info(f"  After merging crypto: {len(merged):,} rows")
        
        # Merge attack metadata
        metadata_cols = ['attack_id', 'category', 'subcategory', 
                        'primary_language', 'complexity_time', 'complexity_space']
        
        merged = merged.merge(
            self.datasets['metadata'][metadata_cols],
            on='attack_id',
            how='left',
            suffixes=('', '_meta')
        )
        logger.info(f"  After merging metadata: {len(merged):,} rows")
        
        # Merge algorithm summary
        summary_cols = [' Algorithm', 'Security_Level', 'Resistance_Score',
                       'Vulnerable_Percent']
        
        merged = merged.merge(
            self.datasets['summary'][summary_cols],
            left_on='algorithm_name',
            right_on='Algorithm',
            how='left',
            suffixes=('', '_summary')
        )
        logger.info(f"  After merging summary: {len(merged):,} rows")
        
        # Verify no row loss
        if len(merged) != initial_rows:
            logger.warning(f"  ⚠ Row count changed from {initial_rows} to {len(merged)}")
        else:
            logger.info(f"  ✓ No rows lost during merge")
        
        logger.info(f"\nFinal merged dataset: {len(merged):,} rows × {len(merged.columns)} columns")
        
        return merged
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics for all datasets.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        for name, df in self.datasets.items():
            summary[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
                'dtypes': df.dtypes.value_counts().to_dict()
            }
        
        return summary


def main():
    """Test data loading."""
    print("\n" + "=" * 60)
    print("DATA LOADER TEST")
    print("=" * 60 + "\n")
    
    # Initialize loader
    loader = DataLoader(data_dir="../dataset_generation")
    
    # Load all datasets
    datasets = loader.load_all()
    
    # Print summary
    print("\nDATASET SUMMARY:")
    print("-" * 60)
    summary = loader.get_data_summary()
    for name, stats in summary.items():
        print(f"\n{name.upper()}:")
        print(f"  Rows: {stats['rows']:,}")
        print(f"  Columns: {stats['columns']}")
        print(f"  Missing: {stats['missing_pct']:.2f}%")
    
    # Test merge
    print("\n" + "=" * 60)
    merged = loader.merge_datasets()
    print("=" * 60)
    
    print(f"\nMerged dataset shape: {merged.shape}")
    print(f"Columns: {list(merged.columns[:10])}... (showing first 10)")
    
    print("\n✓ Data loader test complete!")


if __name__ == "__main__":
    main()
