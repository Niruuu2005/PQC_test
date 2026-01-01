"""
Data Cleaner Module
Handles missing values, outliers, and data quality issues.

Author: AIRAWAT Team
Date: 2026-01-01
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and prepare cryptanalysis datasets."""
    
    def __init__(self):
        """Initialize DataCleaner."""
        self.cleaning_report = {
            'missing_values_handled': {},
            'outliers_capped': {},
            'invalid_records_removed': 0
        }
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on column type.
        
        Strategy:
        - Numeric: Median imputation
        - String: Categorical fill ('no_value')
        - Boolean: Mode imputation
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        df_clean = df.copy()
        
        # Numeric columns: median imputation
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                self.cleaning_report['missing_values_handled'][col] = {
                    'count': missing_count,
                    'method': 'median',
                    'fill_value': median_val
                }
                logger.info(f"  {col}: {missing_count} filled with median={median_val:.2f}")
        
        # String columns: categorical fill
        string_cols = df_clean.select_dtypes(include=['object']).columns
        for col in string_cols:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                fill_value = 'no_value' if 'hex' not in col.lower() else ''
                df_clean[col].fillna(fill_value, inplace=True)
                self.cleaning_report['missing_values_handled'][col] = {
                    'count': missing_count,
                    'method': 'fill',
                    'fill_value': fill_value
                }
                logger.info(f"  {col}: {missing_count} filled with '{fill_value}'")
        
        logger.info(f"✓ Handled missing values in {len(self.cleaning_report['missing_values_handled'])} columns")
        return df_clean
    
    def cap_outliers(self, df: pd.DataFrame, columns: List[str], percentile: float = 99) -> pd.DataFrame:
        """
        Cap outliers at specified percentile.
        
        Args:
            df: Input DataFrame
            columns: List of columns to cap
            percentile: Percentile threshold (default: 99)
            
        Returns:
            DataFrame with outliers capped
        """
        logger.info(f"Capping outliers at {percentile}th percentile...")
        df_clean = df.copy()
        
        for col in columns:
            if col not in df_clean.columns:
                logger.warning(f"  Column not found: {col}")
                continue
                
            upper_bound = df_clean[col].quantile(percentile / 100)
            outliers = (df_clean[col] > upper_bound).sum()
            
            if outliers > 0:
                df_clean[col] = df_clean[col].clip(upper=upper_bound)
                self.cleaning_report['outliers_capped'][col] = {
                    'count': outliers,
                    'upper_bound': upper_bound
                }
                logger.info(f"  {col}: {outliers} outliers capped at {upper_bound:.2f}")
        
        return df_clean
    
    def validate_hex_strings(self, df: pd.DataFrame, hex_columns: List[str]) -> pd.DataFrame:
        """
        Validate and clean hex string columns.
        
        Args:
            df: Input DataFrame
            hex_columns: List of hex string columns
            
        Returns:
            DataFrame with invalid hex strings removed
        """
        logger.info("Validating hex strings...")
        df_clean = df.copy()
        
        def is_valid_hex(hex_str):
            """Check if string is valid hex."""
            if pd.isna(hex_str) or hex_str == '':
                return True  # Allow empty
            try:
                bytes.fromhex(hex_str)
                return len(hex_str) % 2 == 0  # Even length
            except:
                return False
        
        for col in hex_columns:
            if col not in df_clean.columns:
                continue
                
            invalid_mask = ~df_clean[col].apply(is_valid_hex)
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                logger.warning(f"  {col}: {invalid_count} invalid hex strings found")
                # Replace invalid with empty string
                df_clean.loc[invalid_mask, col] = ''
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to check for duplicates (None = all columns)
            
        Returns:
            DataFrame without duplicates
        """
        logger.info("Removing duplicates...")
        initial_rows = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep='first')
        duplicates_removed = initial_rows - len(df_clean)
        
        self.cleaning_report['invalid_records_removed'] = duplicates_removed
        logger.info(f"  Removed {duplicates_removed} duplicate rows")
        
        return df_clean
    
    def clean_dataset(self, df: pd.DataFrame, 
                     outlier_cols: List[str] = None,
                     hex_cols: List[str] = None,
                     duplicate_subset: List[str] = None) -> pd.DataFrame:
        """
        Main cleaning pipeline.
        
        Args:
            df: Input DataFrame
            outlier_cols: Columns to cap outliers
            hex_cols: Hex string columns to validate
            duplicate_subset: Columns for duplicate detection
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA CLEANING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Input: {len(df):,} rows × {len(df.columns)} columns")
        
        # Step 1: Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Step 2: Cap outliers
        if outlier_cols:
            df_clean = self.cap_outliers(df_clean, outlier_cols)
        
        # Step 3: Validate hex strings
        if hex_cols:
            df_clean = self.validate_hex_strings(df_clean, hex_cols)
        
        # Step 4: Remove duplicates
        df_clean = self.remove_duplicates(df_clean, duplicate_subset)
        
        logger.info("=" * 60)
        logger.info(f"Output: {len(df_clean):,} rows × {len(df_clean.columns)} columns")
        logger.info("✓ Cleaning complete")
        logger.info("=" * 60)
        
        return df_clean
    
    def get_report(self) -> Dict:
        """Get cleaning report."""
        return self.cleaning_report


def main():
    """Test data cleaner."""
    print("\n" + "=" * 60)
    print("DATA CLEANER TEST")
    print("=" * 60 + "\n")
    
    # Create test data
    test_data = pd.DataFrame({
        'execution_time_ms': [100, 200, 300, np.nan, 10000],  # Has missing + outlier
        'memory_used_mb': [50, 60, 70, 80, 5000],  # Has outlier
        'plaintext_hex': ['61626', '6162636', '', 'invalid', '616263'],  # Has invalid
        'attack_category': ['A', 'B', 'A', np.nan, 'C']  # Has missing
    })
    
    print("Test data:")
    print(test_data)
    
    # Clean
    cleaner = DataCleaner()
    cleaned = cleaner.clean_dataset(
        test_data,
        outlier_cols=['execution_time_ms', 'memory_used_mb'],
        hex_cols=['plaintext_hex']
    )
    
    print("\nCleaned data:")
    print(cleaned)
    
    print("\nCleaning report:")
    import json
    print(json.dumps(cleaner.get_report(), indent=2, default=str))
    
    print("\n✓ Data cleaner test complete!")


if __name__ == "__main__":
    main()
