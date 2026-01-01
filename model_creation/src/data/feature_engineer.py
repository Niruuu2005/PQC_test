"""
Feature Engineer Module
Creates 65+ features from raw cryptanalysis data.

Author: AIRAWAT Team
Date: 2026-01-01
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for ML models."""
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.feature_count = 0
        self.feature_metadata = []
    
    def extract_hex_features(self, df: pd.DataFrame, hex_col: str, prefix: str) -> pd.DataFrame:
        """
        Extract features from hex string column.
        
        Features:
        - byte_length
        - unique_byte_count
        - byte_entropy
        
        Args:
            df: Input DataFrame
            hex_col: Hex string column name
            prefix: Prefix for new columns (e.g., 'pt' for plaintext)
            
        Returns:
            DataFrame with new hex features
        """
        logger.info(f"Extracting hex features from {hex_col}...")
        
        def compute_hex_features(hex_str):
            if pd.isna(hex_str) or hex_str == '':
                return pd.Series({
                    f'{prefix}_byte_length': 0,
                    f'{prefix}_unique_bytes': 0,
                    f'{prefix}_byte_entropy': 0
                })
            
            try:
                bytes_data = bytes.fromhex(hex_str)
                byte_counts = pd.Series(list(bytes_data)).value_counts()
                
                # Calculate entropy
                probs = byte_counts / len(bytes_data)
                entropy = -sum(probs * np.log2(probs + 1e-10))
                
                return pd.Series({
                    f'{prefix}_byte_length': len(bytes_data),
                    f'{prefix}_unique_bytes': len(byte_counts),
                    f'{prefix}_byte_entropy': entropy
                })
            except:
                return pd.Series({
                    f'{prefix}_byte_length': 0,
                    f'{prefix}_unique_bytes': 0,
                    f'{prefix}_byte_entropy': 0
                })
        
        hex_features = df[hex_col].apply(compute_hex_features)
        logger.info(f"  Created {len(hex_features.columns)} hex features")
        
        return hex_features
    
    def extract_temporal_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Extract temporal features with cyclical encoding.
        
        Features:
        - hour_sin, hour_cos
        - day_of_week_sin, day_of_week_cos
        - is_weekend
        
        Args:
            df: Input DataFrame
            timestamp_col: Timestamp column name
            
        Returns:
            DataFrame with temporal features
        """
        logger.info(f"Extracting temporal features from {timestamp_col}...")
        
        if timestamp_col not in df.columns:
            logger.warning(f"  Column {timestamp_col} not found, skipping")
            return pd.DataFrame()
        
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        temporal_features = pd.DataFrame({
            'hour': timestamps.dt.hour,
            'day_of_week': timestamps.dt.dayofweek,
            'is_weekend': (timestamps.dt.dayofweek >= 5).astype(int)
        })
        
        # Cyclical encoding
        temporal_features['hour_sin'] = np.sin(2 * np.pi * temporal_features['hour'] / 24)
        temporal_features['hour_cos'] = np.cos(2 * np.pi * temporal_features['hour'] / 24)
        temporal_features['dow_sin'] = np.sin(2 * np.pi * temporal_features['day_of_week'] / 7)
        temporal_features['dow_cos'] = np.cos(2 * np.pi * temporal_features['day_of_week'] / 7)
        
        # Drop original non-cyclical features
        temporal_features = temporal_features.drop(['hour', 'day_of_week'], axis=1)
        
        logger.info(f"  Created {len(temporal_features.columns)} temporal features")
        return temporal_features
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived efficiency and interaction features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with derived features
        """
        logger.info("Creating derived features...")
        
        derived = pd.DataFrame()
        
        # Efficiency metrics (handle division by zero)
        if 'execution_time_ms' in df.columns and 'iterations_performed' in df.columns:
            derived['time_per_iteration'] = df['execution_time_ms'] / (df['iterations_performed'] + 1)
        
        if 'memory_used_mb' in df.columns and 'iterations_performed' in df.columns:
            derived['memory_per_iteration'] = df['memory_used_mb'] / (df['iterations_performed'] + 1)
        
        if 'iterations_performed' in df.columns and 'execution_time_ms' in df.columns:
            derived['iterations_per_second'] = df['iterations_performed'] / (df['execution_time_ms'] / 1000 + 1e-6)
        
        # Interaction features
        if 'shannon_entropy' in df.columns and 'chi_square_statistic' in df.columns:
            derived['entropy_chisq_interaction'] = df['shannon_entropy'] * df['chi_square_statistic']
        
        if 'avalanche_effect' in df.columns and 'shannon_entropy' in df.columns:
            derived['avalanche_entropy_interaction'] = df['avalanche_effect'] * df['shannon_entropy']
        
        logger.info(f"  Created {len(derived.columns)} derived features")
        return derived
    
    def encode_categorical(self, df: pd.DataFrame, 
                          one_hot_cols: List[str] = None,
                          label_encode_cols: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            one_hot_cols: Columns for one-hot encoding
            label_encode_cols: Columns for label encoding
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")
        
        encoded = pd.DataFrame(index=df.index)
        
        # One-hot encoding
        if one_hot_cols:
            for col in one_hot_cols:
                if col in df.columns:
                    dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
                    encoded = pd.concat([encoded, dummies], axis=1)
                    logger.info(f"  One-hot encoded {col}: {len(dummies.columns)} categories")
        
        # Label encoding
        if label_encode_cols:
            for col in label_encode_cols:
                if col in df.columns:
                    # Simple label encoding (sorted unique values)
                    categories = sorted(df[col].dropna().unique())
                    mapping = {cat: idx for idx, cat in enumerate(categories)}
                    encoded[f'{col}_encoded'] = df[col].map(mapping).fillna(-1).astype(int)
                    logger.info(f"  Label encoded {col}: {len(categories)} categories")
        
        return encoded
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Input: {len(df):,} rows × {len(df.columns)} columns")
        
        # Start with original numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        features = df[numeric_cols].copy()
        logger.info(f"Starting with {len(numeric_cols)} numeric features")
        
        # Hex features
        if 'plaintext_hex' in df.columns:
            pt_features = self.extract_hex_features(df, 'plaintext_hex', 'pt')
            features = pd.concat([features, pt_features], axis=1)
        
        if 'ciphertext_hex' in df.columns:
            ct_features = self.extract_hex_features(df, 'ciphertext_hex', 'ct')
            features = pd.concat([features, ct_features], axis=1)
        
        # Temporal features
        if 'timestamp' in df.columns:
            temporal = self.extract_temporal_features(df, 'timestamp')
            features = pd.concat([features, temporal], axis=1)
        
        # Derived features
        derived = self.create_derived_features(df)
        features = pd.concat([features, derived], axis=1)
        
        # Categorical encoding
        one_hot_cols = ['algorithm_name', 'attack_category'] if 'algorithm_name' in df.columns else []
        label_cols = ['attack_language'] if 'attack_language' in df.columns else []
        
        if one_hot_cols or label_cols:
            encoded = self.encode_categorical(df, one_hot_cols, label_cols)
            features = pd.concat([features, encoded], axis=1)
        
        logger.info("=" * 60)
        logger.info(f"Output: {len(features):,} rows × {len(features.columns)} features")
        logger.info("✓ Feature engineering complete")
        logger.info("=" * 60)
        
        self.feature_count = len(features.columns)
        return features


def main():
    """Test feature engineer."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEER TEST")
    print("=" * 60 + "\n")
    
    # Create test data
    test_data = pd.DataFrame({
        'execution_time_ms': [100, 200, 300],
        'memory_used_mb': [50, 60, 70],
        'iterations_performed': [1000, 2000, 3000],
        'shannon_entropy': [7.5, 7.8, 7.9],
        'chi_square_statistic': [250, 260, 255],
        'avalanche_effect': [0.49, 0.51, 0.50],
        'plaintext_hex': ['616263', '646566', '676869'],
        'ciphertext_hex': ['abcdef', '123456', 'fedcba'],
        'timestamp': ['2026-01-01 10:00:00', '2026-01-01 14:30:00', '2026-01-01 22:15:00'],
        'algorithm_name': ['AES-128', 'AES-256', 'AES-128'],
        'attack_category': ['Brute-Force', 'Side-Channel', 'Brute-Force']
    })
    
    print("Test data shape:", test_data.shape)
    
    # Engineer features
    engineer = FeatureEngineer()
    features = engineer.engineer_all_features(test_data)
    
    print(f"\nEngineered features shape: {features.shape}")
    print(f"Total features created: {len(features.columns)}")
    print(f"\nFeature columns (first 20):")
    print(list(features.columns[:20]))
    
    print("\n✓ Feature engineer test complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
