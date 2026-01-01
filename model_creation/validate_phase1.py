"""
Phase 1 Validation
Quick validation script for Phase 1 components.

Run this to verify Phase 1 completion.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.cleaner import DataCleaner
from src.data.feature_engineer import FeatureEngineer
from src.data.splitter import DataSplitter


def validate_phase1():
    """Validate Phase 1 completion."""
    print("\n" + "=" * 70)
    print("PHASE 1 VALIDATION")
    print("=" * 70)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Data Cleaner
    print("\n[1/4] Testing Data Cleaner...")
    try:
        test_df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 1000],
            'col2': ['a', 'b', None, 'd', 'e']
        })
        cleaner = DataCleaner()
        cleaned = cleaner.clean_dataset(test_df, outlier_cols=['col1'])
        assert cleaned['col1'].isnull().sum() == 0
        assert cleaned['col1'].max() < 1000
        print("  ✓ Data Cleaner working")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Data Cleaner failed: {e}")
    tests_total += 1
    
    # Test 2: Feature Engineer
    print("\n[2/4] Testing Feature Engineer...")
    try:
        test_df = pd.DataFrame({
            'execution_time_ms': [100, 200, 300],
            'iterations_performed': [1000, 2000, 3000],
            'shannon_entropy': [7.5, 7.8, 7.9],
            'plaintext_hex': ['616263', '646566', '676869']
        })
        engineer = FeatureEngineer()
        features = engineer.engineer_all_features(test_df)
        assert len(features.columns) >= 10
        assert 'time_per_iteration' in features.columns
        print(f"  ✓ Feature Engineer working ({len(features.columns)} features created)")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Feature Engineer failed: {e}")
    tests_total += 1
    
    # Test 3: Data Splitter
    print("\n[3/4] Testing Data Splitter...")
    try:
        test_df = pd.DataFrame({
            'feature': np.random.randn(1000),
            'timestamp': pd.date_range('2026-01-01', periods=1000, freq='h'),
            'target': np.random.choice(['A', 'B'], 1000)
        })
        splitter = DataSplitter()
        train, val, test = splitter.time_based_split(test_df)
        total = len(train) + len(val) + len(test)
        assert total == len(test_df)
        assert abs(len(train)/total - 0.6) < 0.01
        print(f"  ✓ Data Splitter working (Train:{len(train)} Val:{len(val)} Test:{len(test)})")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Data Splitter failed: {e}")
    tests_total += 1
    
    # Test 4: Full Integration
    print("\n[4/4] Testing Full Integration...")
    try:
        n = 500
        test_df = pd.DataFrame({
            'execution_time_ms': np.random.lognormal(5, 1, n),
            'memory_used_mb': np.random.lognormal(4, 0.5, n),
            'iterations_performed': np.random.randint(100, 10000, n),
            'shannon_entropy': np.random.uniform(7, 8, n),
            'plaintext_hex': ['616263' * np.random.randint(1, 5) for _ in range(n)],
            'timestamp': pd.date_range('2026-01-01', periods=n, freq='min'),
            'target': np.random.choice(['A', 'B', 'C'], n)
        })
        
        # Add missing values
        test_df.loc[::10, 'execution_time_ms'] = np.nan
        
        # Pipeline
        cleaner = DataCleaner()
        cleaned = cleaner.clean_dataset(test_df, outlier_cols=['execution_time_ms'])
        
        engineer = FeatureEngineer()
        features = engineer.engineer_all_features(cleaned)
        
        features['target'] = cleaned['target']
        splitter = DataSplitter()
        train, val, test = splitter.time_based_split(features)
        
        assert len(features.columns) >= 15
        assert len(train) + len(val) + len(test) == len(features)
        print(f"  ✓ Integration working ({len(features.columns)} features, {len(train)} train samples)")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Integration failed: {e}")
    tests_total += 1
    
    # Results
    print("\n" + "=" * 70)
    print(f"RESULTS: {tests_passed}/{tests_total} tests passed")
    print("=" * 70)
    
    if tests_passed == tests_total:
        print("\n✓✓✓ PHASE 1 VALIDATION: PASSED ✓✓✓")
        print("\nAll components working correctly!")
        print("Ready to proceed to Phase 2: Classical ML Baseline")
        return True
    else:
        print(f"\n✗✗✗ PHASE 1 VALIDATION: FAILED ✗✗✗")
        print(f"\n{tests_total - tests_passed} test(s) failed. Please fix before continuing.")
        return False


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # Suppress INFO logs for cleaner output
    
    success = validate_phase1()
    sys.exit(0 if success else 1)
