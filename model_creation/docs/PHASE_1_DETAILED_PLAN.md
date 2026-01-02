# PHASE 1: DATA ENGINEERING - DETAILED IMPLEMENTATION PLAN
**Duration:** Weeks 1-2 (14 days)  
**Workspace:** `d:\Dream\AIRAWAT\model_creation`  
**Status:** Ready to Begin

---

## OVERVIEW

Phase 1 transforms raw cryptanalysis datasets into ML-ready features with proper validation, cleaning, engineering, and splits. This is the foundation for all subsequent phases.

**Key Principles:**
- ✅ **No Data Leakage:** Split BEFORE any scaling or imputation based on test set
- ✅ **Reproducibility:** Fix random seeds, document all transformations
- ✅ **Validation:** Test every transformation with assertions
- ✅ **Documentation:** Track all decisions and their rationale

---

## DIRECTORY STRUCTURE SETUP

### Day 0: Initialize Project Structure
```bash
model_creation/
├── data/
│   ├── raw/                 # Original CSVs (do not modify)
│   ├── interim/             # Intermediate processing steps
│   └── processed/           # Final ML-ready data
├── src/
│   └── data/
│       ├── __init__.py
│       ├── loader.py        # Data loading utilities
│       ├── cleaner.py       # Cleaning functions
│       ├── feature_engineer.py  # Feature extraction
│       ├── splitter.py      # Train/val/test splitting
│       └── validators.py    # Data validation functions
├── tests/
│   └── data/
│       ├── __init__.py
│       ├── test_loader.py
│       ├── test_cleaner.py
│       ├── test_feature_engineer.py
│       └── test_splitter.py
├── configs/
│   └── data_config.yaml     # All data processing parameters
├── reports/
│   └── phase1/              # EDA reports, visualizations
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_data_validation.ipynb
├── encoders/                # Saved sklearn encoders/scalers
├── logs/                    # Processing logs
└── docs/                    # This file and others
```

**Create directories:**
```bash
cd d:\Dream\AIRAWAT\model_creation
mkdir -p data/{raw,interim,processed} src/data tests/data configs reports/phase1 notebooks encoders logs
```

---

## WEEK 1: DATA EXPLORATION & ASSESSMENT

### DAY 1: Environment Setup & Data Loading

#### Task 1.1: Create Configuration File

**File:** `configs/data_config.yaml`

```yaml
# Data Configuration
project_name: "QML Cryptanalysis"
random_seed: 42

# Data paths
data:
  raw_dir: "data/raw"
  interim_dir: "data/interim"
  processed_dir: "data/processed"
  
  # Input CSVs (relative to raw_dir)
  attack_dataset: "attack_dataset.csv"
  attack_metadata: "attack_metadata.csv"
  crypto_dataset: "crypto_dataset.csv"
  crypto_summary: "cryptographic_algorithm_summary.csv"

# Feature engineering
features:
  # Quantum circuit features (will be normalized to [0, π])
  quantum_features:
    - execution_time_ms
    - memory_used_mb
    - iterations_performed
    - shannon_entropy
    - chi_square_statistic
    - avalanche_effect
    - efficiency_score
    - entropy_chi_interaction
  
  # All numerical features
  numerical_features:
    - execution_time_ms
    - memory_used_mb
    - cpu_usage_percent
    - iterations_performed
    - confidence_score
    - shannon_entropy
    - chi_square_statistic
    - avalanche_effect
    - severity_score
  
  # Categorical features for encoding
  categorical_features:
    algorithm_name: "onehot"
    attack_category: "onehot"
    attack_language: "onehot"
    vulnerability_type: "onehot"
    key_size_bits: "label"
    security_level: "label"

# Cleaning parameters
cleaning:
  outlier_percentile: 99  # Cap outliers at this percentile
  missing_value_strategy:
    metric_values: "median"
    error_message: "constant:no_error"
    recovered_data_hex: "constant:"
    notes: "constant:no_notes"

# Train/val/test split
split:
  method: "temporal"  # or "stratified"
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  stratify_by:
    - attack_category
    - algorithm_name
```

**Validation:**
```python
import yaml

with open('configs/data_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

assert config['split']['train_ratio'] + config['split']['val_ratio'] + config['split']['test_ratio'] == 1.0
assert config['random_seed'] == 42
print("✅ Config validation passed")
```

#### Task 1.2: Implement Data Loader

**File:** `src/data/loader.py`

```python
"""
Data loading module for QML Cryptanalysis project.
Handles reading CSVs and basic validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate raw datasets."""
    
    def __init__(self, config: dict):
        """
        Initialize loader with configuration.
        
        Args:
            config: Dictionary from data_config.yaml
        """
        self.config = config
        self.raw_dir = Path(config['data']['raw_dir'])
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all 4 datasets.
        
        Returns:
            Dictionary with keys: attack_dataset, attack_metadata,
            crypto_dataset, crypto_summary
        
        Raises:
            FileNotFoundError: If any CSV file is missing
        """
        logger.info("Loading datasets...")
        
        datasets = {}
        
        # Load attack_dataset
        attack_path = self.raw_dir / self.config['data']['attack_dataset']
        datasets['attack_dataset'] = pd.read_csv(attack_path)
        logger.info(f"✓ Loaded attack_dataset: {datasets['attack_dataset'].shape}")
        
        # Load attack_metadata
        metadata_path = self.raw_dir / self.config['data']['attack_metadata']
        datasets['attack_metadata'] = pd.read_csv(metadata_path)
        logger.info(f"✓ Loaded attack_metadata: {datasets['attack_metadata'].shape}")
        
        # Load crypto_dataset
        crypto_path = self.raw_dir / self.config['data']['crypto_dataset']
        datasets['crypto_dataset'] = pd.read_csv(crypto_path)
        logger.info(f"✓ Loaded crypto_dataset: {datasets['crypto_dataset'].shape}")
        
        # Load crypto_summary
        summary_path = self.raw_dir / self.config['data']['crypto_summary']
        datasets['crypto_summary'] = pd.read_csv(summary_path)
        logger.info(f"✓ Loaded crypto_summary: {datasets['crypto_summary'].shape}")
        
        return datasets
    
    def validate_schema(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate that datasets have expected columns.
        
        Args:
            datasets: Dictionary of DataFrames
        
        Returns:
            True if validation passes
        
        Raises:
            ValueError: If required columns are missing
        """
        logger.info("Validating schema...")
        
        # Expected columns (minimum required)
        expected_columns = {
            'attack_dataset': [
                'attack_execution_id', 'encryption_row_id', 'attack_id',
                'algorithm_name', 'execution_time_ms', 'attack_success'
            ],
            'attack_metadata': [
                'attack_id', 'attack_name', 'category'
            ],
            'crypto_dataset': [
                'row_id', 'algorithm_name', 'shannon_entropy', 
                'chi_square_statistic', 'avalanche_effect'
            ],
            'crypto_summary': [
                'Algorithm', 'Total_Attacks_Tested'
            ]
        }
        
        for dataset_name, required_cols in expected_columns.items():
            df = datasets[dataset_name]
            missing_cols = set(required_cols) - set(df.columns)
            
            if missing_cols:
                raise ValueError(
                    f"{dataset_name} missing columns: {missing_cols}"
                )
        
        logger.info("✓ Schema validation passed")
        return True
    
    def validate_foreign_keys(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate foreign key relationships.
        
        Args:
            datasets: Dictionary of DataFrames
        
        Returns:
            True if validation passes
        """
        logger.info("Validating foreign keys...")
        
        attack_df = datasets['attack_dataset']
        metadata_df = datasets['attack_metadata']
        crypto_df = datasets['crypto_dataset']
        
        # Check: attack_dataset.attack_id in attack_metadata.attack_id
        invalid_attack_ids = ~attack_df['attack_id'].isin(metadata_df['attack_id'])
        invalid_count = invalid_attack_ids.sum()
        
        if invalid_count > 0:
            logger.warning(
                f"⚠️  {invalid_count} rows with invalid attack_id (not in metadata)"
            )
            # Log some examples
            examples = attack_df[invalid_attack_ids]['attack_id'].head()
            logger.warning(f"Examples: {examples.tolist()}")
        
        # Check: attack_dataset.encryption_row_id in crypto_dataset.row_id
        invalid_encryption_ids = ~attack_df['encryption_row_id'].isin(crypto_df['row_id'])
        invalid_enc_count = invalid_encryption_ids.sum()
        
        if invalid_enc_count > 0:
            logger.warning(
                f"⚠️  {invalid_enc_count} rows with invalid encryption_row_id"
            )
        
        logger.info("✓ Foreign key validation complete")
        return True


def load_and_validate(config_path: str = 'configs/data_config.yaml') -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load and validate all datasets.
    
    Args:
        config_path: Path to configuration YAML
    
    Returns:
        Dictionary of validated DataFrames
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    loader = DataLoader(config)
    datasets = loader.load_all_datasets()
    loader.validate_schema(datasets)
    loader.validate_foreign_keys(datasets)
    
    return datasets


if __name__ == "__main__":
    # Test loading
    datasets = load_and_validate()
    print("\n✅ Data loading successful!")
    
    for name, df in datasets.items():
        print(f"{name}: {df.shape}")
```

**Test File:** `tests/data/test_loader.py`

```python
import pytest
import pandas as pd
from src.data.loader import DataLoader
import yaml

@pytest.fixture
def config():
    with open('configs/data_config.yaml', 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def loader(config):
    return DataLoader(config)

def test_loader_initialization(loader):
    """Test that loader initializes correctly."""
    assert loader.config is not None
    assert loader.raw_dir.exists()

def test_load_all_datasets(loader):
    """Test loading all datasets."""
    datasets = loader.load_all_datasets()
    
    assert 'attack_dataset' in datasets
    assert 'attack_metadata' in datasets
    assert 'crypto_dataset' in datasets
    assert 'crypto_summary' in datasets
    
    # Check they're DataFrames
    for name, df in datasets.items():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

def test_schema_validation(loader):
    """Test schema validation."""
    datasets = loader.load_all_datasets()
    assert loader.validate_schema(datasets) == True

def test_foreign_key_validation(loader):
    """Test foreign key validation."""
    datasets = loader.load_all_datasets()
    assert loader.validate_foreign_keys(datasets) == True
```

**Run Test:**
```bash
pytest tests/data/test_loader.py -v
```

---

### DAY 2: Exploratory Data Analysis

#### Task 2.1: Basic Statistics

**Jupyter Notebook:** `notebooks/01_data_exploration.ipynb`

```python
# Cell 1: Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loader import load_and_validate

%matplotlib inline
sns.set_style('whitegrid')

# Load data
datasets = load_and_validate()
attack_df = datasets['attack_dataset']
metadata_df = datasets['attack_metadata']
crypto_df = datasets['crypto_dataset']
summary_df = datasets['crypto_summary']

print("Datasets loaded successfully!")

# Cell 2: Dataset Dimensions
print("="*60)
print("DATASET DIMENSIONS")
print("="*60)

for name, df in datasets.items():
    print(f"{name:30s}: {df.shape[0]:,} rows × {df.shape[1]:,} cols")

# Cell 3: Data Types
print("\n" + "="*60)
print("ATTACK DATASET - DATA TYPES")
print("="*60)

print(attack_df.dtypes.value_counts())
print("\nSample columns by type:")
print(attack_df.dtypes.head(20))

# Cell 4: Missing Values
print("\n" + "="*60)
print("MISSING VALUES ANALYSIS")
print("="*60)

missing = attack_df.isnull().sum()
missing_pct = (missing / len(attack_df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

print(missing_df)

# Visualize missing values
plt.figure(figsize=(10, 6))
missing_df['Missing %'].plot(kind='barh')
plt.xlabel('Missing Percentage (%)')
plt.title('Missing Values by Column')
plt.tight_layout()
plt.savefig('reports/phase1/missing_values.png', dpi=300)
plt.show()

# Cell 5: Target Variable Distribution
print("\n" + "="*60)
print("TARGET VARIABLES DISTRIBUTION")
print("="*60)

# Task 1: Attack Category
print("\nAttack Category Distribution:")
if 'attack_category' in attack_df.columns:
    print(attack_df['attack_category'].value_counts())
    
    plt.figure(figsize=(12, 5))
    attack_df['attack_category'].value_counts().plot(kind='bar')
    plt.title('Attack Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('reports/phase1/attack_category_dist.png', dpi=300)
    plt.show()

# Task 2: Attack Success
print("\nAttack Success Distribution:")
if 'attack_success' in attack_df.columns:
    success_dist = attack_df['attack_success'].value_counts()
    print(success_dist)
    success_rate = success_dist[1] / (success_dist[0] + success_dist[1]) * 100
    print(f"Success Rate: {success_rate:.2f}%")
    
    # Check class imbalance
    imbalance_ratio = success_dist.min() / success_dist.max()
    print(f"Imbalance Ratio: {imbalance_ratio:.3f}")
    
    if imbalance_ratio < 0.3:
        print("⚠️  HIGH IMBALANCE DETECTED - Will need SMOTE or class weighting")

# Cell 6: Numerical Features Statistics
print("\n" + "="*60)
print("NUMERICAL FEATURES STATISTICS")
print("="*60)

numerical_cols = [
    'execution_time_ms', 'memory_used_mb', 'cpu_usage_percent',
    'iterations_performed', 'confidence_score'
]

numerical_cols = [col for col in numerical_cols if col in attack_df.columns]

stats = attack_df[numerical_cols].describe()
print(stats)

# Cell 7: Correlation Matrix
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

# Select numerical features for correlation
corr_features = numerical_cols + ['shannon_entropy', 'chi_square_statistic', 'avalanche_effect']
corr_features = [f for f in corr_features if f in attack_df.columns or f in crypto_df.columns]

# Merge with crypto_df to get statistical features
merged_df = attack_df.merge(
    crypto_df[['row_id', 'shannon_entropy', 'chi_square_statistic', 'avalanche_effect']],
    left_on='encryption_row_id',
    right_on='row_id',
    how='left'
)

correlation_matrix = merged_df[corr_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('reports/phase1/correlation_matrix.png', dpi=300)
plt.show()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.95:
            high_corr_pairs.append(
                (correlation_matrix.index[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j])
            )

if high_corr_pairs:
    print("\n⚠️  Highly Correlated Pairs (|r| > 0.95):")
    for feat1, feat2, corr_val in high_corr_pairs:
        print(f"  {feat1} ↔ {feat2}: {corr_val:.4f}")
else:
    print("\n✓ No highly correlated pairs found")

# Cell 8: Save Summary Report
report = f"""
DATA EXPLORATION REPORT
Generated: {pd.Timestamp.now()}

Dataset Dimensions:
- attack_dataset: {attack_df.shape}
- attack_metadata: {metadata_df.shape}
- crypto_dataset: {crypto_df.shape}
- crypto_summary: {summary_df.shape}

Missing Values:
Total columns with missing values: {len(missing_df)}
{missing_df.to_string()}

Target Distribution:
- Attack Success Rate: {success_rate:.2f}%
- Imbalance Ratio: {imbalance_ratio:.3f}

Numerical Features Summary:
{stats.to_string()}

Highly Correlated Pairs:
{high_corr_pairs if high_corr_pairs else "None"}

RECOMMENDATIONS:
1. Handle missing values using median/constant fill strategy
2. {"Apply SMOTE or class weighting due to imbalance" if imbalance_ratio < 0.3 else "Class balance acceptable"}
3. Consider removing highly correlated features
4. Cap outliers at 99th percentile
"""

with open('reports/phase1/exploration_report.txt', 'w') as f:
    f.write(report)

print("\n✅ Report saved to reports/phase1/exploration_report.txt")
```

**Expected Outputs:**
- `reports/phase1/missing_values.png`
- `reports/phase1/attack_category_dist.png`
- `reports/phase1/correlation_matrix.png`
- `reports/phase1/exploration_report.txt`

---

### DAY 3-4: Data Quality Checks & Hex Validation

#### Task 3.1: Implement Validators

**File:** `src/data/validators.py`

```python
"""
Data validation utilities for QML Cryptanalysis.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality and integrity."""
    
    @staticmethod
    def validate_hex_string(hex_str: str) -> bool:
        """
        Validate hex string format.
        
        Args:
            hex_str: Hex string to validate
        
        Returns:
            True if valid, False otherwise
        """
        if pd.isna(hex_str) or hex_str == "":
            return True  # Allow empty/NA
        
        # Check: only hex characters
        if not re.match(r'^[0-9a-fA-F]*$', hex_str):
            return False
        
        # Check: even length (2 chars per byte)
        if len(hex_str) % 2 != 0:
            return False
        
        return True
    
    @staticmethod
    def validate_hex_columns(df: pd.DataFrame, hex_columns: List[str]) -> pd.DataFrame:
        """
        Validate all hex columns in DataFrame.
        
        Args:
            df: DataFrame to validate
            hex_columns: List of column names containing hex strings
        
        Returns:
            DataFrame with validation results
        """
        results = []
        
        for col in hex_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found")
                continue
            
            invalid_mask = ~df[col].apply(DataValidator.validate_hex_string)
            invalid_count = invalid_mask.sum()
            
            results.append({
                'column': col,
                'total': len(df),
                'invalid': invalid_count,
                'invalid_pct': invalid_count / len(df) * 100
            })
            
            if invalid_count > 0:
                logger.warning(f"{col}: {invalid_count} invalid hex strings")
                # Log examples
                examples = df[invalid_mask][col].head(3).tolist()
                logger.warning(f"Examples: {examples}")
        
        return pd.DataFrame(results)
    
    @staticmethod
    def check_length_consistency(df: pd.DataFrame) -> pd.DataFrame:
        """
        Check if plaintext/ciphertext length matches hex string length.
        
        Args:
            df: DataFrame with hex columns and length columns
        
        Returns:
            DataFrame with inconsistency results
        """
        results = []
        
        # Check plaintext
        if 'plaintext_hex' in df.columns and 'plaintext_length' in df.columns:
            df['plaintext_computed_length'] = df['plaintext_hex'].apply(
                lambda x: len(x) // 2 if pd.notna(x) else 0
            )
            
            mismatch = df['plaintext_length'] != df['plaintext_computed_length']
            mismatch_count = mismatch.sum()
            
            results.append({
                'check': 'plaintext_length',
                'mismatches': mismatch_count,
                'mismatch_pct': mismatch_count / len(df) * 100
            })
        
        # Check ciphertext
        if 'ciphertext_hex' in df.columns and 'ciphertext_length' in df.columns:
            df['ciphertext_computed_length'] = df['ciphertext_hex'].apply(
                lambda x: len(x) // 2 if pd.notna(x) else 0
            )
            
            mismatch = df['ciphertext_length'] != df['ciphertext_computed_length']
            mismatch_count = mismatch.sum()
            
            results.append({
                'check': 'ciphertext_length',
                'mismatches': mismatch_count,
                'mismatch_pct': mismatch_count / len(df) * 100
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def detect_outliers_iqr(df: pd.DataFrame, column: str, percentile: float = 99) -> Tuple[pd.Series, float]:
        """
        Detect outliers using IQR method or percentile capping.
        
        Args:
            df: DataFrame
            column: Column name
            percentile: Percentile to cap at (default: 99)
        
        Returns:
            Tuple of (outlier_mask, cap_value)
        """
        values = df[column].dropna()
        
        cap_value = np.percentile(values, percentile)
        outlier_mask = df[column] > cap_value
        
        outlier_count = outlier_mask.sum()
        
        logger.info(f"{column}: {outlier_count} outliers above {percentile}th percentile ({cap_value:.2f})")
        
        return outlier_mask, cap_value


def run_full_validation(datasets: dict) -> dict:
    """
    Run complete validation suite.
    
    Args:
        datasets: Dictionary of DataFrames
    
    Returns:
        Dictionary of validation results
    """
    validator = DataValidator()
    results = {}
    
    attack_df = datasets['attack_dataset']
    
    # 1. Hex validation
    hex_columns = ['plaintext_hex', 'ciphertext_hex', 'key_hex', 'recovered_data_hex']
    hex_validation = validator.validate_hex_columns(attack_df, hex_columns)
    results['hex_validation'] = hex_validation
    
    # 2. Length consistency
    length_check = validator.check_length_consistency(attack_df)
    results['length_check'] = length_check
    
    # 3. Outlier detection
    numerical_cols = ['execution_time_ms', 'memory_used_mb', 'iterations_performed']
    outlier_results = []
    
    for col in numerical_cols:
        if col in attack_df.columns:
            outlier_mask, cap_val = validator.detect_outliers_iqr(attack_df, col)
            outlier_results.append({
                'column': col,
                'outliers': outlier_mask.sum(),
                'cap_value': cap_val
            })
    
    results['outliers'] = pd.DataFrame(outlier_results)
    
    return results


if __name__ == "__main__":
    from src.data.loader import load_and_validate
    
    datasets = load_and_validate()
    results = run_full_validation(datasets)
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    print("\nHex Validation:")
    print(results['hex_validation'])
    
    print("\nLength Consistency:")
    print(results['length_check'])
    
    print("\nOutlier Detection:")
    print(results['outliers'])
```

**Run Validation:**
```bash
python -m src.data.validators
```

---

## WEEK 2: DATA CLEANING & FEATURE ENGINEERING

*Continued in next section due to length...*

---

## SUCCESS CRITERIA FOR PHASE 1

**Data Loading:**
- ✅ All 4 CSVs loaded successfully
- ✅ Schema validation passes
- ✅ Foreign key integrity checked

**Data Quality:**
- ✅ Missing values < 10% per column
- ✅ Hex strings validated (> 95% valid)
- ✅ Length consistency checked
- ✅ Outliers detected and documented

**Feature Engineering:**
- ✅ 85+ features created
- ✅ Categorical encoding applied
- ✅ Temporal features extracted
- ✅ Interaction terms computed

**Data Splits:**
- ✅ Train/val/test: 60/20/20
- ✅ Class balance maintained
- ✅ No data leakage
- ✅ Reproducible (random seed fixed)

**Documentation:**
- ✅ EDA report generated
- ✅ Feature metadata JSON created
- ✅ All transformations logged
- ✅ Test coverage ≥ 80%

---

**Next:** Continue to Week 2 implementation (cleaning, feature engineering, final splits)
