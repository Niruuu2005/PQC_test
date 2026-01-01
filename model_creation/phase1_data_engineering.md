# Phase 1: Data Engineering (Weeks 1-2)
## Quantum-Enhanced Cryptanalysis System

**Duration:** 2 weeks  
**Goal:** Clean, transform, and prepare datasets for ML training

---

## OBJECTIVES

1. Load and merge 4 CSV datasets
2. Perform comprehensive EDA
3. Engineer 65+ features
4. Create train/val/test splits (60/20/20)
5. Handle class imbalance and missing values

---

## INPUT SPECIFICATIONS

**Source Location:** `d:\Dream\AIRAWAT\dataset_generation\`

### Dataset 1: attack_dataset.csv (~50,000 rows)
```
Columns:
- attack_execution_id (primary key)
- encryption_row_id (foreign key → crypto_dataset)
- algorithm_name, attack_id, attack_name, attack_category
- execution_time_ms, memory_used_mb, cpu_usage_percent
- iterations_performed, attack_success, confidence_score
- key_hex, plaintext_hex, ciphertext_hex
- recovered_data_hex, error_message
- metric_1_name, metric_1_value ... metric_10_name, metric_10_value
- vulnerability_detected, vulnerability_type, severity_score
```

### Dataset 2: attack_metadata.csv
```
Columns:
- attack_id (primary key)
- attack_name, category, subcategory
- primary_language, complexity_time, complexity_space
- applicable_to, success_criteria, reference_paper
```

### Dataset 3: crypto_dataset.csv
```
Columns:
- row_id (primary key)
- algorithm_name, key_size_bits
- plaintext_hex, ciphertext_hex
- shannon_entropy, chi_square_statistic, avalanche_effect
- encryption_time_ms, encryption_successful
```

### Dataset 4: Cryptographic_Algorithm_Summary.csv
```
Columns:
- Algorithm, Key_Size_Bits, Security_Level
- Total_Attacks_Tested, Successful_Attacks
- Vulnerable_Percent, Resistance_Score
```

---

## OUTPUT SPECIFICATIONS

**Target Location:** `d:\Dream\AIRAWAT\model_creation\data\processed\`

### Processed Datasets
```
X_train.csv - 30,000 rows × 65 features
y_train.csv - 30,000 labels for 5 tasks
X_val.csv - 10,000 rows × 65 features
y_val.csv - 10,000 labels
X_test.csv - 10,000 rows × 65 features  
y_test.csv - 10,000 labels
```

### Feature Metadata
```json
feature_metadata.json:
{
  "features": [
    {
      "name": "execution_time_ms",
      "type": "numeric",
      "source": "attack_dataset",
      "dtype": "float32", 
      "min": 0.1, "max": 5000.0,
      "scale_method": "none"
    },
    ...
  ],
  "total_features": 65,
  "total_samples": 50000,
  "removed_features": ["redundant_col1", ...]
}
```

### Reports
```
reports/data_exploration_report.pdf:
- Dataset dimensions
- Missing value analysis
- Class distribution
- Feature correlations
- Outlier handling decisions
```

---

## WEEK 1: EXPLORATION & ASSESSMENT

### Day 1-2: Data Loading & Validation

**Step 1.1: Load Datasets**
```python
import pandas as pd

# Load all 4 datasets
attack_df = pd.read_csv('../dataset_generation/attack_dataset.csv')
metadata_df = pd.read_csv('../dataset_generation/attack_metadata.csv')
crypto_df = pd.read_csv('../dataset_generation/crypto_dataset.csv')
summary_df = pd.read_csv('../dataset_generation/Cryptographic_Algorithm_Summary.csv')

# Verify row counts
print(f"Attack dataset: {len(attack_df)} rows")
print(f"Crypto dataset: {len(crypto_df)} rows")
```

**Step 1.2: Data Type Validation**
```python
# Check data types
attack_df.dtypes
attack_df.info()

# Verify hex strings
def validate_hex(hex_str):
    try:
        bytes.fromhex(hex_str)
        return len(hex_str) % 2 == 0
    except:
        return False

attack_df['valid_plaintext'] = attack_df['plaintext_hex'].apply(validate_hex)
attack_df['valid_ciphertext'] = attack_df['ciphertext_hex'].apply(validate_hex)
```

**Step 1.3: Missing Value Analysis**
```python
# Missing value percentages
missing = attack_df.isnull().sum() /  len(attack_df) * 100
print(missing[missing > 0].sort_values(ascending=False))

# Expected missing columns:
# - recovered_data_hex: ~50-80% (only for successful attacks)
# - error_message: ~70-90% (only for failed attacks)
# - metric_*_value: variable (depends on attack type)
```

**Step 1.4: Duplicate Detection**
```python
# Check for duplicates
duplicates = attack_df.duplicated(subset=['attack_execution_id', 'encryption_row_id', 'run_number'])
print(f"Duplicates found: {duplicates.sum()}")

# Remove exact duplicates
attack_df = attack_df.drop_duplicates()
```

**Deliverable:** `notebooks/01_data_loading_validation.ipynb`

---

### Day 3-4: Exploratory Data Analysis (EDA)

**Step 2.1: Target Distribution Analysis**

```python
# Task 1: Attack Classification
attack_df['attack_category'].value_counts().plot(kind='bar')
plt.title('Attack Category Distribution')

# Task 2: Attack Success
attack_df['attack_success'].value_counts()
# Calculate imbalance ratio
ratio = attack_df['attack_success'].value_counts().min() / attack_df['attack_success'].value_counts().max()
print(f"Class imbalance ratio: {ratio:.2f}")
```

**Step 2.2: Feature Correlation Matrix**
```python
# Select numeric features
numeric_features = attack_df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = attack_df[numeric_features].corr()

# Plot heatmap
import seaborn as sns
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
```

**Step 2.3: Statistical Summary**
```python
attack_df.describe()

# Key insights to document:
# - Execution time range: [min, max, median]
# - Memory usage patterns
# - Iteration counts
```

**Deliverable:** `reports/data_exploration_report.pdf` (10-15 pages)
- Executive summary (1 page)
- Dataset statistics (2 pages)
- Missing value analysis (1 page)
- Target distribution (2 pages)
- Feature correlations (2 pages)
- Outlier analysis (2 pages)

---

## WEEK 2: CLEANING & FEATURE ENGINEERING

### Day 5-6: Data Cleaning Pipeline

**Step 3.1: Handle Missing Values**

```python
from sklearn.impute import SimpleImputer

# Strategy by column type:
FILL_STRATEGIES = {
    # Numeric: median imputation
    'execution_time_ms': 'median',
    'memory_used_mb': 'median',
    'metric_1_value': 0,  # or median based on distribution
    
    # Strings: categorical fill
    'recovered_data_hex': '',
    'error_message': 'no_error',
    'notes': 'no_notes',
}

for col, strategy in FILL_STRATEGIES.items():
    if strategy == 'median':
        attack_df[col].fillna(attack_df[col].median(), inplace=True)
    else:
        attack_df[col].fillna(strategy, inplace=True)

# Create missing indicator features
attack_df['missing_recovered_data'] = attack_df['recovered_data_hex'].isna().astype(int)
```

**Step 3.2: Outlier Handling**
```python
def cap_outliers(df, column, percentile=99):
    upper_bound = df[column].quantile(percentile / 100)
    df[column] = df[column].clip(upper=upper_bound)
    return df

# Cap execution metrics at 99th percentile
attack_df = cap_outliers(attack_df, 'execution_time_ms', 99)
attack_df = cap_outliers(attack_df, 'memory_used_mb', 99)
attack_df = cap_outliers(attack_df, 'iterations_performed', 99)
```

**Deliverable:** `src/data/cleaner.py` (reusable module)

---

### Day 7-8: Feature Engineering

**Step 4.1: Statistical Features** (from crypto_dataset)
```python
# Merge crypto statistics
merged_df = attack_df.merge(
    crypto_df[['row_id', 'shannon_entropy', 'chi_square_statistic', 'avalanche_effect']],
    left_on='encryption_row_id',
    right_on='row_id',
    how='left'
)

# Normalize to [0, 1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
merged_df[['shannon_entropy', 'chi_square_statistic', 'avalanche_effect']] = scaler.fit_transform(
    merged_df[['shannon_entropy', 'chi_square_statistic', 'avalanche_effect']]
)
```

**Step 4.2: Hex-based Features**
```python
def extract_hex_features(hex_str):
    if pd.isna(hex_str) or hex_str == '':
        return {
            'byte_length': 0,
            'unique_byte_count': 0,
            'byte_entropy': 0,
            'top_byte_freq': 0
        }
    
    bytes_data = bytes.fromhex(hex_str)
    byte_counts = pd.Series(list(bytes_data)).value_counts()
    
    return {
        'byte_length': len(bytes_data),
        'unique_byte_count': len(byte_counts),
        'byte_entropy': -sum((c/len(bytes_data)) * np.log2(c/len(bytes_data)) for c in byte_counts),
        'top_byte_freq': byte_counts.iloc[0] / len(bytes_data)
    }

# Apply to plaintext and ciphertext
plaintext_features = merged_df['plaintext_hex'].apply(extract_hex_features).apply(pd.Series)
plaintext_features.columns = ['pt_' + col for col in plaintext_features.columns]

ciphertext_features = merged_df['ciphertext_hex'].apply(extract_hex_features).apply(pd.Series)
ciphertext_features.columns = ['ct_' + col for col in ciphertext_features.columns]

merged_df = pd.concat([merged_df, plaintext_features, ciphertext_features], axis=1)
```

**Step 4.3: Temporal Features**
```python
merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
merged_df['hour'] = merged_df['timestamp'].dt.hour
merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
merged_df['is_weekend'] = (merged_df['day_of_week'] >= 5).astype(int)

# Cyclical encoding
merged_df['hour_sin'] = np.sin(2 * np.pi * merged_df['hour'] / 24)
merged_df['hour_cos'] = np.cos(2 * np.pi * merged_df['hour'] / 24)
```

**Step 4.4: Categorical Encoding**
```python
# One-hot encoding
algorithm_dummies = pd.get_dummies(merged_df['algorithm_name'], prefix='algo')
attack_category_dummies = pd.get_dummies(merged_df['attack_category'], prefix='attack')

merged_df = pd.concat([merged_df, algorithm_dummies, attack_category_dummies], axis=1)

# Label encoding for ordinal features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
merged_df['key_size_encoded'] = le.fit_transform(merged_df['key_size_bits'].astype(str))
```

**Step 4.5: Derived Features**
```python
# Efficiency metrics
merged_df['time_per_iteration'] = merged_df['execution_time_ms'] / (merged_df['iterations_performed'] + 1)
merged_df['memory_per_iteration'] = merged_df['memory_used_mb'] / (merged_df['iterations_performed'] + 1)
merged_df['iterations_per_second'] = merged_df['iterations_performed'] / (merged_df['execution_time_ms'] / 1000 + 1e-6)

# Interaction features
merged_df['entropy_chi_sq'] = merged_df['shannon_entropy'] * merged_df['chi_square_statistic']
merged_df['avalanche_entropy'] = merged_df['avalanche_effect'] * merged_df['shannon_entropy']
```

**Deliverable:** `src/data/feature_engineer.py` (~300 lines)

---

### Day 9-10: Train/Val/Test Split

**Step 5.1: Feature Selection**
```python
# Remove highly correlated features (>0.95)
corr_matrix = merged_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
merged_df = merged_df.drop(columns=to_drop)

# Remove zero-variance features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0)
selector.fit(merged_df.select_dtypes(include=['float64', 'int64']))
```

**Step 5.2: Time-based Split**
```python
# Sort by timestamp
merged_df = merged_df.sort_values('timestamp')

# Split indices
n = len(merged_df)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

train_df = merged_df.iloc[:train_end]
val_df = merged_df.iloc[train_end:val_end]
test_df = merged_df.iloc[val_end:]

print(f"Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
print(f"Val: {len(val_df)} ({len(val_df)/n*100:.1f}%)")
print(f"Test: {len(test_df)} ({len(test_df)/n*100:.1f}%)")
```

**Step 5.3: Separate Features and Targets**
```python
# Define feature columns (exclude identifiers and targets)
feature_cols = [col for col in merged_df.columns if col not in [
    'attack_execution_id', 'encryption_row_id', 'timestamp',
    'attack_name', 'attack_category', 'attack_success',  # targets
    'algorithm_name',  # target
    'plaintext_hex', 'ciphertext_hex', 'recovered_data_hex'  # raw data
]]

# Task-specific targets
targets = {
    'task1_attack_classification': 'attack_category',
    'task2_attack_success': 'attack_success',
    'task3_algorithm_id': 'algorithm_name',
    'task4_plaintext_recovery': 'recovered_data_hex',  # special handling
    'task5_key_properties': ['key_size_bits', 'vulnerability_detected']  # multi-target
}

# Save splits
train_df[feature_cols].to_csv('data/processed/X_train.csv', index=False)
train_df[list(targets.values())].to_csv('data/processed/y_train.csv', index=False)
# Repeat for val and test
```

**Deliverable:** `src/data/splitter.py`

---

## WEEK-END DELIVERABLES

### Code Artifacts
- `src/data/loader.py` - Dataset loading utilities
- `src/data/cleaner.py` - Missing value/outlier handling
- `src/data/feature_engineer.py` - 65+ feature engineering
- `src/data/splitter.py` - Train/val/test split logic

### Data Artifacts
- `data/processed/X_train.csv` (30,000 × 65)
- `data/processed/y_train.csv`
- `data/processed/X_val.csv` (10,000 × 65)
- `data/processed/y_val.csv`
- `data/processed/X_test.csv` (10,000 × 65)
- `data/processed/y_test.csv`
- `data/feature_metadata.json`

### Reports
- `reports/data_exploration_report.pdf`
- `reports/feature_engineering_decisions.md`
- `reports/class_imbalance_analysis.md`

---

## SUCCESS CRITERIA

✓ All 4 datasets loaded without errors  
✓ <2% missing values in critical columns  
✓ 65 engineered features created  
✓ No data leakage (split before scaling)  
✓ Class distribution maintained across splits  
✓ Feature correlation matrix documented  
✓ EDA report reviewed and approved

---

**Next Phase:** [Phase 2: Classical ML Baseline](./phase2_classical_baseline.md)
