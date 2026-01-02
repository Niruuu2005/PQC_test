# Quantum-Enhanced Cryptanalysis System
## Phase-Wise Implementation Plan

**Project:** QML Cryptanalysis Pipeline  
**Workspace:** `model_creation`  
**Timeline:** 12 Weeks (6 Phases)  
**Last Updated:** January 2, 2026

---

## EXECUTIVE SUMMARY

This plan implements a **production-grade hybrid quantum-classical machine learning system** for cryptanalysis. The system analyzes attack datasets, identifies encryption algorithms, predicts attack success, and optimizes attack parameters using:

**Classical Components:**
- Ensemble learning (XGBoost, Random Forest, LightGBM, Neural Networks, SVM, CatBoost)
- Stacking meta-learners
- Advanced feature engineering

**Quantum Components:**
- Variational Quantum Circuits (VQC) via TensorFlow Quantum
- QAOA for combinatorial optimization
- Quantum feature extraction

**5 Core Tasks:**
1. **Attack Classification** → Predict attack type from execution metrics
2. **Attack Success Prediction** → Binary prediction of attack success
3. **Algorithm Identification** → Identify encryption algorithm from ciphertext statistics
4. **Plaintext Recovery** → Recover plaintext using QAOA optimization
5. **Key Property Prediction** → Predict key entropy and weakness probability

---

## PHASE OVERVIEW

| Phase | Duration | Focus | Key Deliverables |
|-------|----------|-------|------------------|
| **Phase 1** | Weeks 1-2 | Data Engineering | Clean datasets, engineered features, train/val/test splits |
| **Phase 2** | Weeks 3-4 | Classical ML Baseline | Base learners, stacking ensemble, performance baseline |
| **Phase 3** | Weeks 5-6 | Quantum Circuit Design | Parameterized circuits, QAOA, gradient verification |
| **Phase 4** | Weeks 7-8 | Hybrid Integration | Keras-TFQ models, end-to-end training pipeline |
| **Phase 5** | Weeks 9-10 | Full-Scale Training | Train all 5 tasks, validation, comparison |
| **Phase 6** | Weeks 11-12 | Evaluation & Deployment | Test metrics, documentation, deployment guide |

---

# PHASE 1: DATA ENGINEERING
**Duration:** Weeks 1-2  
**Goal:** Transform raw datasets into ML-ready features with proper splits

## Week 1: Data Exploration & Assessment

### 1.1 Dataset Loading
**Input:** 4 CSV files
- `attack_dataset.csv` (~50,000 rows) - PRIMARY
- `attack_metadata.csv` - Reference for attack definitions
- `crypto_dataset.csv` - Encryption results with statistical properties
- `cryptographic_algorithm_summary.csv` - Algorithm vulnerability summary

**Tasks:**
- [ ] Load all 4 datasets using pandas
- [ ] Verify schema (column names, data types)
- [ ] Check row counts and basic statistics
- [ ] Identify foreign key relationships:
  - `attack_dataset.attack_id` → `attack_metadata.attack_id`
  - `attack_dataset.encryption_row_id` → `crypto_dataset.row_id`
  - `attack_dataset.algorithm_name` → `cryptographic_algorithm_summary.Algorithm`

**Validation Checks:**
```python
# Check for foreign key integrity
assert attack_dataset['attack_id'].isin(attack_metadata['attack_id']).all()
assert attack_dataset['encryption_row_id'].isin(crypto_dataset['row_id']).all()
```

**Output:** `data_loading_report.txt`

### 1.2 Exploratory Data Analysis (EDA)

**Statistical Analysis:**
- [ ] Distribution of target variables:
  - `attack_success` → Calculate success rate (%)
  - `attack_category` → Count per category
  - `algorithm_name` → Most/least tested algorithms
- [ ] Numerical feature ranges (min, max, mean, std):
  - `execution_time_ms`
  - `memory_used_mb`
  - `iterations_performed`
  - `confidence_score`
  - `shannon_entropy`
  - `chi_square_statistic`
  - `avalanche_effect`

**Missing Value Analysis:**
- [ ] Calculate % missing per column
- [ ] Identify patterns (MCAR, MAR, MNAR)
- [ ] Visualize missing data heatmap

**Correlation Analysis:**
- [ ] Correlation matrix (Pearson) for numerical features
- [ ] Identify highly correlated pairs (|r| > 0.95)
- [ ] Feature-target correlations

**Temporal Analysis (if timestamp available):**
- [ ] Parse timestamps to datetime
- [ ] Plot attack frequency over time
- [ ] Check for temporal trends

**Duplicate Detection:**
- [ ] Check for exact duplicates (all columns)
- [ ] Check for logical duplicates:
  - Same `(attack_execution_id, encryption_row_id, run_number)`
- [ ] Decision: Keep first occurrence or aggregate

**Output:**
- `EDA_report.pdf` with:
  - Dataset dimensions table
  - Distribution plots (histograms, box plots)
  - Correlation heatmap
  - Missing value heatmap
  - Key findings and data quality issues

### 1.3 Data Quality Assessment

**Hex String Validation:**
- [ ] Validate `plaintext_hex`, `ciphertext_hex`, `key_hex`, `recovered_data_hex`
- [ ] Check: only characters [0-9, a-f, A-F]
- [ ] Check: even length (2 chars per byte)
- [ ] Flag invalid entries

**Outlier Detection:**
- [ ] Use IQR method for:
  - `execution_time_ms` (cap at 99th percentile)
  - `memory_used_mb` (cap at 99th percentile)
  - `iterations_performed` (cap at 99th percentile)
- [ ] Document outlier handling strategy

**Consistency Checks:**
- [ ] `plaintext_length` == len(plaintext_hex) / 2
- [ ] `ciphertext_length` == len(ciphertext_hex) / 2
- [ ] `key_size_bits` matches algorithm specifications

**Output:** `data_quality_issues.csv` (list of identified problems)

---

## Week 2: Data Cleaning & Feature Engineering

### 2.1 Data Cleaning Pipeline

**Missing Value Imputation:**
```python
# Strategy by column type
IMPUTATION_RULES = {
    'metric_*_value': 'median',  # Numerical metrics
    'error_message': '"no_error"',  # String
    'recovered_data_hex': '""',  # Empty string
    'notes': '"no_notes"',  # String
}
```

**Tasks:**
- [ ] Implement imputation for each column type
- [ ] Create binary flags for imputed values (e.g., `metric_1_was_missing`)
- [ ] Document imputation decisions

**Outlier Handling:**
- [ ] Cap `execution_time_ms` at 99th percentile
- [ ] Cap `memory_used_mb` at 99th percentile
- [ ] Cap `iterations_performed` at 99th percentile
- [ ] Log: Number of capped values per column

**Data Type Conversions:**
- [ ] Timestamps → datetime objects
- [ ] Categorical strings → category dtype (memory optimization)
- [ ] Numerical IDs → integer types

**Duplicate Removal:**
- [ ] Remove exact duplicates
- [ ] Keep first occurrence based on timestamp

**Output:** `cleaned_data.csv`

### 2.2 Feature Engineering

#### Category A: Statistical Features
**Source:** `crypto_dataset` merged via `encryption_row_id`

**Extract:**
- [ ] `shannon_entropy` (information content: 0-8 for random data)
- [ ] `chi_square_statistic` (randomness test)
- [ ] `avalanche_effect` (sensitivity to input changes: ideal ~0.5)

**Normalize:** Min-Max scaling to [0, 1] for interpretability

#### Category B: Hex-Based Features

**Plaintext Features:**
```python
def extract_hex_features(hex_string):
    bytes_data = bytes.fromhex(hex_string)
    return {
        'byte_length': len(bytes_data),
        'unique_byte_count': len(set(bytes_data)),
        'entropy': calculate_entropy(bytes_data),
        'byte_frequency_top5': get_top_5_frequencies(bytes_data),
        'entropy_rate': calculate_entropy(bytes_data) / len(bytes_data)
    }
```

**Tasks:**
- [ ] `plaintext_byte_length`
- [ ] `plaintext_unique_bytes`
- [ ] `plaintext_entropy`
- [ ] `ciphertext_byte_length`
- [ ] `ciphertext_unique_bytes` (expect ~256 for good ciphers)
- [ ] `ciphertext_entropy`
- [ ] `length_ratio` = ciphertext_length / plaintext_length

#### Category C: Temporal Features

**Extract from timestamp:**
```python
# Cyclical encoding (preserves periodicity)
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
day_sin = np.sin(2 * np.pi * day_of_week / 7)
day_cos = np.cos(2 * np.pi * day_of_week / 7)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
```

**Tasks:**
- [ ] `hour_of_day_sin`, `hour_of_day_cos`
- [ ] `day_of_week_sin`, `day_of_week_cos`
- [ ] `month_sin`, `month_cos`
- [ ] `is_weekend` (binary: 0/1)
- [ ] `days_since_epoch` (trend indicator)

#### Category D: Derived Features

**Efficiency Metrics:**
```python
time_per_iteration = execution_time_ms / max(iterations_performed, 1)
memory_per_iteration = memory_used_mb / max(iterations_performed, 1)
iterations_per_second = iterations_performed / (execution_time_ms / 1000)
```

**Interaction Features:**
```python
entropy_chi_interaction = shannon_entropy * chi_square_statistic
avalanche_entropy_interaction = avalanche_effect * shannon_entropy
```

**Performance Score:**
```python
efficiency_score = (1 - memory_used_mb/max_memory) * (1 - execution_time_ms/max_time)
```

**Tasks:**
- [ ] `time_per_iteration`
- [ ] `memory_per_iteration`
- [ ] `iterations_per_second`
- [ ] `efficiency_score`
- [ ] `entropy_chi_interaction`
- [ ] `avalanche_entropy_interaction`
- [ ] `key_entropy_ratio` = key_size_bits / (8 * plaintext_length)

#### Category E: Categorical Encoding

**One-Hot Encoding (for non-ordinal):**
- [ ] `algorithm_name` → [is_AES, is_RSA, is_Kyber, is_ChaCha20, ...]
- [ ] `attack_category` → [is_SideChannel, is_BruteForce, is_Differential, ...]
- [ ] `attack_language` → [is_Python, is_C, is_Java, ...]

**Label Encoding (for ordinal):**
- [ ] `key_size_bits` → {128→1, 192→2, 256→3, 512→4, ...}
- [ ] `vulnerability_type` → Rank by severity
- [ ] `security_level` → {Low→1, Medium→2, High→3, Critical→4}

**Implementation:**
```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# One-hot for algorithm_name
encoder_algo = OneHotEncoder(sparse=False, handle_unknown='ignore')
algo_encoded = encoder_algo.fit_transform(attack_dataset[['algorithm_name']])

# Label encoding for key_size
encoder_key = LabelEncoder()
key_encoded = encoder_key.fit_transform(attack_dataset['key_size_bits'])
```

**Save encoders:** `encoders/one_hot_algorithm.pkl`, `encoders/label_key_size.pkl`

#### Category F: Vulnerability Features
- [ ] `vulnerability_detected` (binary: 0/1)
- [ ] `severity_score` (already [0, 1])
- [ ] `high_severity_count` (from summary table)

### 2.3 Feature Selection

**Correlation-Based Removal:**
- [ ] Compute correlation matrix
- [ ] Remove features with correlation > 0.95 with another feature
- [ ] Keep feature with higher correlation to target

**Variance-Based Removal:**
- [ ] Remove zero-variance features
- [ ] Remove near-zero variance (< 0.01)

**Mutual Information:**
```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X_train, y_train)
# Keep top 80% of features by MI score
```

**Output:**
- `feature_metadata.json`:
```json
{
  "total_features": 85,
  "removed_features": ["feature_x", "feature_y"],
  "feature_list": [
    {
      "name": "execution_time_ms",
      "type": "numeric",
      "source": "attack_dataset",
      "dtype": "float32",
      "min": 0.1,
      "max": 5000.0,
      "mean": 250.5,
      "std": 120.3
    }
  ]
}
```

### 2.4 Train/Val/Test Split

**Strategy:** Temporal split (if timestamps available) or stratified split

**Temporal Split:**
```python
# Sort by timestamp
data_sorted = data.sort_values('timestamp')

# 60% train, 20% val, 20% test
n = len(data_sorted)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

train_data = data_sorted[:train_end]
val_data = data_sorted[train_end:val_end]
test_data = data_sorted[val_end:]
```

**Stratified Split (alternative):**
```python
from sklearn.model_selection import train_test_split

# Stratify by attack_category + algorithm_name
stratify_column = attack_dataset['attack_category'] + "_" + attack_dataset['algorithm_name']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=stratify_column, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
```

**Validation:**
- [ ] Check class balance across splits
- [ ] Verify no data leakage (no overlapping indices)
- [ ] Save splits: `X_train.csv`, `y_train.csv`, `X_val.csv`, `y_val.csv`, `X_test.csv`, `y_test.csv`

**Output:** `train_val_test_split_report.md`

### Phase 1 Deliverables

**Code:**
- `src/data/loader.py` - Load CSVs with validation
- `src/data/cleaner.py` - Handle missing values, outliers
- `src/data/feature_engineer.py` - All feature extraction functions
- `src/data/splitter.py` - Train/val/test split logic

**Data:**
- `data/processed/X_train.csv` (N_train × N_features)
- `data/processed/y_train.csv` (N_train × 1 per task)
- `data/processed/X_val.csv`
- `data/processed/y_val.csv`
- `data/processed/X_test.csv`
- `data/processed/y_test.csv`

**Documentation:**
- `reports/EDA_report.pdf`
- `reports/feature_metadata.json`
- `reports/train_val_test_split_report.md`

**Tests:**
- `tests/test_loader.py`
- `tests/test_feature_engineer.py`

**Success Criteria:**
✅ All 4 datasets loaded successfully  
✅ Missing values < 5% after imputation  
✅ 85+ engineered features created  
✅ Train/val/test split: 60/20/20 with balanced classes  
✅ Data pipeline runs end-to-end without errors  

---

# PHASE 2: CLASSICAL ML BASELINE
**Duration:** Weeks 3-4  
**Goal:** Establish strong classical baseline for all 5 tasks using ensemble methods

## Week 3: Base Learner Implementation

### 3.1 Task 1: Attack Classification (Multi-Class)

**Problem Setup:**
- **Input:** Execution metrics + algorithm features + derived features
- **Target:** `attack_category` (e.g., 5-10 classes)
- **Evaluation Metric:** Macro F1, Accuracy, Weighted F1

**Check Class Imbalance:**
```python
class_counts = y_train['attack_category'].value_counts()
imbalance_ratio = class_counts.min() / class_counts.max()
print(f"Imbalance ratio: {imbalance_ratio:.2f}")

if imbalance_ratio < 0.3:
    print("High imbalance detected - Apply SMOTE")
```

#### Base Learner 1: XGBoost
```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    max_depth=7,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=True
)
```

**Tasks:**
- [ ] Train XGBoost
- [ ] Log training/validation loss per epoch
- [ ] Save model: `models/classical/xgboost_task1.pkl`
- [ ] Get predictions: `xgb_preds = xgb_model.predict_proba(X_val)`

#### Base Learner 2: Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Feature importance
feature_importance = rf_model.feature_importances_
```

**Tasks:**
- [ ] Train Random Forest
- [ ] Extract and save feature importances
- [ ] OOB score (if oob_score=True)
- [ ] Save model: `models/classical/random_forest_task1.pkl`

#### Base Learner 3: LightGBM
```python
from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1,
    num_leaves=63,
    max_depth=7,
    colsample_bytree=0.8,
    random_state=42
)

lgbm_model.fit(X_train, y_train)
```

#### Base Learner 4: CatBoost
```python
from catboost import CatBoostClassifier

catboost_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    verbose=False,
    random_state=42
)

catboost_model.fit(X_train, y_train)
```

#### Base Learner 5: SVM
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# SVM requires scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,  # Enable predict_proba
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)
```

#### Base Learner 6: Neural Network
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

nn_model = Sequential([
    Dense(256, activation='relu', input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])

nn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = nn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)
```

#### Base Learner 7: Logistic Regression (Baseline)
```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    max_iter=1000,
    penalty='l2',
    C=1.0,
    multi_class='multinomial',
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)
```

### 3.2 Evaluation of Base Learners

**Compute Metrics:**
```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

models = {
    'XGBoost': xgb_model,
    'RandomForest': rf_model,
    'LightGBM': lgbm_model,
    'CatBoost': catboost_model,
    'SVM': svm_model,
    'NeuralNet': nn_model,
    'LogisticRegression': lr_model
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    weighted_f1 = f1_score(y_val, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Macro F1': macro_f1,
        'Weighted F1': weighted_f1
    })

results_df = pd.DataFrame(results)
print(results_df)
```

**Output:** `reports/task1_base_learners_performance.csv`

### 3.3 Repeat for Other Tasks

#### Task 2: Attack Success Prediction (Binary Classification)

**Modifications:**
- Target: `attack_success` (0/1)
- Check imbalance: Apply SMOTE if ratio < 0.3
- XGBoost: `scale_pos_weight = majority_count / minority_count`
- Evaluation: ROC-AUC, F1, Precision @ 90% Recall

**Implementation:**
```python
from imblearn.over_sampling import SMOTE

if imbalance_ratio < 0.3:
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
else:
    X_train_resampled, y_train_resampled = X_train, y_train
```

#### Task 3: Algorithm Identification (Multi-Class)

**Features:** Statistical properties (entropy, chi_square, avalanche) + frequency distribution  
**Target:** `algorithm_name`  
**Evaluation:** Top-1 accuracy, Top-2 accuracy

#### Tasks 4 & 5: Simplified Baselines

**Task 4 (Plaintext Recovery):** Train simple XGBoost regression to predict Hamming distance (quantum will enhance this)

**Task 5 (Key Properties):** Multi-task learning:
```python
# Two outputs: key_entropy (regression), weak_key (classification)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

inputs = Input(shape=(n_features,))
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)

# Output branches
entropy_output = Dense(1, name='entropy')(x)  # Regression
weak_key_output = Dense(1, activation='sigmoid', name='weak_key')(x)  # Binary classification

model = Model(inputs=inputs, outputs=[entropy_output, weak_key_output])

model.compile(
    optimizer='adam',
    loss={'entropy': 'mse', 'weak_key': 'binary_crossentropy'},
    loss_weights={'entropy': 0.5, 'weak_key': 0.5},
    metrics={'entropy': 'mae', 'weak_key': 'accuracy'}
)
```

---

## Week 4: Ensemble & Hyperparameter Tuning

### 4.1 Hyperparameter Optimization (Optuna)

**Task 1 Example:**
```python
import optuna

def objective(trial):
    # XGBoost hyperparameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
    }
    
    model = XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    
    return f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
print(f"Best F1: {study.best_value}")
print(f"Best params: {best_params}")
```

**Tasks:**
- [ ] Optimize XGBoost for all 5 tasks
- [ ] Optimize Random Forest
- [ ] Optimize Neural Network
- [ ] Save best hyperparameters: `configs/best_params_task{i}.json`

### 4.2 Stacking Ensemble

**Level 0: Base Learner Predictions**
```python
# Collect probability predictions from all base learners
base_predictions_train = []
base_predictions_val = []

for name, model in models.items():
    train_preds = model.predict_proba(X_train)
    val_preds = model.predict_proba(X_val)
    
    base_predictions_train.append(train_preds)
    base_predictions_val.append(val_preds)

# Stack predictions
X_meta_train = np.hstack(base_predictions_train)  # (n_train, 7 * n_classes)
X_meta_val = np.hstack(base_predictions_val)  # (n_val, 7 * n_classes)
```

**Level 1: Meta-Learner**
```python
# Train meta-learner on base predictions
meta_learner = XGBClassifier(
    max_depth=3,  # Shallow to avoid overfitting
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

meta_learner.fit(X_meta_train, y_train)

# Final predictions
y_stacked_pred = meta_learner.predict(X_meta_val)
```

**Evaluation:**
```python
stacked_accuracy = accuracy_score(y_val, y_stacked_pred)
stacked_f1 = f1_score(y_val, y_stacked_pred, average='macro')

print(f"Stacked Ensemble - Accuracy: {stacked_accuracy:.4f}, Macro F1: {stacked_f1:.4f}")
```

### 4.3 Cross-Validation

**Stratified K-Fold:**
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    xgb_model, X_train, y_train,
    cv=skf,
    scoring='f1_macro',
    n_jobs=-1
)

print(f"CV Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

### Phase 2 Deliverables

**Code:**
- `src/classical/base_learners.py` - All 7 base learner classes
- `src/classical/ensemble.py` - Stacking implementation
- `src/classical/hyperopt.py` - Optuna optimization
- `src/classical/evaluator.py` - Metrics computation

**Models:**
- `models/classical/xgboost_task{1-5}.pkl`
- `models/classical/random_forest_task{1-5}.pkl`
- `models/classical/stacked_ensemble_task{1-5}.pkl`
- ... (all base learners for all tasks)

**Reports:**
- `reports/classical_baseline_performance.pdf`:
  - Performance comparison table (all models, all tasks)
  - Learning curves (train/val loss)
  - Confusion matrices
  - Feature importance plots
  - ROC-AUC curves

**Success Criteria:**
✅ Task 1: Accuracy ≥ 85%, Macro F1 ≥ 0.84  
✅ Task 2: ROC-AUC ≥ 0.80, F1 ≥ 0.75  
✅ Task 3: Top-1 Accuracy ≥ 83%  
✅ Stacking improves over best base learner by ≥1%  
✅ All models reproducible (saved successfully)  

---

# PHASE 3: QUANTUM CIRCUIT DESIGN
**Duration:** Weeks 5-6  
**Goal:** Design and validate parameterized quantum circuits for hybrid models

## Week 5: Ansatz Design & Implementation

### 5.1 Setup TensorFlow Quantum

**Installation:**
```bash
pip install tensorflow-quantum cirq matplotlib
```

**Verify:**
```python
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
import sympy

print(f"Cirq version: {cirq.__version__}")
print(f"TFQ version: {tfq.__version__}")
```

### 5.2 Task 1: Attack Classification Circuit

**Design Decisions:**
- **n_qubits:** 8 (balance expressivity vs hardware constraints)
- **n_layers:** 4 (avoid barren plateaus)
- **Entanglement:** Ring topology (hardware-efficient)
- **Data encoding:** Angle encoding (Ry rotations)

**Implementation:**
```python
import cirq
import sympy
import tensorflow_quantum as tfq

def create_attack_classification_circuit(n_qubits=8, n_layers=4):
    """
    Create variational quantum circuit for attack classification.
    
    Returns:
        circuit: Cirq Circuit
        data_symbols: List of sympy symbols for data encoding
        param_symbols: List of sympy symbols for trainable parameters
    """
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # Data encoding symbols
    data_symbols = [sympy.Symbol(f'x{i}') for i in range(n_qubits)]
    param_symbols = []
    
    # Layer 1: Data encoding (angle encoding)
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.ry(data_symbols[i])(qubit))
    
    # Variational layers
    for layer in range(n_layers):
        # Parameterized single-qubit rotations
        layer_symbols = [sympy.Symbol(f'theta_{layer}_{i}') for i in range(n_qubits)]
        param_symbols.extend(layer_symbols)
        
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.ry(layer_symbols[i])(qubit))
        
        # Entanglement: Ring topology
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
        
        # Close the ring
        circuit.append(cirq.CNOT(qubits[-1], qubits[0]))
    
    return circuit, data_symbols, param_symbols

# Create circuit
circuit, x_syms, theta_syms = create_attack_classification_circuit()

# Visualize
print(circuit)
print(f"Data symbols: {len(x_syms)}")
print(f"Trainable parameters: {len(theta_syms)}")
```

**Tasks:**
- [ ] Implement circuit
- [ ] Verify parameter count: n_layers × n_qubits = 4 × 8 = 32 parameters
- [ ] Visualize circuit diagram (print or SVG)
- [ ] Test 1 forward pass (simulate)

### 5.3 Task 2: Attack Success Prediction Circuit

**Simpler Binary Circuit:**
```python
def create_attack_success_circuit(n_qubits=6, n_layers=3):
    """Binary classification circuit with single measurement."""
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    data_symbols = [sympy.Symbol(f'x{i}') for i in range(n_qubits)]
    param_symbols = []
    
    # Data encoding
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.ry(data_symbols[i])(qubit))
    
    # Variational layers
    for layer in range(n_layers):
        layer_symbols = [sympy.Symbol(f'theta_{layer}_{i}') for i in range(n_qubits)]
        param_symbols.extend(layer_symbols)
        
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.ry(layer_symbols[i])(qubit))
        
        # Entanglement
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
    
    return circuit, data_symbols, param_symbols
```

### 5.4 Task 3: Algorithm Identification Circuit

**Multiple Measurements for Multi-Class:**
```python
def create_algorithm_id_circuit(n_qubits=10, n_layers=4, n_measurements=4):
    """
    Circuit with multiple Pauli measurements for feature extraction.
    """
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # Similar structure but with amplitude encoding option
    # ... (same pattern as above)
    
    return circuit, data_symbols param_symbols, qubits[:n_measurements]
```

### 5.5 Task 4: QAOA Circuit (Plaintext Recovery)

**QAOA for Combinatorial Optimization:**
```python
def create_qaoa_circuit(n_qubits, problem_hamiltonian, p=2):
    """
    QAOA circuit for plaintext recovery.
    
    Args:
        n_qubits: Number of qubits (plaintext length in bits)
        problem_hamiltonian: Cost Hamiltonian encoding plaintext-ciphertext relation
        p: Number of QAOA layers
    
    Returns:
        circuit: Cirq Circuit
        gamma_symbols: Cost Hamiltonian rotation angles (trainable)
        beta_symbols: Mixer Hamiltonian rotation angles (trainable)
    """
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    gamma_symbols = [sympy.Symbol(f'gamma_{i}') for i in range(p)]
    beta_symbols = [sympy.Symbol(f'beta_{i}') for i in range(p)]
    
    # Initialize to superposition
    circuit.append([cirq.H(q) for q in qubits])
    
    for layer in range(p):
        # Cost layer: exp(-i * gamma * H_C)
        # Implement problem_hamiltonian gates
        # Example: ZZ interactions
        for i in range(n_qubits - 1):
            circuit.append(cirq.ZZ(qubits[i], qubits[i+1])**(gamma_symbols[layer]))
        
        # Mixer layer: exp(-i * beta * H_M) where H_M = sum(X_i)
        for q in qubits:
            circuit.append(cirq.rx(2 * beta_symbols[layer])(q))
    
    return circuit, gamma_symbols, beta_symbols
```

### 5.6 Task 5: Key Property Prediction Circuit

**Mixed Pauli Measurements:**
```python
def create_key_property_circuit(n_qubits=8, n_layers=4):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # Standard angle encoding + variational layers
    # ... (similar to Task 1)
    
    # Define mixed measurements
    observables = [
        cirq.Z(qubits[0]),  # Z measurement
        cirq.Z(qubits[1]),
        cirq.X(qubits[2]),  # X measurement
        cirq.Y(qubits[3]),  # Y measurement
    ]
    
    return circuit, data_symbols, param_symbols, observables
```

---

## Week 6: Quantum Simulation & Validation

### 6.1 Gradient Verification (Parameter Shift Rule)

**Theory:**
```
For Pauli rotation gates:
∂⟨ψ|H|ψ⟩/∂θ = [f(θ + π/2) - f(θ - π/2)] / 2
```

**Implementation:**
```python
def parameter_shift_gradient(circuit, symbol, input_data, observable):
    """
    Compute gradient using parameter shift rule.
    
    Args:
        circuit: Parameterized quantum circuit
        symbol: Parameter to differentiate
        input_data: Quantum state encoding
        observable: Measurement operator
    
    Returns:
        gradient: ∂f/∂θ
    """
    # Shift by +π/2
    resolver_plus = {symbol: input_data + np.pi/2}
    f_plus = tfq.layers.Expectation()(circuit, symbol_names=[symbol], symbol_values=resolver_plus, operators=observable)
    
    # Shift by -π/2
    resolver_minus = {symbol: input_data - np.pi/2}
    f_minus = tfq.layers.Expectation()(circuit, symbol_names=[symbol], symbol_values=resolver_minus, operators=observable)
    
    gradient = (f_plus - f_minus) / 2.0
    
    return gradient

# Verify against numerical gradient
def numerical_gradient(circuit, symbol, input_data, observable, epsilon=1e-5):
    """Finite difference approximation."""
    resolver_plus = {symbol: input_data + epsilon}
    f_plus = tfq.layers.Expectation()(circuit, symbol_names=[symbol], symbol_values=resolver_plus, operators=observable)
    
    resolver_minus = {symbol: input_data - epsilon}
    f_minus = tfq.layers.Expectation()(circuit, symbol_names=[symbol], symbol_values=resolver_minus, operators=observable)
    
    gradient = (f_plus - f_minus) / (2 * epsilon)
    
    return gradient

# Test
grad_ps = parameter_shift_gradient(circuit, theta_syms[0], x_test, observable)
grad_num = numerical_gradient(circuit, theta_syms[0], x_test, observable)

assert np.abs(grad_ps - grad_num) < 1e-6, "Gradient mismatch!"
print("✅ Gradient verification passed")
```

**Tasks:**
- [ ] Verify gradients for all circuits
- [ ] Ensure error < 1e-6 between parameter shift and numerical
- [ ] Document results

### 6.2 Barren Plateau Analysis

**Check Gradient Norms:**
```python
def check_barren_plateau(circuit, param_symbols, input_data, observable, n_samples=100):
    """
    Compute gradient norms for random parameter initializations.
    
    If gradient norms → 0 exponentially with depth, barren plateau exists.
    """
    gradient_norms = []
    
    for _ in range(n_samples):
        # Random parameter initialization
        params = np.random.uniform(-np.pi, np.pi, size=len(param_symbols))
        
        # Compute gradients for all parameters
        grads = []
        for i, symbol in enumerate(param_symbols):
            grad = parameter_shift_gradient(circuit, symbol, input_data, observable)
            grads.append(grad)
        
        grad_norm = np.linalg.norm(grads)
        gradient_norms.append(grad_norm)
    
    mean_grad_norm = np.mean(gradient_norms)
    std_grad_norm = np.std(gradient_norms)
    
    print(f"Mean gradient norm: {mean_grad_norm:.6f}")
    print(f"Std gradient norm: {std_grad_norm:.6f}")
    
    if mean_grad_norm < 1e-4:
        print("⚠️ WARNING: Potential barren plateau detected")
    else:
        print("✅ Gradient flow OK")
    
    return gradient_norms

# Test
grad_norms = check_barren_plateau(circuit, theta_syms, x_sample, observable)

# Plot distribution
import matplotlib.pyplot as plt
plt.hist(grad_norms, bins=30)
plt.xlabel('Gradient Norm')
plt.ylabel('Frequency')
plt.title('Gradient Distribution (Barren Plateau Check)')
plt.savefig('reports/barren_plateau_analysis.png')
```

**Tasks:**
- [ ] Run barren plateau check for all circuits
- [ ] If detected: Reduce n_layers or use problem-inspired ansatz
- [ ] Plot gradient distributions

### 6.3 Simulation Benchmarks

**Forward Pass Time:**
```python
import time

def benchmark_forward_pass(circuit, input_data, observable, batch_size=100, n_runs=10):
    """Measure simulation time."""
    
    times = []
    for _ in range(n_runs):
        start = time.time()
        
        # Simulate circuit
        outputs = tfq.layers.Expectation()(
            circuit_batch=[circuit] * batch_size,
            symbol_names=data_symbols,
            symbol_values=np.random.randn(batch_size, len(data_symbols)),
            operators=observable
        )
        
        end = time.time()
        times.append(end - start)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Forward pass time (batch={batch_size}): {mean_time:.4f} ± {std_time:.4f} seconds")
    print(f"Time per sample: {mean_time / batch_size * 1000:.2f} ms")
    
    return mean_time

# Benchmark
benchmark_forward_pass(circuit, x_sample, observable, batch_size=100)
```

**Memory Usage:**
```python
import tracemalloc

tracemalloc.start()

# Run forward pass
outputs = tfq.layers.Expectation()(...)

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Current memory: {current / 1024**2:.2f} MB")
print(f"Peak memory: {peak / 1024**2:.2f} MB")
```

**Tasks:**
- [ ] Benchmark forward pass (batch size: 32, 64, 128)
- [ ] Benchmark gradient computation
- [ ] Measure memory usage
- [ ] Test scaling: Time vs batch size (should be linear)

### Phase 3 Deliverables

**Code:**
- `src/quantum/circuits.py` - All circuit definitions
- `src/quantum/validators.py` - Gradient verification, barren plateau checks
- `src/quantum/benchmarks.py` - Performance testing

**Tests:**
- `tests/test_circuits.py` - Unit tests for each circuit
- `tests/test_gradients.py` - Gradient verification tests

**Reports:**
- `reports/quantum_circuits_design.pdf`:
  - Circuit diagrams (ASCII or image)
  - Parameter counts per circuit
  - Gradient verification results
  - Barren plateau analysis
  - Simulation benchmarks (time, memory)
  - Recommendations for circuit depth and qubit count

**Success Criteria:**
✅ All 5 circuits implemented and tested  
✅ Gradient verification: error < 1e-6  
✅ No barren plateaus detected (gradient norm > 1e-4)  
✅ Forward pass < 100ms per sample (batch=32)  
✅ Memory usage < 4GB for typical batch  

---

# Continue to `docs/PHASE_4_5_6.md` for remaining phases...
