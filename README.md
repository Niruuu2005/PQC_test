# Quantum-Enhanced Cryptanalysis System

**AI-driven Research for Advanced Weaponization of Attack Reconnaissance and Analysis Technologies**

A production-ready hybrid quantum-classical machine learning system for cryptanalytic attack analysis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Generation](#dataset-generation)
- [Model Creation](#model-creation)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Overview

AIRAWAT implements a complete ML pipeline for analyzing cryptographic attacks with:
- **427K+ attack scenarios** across 83 attack types and 111 algorithms
- **9 Classical ML models** (Random Forest, XGBoost, LightGBM, CatBoost, etc.)
- **5 Quantum circuits** (116 trainable parameters using Cirq)
- **Hybrid quantum-classical** architecture
- **REST API** for model serving
- **100% test success rate** - Production ready

### 5 Prediction Tasks
1. **Attack Classification** - Identify attack type (multi-class)
2. **Attack Success Prediction** - Predict success probability (binary)
3. **Algorithm Identification** - Identify crypto algorithm (multi-class)
4. **Plaintext Recovery** - QAOA-based recovery (optimization)
5. **Key Properties** - Predict key characteristics (regression)

---

## Quick Start

```bash
# Clone and install
git clone [repository-url]
cd AIRAWAT
pip install -r requirements.txt

# Run complete pipeline (generate data + train models + evaluate)
python main.py --mode all
```

**Expected Output**: Trained models in `model_creation/models/final/` and reports in `model_creation/reports/`

---

## Installation

### Requirements
- Python 3.9+
- 8GB+ RAM
- Windows/Linux/macOS

### Dependencies
```bash
pip install numpy pandas scikit-learn
pip install xgboost lightgbm catboost
pip install cirq optuna flask
pip install matplotlib seaborn
```

Or simply:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
AIRAWAT/
├── main.py                          # Main pipeline entry point
├── README.md                        # This file
├── requirements.txt                 # Dependencies
│
├── dataset_generation/              # Dataset generation module
│   ├── README.md                    # Data generation docs
│   ├── generate_attack_dataset.py  # Main generation script
│   ├── attack_dataset.csv          # Generated data (~370MB)
│   └── src/                         # Generation source code
│
└── model_creation/                  # Model creation module
    ├── README.md                    # Model creation docs
    ├── src/
    │   ├── data/                    # Data pipeline (Phase 1)
    │   │   ├── loader.py
    │   │   ├── validator.py
    │   │   ├── eda_reporter.py
    │   │   └── enhanced_feature_engineer.py
    │   ├── classical/               # Classical ML (Phase 2)
    │   │   ├── base_models.py       # 9 ML models
    │   │   ├── trainer.py
    │   │   ├── evaluator.py
    │   │   └── hyperopt.py
    │   ├── quantum/                 # Quantum circuits (Phase 3)
    │   │   └── circuits.py          # 5 quantum circuits
    │   ├── hybrid/                  # Hybrid models (Phase 4)
    │   │   └── hybrid_model.py
    │   └── evaluation/              # Evaluation (Phase 6)
    │       └── performance_evaluator.py
    ├── models/                      # Trained models
    ├── reports/                     # Performance reports
    └── deployment/                  # REST API
        └── model_server.py
```

---

## Dataset Generation

### Overview
Generates realistic cryptographic attack datasets with execution metrics.

### Data Generated
- **427,950 samples** across multiple CSVs
- **83 attack types**: Timing, Fault Injection, Power Analysis, etc.
- **111 algorithms**: AES, RSA, DES, ECC, etc.
- **Execution metrics**: Time, memory, CPU usage, success rates

### Generate Data
```bash
python dataset_generation/generate_attack_dataset.py
```

**Output**: `attack_dataset.csv` (~370MB)

### Schema
- `attack_id`, `attack_name`, `attack_category`, `attack_success`
- `algorithm_name`, `key_size_bits`, `algorithm_type`
- `execution_time_ms`, `memory_used_mb`, `cpu_usage_percent`
- Ciphertext, plaintext, key information

See `dataset_generation/README.md` for detailed documentation.

---

## Model Creation

The model creation pipeline implements 6 phases:

### **Phase 1: Data Engineering**
- **Data Loader**: Merges CSVs, handles missing values
- **Validator**: Schema validation, quality scoring
- **EDA Reporter**: Automated analysis with visualizations
- **Feature Engineer**: Generates 65+ features

```python
from model_creation.src.data.loader import DataLoader
from model_creation.src.data.enhanced_feature_engineer import EnhancedFeatureEngineer

loader = DataLoader(data_dir='dataset_generation')
df = loader.load_attack_dataset()

engineer = EnhancedFeatureEngineer()
df = engineer.engineer_features(df)  # 65+ features
```

### **Phase 2: Classical ML Models**

**9 Models Implemented**:
1. Random Forest
2. XGBoost
3. LightGBM
4. CatBoost
5. Gradient Boosting
6. Logistic Regression
7. SVM
8. Decision Tree
9. Extra Trees

```python
from model_creation.src.classical import create_model

model = create_model('xgboost', task_name='attack_classification')
model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
model.save('models/my_xgboost')
```

**Training Infrastructure**:
- `ModelTrainer`: Multi-task orchestration, SMOTE, cross-validation
- `ModelEvaluator`: Metrics, confusion matrices, ROC curves
- `HyperparameterOptimizer`: Bayesian optimization with Optuna

### **Phase 3: Quantum Circuits**

**5 Parameterized Quantum Circuits** (Cirq framework):

1. **Attack Classification**: 8 qubits, 32 parameters
2. **Attack Success**: 6 qubits, 18 parameters
3. **Algorithm ID**: 10 qubits, 40 parameters
4. **Plaintext Recovery (QAOA)**: 8 qubits, 4 parameters
5. **Key Properties**: 8 qubits, 24 parameters

**Total**: 116 trainable quantum parameters

```python
from model_creation.src.quantum import get_task_circuit

circuit, data_syms, param_syms, qubits = get_task_circuit('attack_classification')
print(f"Circuit: {len(qubits)} qubits, {len(param_syms)} parameters")
```

### **Phase 4: Hybrid Integration**

Combines classical preprocessing with quantum circuit processing:

```
Input (65 features) 
    → Classical Preprocessing (Feature Selection → Top 8)
    → Quantum Circuit Processing
    → Classical Postprocessing
    → Output (Predictions)
```

```python
from model_creation.src.hybrid import create_hybrid_model

hybrid = create_hybrid_model(
    task_name='attack_classification',
    classical_model_type='random_forest',
    n_quantum_features=8
)
hybrid.fit(X_train, y_train, X_val, y_val)
```

### **Phase 5: System Validation**
- Comprehensive testing: **9/9 tests passing (100%)**
- Production readiness verified

### **Phase 6: Evaluation & Deployment**
- Performance evaluator with multi-metric comparison
- REST API for model serving

See `model_creation/README.md` for complete phase documentation.

---

## Usage Examples

### Complete Pipeline
```bash
python main.py --mode all  # Generate + Train + Evaluate
```

### Individual Steps
```bash
python main.py --mode generate  # Generate data only
python main.py --mode train     # Train models only
python main.py --mode evaluate  # Evaluate only
```

### Use Trained Models
```python
from model_creation.src.classical import create_model

# Load and use model
model = create_model('xgboost', 'attack_classification')
model.load('model_creation/models/final/xgboost')
predictions = model.predict(X_test)
```

### Model Comparison
```python
from model_creation.src.evaluation import PerformanceEvaluator

evaluator = PerformanceEvaluator()

# Evaluate multiple models
for name in ['xgboost', 'random_forest', 'lightgbm']:
    model = create_model(name, 'attack_classification')
    model.fit(X_train, y_train)
    evaluator.evaluate_model(model, X_test, y_test, name, 'attack_classification')

# Generate comparison report
report = evaluator.generate_report()
comparison = evaluator.compare_models('attack_classification')
print(comparison)
```

---

## API Reference

### Start API Server
```bash
python model_creation/deployment/model_server.py
```

Server runs on `http://localhost:5000`

### Endpoints

#### Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{"status": "healthy", "models_loaded": 3, "version": "1.0.0"}
```

#### Make Predictions
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "xgboost",
    "task_name": "attack_classification",
    "features": [[1.2, 3.4, 5.6, ...]]
  }'
```

Response:
```json
{
  "predictions": [0, 1, 2],
  "probabilities": [[0.7, 0.2, 0.1], ...],
  "model": "xgboost",
  "n_samples": 3
}
```

#### List Models
```bash
curl http://localhost:5000/models
```

#### System Info
```bash
curl http://localhost:5000/info
```

---

## Architecture

### High-Level Flow
```
CSV Data → Data Pipeline → Feature Engineering (65+) 
    → Classical ML (9 models) 
    → Quantum Circuits (5) 
    → Hybrid Models (5)
    → Evaluation & Reports
    → REST API Serving
```

### Technology Stack
| Layer | Technologies |
|-------|-------------|
| Data | Pandas, NumPy |
| Classical ML | XGBoost, LightGBM, CatBoost, Scikit-learn |
| Quantum | Cirq |
| Optimization | Optuna |
| API | Flask |
| Visualization | Matplotlib, Seaborn |

---

## Performance

### Test Results
| Component | Status | Tests |
|-----------|--------|-------|
| Data Pipeline | ✅ Operational | 2/2 (100%) |
| Classical ML | ✅ Operational | 3/3 (100%) |
| Quantum Circuits | ✅ Operational | 2/2 (100%) |
| Hybrid Integration | ✅ Operational | 2/2 (100%) |

**Overall**: 9/9 tests passing (100% success rate)

### Model Performance
- **Attack Classification**: ~85-90% accuracy
- **Attack Success**: ~0.80-0.85 AUC
- **Algorithm ID**: ~85-87% accuracy

---

## Troubleshooting

### Issue: Import Errors
```bash
# Install missing packages
pip install xgboost lightgbm catboost cirq optuna
```

### Issue: Out of Memory
```python
# Sample data to reduce size
df = df.sample(n=100000, random_state=42)
```

### Issue: Port Already in Use
```python
# Change port in model_server.py
app.run(host='0.0.0.0', port=8080)  # Use different port
```

### Issue: Data Not Found
```bash
# Generate data first
python dataset_generation/generate_attack_dataset.py
```

---

## License

[Your License Here]

---

## Contact

For issues, questions, or contributions, please contact [Your Contact Information]

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**Last Updated**: 2026-01-02
