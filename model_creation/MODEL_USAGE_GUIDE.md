# Model Usage & TensorFlow Quantum Guide

## 📦 Where are Models Stored?

### Currently: In-Memory Only
The models from the recent training are **not saved to disk** yet. They were trained and evaluated in memory.

### Save Models to Disk

Run this command to train and save models:
```bash
cd model_creation
python model_inference.py --train
```

**Models will be saved to:**
```
model_creation/models/saved/
├── attack_success_random_forest.pkl
├── attack_success_gradient_boosting.pkl
├── attack_success_logistic_regression.pkl
└── attack_success_metadata.json
```

---

## 🔮 How to Use Models on Random Data

### 1. Show Required Features
```bash
python model_inference.py --features
```

**Output:** Lists all 37 required features

### 2. Make Predictions
```bash
python model_inference.py --predict
```

**Example Code:**
```python
from model_inference import ModelManager
import numpy as np

# Initialize manager
manager = ModelManager()

# Create random data (37 features)
X_random = np.random.randn(10, 37)

# Predict
predictions, probabilities = manager.predict('attack_success', X_random)

# Results
for i, pred in enumerate(predictions):
    result = "SUCCESS" if pred else "FAILURE"
    confidence = probabilities[i][1]
    print(f"Sample {i+1}: {result} ({confidence:.2%} confidence)")
```

---

## 📊 Required Input Format

### Input Shape
- **Array:** `(n_samples, 37)`
- **Type:** `numpy.ndarray` or `pandas.DataFrame`
- **Features:** 37 numeric values

### Feature List (37 total)
```python
[
    'execution_time_ms',          # Attack execution time
    'memory_used_mb',              # Memory consumption
    'cpu_usage_percent',           # CPU utilization
    'iterations_performed',        # Number of iterations
    'confidence_score',            # Attack confidence
    'metric_1_value',              # Custom metric 1
    # ... metrics 2-10 ...
    'vulnerability_detected',      # Binary flag
    'severity_score',              # Vulnerability severity
    # ... and 22 more features
]
```

### Example Input
```python
import pandas as pd

# Single sample
sample = {
    'execution_time_ms': 245.3,
    'memory_used_mb': 128.5,
    'cpu_usage_percent': 75.2,
    'iterations_performed': 5000,
    'confidence_score': 0.89,
    # ... 32 more features ...
}

# Convert to DataFrame
df = pd.DataFrame([sample])

# Predict
predictions, probs = manager.predict('attack_success', df)
```

---

## 🔬 TensorFlow Quantum Implementation

### Prerequisites
```bash
pip install tensorflow-quantum cirq-google tensorflow
```

### Quick Start
```bash
cd model_creation
python train_tfq_hybrid.py
```

### What It Does
1. Loads attack dataset (427K samples)
2. Selects first 8 features for quantum encoding
3. Normalizes to [0, 2π] range
4. Creates 8-qubit, 4-layer quantum circuit
5. Integrates with classical neural network
6. Trains hybrid model for 20 epochs
7. Saves to `models/hybrid/quantum_hybrid_model.h5`

### Architecture
```
Input (8 features)
    ↓
[Quantum Circuit]
- 8 qubits
- 4 variational layers
- 32 parameters
- Ring entanglement
    ↓
[Measurement]
- 3 observables (Z₀, Z₁, Z₂)
    ↓
[Classical NN]
- Dense(32, relu)
- Dropout(0.3)
- Dense(16, relu)
- Dense(1, sigmoid)
    ↓
Output (attack success probability)
```

### Custom Training Example
```python
from train_tfq_hybrid import QuantumHybridModel
import numpy as np

# Prepare data (8 features, normalized to [0, 2π])
X_train = np.random.uniform(0, 2*np.pi, (1000, 8))
y_train = np.random.randint(0, 2, 1000)

# Create model
model = QuantumHybridModel(
    n_qubits=8,
    n_layers=4,
    n_classes=2
)

# Train
history = model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=32
)

# Save
model.save('my_quantum_model.h5')
```

---

## 🎯 Complete Workflow

### 1. Train Classical Models
```bash
python model_inference.py --train
```
**Output:** 3 models saved (RF, GB, LogReg)

### 2. Make Predictions
```bash
python model_inference.py --predict
```
**Output:** Predictions on 10 random samples

### 3. Train Quantum Hybrid
```bash
python train_tfq_hybrid.py
```
**Output:** TFQ model saved

### 4. Compare Results
- Classical RF: 99.78% accuracy ✅
- Classical GB: 98.95% accuracy
- Quantum Hybrid: TBD (requires TFQ installation)

---

## 📁 File Locations

| File | Purpose | Location |
|------|---------|----------|
| Trained models | Saved classifiers | `models/saved/*.pkl` |
| Model metadata | Feature info | `models/saved/*_metadata.json` |
| TFQ model | Quantum hybrid | `models/hybrid/quantum_hybrid_model.h5` |
| Training results | Performance metrics | `results/training_results_*.json` |

---

## 🔧 Troubleshooting

### "TensorFlow Quantum not found"
```bash
pip install tensorflow-quantum cirq-google
```

### "Model not found"
Run training first:
```bash
python model_inference.py --train
```

### "Feature mismatch"
Check expected features:
```bash
python model_inference.py --features
```

---

## 🚀 Next Steps

1. **Save Current Models:** `python model_inference.py --train`
2. **Test Inference:** `python model_inference.py --predict`
3. **Install TFQ:** `pip install tensorflow-quantum`
4. **Train Quantum Model:** `python train_tfq_hybrid.py`
5. **Compare Performance:** Classical vs Quantum
