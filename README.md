# AIRAWAT - Quantum ML Cryptanalysis System

**Post-Quantum Cryptography Testing & Analysis**

A comprehensive hybrid quantum-classical machine learning system for cryptographic algorithm analysis, attack pattern detection, and security assessment.

## 🎯 Project Overview

AIRAWAT combines classical machine learning with quantum computing to analyze cryptographic systems:

- **427,950+ attack execution samples** analyzed
- **99.78% accuracy** in attack success prediction
- **111 cryptographic algorithms** supported
- **Quantum-enhanced ML models** for cryptanalysis

## 🚀 Quick Start

### Prerequisites
```bash
python 3.9+
pip install -r requirements.txt
```

### Run Training
```bash
cd model_creation
python train_real_data.py
```

### Results
- **Random Forest:** 99.78% accuracy ✅
- **Gradient Boosting:** 98.95% accuracy
- **Logistic Regression:** 98.77% accuracy

## 📊 Project Structure

```
AIRAWAT/
├── dataset_generation/     # Cryptographic dataset generation
│   ├── crypto_dataset.csv  # 890 encryption tests
│   └── attack_dataset.csv  # 427K attack executions
├── model_creation/         # ML model training & inference
│   ├── src/
│   │   ├── data/          # Data processing pipeline
│   │   ├── classical/     # Classical ML models
│   │   └── quantum/       # Quantum circuits & hybrid models
│   ├── models/            # Trained models
│   ├── train_real_data.py # Complete training pipeline
│   └── model_inference.py # Model usage & inference
└── docs/                  # Documentation & guides
```

## 🔬 Features

### Data Engineering
- Multi-source dataset loading & merging
- 65+ engineered features
- Missing value imputation
- Outlier detection & capping

### Classical ML
- Random Forest
- Gradient Boosting
- Logistic Regression
- SVM, Neural Networks

### Quantum ML (TensorFlow Quantum)
- Parameterized Quantum Circuits (PQC)
- 8-qubit hybrid models
- Variational quantum algorithms
- Classical-quantum integration

### PQC Support
- ML-KEM (Kyber)
- Dilithium, Falcon
- SPHINCS+, Classic McEliece
- 60+ post-quantum algorithms

## 📈 Results

**Attack Success Prediction:**
- Dataset: 427,950 samples
- Features: 37 numeric features
- Accuracy: 99.78% (Random Forest)
- Training: <2 minutes

**Supported Tasks:**
1. Attack Classification
2. Attack Success Prediction ✅
3. Algorithm Identification
4. Plaintext Recovery (QAOA)
5. Key Properties Prediction

## 🛠️ Installation

### Basic Setup
```bash
git clone https://github.com/yourusername/PQC_test.git
cd PQC_test
pip install -r requirements.txt
```

### TensorFlow Quantum (Optional)
```bash
pip install tensorflow-quantum cirq-google
```

### PQC Libraries (Windows)
See `PQC_INSTALLATION_WINDOWS.md` for detailed setup.

## 📖 Documentation

- **[Implementation Plan](implementation-checklist.md)** - 12-week development roadmap
- **[Quick Start Guide](quick-start-guide.md)** - Getting started quickly
- **[QML Plan](qml-cryptanalysis-plan.md)** - Quantum ML architecture
- **[Model Usage Guide](model_creation/MODEL_USAGE_GUIDE.md)** - Using trained models
- **[PQC Installation](PQC_INSTALLATION_WINDOWS.md)** - Windows PQC setup

## 🎓 Usage Examples

### Train Models
```bash
cd model_creation
python train_real_data.py
```

### Make Predictions
```python
from model_inference import ModelManager
import numpy as np

manager = ModelManager()
X_random = np.random.randn(10, 37)  # 10 samples, 37 features
predictions, probs = manager.predict('attack_success', X_random)
```

### Train Quantum Model
```bash
python train_tfq_hybrid.py
```

## 🔧 Technologies

- **ML:** scikit-learn, pandas, numpy
- **Quantum:** TensorFlow Quantum, Cirq
- **Crypto:** liboqs, PyCryptodome
- **Deployment:** Flask, Docker

## 📊 Performance

| Model | Accuracy | Dataset Size | Training Time |
|-------|----------|--------------|---------------|
| Random Forest | 99.78% | 427,950 | ~45s |
| Gradient Boosting | 98.95% | 427,950 | ~90s |
| Logistic Regression | 98.77% | 427,950 | ~30s |

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- NIST Post-Quantum Cryptography Project
- Open Quantum Safe (liboqs)
- TensorFlow Quantum team

## 📧 Contact

For questions or collaboration: [Your Email]

---

**Status:** ✅ Production-ready framework with 99.78% accuracy on real cryptanalysis data
