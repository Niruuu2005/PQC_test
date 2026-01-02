# Dataset Generation Module

Complete documentation for the AIRAWAT dataset generation pipeline.

---

## Overview

Generates realistic cryptographic attack datasets for ML training.

**Output**: 427,950+ samples with execution metrics across 83 attack types and 111 algorithms

---

## Quick Start

```bash
python generate_attack_dataset.py
```

**Output**: `attack_dataset.csv` (~370MB)

---

## Dataset Schema

### Core Columns
- `attack_id` - Unique attack identifier
- `attack_name` - Attack type name
- `attack_category` - Category classification  
- `attack_success` - Binary success indicator (0/1)
- `algorithm_name` - Cryptographic algorithm (e.g., "AES-256")
- `algorithm_type` - Symmetric/Asymmetric
- `key_size_bits` - Key size in bits
- `execution_time_ms` - Execution time (milliseconds)
- `memory_used_mb` - Memory usage (MB)
- `cpu_usage_percent` - CPU utilization (%)

---

## Attack Types (83 total)

### Categories
- **Side-Channel**: Timing, Power Analysis, EM Analysis
- **Fault Injection**: Clock glitching, Voltage manipulation
- **Cryptanalytic**: Differential, Linear, Meet-in-the-Middle
- **Implementation**: Padding Oracle, Cache-timing

---

## Algorithms (111 total)

### Symmetric Ciphers
AES (128/192/256), DES, 3DES, Blowfish, Twofish, ChaCha20, Serpent, Camellia, etc.

### Asymmetric Ciphers
RSA (1024/2048/4096), ECC (P-256/P-384/P-521), ElGamal, DSA, etc.

### Hash Functions
SHA-256, SHA-512, MD5, SHA-1, SHA-3, BLAKE2, etc.

---

## Generation Process

1. **Algorithm Definition** - Load 111 algorithms with specifications
2. **Attack Simulation** - Simulate 83 attack types
3. **Execution Metrics** - Generate realistic timing/memory/CPU data
4. **Success Modeling** - Calculate attack success probability
5. **CSV Export** - Export to structured CSV format

---

## Data Statistics

- **Total Samples**: 427,950
- **Attack Types**: 83
- **Algorithms**: 111
- **File Size**: ~370MB
- **Quality Score**: >98% complete, >99% valid

---

## Usage in Pipeline

```python
from model_creation.src.data.loader import DataLoader

loader = DataLoader(data_dir='dataset_generation')
df = loader.load_attack_dataset()

print(f"Loaded {len(df)} samples")
```

---

## Regeneration

To regenerate the dataset:

```bash
# Delete existing (if needed)
rm attack_dataset.csv

# Generate fresh
python generate_attack_dataset.py
```

**Expected Time**: 2-5 minutes

---

## Future Enhancements

- Real attack measurements from actual systems
- Temporal attack sequences
- Multi-party attack scenarios
- Extended algorithm coverage

---

**Status**: Production-ready dataset generation  
**Last Updated**: 2026-01-02
