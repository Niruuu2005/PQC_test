# AIRAWAT - Comprehensive Usage Manual

**Version**: 1.0  
**Last Updated**: December 31, 2025  
**Status**: Production Ready ✅

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Usage Guide](#usage-guide)
5. [Configuration](#configuration)
6. [Output Files](#output-files)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## Overview

AIRAWAT is a comprehensive cryptographic dataset generator that creates production-grade datasets for cryptanalysis research, machine learning, and security analysis.

### Key Features

- **51 Cryptographic Algorithms**: Symmetric, asymmetric, legacy ciphers
- **90 Attack Types**: Comprehensive attack simulation across 9 categories
- **Multi-Language Support**: Python (primary), C, Java, Rust (extensible)
- **Automated Pipeline**: One-command execution for all phases
- **Robust Error Handling**: 3-attempt retry logic with detailed logging
- **Checkpoint Resume**: Automatic recovery from interruptions
- **Production Ready**: Battle-tested with 139,050+ attack simulations

### System Requirements

#### Minimum
- Python 3.8+
- 8 GB RAM
- 500 MB disk space
- Windows/Linux/macOS

#### Recommended
- Python 3.10+
- 16 GB RAM
- 2 GB disk space
- Multi-core CPU (4+ cores)

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone/download the project
cd AIRAWAT

# Install required packages
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Generate all datasets in one command
python run_complete_pipeline.py
```

This will execute three phases:
1. **Phase 1**: Crypto dataset (51 algorithms × 10 samples)
2. **Phase 2**: Attack dataset (90 attacks × 510 encryptions × 3 runs)
3. **Phase 3**: Cryptanalysis summary with recommendations

**Duration**: ~3-6 hours (depends on system)

### 3. Check Output

```bash
# Generated files
ls -lh *.csv

# Output:
# crypto_dataset.csv              - 280 KB
# attack_dataset.csv              - 96.4 MB
# Cryptographic_Algorithm_Summary.csv - 7 KB
```

---

## Installation

### Step-by-Step Installation

```bash
# 1. Ensure Python 3.8+ is installed
python --version

# 2. Create virtual environment (recommended)
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "from src.crypto_dataset_generator.crypto import cipher_factory; print('✅ Installation successful')"
```

### Dependencies

```
cryptography>=41.0.0      # Core crypto operations
pycryptodome>=3.18.0      # Additional algorithms
numpy>=1.24.0             # Numerical operations
pandas>=2.0.0             # Data processing
psutil>=5.9.0             # System monitoring
scipy>=1.10.0             # Statistical analysis
```

---

## Usage Guide

### Method 1: Complete Automated Pipeline (Recommended)

**Command**:
```bash
python run_complete_pipeline.py
```

**Features**:
- Automated execution of all 3 phases
- Detailed batch logging ("90 attacks for encryption X completed")
- 3-attempt retry on failures
- Real-time progress streaming
- Comprehensive error handling

**Output**: All three CSV files generated sequentially

**Logging**: `logs/complete_pipeline_YYYYMMDD_HHMMSS.log`

---

### Method 2: Individual Phase Execution

#### Phase 1: Crypto Dataset

```bash
# Basic usage
python main.py

# Custom configuration
python main.py \
    --samples 20 \
    --algorithms AES-256-GCM,ChaCha20,RSA-2048-OAEP \
    --output custom_crypto.csv \
    --seed 42 \
    --threads 4
```

**Options**:
- `--samples N`: Samples per algorithm (default: 10)
- `--algorithms ALGO1,ALGO2`: Specific algorithms (default: all)
- `--output FILE`: Output path (default: crypto_dataset.csv)
- `--seed N`: RNG seed for reproducibility (default: 42)
- `--threads N`: Worker threads (default: 4)

**Example Output**:
```
[1/51] Processing AES-256-GCM...
  Progress: 2.0% (10/510 samples)
[2/51] Processing ChaCha20...
  Progress: 3.9% (20/510 samples)
...
✅ Generated crypto_dataset.csv (510 rows, 280 KB)
```

---

#### Phase 2: Attack Dataset

```bash
# Basic usage
python generate_attack_dataset.py

# With custom configuration
python generate_attack_dataset.py \
    --input crypto_dataset.csv \
    --output attack_dataset.csv \
    --workers 4 \
    --resume \
    --timeout 120
```

**Options**:
- `--input FILE`: Input crypto dataset (default: crypto_dataset.csv)
- `--output FILE`: Output path (default: attack_dataset.csv)
- `--workers N`: Parallel workers (default: 1)
- `--resume`: Resume from checkpoint (recommended)
- `--timeout N`: Attack timeout in seconds (default: 120)

**Example Output with Batch Logging**:
```
[Encryption 1/510] Running 90 attacks...
  ✅ 90 attacks for encryption 1 completed (3.5% success rate)
[Encryption 2/510] Running 90 attacks...
  ✅ 90 attacks for encryption 2 completed (4.1% success rate)
...
🎯 Milestone: 10 encryptions completed
✅ Batch completed: Total 900 attacks executed
...
[Encryption 510/510] Running 90 attacks...
  ✅ 90 attacks for encryption 510 completed
✅ Generated attack_dataset.csv (139,050 rows, 96.4 MB)
```

---

#### Phase 3: Cryptanalysis Summary

This phase is automatically executed in the complete pipeline, or you can generate it manually:

```python
python -c "
from run_complete_pipeline import phase_3_cryptanalysis_summary
phase_3_cryptanalysis_summary()
"
```

---

### Method 3: Programmatic API

```python
from src.crypto_dataset_generator.crypto.cipher_factory import create_cipher
from src.crypto_dataset_generator.crypto.multilang_crypto import MultiLangCrypto

# Method 1: Traditional approach
cipher = create_cipher('AES-256-GCM', seed=42)
cipher.generate_key()
ciphertext, metadata = cipher.encrypt(b"Hello, World!")
plaintext, decrypt_meta = cipher.decrypt(ciphertext, 
                                         iv=bytes.fromhex(metadata.iv),
                                         tag=bytes.fromhex(metadata.tag))

# Method 2: Multi-language approach
crypto = MultiLangCrypto('AES-256-GCM', language='python')
crypto.generate_key()
ciphertext, metadata = crypto.encrypt(b"Hello, World!")
plaintext, decrypt_meta = crypto.decrypt(ciphertext,
                                         iv=bytes.fromhex(metadata.iv),
                                         tag=bytes.fromhex(metadata.tag))

print(f"Active language: {crypto.get_active_language()}")  # python
```

---

## Configuration

### Algorithm Selection

```bash
# Use all available algorithms (default)
python main.py

# Use specific algorithms
python main.py --algorithms AES-256-GCM,ChaCha20

# Use algorithm categories
python main.py --algorithms "AES-*,RSA-*"
```

### Performance Tuning

```bash
# Single-threaded (safest, slowest)
python main.py --threads 1

# Multi-threaded (faster, more memory)
python main.py --threads 8

# Parallel attack execution
python generate_attack_dataset.py --workers 4
```

### Checkpointing

```bash
# Enable automatic checkpointing (recommended)
python generate_attack_dataset.py --resume

# Checkpoint frequency (default: every 10 encryptions)
# Configure in generate_attack_dataset.py:
CHECKPOINT_EVERY = 10
```

---

## Output Files

### 1. crypto_dataset.csv

**Size**: ~280 KB  
**Rows**: 510 (51 algorithms × 10 samples)  
**Columns**: 45

**Schema**:
```csv
row_id,algorithm_name,key_hex,key_size_bits,plaintext_hex,plaintext_length,
ciphertext_hex,ciphertext_length,iv_or_nonce,tag,encryption_time_ms,
decryption_time_ms,plaintext_hash_sha256,entropy,chi_square_statistic,
avalanche_effect,padding_size,compression_ratio,key_schedule_rounds,
cipher_mode,nist_approved,quantum_resistant,...
```

**Use Cases**:
- Algorithm performance analysis
- Encryption overhead comparison
- Security metric baseline

---

### 2. attack_dataset.csv

**Size**: ~96.4 MB  
**Rows**: 139,050 (510 encryptions × 90 attacks × 3 runs)  
**Columns**: 55

**Schema**:
```csv
attack_execution_id,encryption_row_id,algorithm_name,attack_id,attack_name,
attack_category,run_number,parameter_set,timestamp,key_hex,plaintext_hex,
ciphertext_hex,attack_success,confidence_score,execution_time_ms,
memory_used_mb,cpu_usage_percent,iterations_performed,recovered_data_hex,
vulnerability_detected,vulnerability_type,severity_score,recommendation,...
```

**Use Cases**:
- Machine learning training data
- Attack effectiveness analysis
- Vulnerability pattern recognition
- Security research

---

### 3. Cryptographic_Algorithm_Summary.csv

**Size**: ~7 KB  
**Rows**: 51 (one per algorithm)  
**Columns**: 8

**Schema**:
```csv
Algorithm,Key_Size_Bits,Block_Size_Bits,Total_Attacks_Tested,
Successful_Attacks,Success_Rate_Percent,Avg_Encryption_Time_ms,
Recommendation
```

**Recommendations**:
- `RECOMMENDED`: < 10% attack success (e.g., AES-256, ChaCha20)
- `CAUTION`: 10-40% attack success
- `PHASE_OUT`: 40-80% attack success
- `AVOID`: > 80% attack success (e.g., DES, RC4, MD5)

---

## Advanced Usage

### Multi-Language Crypto Implementation

```python
from src.crypto_dataset_generator.crypto.multilang_crypto import (
    MultiLangCrypto,
    get_available_languages
)

# Check available languages for an algorithm
languages = get_available_languages('AES-256-GCM')
print(f"Available: {languages}")  # ['python', 'c', 'java', 'rust']

# Try C implementation (falls back to Python if unavailable)
crypto_c = MultiLangCrypto('AES-256-GCM', language='c')
print(f"Using: {crypto_c.get_active_language()}")  # python (fallback)

# Encrypt/decrypt
crypto_c.generate_key()
ct, meta = crypto_c.encrypt(b"Test data")
pt, decrypt_meta = crypto_c.decrypt(ct, iv=bytes.fromhex(meta.iv))
```

### Custom Attack Implementation

```python
# Create custom attack in src/crypto_dataset_generator/attacks/
# See existing attacks for template

from src.crypto_dataset_generator.attacks.base_attack import BaseAttack

class MyCustomAttack(BaseAttack):
    def execute(self, ciphertext, algorithm_info):
        # Your attack logic here
        success = self.attempt_break(ciphertext)
        return {
            'success': success,
            'confidence': 0.95,
            'execution_time_ms': 123.45
        }
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Import Error

```
ModuleNotFoundError: No module named 'cryptography'
```

**Solution**:
```bash
pip install -r requirements.txt
```

---

#### Issue 2: Memory Error During Attack Generation

```
MemoryError: Unable to allocate array
```

**Solution**:
```bash
# Use single worker
python generate_attack_dataset.py --workers 1

# Or increase system swap space
```

---

#### Issue 3: Checkpoint Resume Not Working

**Solution**:
```bash
# Manually specify resume
python generate_attack_dataset.py --resume

# Check for checkpoint file
ls attack_dataset.checkpoint.json
```

---

#### Issue 4: Slow Performance

**Solutions**:
```bash
# 1. Increase workers (if you have RAM)
python generate_attack_dataset.py --workers 4

# 2. Reduce timeout
python generate_attack_dataset.py --timeout 60

# 3. Run overnight
nohup python run_complete_pipeline.py > pipeline.log 2>&1 &
```

---

## API Reference

### Core Functions

#### create_cipher()
```python
from src.crypto_dataset_generator.crypto.cipher_factory import create_cipher

cipher = create_cipher(
    algorithm_name: str,  # e.g., 'AES-256-GCM'
    seed: Optional[int] = None  # RNG seed for determinism
) -> BaseCipherSystem
```

#### get_available_algorithms()
```python
from src.crypto_dataset_generator.crypto.cipher_factory import get_available_algorithms

algorithms: List[str] = get_available_algorithms()
# Returns: ['AES-128-CBC', 'AES-128-GCM', 'ChaCha20', ...]
```

#### MultiLangCrypto()
```python
from src.crypto_dataset_generator.crypto.multilang_crypto import MultiLangCrypto

crypto = MultiLangCrypto(
    algorithm: str,  # e.g., 'AES-256-GCM'
    language: str = 'python',  # 'python', 'c', 'java', 'rust'
    **kwargs
)

crypto.generate_key()
ciphertext, metadata = crypto.encrypt(plaintext: bytes)
plaintext, metadata = crypto.decrypt(ciphertext: bytes, **kwargs)
active_lang: str = crypto.get_active_language()
```

---

## Support & Resources

- **Documentation**: `Docs/` folder
- **Examples**: See usage examples above
- **Issues**: Check troubleshooting section
- **Logs**: `logs/complete_pipeline_*.log`

---

## License

See LICENSE file in project root.

---

**End of Usage Manual**  
Generated by AIRAWAT v1.0
