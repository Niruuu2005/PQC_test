# AIRAWAT - System Architecture

**Version**: 1.0  
**Last Updated**: December 31, 2025

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Components](#system-components)
3. [Data Flow](#data-flow)
4. [Module Design](#module-design)
5. [Design Patterns](#design-patterns)
6. [Security Architecture](#security-architecture)
7. [Scalability](#scalability)

---

## Architecture Overview

AIRAWAT follows a **modular, layered architecture** with clear separation of concerns.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ run_complete_   │  │   main.py    │  │ generate_      │ │
│  │ pipeline.py     │  │              │  │ attack_        │ │
│  │ (Orchestrator)  │  │ (Phase 1)    │  │ dataset.py     │ │
│  └────────┬────────┘  └──────┬───────┘  └───────┬────────┘ │
└───────────┼────────────────┼────────────────────┼──────────┘
            │                 │                    │
┌───────────┴─────────────────┴────────────────────┴──────────┐
│                  ORCHESTRATION LAYER                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Retry Logic │ Logging │ Checkpoints │ Error Handling│   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────────┬───────────────────────────────┘
                                │
┌───────────────────────────────┴───────────────────────────────┐
│                     BUSINESS LOGIC LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Cipher       │  │ Attack       │  │ Analysis         │   │
│  │ Factory      │  │ Engine       │  │ Engine           │   │
│  └──────┬───────┘  └──────┬───────┘  └─────┬────────────┘   │
└─────────┼──────────────────┼─────────────────┼────────────────┘
          │                  │                 │
┌─────────┴──────────────────┴─────────────────┴────────────────┐
│                     ALGORITHM LAYER                             │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐    │
│  │ Symmetric  │  │ Asymmetric │  │ Multi-Language       │    │
│  │ Ciphers    │  │ Ciphers    │  │ Router               │    │
│  ├────────────┤  ├────────────┤  ├──────────────────────┤    │
│  │ AES        │  │ RSA        │  │ Python (primary)     │    │
│  │ ChaCha20   │  │ ECC        │  │ C (via ctypes)       │    │
│  │ 3DES       │  │            │  │ Java (via JPype)     │    │
│  │ Blowfish   │  │            │  │ Rust (via PyO3)      │    │
│  └────────────┘  └────────────┘  └──────────────────────┘    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────┐
│                      DATA LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐    │
│  │ crypto_      │  │ attack_      │  │ Cryptographic_    │    │
│  │ dataset.csv  │  │ dataset.csv  │  │ Algorithm_        │    │
│  │              │  │              │  │ Summary.csv       │    │
│  └──────────────┘  └──────────────┘  └───────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Components

### 1. Orchestration Layer

**Purpose**: Manages pipeline execution, error handling, and logging

**Components**:
- `run_complete_pipeline.py` - Main orchestrator
- `complete_pipeline.py` - Legacy pipeline
- Retry mechanism (3 attempts)
- Checkpoint management
- Comprehensive logging

**Design Pattern**: **Template Method** + **Chain of Responsibility**

```python
def execute_with_retry(phase_func, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            return phase_func()
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Retry {attempt}/{max_retries}")
                time.sleep(retry_delay)
            else:
                raise
```

---

### 2. Cipher Factory Layer

**Purpose**: Abstract algorithm instantiation using Factory pattern

**File**: `src/crypto_dataset_generator/crypto/cipher_factory.py`

**Design Pattern**: **Factory Method** + **Registry Pattern**

```python
# Registry Pattern
ALGORITHM_REGISTRY = {
    "AES-256-GCM": (AESCipherSystem, {"key_size": 256, "mode": "GCM"}),
    "ChaCha20": (ChaCha20CipherSystem, {}),
    # ... 51 total algorithms
}

# Factory Method
def create_cipher(algorithm_name: str, seed=None) -> BaseCipherSystem:
    cipher_class, constructor_args = ALGORITHM_REGISTRY[algorithm_name]
    return cipher_class(**constructor_args, seed=seed)
```

**Benefits**:
- Decouples algorithm selection from implementation
- Easy to add new algorithms
- Centralized algorithm management

---

### 3. Multi-Language Router

**Purpose**: Support multiple implementation languages with automatic fallback

**File**: `src/crypto_dataset_generator/crypto/multilang_crypto.py`

**Design Pattern**: **Strategy Pattern** + **Adapter Pattern**

```python
class MultiLangCrypto:
    def _load_implementation(self):
        # Try requested language
        if self.requested_language == 'python':
            return self._load_python()
        elif self.requested_language == 'c':
            return self._load_c() or self._load_python()  # Fallback
        elif self.requested_language == 'java':
            return self._load_java() or self._load_python()
        elif self.requested_language == 'rust':
            return self._load_rust() or self._load_python()
```

**Benefits**:
- Transparent language switching
- Graceful degradation
- Future-proof for performance optimization

---

### 4. Attack Engine

**Purpose**: Execute cryptanalytic attacks with consistent interface

**Files**:
- `src/crypto_dataset_generator/attacks/`
- `src/crypto_dataset_generator/orchestrator/attack_orchestrator.py`

**Design Pattern**: **Template Method** + **Observer Pattern**

```python
class BaseAttack(ABC):
    @abstractmethod
    def execute(self, ciphertext, algorithm_info):
        """Template method for attack execution"""
        pass

class AttackOrchestrator:
    def run_attacks_for_encryption(self, encryption_data):
        results = []
        for attack in self.get_all_attacks():
            result = attack.execute(encryption_data)
            results.append(result)
            self.notify_progress(attack, result)  # Observer
        return results
```

---

## Data Flow

### Phase 1: Crypto Dataset Generation

```
┌──────────┐     ┌────────────┐     ┌──────────────┐     ┌─────────┐
│ main.py  │────▶│  Cipher    │────▶│  Algorithm   │────▶│  CSV    │
│          │     │  Factory   │     │  Execution   │     │  Output │
└──────────┘     └────────────┘     └──────────────┘     └─────────┘
                       │                    │
                       ▼                    ▼
              ┌──────────────┐     ┌──────────────┐
              │ Algorithm    │     │ Encryption   │
              │ Registry     │     │ Metrics      │
              └──────────────┘     └──────────────┘
```

**Flow**:
1. User executes `main.py` with configuration
2. Cipher Factory loads algorithm registry
3. For each algorithm:
   - Instantiate cipher object
   - Generate key
   - Encrypt test strings
   - Record metrics
4. Write to `crypto_dataset.csv`

---

### Phase 2: Attack Dataset Generation

```
┌──────────────┐     ┌────────────┐     ┌─────────────┐
│ generate_    │────▶│  Load      │────▶│  Attack     │
│ attack_      │     │  Crypto    │     │  Orchestr.  │
│ dataset.py   │     │  Dataset   │     │             │
└──────────────┘     └────────────┘     └──────┬──────┘
                                               │
                          ┌────────────────────┴─────────────┐
                          ▼                                  ▼
                  ┌──────────────┐                  ┌──────────────┐
                  │  Execute     │                  │  Checkpoint  │
                  │  90 Attacks  │                  │  Manager     │
                  │  × 3 Runs    │                  │              │
                  └──────┬───────┘                  └──────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  Record      │
                  │  Results     │
                  │  to CSV      │
                  └──────────────┘
```

**Flow**:
1. Load `crypto_dataset.csv` (510 encryptions)
2. For each encryption (with checkpoint resume):
   - Load 90 attack implementations
   - For each attack:
     - Run 3 parameter sets (baseline, aggressive, stress)
     - Record success, confidence, metrics
   - Log batch completion ("90 attacks for encryption X done")
   - Save checkpoint every 10 encryptions
3. Write to `attack_dataset.csv` (139,050 rows)

---

### Phase 3: Cryptanalysis Summary

```
┌──────────────┐     ┌────────────┐     ┌─────────────┐
│ Analysis     │────▶│  Aggregate │────▶│  Generate   │
│ Engine       │     │  by Algo   │     │  Recomm.    │
└──────────────┘     └────────────┘     └─────────────┘
        │                   │                    │
        ▼                   ▼                    ▼
┌──────────────┐    ┌────────────┐     ┌─────────────┐
│ Load Attack  │    │ Calculate  │     │ Classify    │
│ Dataset      │    │ Success    │     │ Security    │
│              │    │ Rates      │     │ Rating      │
└──────────────┘    └────────────┘     └─────────────┘
```

---

## Module Design

### Module Structure

```
src/crypto_dataset_generator/
├── crypto/                    # Cryptographic implementations
│   ├── __init__.py
│   ├── base_cipher.py        # Abstract base classes
│   ├── cipher_factory.py     # Factory + Registry
│   ├── symmetric_ciphers.py  # AES, ChaCha20, etc.
│   ├── asymmetric_ciphers.py # RSA, ECC
│   ├── multilang_crypto.py   # Multi-language router
│   ├── hash_functions.py     # SHA, BLAKE, etc.
│   ├── mac_functions.py      # HMAC, CMAC, etc.
│   ├── kdf_functions.py      # PBKDF2, Argon2, etc.
│   ├── state_management.py   # Cipher state handling
│   └── pqc_ciphers.py        # Post-quantum algorithms
│
├── attacks/                   # Attack implementations
│   ├── __init__.py
│   ├── base_attack.py        # Abstract attack base
│   ├── brute_force/          # Brute force attacks
│   ├── statistical/          # Statistical attacks
│   ├── cryptanalysis/        # Cryptanalytic attacks
│   ├── algebraic/            # Algebraic attacks
│   ├── side_channel/         # Side-channel attacks
│   └── attack_executor_v2.py # Attack execution engine
│
├── orchestrator/              # Execution orchestration
│   ├── __init__.py
│   └── attack_orchestrator.py
│
└── analysis/                  # Data analysis
    ├── __init__.py
    └── validator.py
```

---

## Design Patterns Used

### 1. Factory Method Pattern

**Where**: `cipher_factory.py`  
**Purpose**: Create cipher objects without exposing instantiation logic

```python
cipher = create_cipher("AES-256-GCM")  # Factory hides complexity
```

---

### 2. Strategy Pattern

**Where**: `multilang_crypto.py`  
**Purpose**: Switch between implementation languages at runtime

```python
crypto = MultiLangCrypto('AES-256-GCM', language='python')  # Strategy
crypto = MultiLangCrypto('AES-256-GCM', language='c')       # Different strategy
```

---

### 3. Template Method Pattern

**Where**: `base_attack.py`, `base_cipher.py`  
**Purpose**: Define algorithm skeleton, let subclasses fill in details

```python
class BaseCipherSystem(ABC):
    def encrypt(self, plaintext):
        # Template steps
        self._validate_key()
        data = self._prepare_plaintext(plaintext)
        ciphertext = self._do_encrypt(data)  # Subclass implements
        return self._package_result(ciphertext)
```

---

### 4. Observer Pattern

**Where**: Attack orchestrator  
**Purpose**: Notify on progress updates

```python
# Batch logging uses observer pattern
for encryption in encryptions:
    result = run_attacks(encryption)
    notify_observers(f"90 attacks for encryption {i} completed")
```

---

### 5. Registry Pattern

**Where**: Algorithm registry  
**Purpose**: Centralized algorithm management

```python
ALGORITHM_REGISTRY = {
    "AES-256-GCM": (AESCipherSystem, {...}),
    # ... 50 more
}
```

---

### 6. Checkpoint Pattern

**Where**: Attack generation  
**Purpose**: Resume from interruption

```python
checkpoint = load_checkpoint()
for i in range(checkpoint['last_processed'] + 1, total):
    process(i)
    if i % 10 == 0:
        save_checkpoint({'last_processed': i})
```

---

### 7. Retry Pattern

**Where**: `run_complete_pipeline.py`  
**Purpose**: Gracefully handle transient failures

```python
@retry(max_attempts=3, delay=5)
def execute_phase():
    # Phase logic
    pass
```

---

## Security Architecture

### 1. Cryptographic Key Management

- **Key Generation**: Uses cryptographically secure RNG (`os.urandom`)
- **Key Storage**: In-memory only, never persisted
- **Key Lifecycle**: Generated → Used → Discarded

### 2. Input Validation

- Algorithm names validated against registry
- File paths sanitized
- Configuration parameters range-checked

### 3. Error Handling

- Exceptions caught and logged
- Sensitive data excluded from logs
- Graceful degradation (fallback to Python implementation)

---

## Scalability

### Horizontal Scaling

```python
# Multi-worker attack execution
python generate_attack_dataset.py --workers 8
```

- Process-based parallelism
- Independent worker processes
- Shared-nothing architecture

### Vertical Scaling

- Checkpoint resume enables batch processing
- Configurable memory usage
- Stream processing for large datasets

---

**End of Architecture Document**  
Generated by AIRAWAT v1.0
