"""
Cryptographic Dataset Generator - Crypto Subsystem

This module handles all cryptographic operations for the dataset generator.

Version: 3.1
Date: December 30, 2025
"""

__version__ = "3.1.0"

# Import base classes
from .base_cipher import (
    BaseCipherSystem,
    SymmetricBlockCipherSystem,
    SymmetricStreamCipherSystem,
    AsymmetricCipherSystem,
    PQCKEMSystem,
)

# Import data models
from .state_management import (
    EncryptionMetadata,
    DecryptionMetadata,
    KeyInfo,
    CipherState,
    compute_plaintext_hash,
    estimate_entropy,
)

# Import concrete cipher implementations
from .symmetric_ciphers import (
    AESCipherSystem,
    ChaCha20CipherSystem,
)

# Import factory functions
from .cipher_factory import (
    create_cipher,
    get_available_algorithms,
    get_algorithm_metadata,
    filter_algorithms,
    validate_algorithm,
)

# Import key generation utilities
from .key_generator import (
    SymmetricKeyGenerator,
    AsymmetricKeyGenerator,
    PQCKeyGenerator,
    generate_random_bytes,
    estimate_key_entropy,
    secure_erase_key,
    seed_rng,
)

# Export all public classes and functions
__all__ = [
    # Base classes
    "BaseCipherSystem",
    "SymmetricBlockCipherSystem",
    "SymmetricStreamCipherSystem",
    "AsymmetricCipherSystem",
    "PQCKEMSystem",
    # Data models
    "EncryptionMetadata",
    "DecryptionMetadata",
    "KeyInfo",
    "CipherState",
    "compute_plaintext_hash",
    "estimate_entropy",
    # Concrete ciphers
    "AESCipherSystem",
    "ChaCha20CipherSystem",
    # Factory functions
    "create_cipher",
    "get_available_algorithms",
    "get_algorithm_metadata",
    "filter_algorithms",
    "validate_algorithm",
    # Key generation
    "SymmetricKeyGenerator",
    "AsymmetricKeyGenerator",
    "PQCKeyGenerator",
    "generate_random_bytes",
    "estimate_key_entropy",
    "secure_erase_key",
    "seed_rng",
]

