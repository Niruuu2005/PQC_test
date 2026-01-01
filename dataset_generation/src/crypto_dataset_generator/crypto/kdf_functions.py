"""
KDF (Key Derivation Function) Module

This module provides implementations of various key derivation and password hashing functions.

Algorithms Implemented:
- PBKDF1 (legacy)
- PBKDF2 (10K and 100K+ iterations)
- bcrypt
- scrypt
- Argon2 (i, d, id variants)
- HKDF

Version: 1.0
Date: December 31, 2025
"""

import hashlib
import hmac
import logging
import secrets
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

from .base_cipher import BaseCipherSystem
from .state_management import (
    EncryptionMetadata,
    DecryptionMetadata,
    CipherState,
    compute_plaintext_hash
)

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from Crypto.Protocol.KDF import PBKDF1 as CryptoPBKDF1
    PYCRYPTODOME_KDF_AVAILABLE = True
except ImportError:
    PYCRYPTODOME_KDF_AVAILABLE = False
    logger.warning("PyCryptodome not available - PBKDF1 may not work")

try:
    import bcrypt as bcrypt_lib
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not available")

try:
    from argon2 import PasswordHasher, Type
    from argon2.low_level import hash_secret_raw
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False
    logger.warning("argon2-cffi not available")

try:
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    CRYPTOGRAPHY_KDF_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_KDF_AVAILABLE = False
    logger.warning("cryptography library not available - HKDF may not work")


class BaseKDFSystem(BaseCipherSystem):
    """
    Base class for Key Derivation Function (KDF) systems.
    
    KDFs derive cryptographic keys from passwords or other key material.
    """
    
    def __init__(self, kdf_algorithm: str, output_length: int, seed: Optional[int] = None):
        """
        Initialize KDF system.
        
        Args:
            kdf_algorithm: Name of KDF algorithm
            output_length: Length of derived key in bytes
            seed: Optional seed for deterministic behavior
        """
        super().__init__(algorithm_name=kdf_algorithm, key_size_bits=output_length * 8, seed=seed)
        self.kdf_algorithm = kdf_algorithm
        self.output_length = output_length
    
    def generate_key(self) -> bytes:
        """Generate random salt for KDF."""
        if self.cipher_state.seed is not None:
            # Deterministic salt generation
            import numpy as np
            rng = np.random.RandomState(self.cipher_state.seed)
            salt = rng.bytes(16)  # 128-bit salt
        else:
            salt = secrets.token_bytes(16)
        
        self.cipher_state.key = salt  # Store salt as "key"
        return salt
    
    @abstractmethod
    def derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive key from password and salt."""
        pass
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        'Encrypt' for KDF means derive key from password (plaintext).
        Returns: (derived_key, metadata)
        """
        if not self.cipher_state.key:
            self.generate_key()  # Generate salt
        
        derived_key = self.derive_key(plaintext, self.cipher_state.key)
        
        metadata = EncryptionMetadata(
            algorithm=self.kdf_algorithm,
            plaintext_hash=compute_plaintext_hash(plaintext),
            plaintext_length=len(plaintext),
            key_size_bits=self.output_length * 8,
            ciphertext_length=len(derived_key),
            encryption_time_ms=0.0,
            success=True,
            nonce=self.cipher_state.key.hex()  # Salt in nonce field
        )
        
        return derived_key, metadata
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        """
        KDFs are one-way functions - cannot reverse.
        Returns empty plaintext to maintain interface compatibility.
        """
        metadata = DecryptionMetadata(
            success=False,
            decryption_time_ms=0.0,
            recovered_size=0,
            error_message="KDFs are one-way functions"
        )
        return b'', metadata


# ============================================================================
# PBKDF1 (Legacy - RFC 2898)
# ============================================================================

class PBKDF1System(BaseKDFSystem):
    """PBKDF1 - Legacy password-based KDF (AVOID: Use PBKDF2 instead)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("PBKDF1", 16, seed)  # 128-bit output
        if not PYCRYPTODOME_KDF_AVAILABLE:
            raise ImportError("PBKDF1 requires PyCryptodome")
    
    def derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive key using PBKDF1 with 1000 iterations."""
        return CryptoPBKDF1(password, salt, 16, count=1000, hashAlgo=None)


# ============================================================================
# PBKDF2 (Standard password-based KDF - RFC 2898)
# ============================================================================

class PBKDF2_10KSystem(BaseKDFSystem):
    """PBKDF2 with 10,000 iterations (minimum recommended)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("PBKDF2-10K", 32, seed)  # 256-bit output
        self.iterations = 10000
    
    def derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive key using PBKDF2-HMAC-SHA256."""
        return hashlib.pbkdf2_hmac('sha256', password, salt, self.iterations, dklen=32)


class PBKDF2_100KSystem(BaseKDFSystem):
    """PBKDF2 with 100,000+ iterations (recommended for high security)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("PBKDF2-100K+", 32, seed)  # 256-bit output
        self.iterations = 100000
    
    def derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive key using PBKDF2-HMAC-SHA256 with 100K iterations."""
        return hashlib.pbkdf2_hmac('sha256', password, salt, self.iterations, dklen=32)


# ============================================================================
# bcrypt (Password hashing - Blowfish-based)
# ============================================================================

class BcryptSystem(BaseKDFSystem):
    """bcrypt - Adaptive password hashing function"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("bcrypt", 24, seed)  # bcrypt outputs 24 bytes (truncated to fit)
        if not BCRYPT_AVAILABLE:
            raise ImportError("bcrypt requires bcrypt library: pip install bcrypt")
        self.cost_factor = 12  # 2^12 iterations
    
    def derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive key using bcrypt."""
        # bcrypt requires 16-byte salt
        salt_16 = (salt + b'\x00' * 16)[:16]
        
        # bcrypt generates its own salt, but we can extract the hash
        # For consistency, we'll use bcrypt.kdf
        try:
            derived = bcrypt_lib.kdf(
                password=password,
                salt=salt_16,
                desired_key_bytes=32,
                rounds=100  # bcrypt KDF rounds
            )
            return derived[:24]  # bcrypt traditional output is 24 bytes
        except AttributeError:
            # Fallback: use standard bcrypt hash
            # bcrypt.hashpw requires salt in specific format
            salt_formatted = bcrypt_lib.gensalt(rounds=self.cost_factor)
            hashed = bcrypt_lib.hashpw(password, salt_formatted)
            return hashed[:24]


# ============================================================================
# scrypt (Memory-hard KDF)
# ============================================================================

class ScryptSystem(BaseKDFSystem):
    """scrypt - Memory-hard password-based KDF"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("scrypt", 32, seed)  # 256-bit output
        # Moderate parameters: N=2^14, r=8, p=1
        self.n = 16384  # CPU/memory cost (2^14)
        self.r = 8      # Block size
        self.p = 1      # Parallelization
    
    def derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive key using scrypt."""
        return hashlib.scrypt(
            password=password,
            salt=salt,
            n=self.n,
            r=self.r,
            p=self.p,
            dklen=32
        )


# ============================================================================
# Argon2 (Modern password hashing - Winner of Password Hashing Competition)
# ============================================================================

class Argon2iSystem(BaseKDFSystem):
    """Argon2i - Data-independent variant (resistant to side-channel attacks)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Argon2i", 32, seed)  # 256-bit output
        if not ARGON2_AVAILABLE:
            raise ImportError("Argon2 requires argon2-cffi: pip install argon2-cffi")
    
    def derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive key using Argon2i."""
        # Ensure salt is 16 bytes
        salt_16 = (salt + b'\x00' * 16)[:16]
        
        return hash_secret_raw(
            secret=password,
            salt=salt_16,
            time_cost=2,       # Number of iterations
            memory_cost=65536, # Memory usage in KiB (64 MB)
            parallelism=4,     # Number of parallel threads
            hash_len=32,       # Output length
            type=Type.I        # Argon2i variant
        )


class Argon2dSystem(BaseKDFSystem):
    """Argon2d - Data-dependent variant (faster, but vulnerable to side-channels)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Argon2d", 32, seed)
        if not ARGON2_AVAILABLE:
            raise ImportError("Argon2 requires argon2-cffi: pip install argon2-cffi")
    
    def derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive key using Argon2d."""
        salt_16 = (salt + b'\x00' * 16)[:16]
        
        return hash_secret_raw(
            secret=password,
            salt=salt_16,
            time_cost=2,
            memory_cost=65536,
            parallelism=4,
            hash_len=32,
            type=Type.D        # Argon2d variant
        )


class Argon2idSystem(BaseKDFSystem):
    """Argon2id - Hybrid variant (RECOMMENDED: Best of both i and d)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Argon2id", 32, seed)
        if not ARGON2_AVAILABLE:
            raise ImportError("Argon2 requires argon2-cffi: pip install argon2-cffi")
    
    def derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive key using Argon2id (recommended variant)."""
        salt_16 = (salt + b'\x00' * 16)[:16]
        
        return hash_secret_raw(
            secret=password,
            salt=salt_16,
            time_cost=2,
            memory_cost=65536,
            parallelism=4,
            hash_len=32,
            type=Type.ID       # Argon2id variant
        )


# ============================================================================
# HKDF (HMAC-based Extract-and-Expand KDF - RFC 5869)
# ============================================================================

class HKDFSystem(BaseKDFSystem):
    """HKDF - HMAC-based Extract-and-Expand Key Derivation Function"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("HKDF", 32, seed)  # 256-bit output
        if not CRYPTOGRAPHY_KDF_AVAILABLE:
            raise ImportError("HKDF requires cryptography library")
    
    def derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive key using HKDF with SHA-256."""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b'airawat-kdf-context',
        )
        return hkdf.derive(password)


# ============================================================================
# KDF ALGORITHM REGISTRY
# ============================================================================

KDF_ALGORITHM_REGISTRY: Dict[str, type] = {
    # PBKDF family
    "PBKDF1": PBKDF1System,
    "PBKDF2-10K": PBKDF2_10KSystem,
    "PBKDF2-100K": PBKDF2_100KSystem,
    "PBKDF2-100K+": PBKDF2_100KSystem,
    
    # bcrypt
    "BCRYPT": BcryptSystem,
    "bcrypt": BcryptSystem,
    
    # scrypt
    "SCRYPT": ScryptSystem,
    "scrypt": ScryptSystem,
    
    # Argon2 variants
    "ARGON2I": Argon2iSystem,
    "ARGON2D": Argon2dSystem,
    "ARGON2ID": Argon2idSystem,
    "Argon2i": Argon2iSystem,
    "Argon2d": Argon2dSystem,
    "Argon2id": Argon2idSystem,
    
    # HKDF
    "HKDF": HKDFSystem,
}


def create_kdf_system(algorithm: str, seed: Optional[int] = None) -> BaseKDFSystem:
    """
    Factory function to create KDF system instances.
    
    Args:
        algorithm: Name of KDF algorithm
        seed: Optional seed for deterministic behavior
        
    Returns:
        Instance of appropriate KDF system
        
    Raises:
        ValueError: If algorithm is not recognized
        ImportError: If required library is not installed
    """
    normalized_name = algorithm.upper().replace(" ", "").replace("_", "-")
    
    if normalized_name not in KDF_ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown KDF algorithm: {algorithm}. "
            f"Available: {', '.join(sorted(set(KDF_ALGORITHM_REGISTRY.keys())))}"
        )
    
    kdf_class = KDF_ALGORITHM_REGISTRY[normalized_name]
    
    try:
        return kdf_class(seed=seed)
    except ImportError as e:
        raise ImportError(f"Cannot create {algorithm}: {e}")


def get_available_kdf_algorithms() -> list:
    """Get list of available KDF algorithms."""
    available = []
    for algo_name, kdf_class in KDF_ALGORITHM_REGISTRY.items():
        try:
            kdf_class()
            if algo_name not in available:
                available.append(algo_name)
        except (ImportError, Exception):
            pass
    return sorted(set(available))

