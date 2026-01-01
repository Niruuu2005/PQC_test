"""
Hash Functions Module - Comprehensive Hash Algorithm Implementations

This module provides implementations of all major cryptographic hash functions
including MD family, SHA family, BLAKE family, and others.

Algorithms Implemented:
- MD2, MD4, MD5
- SHA-1, SHA-224, SHA-256, SHA-384, SHA-512
- SHA3-224, SHA3-256, SHA3-384, SHA3-512
- RIPEMD-160
- Whirlpool
- BLAKE2s, BLAKE2b
- BLAKE3

Version: 1.0
Date: December 31, 2025
"""

import hashlib
import logging
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
    from Crypto.Hash import MD2, MD4, RIPEMD160, whirlpool
    PYCRYPTODOME_HASH_AVAILABLE = True
except ImportError:
    PYCRYPTODOME_HASH_AVAILABLE = False
    logger.warning("PyCryptodome not available - MD2, MD4, RIPEMD-160, Whirlpool will not be available")

try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False
    logger.warning("blake3 not available - BLAKE3 will not be available")


class BaseHashSystem(BaseCipherSystem):
    """
    Base class for hash function systems.
    
    Hash functions are one-way functions that produce a fixed-size output
    from variable-size input. They don't use keys or encryption/decryption.
    """
    
    def __init__(self, hash_algorithm: str, output_size: int, seed: Optional[int] = None):
        """
        Initialize hash system.
        
        Args:
            hash_algorithm: Name of hash algorithm
            output_size: Size of hash output in bits
            seed: Optional seed for deterministic behavior
        """
        super().__init__(algorithm_name=hash_algorithm, key_size_bits=output_size, seed=seed)
        self.hash_algorithm = hash_algorithm
        self.output_size = output_size
    
    def generate_key(self) -> bytes:
        """Hash functions don't use keys."""
        return b''
    
    @abstractmethod
    def hash_data(self, data: bytes) -> bytes:
        """Compute hash of data."""
        pass
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        'Encrypt' for hash functions means compute hash.
        
        Args:
            plaintext: Data to hash
            
        Returns:
            Tuple of (hash_output, metadata)
        """
        hash_output = self.hash_data(plaintext)
        
        metadata = EncryptionMetadata(
            algorithm=self.hash_algorithm,
            plaintext_hash=compute_plaintext_hash(plaintext),
            plaintext_length=len(plaintext),
            key_size_bits=self.output_size,
            ciphertext_length=len(hash_output),
            encryption_time_ms=0.0,
            success=True,
            tag=hash_output.hex()[:64]  # Store hash in tag field
        )
        
        return hash_output, metadata
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        """
        Hash functions are one-way - cannot decrypt.
        Returns empty plaintext to maintain interface compatibility.
        """
        metadata = DecryptionMetadata(
            success=False,
            decryption_time_ms=0.0,
            recovered_size=0,
            error_message="Hash functions are one-way"
        )
        return b'', metadata


# ============================================================================
# MD FAMILY (Legacy - avoid in production)
# ============================================================================

class MD5HashSystem(BaseHashSystem):
    """MD5 hash function (128-bit output) - AVOID: Cryptographically broken"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("MD5", 128, seed)
    
    def hash_data(self, data: bytes) -> bytes:
        return hashlib.md5(data).digest()


class MD2HashSystem(BaseHashSystem):
    """MD2 hash function (128-bit output) - AVOID: Obsolete"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("MD2", 128, seed)
        if not PYCRYPTODOME_HASH_AVAILABLE:
            raise ImportError("MD2 requires PyCryptodome")
    
    def hash_data(self, data: bytes) -> bytes:
        return MD2.new(data).digest()


class MD4HashSystem(BaseHashSystem):
    """MD4 hash function (128-bit output) - AVOID: Cryptographically broken"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("MD4", 128, seed)
        if not PYCRYPTODOME_HASH_AVAILABLE:
            raise ImportError("MD4 requires PyCryptodome")
    
    def hash_data(self, data: bytes) -> bytes:
        return MD4.new(data).digest()


# ============================================================================
# SHA-1 (Legacy - phase out)
# ============================================================================

class SHA1HashSystem(BaseHashSystem):
    """SHA-1 hash function (160-bit output) - AVOID: Collision attacks exist"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("SHA-1", 160, seed)
    
    def hash_data(self, data: bytes) -> bytes:
        return hashlib.sha1(data).digest()


# ============================================================================
# SHA-2 FAMILY (Modern, widely used)
# ============================================================================

class SHA224HashSystem(BaseHashSystem):
    """SHA-224 hash function (224-bit output)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("SHA-224", 224, seed)
    
    def hash_data(self, data: bytes) -> bytes:
        return hashlib.sha224(data).digest()


class SHA256HashSystem(BaseHashSystem):
    """SHA-256 hash function (256-bit output) - RECOMMENDED"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("SHA-256", 256, seed)
    
    def hash_data(self, data: bytes) -> bytes:
        return hashlib.sha256(data).digest()


class SHA384HashSystem(BaseHashSystem):
    """SHA-384 hash function (384-bit output)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("SHA-384", 384, seed)
    
    def hash_data(self, data: bytes) -> bytes:
        return hashlib.sha384(data).digest()


class SHA512HashSystem(BaseHashSystem):
    """SHA-512 hash function (512-bit output) - RECOMMENDED"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("SHA-512", 512, seed)
    
    def hash_data(self, data: bytes) -> bytes:
        return hashlib.sha512(data).digest()


# ============================================================================
# SHA-3 FAMILY (Modern, standardized)
# ============================================================================

class SHA3_224HashSystem(BaseHashSystem):
    """SHA3-224 hash function (224-bit output)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("SHA3-224", 224, seed)
    
    def hash_data(self, data: bytes) -> bytes:
        return hashlib.sha3_224(data).digest()


class SHA3_256HashSystem(BaseHashSystem):
    """SHA3-256 hash function (256-bit output)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("SHA3-256", 256, seed)
    
    def hash_data(self, data: bytes) -> bytes:
        return hashlib.sha3_256(data).digest()


class SHA3_384HashSystem(BaseHashSystem):
    """SHA3-384 hash function (384-bit output)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("SHA3-384", 384, seed)
    
    def hash_data(self, data: bytes) -> bytes:
        return hashlib.sha3_384(data).digest()


class SHA3_512HashSystem(BaseHashSystem):
    """SHA3-512 hash function (512-bit output)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("SHA3-512", 512, seed)
    
    def hash_data(self, data: bytes) -> bytes:
        return hashlib.sha3_512(data).digest()


# ============================================================================
# RIPEMD (Alternative hash function)
# ============================================================================

class RIPEMD160HashSystem(BaseHashSystem):
    """RIPEMD-160 hash function (160-bit output)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("RIPEMD-160", 160, seed)
        if not PYCRYPTODOME_HASH_AVAILABLE:
            raise ImportError("RIPEMD-160 requires PyCryptodome")
    
    def hash_data(self, data: bytes) -> bytes:
        return RIPEMD160.new(data).digest()


# ============================================================================
# WHIRLPOOL (512-bit hash)
# ============================================================================

class WhirlpoolHashSystem(BaseHashSystem):
    """Whirlpool hash function (512-bit output)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Whirlpool", 512, seed)
        if not PYCRYPTODOME_HASH_AVAILABLE:
            raise ImportError("Whirlpool requires PyCryptodome")
    
    def hash_data(self, data: bytes) -> bytes:
        return whirlpool.new(data).digest()


# ============================================================================
# BLAKE2 FAMILY (Fast, secure)
# ============================================================================

class BLAKE2sHashSystem(BaseHashSystem):
    """BLAKE2s hash function (256-bit output) - Fast, secure"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("BLAKE2s", 256, seed)
    
    def hash_data(self, data: bytes) -> bytes:
        return hashlib.blake2s(data).digest()


class BLAKE2bHashSystem(BaseHashSystem):
    """BLAKE2b hash function (512-bit output) - Fast, secure"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("BLAKE2b", 512, seed)
    
    def hash_data(self, data: bytes) -> bytes:
        return hashlib.blake2b(data).digest()


# ============================================================================
# BLAKE3 (Next-generation, very fast)
# ============================================================================

class BLAKE3HashSystem(BaseHashSystem):
    """BLAKE3 hash function (256-bit output) - RECOMMENDED: Very fast, very secure"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("BLAKE3", 256, seed)
        if not BLAKE3_AVAILABLE:
            raise ImportError("BLAKE3 requires blake3 library: pip install blake3")
    
    def hash_data(self, data: bytes) -> bytes:
        return blake3.blake3(data).digest()


# ============================================================================
# HASH ALGORITHM REGISTRY
# ============================================================================

HASH_ALGORITHM_REGISTRY: Dict[str, type] = {
    # MD family
    "MD2": MD2HashSystem,
    "MD4": MD4HashSystem,
    "MD5": MD5HashSystem,
    
    # SHA-1
    "SHA-1": SHA1HashSystem,
    "SHA1": SHA1HashSystem,
    
    # SHA-2 family
    "SHA-224": SHA224HashSystem,
    "SHA-256": SHA256HashSystem,
    "SHA-384": SHA384HashSystem,
    "SHA-512": SHA512HashSystem,
    "SHA224": SHA224HashSystem,
    "SHA256": SHA256HashSystem,
    "SHA384": SHA384HashSystem,
    "SHA512": SHA512HashSystem,
    
    # SHA-3 family
    "SHA3-224": SHA3_224HashSystem,
    "SHA3-256": SHA3_256HashSystem,
    "SHA3-384": SHA3_384HashSystem,
    "SHA3-512": SHA3_512HashSystem,
    
    # Other hash functions
    "RIPEMD-160": RIPEMD160HashSystem,
    "RIPEMD160": RIPEMD160HashSystem,
    "WHIRLPOOL": WhirlpoolHashSystem,
    "Whirlpool": WhirlpoolHashSystem,
    
    # BLAKE family
    "BLAKE2s": BLAKE2sHashSystem,
    "BLAKE2b": BLAKE2bHashSystem,
    "BLAKE3": BLAKE3HashSystem,
}


def create_hash_system(algorithm: str, seed: Optional[int] = None) -> BaseHashSystem:
    """
    Factory function to create hash system instances.
    
    Args:
        algorithm: Name of hash algorithm
        seed: Optional seed for deterministic behavior
        
    Returns:
        Instance of appropriate hash system
        
    Raises:
        ValueError: If algorithm is not recognized
        ImportError: If required library is not installed
        
    Examples:
        >>> hasher = create_hash_system("SHA-256")
        >>> hash_output, metadata = hasher.encrypt(b"Hello, World!")
        >>> print(hash_output.hex()[:32])
    """
    normalized_name = algorithm.upper().replace(" ", "").replace("_", "-")
    
    if normalized_name not in HASH_ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown hash algorithm: {algorithm}. "
            f"Available: {', '.join(sorted(set(HASH_ALGORITHM_REGISTRY.keys())))}"
        )
    
    hash_class = HASH_ALGORITHM_REGISTRY[normalized_name]
    
    try:
        return hash_class(seed=seed)
    except ImportError as e:
        raise ImportError(f"Cannot create {algorithm}: {e}")


def get_available_hash_algorithms() -> list:
    """Get list of available hash algorithms."""
    available = []
    for algo_name, hash_class in HASH_ALGORITHM_REGISTRY.items():
        try:
            # Try to instantiate to check if dependencies are available
            hash_class()
            if algo_name not in available:  # Avoid duplicates from aliases
                available.append(algo_name)
        except (ImportError, Exception):
            pass
    return sorted(set(available))

