"""
MAC (Message Authentication Code) Functions Module

This module provides implementations of various MAC algorithms for data authentication.

Algorithms Implemented:
- HMAC (with SHA-256, SHA-512)
- CMAC (AES-based)
- GMAC (GCM-based)
- KMAC (Keccak-based, 128 and 256)
- Poly1305

Version: 1.0
Date: December 31, 2025
"""

import hmac
import hashlib
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
    from Crypto.Hash import CMAC
    from Crypto.Cipher import AES
    PYCRYPTODOME_MAC_AVAILABLE = True
except ImportError:
    PYCRYPTODOME_MAC_AVAILABLE = False
    logger.warning("PyCryptodome not available - CMAC, GMAC will not be available")

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.poly1305 import Poly1305
    CRYPTOGRAPHY_MAC_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_MAC_AVAILABLE = False
    logger.warning("cryptography library not available - some MACs may not work")


class BaseMACSystem(BaseCipherSystem):
    """
    Base class for MAC (Message Authentication Code) systems.
    
    MACs provide data authentication using a shared secret key.
    """
    
    def __init__(self, mac_algorithm: str, key_size: int, seed: Optional[int] = None):
        """
        Initialize MAC system.
        
        Args:
            mac_algorithm: Name of MAC algorithm
            key_size: Size of MAC key in bits
            seed: Optional seed for deterministic key generation
        """
        super().__init__(algorithm_name=mac_algorithm, key_size_bits=key_size, seed=seed)
        self.mac_algorithm = mac_algorithm
        self.key_size_bytes = key_size // 8
    
    def generate_key(self) -> bytes:
        """Generate random MAC key."""
        if self.cipher_state.seed is not None:
            # Deterministic key generation
            import numpy as np
            rng = np.random.RandomState(self.cipher_state.seed)
            key = rng.bytes(self.key_size_bytes)
        else:
            # Secure random key generation
            key = secrets.token_bytes(self.key_size_bytes)
        
        self.cipher_state.key = key
        return key
    
    @abstractmethod
    def compute_mac(self, data: bytes, key: bytes) -> bytes:
        """Compute MAC tag for data."""
        pass
    
    @abstractmethod
    def verify_mac(self, data: bytes, mac_tag: bytes, key: bytes) -> bool:
        """Verify MAC tag for data."""
        pass
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        'Encrypt' for MAC means compute authentication tag.
        Returns: (MAC tag, metadata)
        """
        if not self.cipher_state.key:
            self.generate_key()
        
        mac_tag = self.compute_mac(plaintext, self.cipher_state.key)
        
        metadata = EncryptionMetadata(
            algorithm=self.mac_algorithm,
            plaintext_hash=compute_plaintext_hash(plaintext),
            plaintext_length=len(plaintext),
            key_size_bits=self.key_size_bits,
            ciphertext_length=len(mac_tag),
            encryption_time_ms=0.0,
            success=True,
            tag=mac_tag.hex()
        )
        
        return mac_tag, metadata
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        """
        'Decrypt' for MAC means verify tag.
        ciphertext is the MAC tag, kwargs should contain 'data' to verify.
        """
        data = kwargs.get('data', b'')
        is_valid = self.verify_mac(data, ciphertext, self.cipher_state.key)
        
        metadata = DecryptionMetadata(
            success=is_valid,
            decryption_time_ms=0.0,
            recovered_size=len(data) if is_valid else 0,
            error_message=None if is_valid else "MAC verification failed"
        )
        
        return data if is_valid else b'', metadata


# ============================================================================
# HMAC (Hash-based MAC)
# ============================================================================

class HMACSHA256System(BaseMACSystem):
    """HMAC with SHA-256 (256-bit key, 256-bit output)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("HMAC-SHA256", 256, seed)
    
    def compute_mac(self, data: bytes, key: bytes) -> bytes:
        return hmac.new(key, data, hashlib.sha256).digest()
    
    def verify_mac(self, data: bytes, mac_tag: bytes, key: bytes) -> bool:
        expected_mac = self.compute_mac(data, key)
        return hmac.compare_digest(expected_mac, mac_tag)


class HMACSHA512System(BaseMACSystem):
    """HMAC with SHA-512 (512-bit key, 512-bit output)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("HMAC-SHA512", 512, seed)
    
    def compute_mac(self, data: bytes, key: bytes) -> bytes:
        return hmac.new(key, data, hashlib.sha512).digest()
    
    def verify_mac(self, data: bytes, mac_tag: bytes, key: bytes) -> bool:
        expected_mac = self.compute_mac(data, key)
        return hmac.compare_digest(expected_mac, mac_tag)


# ============================================================================
# CMAC (Cipher-based MAC using AES)
# ============================================================================

class CMACAESSystem(BaseMACSystem):
    """CMAC with AES (128-bit key, 128-bit output)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("CMAC-AES", 128, seed)
        if not PYCRYPTODOME_MAC_AVAILABLE:
            raise ImportError("CMAC-AES requires PyCryptodome")
    
    def compute_mac(self, data: bytes, key: bytes) -> bytes:
        cipher = CMAC.new(key, ciphermod=AES)
        cipher.update(data)
        return cipher.digest()
    
    def verify_mac(self, data: bytes, mac_tag: bytes, key: bytes) -> bool:
        try:
            cipher = CMAC.new(key, ciphermod=AES)
            cipher.update(data)
            cipher.verify(mac_tag)
            return True
        except ValueError:
            return False


# ============================================================================
# GMAC (GCM-based MAC)
# ============================================================================

class GMACSystem(BaseMACSystem):
    """GMAC (GCM-based MAC, 128-bit key)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("GMAC", 128, seed)
        if not CRYPTOGRAPHY_MAC_AVAILABLE:
            raise ImportError("GMAC requires cryptography library")
    
    def compute_mac(self, data: bytes, key: bytes) -> bytes:
        """
        GMAC is essentially GCM mode with empty plaintext.
        Returns the authentication tag.
        """
        # Use AES-GCM with empty plaintext
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        aesgcm = AESGCM(key)
        
        # Encrypt empty data with AAD=data, get tag
        ciphertext = aesgcm.encrypt(nonce, b'', data)
        # The last 16 bytes are the authentication tag
        tag = ciphertext[-16:]
        
        # Store nonce for verification
        self._last_nonce = nonce
        return tag
    
    def verify_mac(self, data: bytes, mac_tag: bytes, key: bytes) -> bool:
        """Verify GMAC tag."""
        if not hasattr(self, '_last_nonce'):
            return False
        
        try:
            aesgcm = AESGCM(key)
            # Try to decrypt empty ciphertext with the tag
            aesgcm.decrypt(self._last_nonce, b'' + mac_tag, data)
            return True
        except Exception:
            return False


# ============================================================================
# KMAC (Keccak-based MAC, SHA-3 derived)
# ============================================================================

class KMAC128System(BaseMACSystem):
    """KMAC128 (Keccak-based MAC, 128-bit security)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("KMAC128", 128, seed)
    
    def compute_mac(self, data: bytes, key: bytes) -> bytes:
        """
        KMAC128 using SHAKE128 (closest approximation in standard library).
        True KMAC requires full Keccak implementation.
        """
        # Simplified KMAC using SHAKE128
        # Format: SHAKE128(key || data)
        shake = hashlib.shake_128()
        shake.update(key)
        shake.update(data)
        return shake.digest(16)  # 128-bit output
    
    def verify_mac(self, data: bytes, mac_tag: bytes, key: bytes) -> bool:
        expected_mac = self.compute_mac(data, key)
        return hmac.compare_digest(expected_mac, mac_tag)


class KMAC256System(BaseMACSystem):
    """KMAC256 (Keccak-based MAC, 256-bit security)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("KMAC256", 256, seed)
    
    def compute_mac(self, data: bytes, key: bytes) -> bytes:
        """
        KMAC256 using SHAKE256 (closest approximation in standard library).
        """
        shake = hashlib.shake_256()
        shake.update(key)
        shake.update(data)
        return shake.digest(32)  # 256-bit output
    
    def verify_mac(self, data: bytes, mac_tag: bytes, key: bytes) -> bool:
        expected_mac = self.compute_mac(data, key)
        return hmac.compare_digest(expected_mac, mac_tag)


# ============================================================================
# Poly1305 (Fast one-time MAC)
# ============================================================================

class Poly1305System(BaseMACSystem):
    """Poly1305 (Fast one-time MAC, 256-bit key, 128-bit output)"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Poly1305", 256, seed)
        if not CRYPTOGRAPHY_MAC_AVAILABLE:
            raise ImportError("Poly1305 requires cryptography library")
    
    def compute_mac(self, data: bytes, key: bytes) -> bytes:
        """Compute Poly1305 MAC tag."""
        # Poly1305 requires exactly 32-byte key
        if len(key) != 32:
            # Pad or truncate key
            key = (key + b'\x00' * 32)[:32]
        
        p = Poly1305.generate_tag(key, data)
        return p
    
    def verify_mac(self, data: bytes, mac_tag: bytes, key: bytes) -> bool:
        """Verify Poly1305 MAC tag."""
        try:
            if len(key) != 32:
                key = (key + b'\x00' * 32)[:32]
            Poly1305.verify_tag(key, data, mac_tag)
            return True
        except Exception:
            return False


# ============================================================================
# MAC ALGORITHM REGISTRY
# ============================================================================

MAC_ALGORITHM_REGISTRY: Dict[str, type] = {
    # HMAC variants
    "HMAC-SHA256": HMACSHA256System,
    "HMAC-SHA512": HMACSHA512System,
    
    # CMAC
    "CMAC-AES": CMACAESSystem,
    "CMAC": CMACAESSystem,
    
    # GMAC
    "GMAC": GMACSystem,
    
    # KMAC
    "KMAC128": KMAC128System,
    "KMAC256": KMAC256System,
    
    # Poly1305
    "POLY1305": Poly1305System,
    "Poly1305": Poly1305System,
}


def create_mac_system(algorithm: str, seed: Optional[int] = None) -> BaseMACSystem:
    """
    Factory function to create MAC system instances.
    
    Args:
        algorithm: Name of MAC algorithm
        seed: Optional seed for deterministic behavior
        
    Returns:
        Instance of appropriate MAC system
        
    Raises:
        ValueError: If algorithm is not recognized
        ImportError: If required library is not installed
    """
    normalized_name = algorithm.upper().replace(" ", "").replace("_", "-")
    
    if normalized_name not in MAC_ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown MAC algorithm: {algorithm}. "
            f"Available: {', '.join(sorted(MAC_ALGORITHM_REGISTRY.keys()))}"
        )
    
    mac_class = MAC_ALGORITHM_REGISTRY[normalized_name]
    
    try:
        return mac_class(seed=seed)
    except ImportError as e:
        raise ImportError(f"Cannot create {algorithm}: {e}")


def get_available_mac_algorithms() -> list:
    """Get list of available MAC algorithms."""
    available = []
    for algo_name, mac_class in MAC_ALGORITHM_REGISTRY.items():
        try:
            mac_class()
            if algo_name not in available:
                available.append(algo_name)
        except (ImportError, Exception):
            pass
    return sorted(set(available))

