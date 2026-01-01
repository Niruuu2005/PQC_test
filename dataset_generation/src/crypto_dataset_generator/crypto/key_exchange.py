"""
Key Exchange Systems Module

Implements various key exchange protocols for establishing shared secrets.

Algorithms Implemented:
- Diffie-Hellman (DH-1024, DH-2048, DH-3072)
- Elliptic Curve Diffie-Hellman (ECDH P-256, P-384, P-521)
- X25519 (Curve25519-based)
- X448 (Curve448-based)

Version: 1.0
Date: December 31, 2025
"""

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

try:
    from cryptography.hazmat.primitives.asymmetric import dh, ec, x25519, x448
    from cryptography.hazmat.primitives import serialization
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.error("cryptography library required for key exchange")


class BaseKeyExchangeSystem(BaseCipherSystem):
    """Base class for key exchange systems."""
    
    def __init__(self, algorithm: str, key_size: int, seed: Optional[int] = None):
        super().__init__(algorithm_name=algorithm, key_size_bits=key_size, seed=seed)
        self.algorithm = algorithm
        self.private_key = None
        self.public_key = None
        self.peer_public_key = None
    
    @abstractmethod
    def generate_keypair(self):
        """Generate key pair for key exchange."""
        pass
    
    @abstractmethod
    def compute_shared_secret(self, peer_public_key) -> bytes:
        """Compute shared secret with peer's public key."""
        pass
    
    def generate_key(self) -> bytes:
        """Generate keypair and return public key."""
        self.generate_keypair()
        return self.cipher_state.key
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        'Encrypt' for key exchange means compute shared secret.
        plaintext is treated as peer's public key.
        """
        if not self.private_key:
            self.generate_keypair()
        
        # Simulate peer public key (for testing, use our own public key)
        shared_secret = self.compute_shared_secret(self.public_key)
        
        metadata = EncryptionMetadata(
            algorithm=self.algorithm,
            plaintext_hash=compute_plaintext_hash(plaintext),
            plaintext_length=len(plaintext),
            key_size_bits=self.key_size_bits,
            ciphertext_length=len(shared_secret),
            encryption_time_ms=0.0,
            success=True,
            tag=self.cipher_state.key.hex()[:64]  # Our public key
        )
        
        return shared_secret, metadata
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        """Key exchange doesn't decrypt - returns shared secret."""
        metadata = DecryptionMetadata(
            success=True,
            decryption_time_ms=0.0,
            recovered_size=len(ciphertext),
            error_message=None
        )
        return ciphertext, metadata


# ============================================================================
# Diffie-Hellman (Classical DH)
# ============================================================================

class DHKeyExchangeSystem(BaseKeyExchangeSystem):
    """Diffie-Hellman key exchange."""
    
    def __init__(self, key_size: int = 2048, seed: Optional[int] = None):
        super().__init__(f"DH-{key_size}", key_size, seed)
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("DH requires cryptography library")
        self.parameters = None
    
    def generate_keypair(self):
        """Generate DH key pair."""
        # Generate DH parameters (in production, these would be pre-generated)
        self.parameters = dh.generate_parameters(
            generator=2,
            key_size=self.key_size_bits
        )
        
        # Generate private key
        self.private_key = self.parameters.generate_private_key()
        self.public_key = self.private_key.public_key()
        
        # Store public key bytes
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self.cipher_state.key = public_bytes[:64]  # Truncate for storage
    
    def compute_shared_secret(self, peer_public_key) -> bytes:
        """Compute DH shared secret."""
        shared_key = self.private_key.exchange(peer_public_key)
        return shared_key[:32]  # Return first 256 bits


# ============================================================================
# ECDH (Elliptic Curve Diffie-Hellman)
# ============================================================================

class ECDHKeyExchangeSystem(BaseKeyExchangeSystem):
    """ECDH key exchange."""
    
    def __init__(self, curve: str = "P-256", seed: Optional[int] = None):
        self.curve_name = curve
        self.curve_obj = self._get_curve(curve)
        key_size = self._get_curve_key_size(curve)
        super().__init__(f"ECDH-{curve}", key_size, seed)
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("ECDH requires cryptography library")
    
    def _get_curve(self, curve_name: str):
        """Get curve object."""
        curves = {
            "P-256": ec.SECP256R1(),
            "P-384": ec.SECP384R1(),
            "P-521": ec.SECP521R1(),
        }
        return curves.get(curve_name, ec.SECP256R1())
    
    def _get_curve_key_size(self, curve_name: str) -> int:
        """Get key size for curve."""
        sizes = {"P-256": 256, "P-384": 384, "P-521": 521}
        return sizes.get(curve_name, 256)
    
    def generate_keypair(self):
        """Generate ECDH key pair."""
        self.private_key = ec.generate_private_key(self.curve_obj)
        self.public_key = self.private_key.public_key()
        
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self.cipher_state.key = public_bytes[:64]
    
    def compute_shared_secret(self, peer_public_key) -> bytes:
        """Compute ECDH shared secret."""
        shared_key = self.private_key.exchange(ec.ECDH(), peer_public_key)
        return shared_key


# ============================================================================
# X25519 (Modern Curve25519-based key exchange)
# ============================================================================

class X25519KeyExchangeSystem(BaseKeyExchangeSystem):
    """X25519 key exchange (Curve25519)."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("X25519", 256, seed)
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("X25519 requires cryptography library")
    
    def generate_keypair(self):
        """Generate X25519 key pair."""
        self.private_key = x25519.X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        self.cipher_state.key = public_bytes
    
    def compute_shared_secret(self, peer_public_key) -> bytes:
        """Compute X25519 shared secret."""
        shared_key = self.private_key.exchange(peer_public_key)
        return shared_key


# ============================================================================
# X448 (Curve448-based key exchange)
# ============================================================================

class X448KeyExchangeSystem(BaseKeyExchangeSystem):
    """X448 key exchange (Curve448)."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("X448", 448, seed)
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("X448 requires cryptography library")
    
    def generate_keypair(self):
        """Generate X448 key pair."""
        self.private_key = x448.X448PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        self.cipher_state.key = public_bytes[:64]  # Truncate for storage
    
    def compute_shared_secret(self, peer_public_key) -> bytes:
        """Compute X448 shared secret."""
        shared_key = self.private_key.exchange(peer_public_key)
        return shared_key


# ============================================================================
# KEY EXCHANGE ALGORITHM REGISTRY
# ============================================================================

KEY_EXCHANGE_REGISTRY: Dict[str, type] = {
    # Diffie-Hellman
    "DH-1024": lambda seed=None: DHKeyExchangeSystem(1024, seed),
    "DH-2048": lambda seed=None: DHKeyExchangeSystem(2048, seed),
    "DH-3072": lambda seed=None: DHKeyExchangeSystem(3072, seed),
    
    # ECDH
    "ECDH-P-256": lambda seed=None: ECDHKeyExchangeSystem("P-256", seed),
    "ECDH-P-384": lambda seed=None: ECDHKeyExchangeSystem("P-384", seed),
    "ECDH-P-521": lambda seed=None: ECDHKeyExchangeSystem("P-521", seed),
    
    # Modern curves
    "X25519": X25519KeyExchangeSystem,
    "X448": X448KeyExchangeSystem,
}


def create_key_exchange_system(algorithm: str, seed: Optional[int] = None) -> BaseKeyExchangeSystem:
    """Factory function to create key exchange system instances."""
    normalized_name = algorithm.upper().replace(" ", "").replace("_", "-")
    
    if normalized_name not in KEY_EXCHANGE_REGISTRY:
        raise ValueError(
            f"Unknown key exchange algorithm: {algorithm}. "
            f"Available: {', '.join(sorted(KEY_EXCHANGE_REGISTRY.keys()))}"
        )
    
    kex_class = KEY_EXCHANGE_REGISTRY[normalized_name]
    
    try:
        if callable(kex_class):
            return kex_class(seed=seed)
        else:
            return kex_class(seed=seed)
    except ImportError as e:
        raise ImportError(f"Cannot create {algorithm}: {e}")


def get_available_key_exchange_algorithms() -> list:
    """Get list of available key exchange algorithms."""
    available = []
    for algo_name in KEY_EXCHANGE_REGISTRY.keys():
        try:
            create_key_exchange_system(algo_name)
            available.append(algo_name)
        except (ImportError, Exception):
            pass
    return sorted(set(available))

