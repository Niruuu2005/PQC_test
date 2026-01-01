"""
Digital Signature Systems Module

This module provides implementations of digital signature algorithms.

Algorithms Implemented:
- DSA (1024, 2048-bit)
- ECDSA (P-256, P-384)
- Ed25519, Ed448 (EdDSA)
- RSA-PSS (2048, 3072-bit)

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

# Check for cryptography library
try:
    from cryptography.hazmat.primitives.asymmetric import dsa, rsa, ec, ed25519, ed448, padding
    from cryptography.hazmat.primitives import hashes, serialization
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.error("cryptography library required for signatures")


class BaseSignatureSystem(BaseCipherSystem):
    """Base class for digital signature systems."""
    
    def __init__(self, algorithm: str, key_size: int, seed: Optional[int] = None):
        """Initialize signature system."""
        super().__init__(algorithm_name=algorithm, key_size_bits=key_size, seed=seed)
        self.algorithm = algorithm
        self.private_key = None
        self.public_key = None
    
    @abstractmethod
    def generate_keypair(self):
        """Generate public/private key pair."""
        pass
    
    @abstractmethod
    def sign(self, data: bytes) -> bytes:
        """Sign data and return signature."""
        pass
    
    @abstractmethod
    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify signature."""
        pass
    
    def generate_key(self) -> bytes:
        """Generate key pair and return public key."""
        self.generate_keypair()
        return self.cipher_state.key
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """'Encrypt' means sign the data."""
        if not self.private_key:
            self.generate_keypair()
        
        signature = self.sign(plaintext)
        
        metadata = EncryptionMetadata(
            algorithm=self.algorithm,
            plaintext_hash=compute_plaintext_hash(plaintext),
            plaintext_length=len(plaintext),
            key_size_bits=self.key_size_bits,
            ciphertext_length=len(signature),
            encryption_time_ms=0.0,
            success=True,
            tag=signature.hex()
        )
        
        return signature, metadata
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        """'Decrypt' means verify the signature."""
        data = kwargs.get('data', b'')
        is_valid = self.verify(data, ciphertext)
        
        metadata = DecryptionMetadata(
            success=is_valid,
            decryption_time_ms=0.0,
            recovered_size=len(data) if is_valid else 0,
            error_message=None if is_valid else "Signature verification failed"
        )
        
        return data if is_valid else b'', metadata


# ============================================================================
# DSA (Digital Signature Algorithm)
# ============================================================================

class DSASignatureSystem(BaseSignatureSystem):
    """DSA digital signature system."""
    
    def __init__(self, key_size: int = 2048, seed: Optional[int] = None):
        """Initialize DSA system with specified key size."""
        super().__init__(f"DSA-{key_size}", key_size, seed)
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("DSA requires cryptography library")
    
    def generate_keypair(self):
        """Generate DSA key pair."""
        self.private_key = dsa.generate_private_key(key_size=self.key_size_bits)
        self.public_key = self.private_key.public_key()
        
        # Store public key bytes
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self.cipher_state.key = public_bytes[:64]  # Truncate for storage
    
    def sign(self, data: bytes) -> bytes:
        """Sign data using DSA."""
        signature = self.private_key.sign(data, hashes.SHA256())
        return signature
    
    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify DSA signature."""
        try:
            self.public_key.verify(signature, data, hashes.SHA256())
            return True
        except Exception:
            return False


# ============================================================================
# ECDSA (Elliptic Curve DSA)
# ============================================================================

class ECDSASignatureSystem(BaseSignatureSystem):
    """ECDSA digital signature system."""
    
    def __init__(self, curve: str = "P-256", seed: Optional[int] = None):
        """Initialize ECDSA with specified curve."""
        self.curve_name = curve
        self.curve_obj = self._get_curve(curve)
        key_size = self._get_curve_key_size(curve)
        super().__init__(f"ECDSA-{curve}", key_size, seed)
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("ECDSA requires cryptography library")
    
    def _get_curve(self, curve_name: str):
        """Get curve object by name."""
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
        """Generate ECDSA key pair."""
        self.private_key = ec.generate_private_key(self.curve_obj)
        self.public_key = self.private_key.public_key()
        
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self.cipher_state.key = public_bytes[:64]
    
    def sign(self, data: bytes) -> bytes:
        """Sign data using ECDSA."""
        signature = self.private_key.sign(data, ec.ECDSA(hashes.SHA256()))
        return signature
    
    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify ECDSA signature."""
        try:
            self.public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception:
            return False


# ============================================================================
# EdDSA (Edwards-curve Digital Signature Algorithm)
# ============================================================================

class Ed25519SignatureSystem(BaseSignatureSystem):
    """Ed25519 signature system (EdDSA on Curve25519)."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Ed25519", 256, seed)
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("Ed25519 requires cryptography library")
    
    def generate_keypair(self):
        """Generate Ed25519 key pair."""
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        self.cipher_state.key = public_bytes
    
    def sign(self, data: bytes) -> bytes:
        """Sign data using Ed25519."""
        return self.private_key.sign(data)
    
    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify Ed25519 signature."""
        try:
            self.public_key.verify(signature, data)
            return True
        except Exception:
            return False


class Ed448SignatureSystem(BaseSignatureSystem):
    """Ed448 signature system (EdDSA on Curve448)."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Ed448", 448, seed)
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("Ed448 requires cryptography library")
    
    def generate_keypair(self):
        """Generate Ed448 key pair."""
        self.private_key = ed448.Ed448PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        self.cipher_state.key = public_bytes[:64]  # Truncate for storage
    
    def sign(self, data: bytes) -> bytes:
        """Sign data using Ed448."""
        return self.private_key.sign(data)
    
    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify Ed448 signature."""
        try:
            self.public_key.verify(signature, data)
            return True
        except Exception:
            return False


# ============================================================================
# RSA-PSS (RSA with Probabilistic Signature Scheme)
# ============================================================================

class RSAPSSSignatureSystem(BaseSignatureSystem):
    """RSA-PSS signature system."""
    
    def __init__(self, key_size: int = 2048, seed: Optional[int] = None):
        """Initialize RSA-PSS with specified key size."""
        super().__init__(f"RSA-{key_size}-PSS", key_size, seed)
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("RSA-PSS requires cryptography library")
    
    def generate_keypair(self):
        """Generate RSA key pair."""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size_bits
        )
        self.public_key = self.private_key.public_key()
        
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self.cipher_state.key = public_bytes[:64]  # Truncate for storage
    
    def sign(self, data: bytes) -> bytes:
        """Sign data using RSA-PSS."""
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify RSA-PSS signature."""
        try:
            self.public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


# ============================================================================
# SIGNATURE ALGORITHM REGISTRY
# ============================================================================

SIGNATURE_ALGORITHM_REGISTRY: Dict[str, type] = {
    # DSA
    "DSA-1024": lambda seed=None: DSASignatureSystem(1024, seed),
    "DSA-2048": lambda seed=None: DSASignatureSystem(2048, seed),
    
    # ECDSA
    "ECDSA-P-256": lambda seed=None: ECDSASignatureSystem("P-256", seed),
    "ECDSA-P-384": lambda seed=None: ECDSASignatureSystem("P-384", seed),
    "ECDSA-P-521": lambda seed=None: ECDSASignatureSystem("P-521", seed),
    
    # EdDSA
    "ED25519": Ed25519SignatureSystem,
    "Ed25519": Ed25519SignatureSystem,
    "ED448": Ed448SignatureSystem,
    "Ed448": Ed448SignatureSystem,
    
    # RSA-PSS
    "RSA-2048-PSS": lambda seed=None: RSAPSSSignatureSystem(2048, seed),
    "RSA-3072-PSS": lambda seed=None: RSAPSSSignatureSystem(3072, seed),
    "RSA-4096-PSS": lambda seed=None: RSAPSSSignatureSystem(4096, seed),
}


def create_signature_system(algorithm: str, seed: Optional[int] = None) -> BaseSignatureSystem:
    """Factory function to create signature system instances."""
    normalized_name = algorithm.upper().replace(" ", "").replace("_", "-")
    
    if normalized_name not in SIGNATURE_ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown signature algorithm: {algorithm}. "
            f"Available: {', '.join(sorted(SIGNATURE_ALGORITHM_REGISTRY.keys()))}"
        )
    
    sig_class = SIGNATURE_ALGORITHM_REGISTRY[normalized_name]
    
    try:
        if callable(sig_class):
            return sig_class(seed=seed)
        else:
            return sig_class(seed=seed)
    except ImportError as e:
        raise ImportError(f"Cannot create {algorithm}: {e}")


def get_available_signature_algorithms() -> list:
    """Get list of available signature algorithms."""
    available = []
    for algo_name in SIGNATURE_ALGORITHM_REGISTRY.keys():
        try:
            create_signature_system(algo_name)
            available.append(algo_name)
        except (ImportError, Exception):
            pass
    return sorted(set(available))

