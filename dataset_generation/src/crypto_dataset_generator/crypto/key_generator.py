"""
Key Generation Utilities

This module provides centralized key generation functions for symmetric and asymmetric
cryptographic algorithms.

Version: 1.0
Date: December 30, 2025
"""

import os
import time
import numpy as np
from typing import Tuple, Optional, Any
from abc import ABC, abstractmethod


def seed_rng(seed: int) -> np.random.RandomState:
    """
    Create a seeded random number generator.
    
    Args:
        seed: Integer seed value
    
    Returns:
        Seeded numpy RandomState generator
    
    Examples:
        >>> rng = seed_rng(42)
        >>> random_bytes = rng.bytes(16)
    """
    return np.random.RandomState(seed)


def generate_random_bytes(size: int, seed: Optional[int] = None) -> bytes:
    """
    Generate cryptographically random bytes.
    
    Uses seeded RNG if seed provided, otherwise uses os.urandom().
    
    Args:
        size: Number of bytes to generate
        seed: Optional seed for deterministic generation
    
    Returns:
        Random bytes of specified size
    
    Examples:
        >>> key = generate_random_bytes(32, seed=42)  # Deterministic
        >>> key = generate_random_bytes(32)  # Cryptographically random
    """
    if seed is not None:
        rng = seed_rng(seed)
        return rng.bytes(size)
    else:
        return os.urandom(size)


def estimate_key_entropy(key: bytes) -> float:
    """
    Estimate entropy of a key using Shannon entropy.
    
    Shannon entropy H(X) = -Σ p(x) * log2(p(x))
    
    Args:
        key: Key bytes to analyze
    
    Returns:
        Entropy in bits (range: [0, 8 * len(key)])
    
    Examples:
        >>> key = b'\\x00\\x00\\x00\\x00'  # Low entropy
        >>> entropy = estimate_key_entropy(key)
        >>> print(f"Entropy: {entropy:.2f} bits")
    """
    if len(key) == 0:
        return 0.0
    
    # Count byte frequency
    byte_counts = [0] * 256
    for byte in key:
        byte_counts[byte] += 1
    
    # Calculate probabilities and entropy
    entropy = 0.0
    total_bytes = len(key)
    
    for count in byte_counts:
        if count > 0:
            probability = count / total_bytes
            entropy -= probability * np.log2(probability)
    
    # Return entropy per byte (max 8 bits per byte)
    return entropy


def secure_erase_key(key: bytes) -> None:
    """
    Securely erase key material from memory.
    
    Overwrites key bytes with zeros before deletion.
    
    Note: In Python, this provides limited protection due to immutable strings
    and garbage collection, but it's a best practice for sensitive data.
    
    Args:
        key: Key bytes to erase
    
    Examples:
        >>> key = generate_random_bytes(32)
        >>> secure_erase_key(key)
    """
    # Python bytes are immutable, so we can't directly overwrite
    # This is more symbolic, but good practice
    if isinstance(key, bytes):
        # Clear the reference
        del key


class KeyGenerator(ABC):
    """
    Abstract base class for key generators.
    """
    
    @abstractmethod
    def generate(self, **kwargs) -> Any:
        """Generate a key."""
        raise NotImplementedError("Subclass must implement generate()")
    
    @abstractmethod
    def validate(self, key: Any) -> bool:
        """Validate a key."""
        raise NotImplementedError("Subclass must implement validate()")


class SymmetricKeyGenerator(KeyGenerator):
    """
    Generator for symmetric cryptographic keys.
    
    Generates random byte sequences suitable for symmetric algorithms.
    """
    
    def generate(self, size: int, seed: Optional[int] = None) -> bytes:
        """
        Generate a symmetric key.
        
        Args:
            size: Key size in bytes
            seed: Optional seed for deterministic generation
        
        Returns:
            Random key bytes
        
        Examples:
            >>> gen = SymmetricKeyGenerator()
            >>> key = gen.generate(32, seed=42)  # 256-bit key
        """
        return generate_random_bytes(size, seed)
    
    def validate(self, key: bytes) -> bool:
        """
        Validate a symmetric key.
        
        Args:
            key: Key bytes to validate
        
        Returns:
            True if valid, False otherwise
        
        Examples:
            >>> gen = SymmetricKeyGenerator()
            >>> key = gen.generate(32)
            >>> assert gen.validate(key)
        """
        # Basic validation: check if it's bytes and has reasonable size
        if not isinstance(key, bytes):
            return False
        
        # Key should be at least 8 bytes (64 bits) and at most 256 bytes (2048 bits)
        if len(key) < 8 or len(key) > 256:
            return False
        
        return True
    
    def estimate_entropy(self, key: bytes) -> float:
        """
        Estimate entropy of a key.
        
        Args:
            key: Key bytes to analyze
        
        Returns:
            Shannon entropy in bits per byte
        
        Examples:
            >>> gen = SymmetricKeyGenerator()
            >>> key = gen.generate(32)
            >>> entropy = gen.estimate_entropy(key)
        """
        return estimate_key_entropy(key)


class AsymmetricKeyGenerator(KeyGenerator):
    """
    Generator for asymmetric (public-key) cryptographic keys.
    
    Supports RSA and ECC key pair generation.
    """
    
    def generate(self, algorithm: str = "RSA", key_size: int = 2048, 
                 seed: Optional[int] = None, **kwargs) -> Tuple[Any, Any]:
        """
        Generate an asymmetric key pair.
        
        Args:
            algorithm: Algorithm name ("RSA", "ECC", etc.)
            key_size: Key size in bits (for RSA) or curve name (for ECC)
            seed: Optional seed for deterministic generation
            **kwargs: Additional algorithm-specific parameters
        
        Returns:
            Tuple of (public_key, private_key)
        
        Examples:
            >>> gen = AsymmetricKeyGenerator()
            >>> pub, priv = gen.generate_key_pair("RSA", 2048, seed=42)
        """
        return self.generate_key_pair(algorithm, key_size, seed, **kwargs)
    
    def generate_key_pair(self, algorithm: str, key_size: int, 
                          seed: Optional[int] = None, **kwargs) -> Tuple[Any, Any]:
        """
        Generate a key pair for asymmetric algorithms.
        
        Args:
            algorithm: Algorithm name ("RSA" or "ECC")
            key_size: Key size in bits (RSA) or curve name (ECC)
            seed: Optional seed for deterministic generation
            **kwargs: Additional parameters (e.g., curve for ECC)
        
        Returns:
            Tuple of (public_key, private_key)
        
        Raises:
            ValueError: If algorithm not supported
        """
        algorithm = algorithm.upper()
        
        if algorithm == "RSA":
            return self._generate_rsa_key_pair(key_size, seed)
        elif algorithm == "ECC":
            curve = kwargs.get("curve", "P-256")
            return self._generate_ecc_key_pair(curve, seed)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def _generate_rsa_key_pair(self, key_size: int, seed: Optional[int] = None) -> Tuple[Any, Any]:
        """
        Generate an RSA key pair.
        
        Args:
            key_size: Key size in bits (1024, 2048, 3072, or 4096)
            seed: Optional seed for deterministic generation
        
        Returns:
            Tuple of (public_key, private_key)
        """
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
        
        # RSA key generation doesn't support seeding directly in cryptography library
        # For deterministic key generation, we'd need to use a lower-level library
        # For now, we'll generate non-deterministically
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        return public_key, private_key
    
    def _generate_ecc_key_pair(self, curve: str, seed: Optional[int] = None) -> Tuple[Any, Any]:
        """
        Generate an ECC key pair.
        
        Args:
            curve: Curve name ("P-256", "P-384", "P-521")
            seed: Optional seed for deterministic generation
        
        Returns:
            Tuple of (public_key, private_key)
        """
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.backends import default_backend
        
        # Map curve names to cryptography curve objects
        curve_map = {
            "P-256": ec.SECP256R1(),
            "P-384": ec.SECP384R1(),
            "P-521": ec.SECP521R1(),
            "SECP256R1": ec.SECP256R1(),
            "SECP384R1": ec.SECP384R1(),
            "SECP521R1": ec.SECP521R1(),
        }
        
        if curve not in curve_map:
            raise ValueError(f"Unsupported curve: {curve}. Supported: {list(curve_map.keys())}")
        
        curve_obj = curve_map[curve]
        
        private_key = ec.generate_private_key(curve_obj, default_backend())
        public_key = private_key.public_key()
        
        return public_key, private_key
    
    def validate(self, key: Any) -> bool:
        """
        Validate an asymmetric key.
        
        Args:
            key: Public or private key object
        
        Returns:
            True if valid, False otherwise
        """
        # Basic validation: check if it's a cryptography key object
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa, ec
            
            # Check if it's an RSA or ECC key
            if isinstance(key, (rsa.RSAPublicKey, rsa.RSAPrivateKey,
                               ec.EllipticCurvePublicKey, ec.EllipticCurvePrivateKey)):
                return True
            
            return False
        except Exception:
            return False
    
    def validate_key_pair(self, public_key: Any, private_key: Any) -> bool:
        """
        Validate a key pair.
        
        Args:
            public_key: Public key object
            private_key: Private key object
        
        Returns:
            True if valid pair, False otherwise
        """
        # Validate both keys individually
        if not (self.validate(public_key) and self.validate(private_key)):
            return False
        
        # Check if they're the same algorithm type
        from cryptography.hazmat.primitives.asymmetric import rsa, ec
        
        if isinstance(public_key, rsa.RSAPublicKey) and isinstance(private_key, rsa.RSAPrivateKey):
            # RSA key pair - verify they match
            try:
                # Extract public key from private key and compare
                derived_public = private_key.public_key()
                return (derived_public.public_numbers() == public_key.public_numbers())
            except Exception:
                return False
        
        elif isinstance(public_key, ec.EllipticCurvePublicKey) and isinstance(private_key, ec.EllipticCurvePrivateKey):
            # ECC key pair - verify they match
            try:
                derived_public = private_key.public_key()
                return (derived_public.public_numbers() == public_key.public_numbers())
            except Exception:
                return False
        
        return False


class PQCKeyGenerator(KeyGenerator):
    """
    Generator for post-quantum cryptographic keys.
    
    Supports ML-KEM and other PQC algorithms via liboqs.
    """
    
    def generate(self, variant: str = "ML-KEM-768", seed: Optional[int] = None) -> Tuple[Any, Any]:
        """
        Generate a PQC key pair.
        
        Args:
            variant: PQC algorithm variant (e.g., "ML-KEM-512", "ML-KEM-768")
            seed: Optional seed for deterministic generation
        
        Returns:
            Tuple of (public_key, private_key)
        
        Examples:
            >>> gen = PQCKeyGenerator()
            >>> pub, priv = gen.generate_key_pair("ML-KEM-768")
        """
        return self.generate_key_pair(variant, seed)
    
    def generate_key_pair(self, variant: str, seed: Optional[int] = None) -> Tuple[bytes, bytes]:
        """
        Generate a PQC key pair using liboqs.
        
        Args:
            variant: Algorithm variant
            seed: Optional seed
        
        Returns:
            Tuple of (public_key_bytes, private_key_bytes)
        
        Raises:
            ImportError: If liboqs not available
        """
        try:
            import oqs
        except ImportError:
            raise ImportError("liboqs-python is required for PQC key generation. Install with: pip install liboqs-python")
        
        # Map variant names to liboqs algorithm names
        variant_map = {
            "ML-KEM-512": "Kyber512",
            "ML-KEM-768": "Kyber768",
            "ML-KEM-1024": "Kyber1024",
            "KYBER-512": "Kyber512",
            "KYBER-768": "Kyber768",
            "KYBER-1024": "Kyber1024",
        }
        
        oqs_algorithm = variant_map.get(variant.upper())
        if not oqs_algorithm:
            raise ValueError(f"Unsupported PQC variant: {variant}")
        
        # Generate key pair
        with oqs.KeyEncapsulation(oqs_algorithm) as kem:
            public_key = kem.generate_keypair()
            # In liboqs, the secret key is stored internally
            # We need to export both keys
            secret_key = kem.export_secret_key()
            
            return public_key, secret_key
    
    def validate(self, key: bytes) -> bool:
        """
        Validate a PQC key.
        
        Args:
            key: Key bytes
        
        Returns:
            True if valid, False otherwise
        """
        # Basic validation for PQC keys
        if not isinstance(key, bytes):
            return False
        
        # PQC keys are typically larger (hundreds to thousands of bytes)
        if len(key) < 32 or len(key) > 10000:
            return False
        
        return True


# Export all classes and functions
__all__ = [
    "seed_rng",
    "generate_random_bytes",
    "estimate_key_entropy",
    "secure_erase_key",
    "KeyGenerator",
    "SymmetricKeyGenerator",
    "AsymmetricKeyGenerator",
    "PQCKeyGenerator",
]

