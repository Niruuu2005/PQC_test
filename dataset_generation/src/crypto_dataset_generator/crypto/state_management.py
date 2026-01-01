"""
State Management and Data Models for Cryptographic Operations

This module defines dataclasses and state management for cipher operations.

Version: 3.1
Date: December 30, 2025
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
import hashlib


@dataclass
class EncryptionMetadata:
    """
    Metadata captured during encryption operation.
    
    Attributes:
        algorithm: Algorithm name (e.g., "AES-256-GCM")
        plaintext_hash: SHA-256 hash of plaintext (hex string)
        plaintext_length: Length of plaintext in bytes
        key_size_bits: Key size in bits
        ciphertext_length: Length of ciphertext in bytes
        iv: Initialization vector (hex string, optional)
        tag: Authentication tag (hex string, optional for AEAD)
        nonce: Nonce (hex string, optional for stream ciphers)
        encryption_time_ms: Encryption time in milliseconds
        success: Whether encryption succeeded
        error_message: Error message if failed
        cpu_cycles: CPU cycles used (optional)
        memory_peak_mb: Peak memory usage in MB (optional)
        timestamp: ISO 8601 timestamp
    """
    algorithm: str
    plaintext_hash: str
    plaintext_length: int
    key_size_bits: int
    ciphertext_length: int
    encryption_time_ms: float
    success: bool
    iv: Optional[str] = None
    tag: Optional[str] = None
    nonce: Optional[str] = None
    error_message: Optional[str] = None
    cpu_cycles: Optional[int] = None
    memory_peak_mb: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        if self.encryption_time_ms < 0:
            raise ValueError("Encryption time cannot be negative")
        if self.ciphertext_length < 0:
            raise ValueError("Ciphertext length cannot be negative")
        if self.key_size_bits <= 0:
            raise ValueError("Key size must be positive")


@dataclass
class DecryptionMetadata:
    """
    Metadata captured during decryption operation.
    
    Attributes:
        success: Whether decryption succeeded
        decryption_time_ms: Decryption time in milliseconds
        error_message: Error message if failed
        recovered_size: Size of recovered plaintext in bytes
    """
    success: bool
    decryption_time_ms: float
    recovered_size: int = 0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        if self.decryption_time_ms < 0:
            raise ValueError("Decryption time cannot be negative")


@dataclass
class KeyInfo:
    """
    Information about generated cryptographic key.
    
    Attributes:
        algorithm: Algorithm name
        key_size_bits: Key size in bits
        key_entropy: Estimated entropy in bits
        generation_time_ms: Key generation time in milliseconds
        key_format: Format of key (e.g., "raw", "pem", "der")
    """
    algorithm: str
    key_size_bits: int
    key_entropy: float
    generation_time_ms: float
    key_format: str = "raw"
    
    def __post_init__(self):
        """Validate key info after initialization."""
        if self.key_size_bits <= 0:
            raise ValueError("Key size must be positive")
        if self.key_entropy < 0:
            raise ValueError("Key entropy cannot be negative")
        if self.generation_time_ms < 0:
            raise ValueError("Generation time cannot be negative")


class CipherState:
    """
    Manages the state and lifecycle of a cipher instance.
    
    States:
        - UNINITIALIZED: Cipher created but no key generated
        - READY: Key generated, ready for operations
        - ENCRYPTED: Encryption completed
    
    Attributes:
        algorithm_name: Name of the algorithm
        seed: RNG seed for deterministic key generation (optional)
        key: Symmetric key bytes (optional)
        private_key: Private key for asymmetric (optional)
        public_key: Public key for asymmetric (optional)
        key_generation_time_ms: Time taken to generate key
        key_entropy: Estimated entropy of key
        state: Current state of cipher
    """
    
    # State constants
    STATE_UNINITIALIZED = "UNINITIALIZED"
    STATE_READY = "READY"
    STATE_ENCRYPTED = "ENCRYPTED"
    
    def __init__(self, algorithm_name: str, seed: Optional[int] = None):
        """
        Initialize cipher state.
        
        Args:
            algorithm_name: Name of the cryptographic algorithm
            seed: Optional RNG seed for deterministic key generation
        """
        self.algorithm_name = algorithm_name
        self.seed = seed
        self.key: Optional[bytes] = None
        self.private_key: Optional[any] = None
        self.public_key: Optional[any] = None
        self.key_generation_time_ms: float = 0.0
        self.key_entropy: float = 0.0
        self.state: str = self.STATE_UNINITIALIZED
    
    def set_key(self, key: bytes, entropy: float, generation_time_ms: float) -> None:
        """
        Set the cipher key and update state.
        
        Args:
            key: Key bytes
            entropy: Estimated key entropy
            generation_time_ms: Time taken to generate key
        """
        self.key = key
        self.key_entropy = entropy
        self.key_generation_time_ms = generation_time_ms
        self.state = self.STATE_READY
    
    def set_key_pair(self, private_key: any, public_key: any, 
                     generation_time_ms: float) -> None:
        """
        Set asymmetric key pair and update state.
        
        Args:
            private_key: Private key object
            public_key: Public key object
            generation_time_ms: Time taken to generate key pair
        """
        self.private_key = private_key
        self.public_key = public_key
        self.key_generation_time_ms = generation_time_ms
        self.state = self.STATE_READY
    
    def mark_encrypted(self) -> None:
        """Mark state as encrypted."""
        if self.state != self.STATE_READY:
            raise RuntimeError(f"Cannot encrypt from state: {self.state}")
        self.state = self.STATE_ENCRYPTED
    
    def cleanup(self) -> None:
        """
        Securely erase key material from memory.
        
        Note: This is a best-effort cleanup. Complete memory wiping
        requires low-level memory management beyond Python's scope.
        """
        if self.key:
            # Overwrite key bytes with zeros
            key_len = len(self.key)
            self.key = b'\x00' * key_len
            self.key = None
        
        # Clear key pair references
        self.private_key = None
        self.public_key = None
        
        self.state = self.STATE_UNINITIALIZED
    
    def __repr__(self) -> str:
        """String representation of cipher state."""
        return (f"CipherState(algorithm={self.algorithm_name}, "
                f"state={self.state}, "
                f"has_key={self.key is not None}, "
                f"seed={self.seed})")


def compute_plaintext_hash(plaintext: bytes) -> str:
    """
    Compute SHA-256 hash of plaintext.
    
    Args:
        plaintext: Plaintext bytes
    
    Returns:
        Hex string of SHA-256 hash
    """
    return hashlib.sha256(plaintext).hexdigest()


def estimate_entropy(data: bytes) -> float:
    """
    Estimate entropy of byte data (Shannon entropy).
    
    Args:
        data: Byte data
    
    Returns:
        Entropy in bits per byte (0.0 - 8.0)
    """
    import math
    
    if not data:
        return 0.0
    
    # Count byte frequencies
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    
    # Calculate Shannon entropy
    entropy = 0.0
    data_len = len(data)
    
    for count in freq.values():
        if count > 0:
            p = count / data_len
            entropy -= p * math.log2(p)
    
    return min(entropy, 8.0)  # Cap at 8 bits per byte


# Export all classes
__all__ = [
    "EncryptionMetadata",
    "DecryptionMetadata",
    "KeyInfo",
    "CipherState",
    "compute_plaintext_hash",
    "estimate_entropy",
]

