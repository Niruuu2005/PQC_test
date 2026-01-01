"""
Base Cipher System - Abstract Base Classes

This module defines abstract base classes for all cipher implementations.

Version: 3.1
Date: December 30, 2025
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
from .state_management import (
    EncryptionMetadata,
    DecryptionMetadata,
    CipherState,
    compute_plaintext_hash,
)


class BaseCipherSystem(ABC):
    """
    Abstract base class for all cipher systems.
    
    This class defines the interface that all cipher implementations must follow.
    Provides state management, key generation, encryption, and decryption.
    
    Attributes:
        algorithm_name: Name of the algorithm
        key_size_bits: Key size in bits
        cipher_state: State management object
    """
    
    def __init__(self, algorithm_name: str, key_size_bits: int, seed: Optional[int] = None):
        """
        Initialize base cipher system.
        
        Args:
            algorithm_name: Name of the algorithm (e.g., "AES-256-GCM")
            key_size_bits: Key size in bits
            seed: Optional RNG seed for deterministic key generation
        """
        self.algorithm_name = algorithm_name
        self.key_size_bits = key_size_bits
        self.cipher_state = CipherState(algorithm_name, seed)
    
    @abstractmethod
    def generate_key(self) -> None:
        """
        Generate cryptographic key.
        
        Must set cipher_state.key or cipher_state.key_pair after generation.
        Must update cipher_state.state to READY.
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclass must implement generate_key()")
    
    @abstractmethod
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        Encrypt plaintext.
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            Tuple of (ciphertext, encryption_metadata)
        
        Raises:
            RuntimeError: If key not generated
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclass must implement encrypt()")
    
    @abstractmethod
    def decrypt(self, ciphertext: bytes, iv: Optional[bytes] = None, 
                tag: Optional[bytes] = None) -> Tuple[bytes, DecryptionMetadata]:
        """
        Decrypt ciphertext.
        
        Args:
            ciphertext: Data to decrypt
            iv: Initialization vector (optional, for modes that need it)
            tag: Authentication tag (optional, for AEAD modes)
        
        Returns:
            Tuple of (plaintext, decryption_metadata)
        
        Raises:
            RuntimeError: If key not generated
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclass must implement decrypt()")
    
    def validate_key(self) -> bool:
        """
        Validate that key is generated and ready.
        
        Returns:
            True if key is valid and ready
        """
        return self.cipher_state.state == CipherState.STATE_READY
    
    def cleanup(self) -> None:
        """
        Securely erase key material from memory.
        
        This should be called when the cipher is no longer needed.
        """
        self.cipher_state.cleanup()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup keys."""
        self.cleanup()
        return False
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}(algorithm={self.algorithm_name}, "
                f"key_size={self.key_size_bits} bits, "
                f"state={self.cipher_state.state})")


class SymmetricBlockCipherSystem(BaseCipherSystem):
    """
    Abstract base class for symmetric block ciphers.
    
    Extends BaseCipherSystem with block cipher specific properties.
    
    Additional Attributes:
        block_size: Block size in bits
        mode: Cipher mode (e.g., CBC, GCM, CTR)
    """
    
    def __init__(self, algorithm_name: str, key_size_bits: int, 
                 block_size: int, mode: str, seed: Optional[int] = None):
        """
        Initialize symmetric block cipher.
        
        Args:
            algorithm_name: Name of the algorithm
            key_size_bits: Key size in bits
            block_size: Block size in bits
            mode: Cipher mode (ECB, CBC, CTR, GCM, etc.)
            seed: Optional RNG seed
        """
        super().__init__(algorithm_name, key_size_bits, seed)
        self.block_size = block_size
        self.mode = mode.upper()
    
    def generate_iv(self, size: Optional[int] = None) -> bytes:
        """
        Generate random initialization vector (IV).
        
        Args:
            size: IV size in bytes (defaults to block_size // 8)
        
        Returns:
            Random IV bytes
        """
        import os
        if size is None:
            size = self.block_size // 8
        
        # Use seeded RNG if seed provided, otherwise use os.urandom
        if self.cipher_state.seed is not None:
            import numpy as np
            rng = np.random.RandomState(self.cipher_state.seed)
            return rng.bytes(size)
        else:
            return os.urandom(size)
    
    def pad_plaintext(self, plaintext: bytes) -> bytes:
        """
        Apply PKCS#7 padding to plaintext.
        
        Args:
            plaintext: Data to pad
        
        Returns:
            Padded plaintext
        """
        block_size_bytes = self.block_size // 8
        padding_len = block_size_bytes - (len(plaintext) % block_size_bytes)
        padding = bytes([padding_len] * padding_len)
        return plaintext + padding
    
    def unpad_ciphertext(self, padded_data: bytes) -> bytes:
        """
        Remove PKCS#7 padding from data.
        
        Args:
            padded_data: Padded data
        
        Returns:
            Unpadded data
        
        Raises:
            ValueError: If padding is invalid
        """
        if not padded_data:
            raise ValueError("Cannot unpad empty data")
        
        padding_len = padded_data[-1]
        
        # Validate padding
        if padding_len > len(padded_data) or padding_len == 0:
            raise ValueError("Invalid padding length")
        
        # Check all padding bytes are correct
        if padded_data[-padding_len:] != bytes([padding_len] * padding_len):
            raise ValueError("Invalid padding bytes")
        
        return padded_data[:-padding_len]
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}(algorithm={self.algorithm_name}, "
                f"key_size={self.key_size_bits} bits, "
                f"block_size={self.block_size} bits, "
                f"mode={self.mode}, "
                f"state={self.cipher_state.state})")


class SymmetricStreamCipherSystem(BaseCipherSystem):
    """
    Abstract base class for symmetric stream ciphers.
    
    Extends BaseCipherSystem with stream cipher specific properties.
    
    Additional Attributes:
        nonce_size: Nonce size in bits
        stateful: Whether cipher maintains state between operations
    """
    
    def __init__(self, algorithm_name: str, key_size_bits: int, 
                 nonce_size: int, stateful: bool = False, seed: Optional[int] = None):
        """
        Initialize symmetric stream cipher.
        
        Args:
            algorithm_name: Name of the algorithm
            key_size_bits: Key size in bits
            nonce_size: Nonce size in bits
            stateful: Whether cipher is stateful
            seed: Optional RNG seed
        """
        super().__init__(algorithm_name, key_size_bits, seed)
        self.nonce_size = nonce_size
        self.stateful = stateful
    
    def generate_nonce(self) -> bytes:
        """
        Generate random nonce.
        
        Returns:
            Random nonce bytes
        """
        import os
        nonce_size_bytes = self.nonce_size // 8
        
        # Use seeded RNG if seed provided
        if self.cipher_state.seed is not None:
            import numpy as np
            rng = np.random.RandomState(self.cipher_state.seed)
            return rng.bytes(nonce_size_bytes)
        else:
            return os.urandom(nonce_size_bytes)
    
    def reset_state(self) -> None:
        """
        Reset cipher state (for stateful stream ciphers).
        
        For stateless ciphers, this is a no-op.
        """
        if self.stateful:
            # Subclass should override if stateful
            pass
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}(algorithm={self.algorithm_name}, "
                f"key_size={self.key_size_bits} bits, "
                f"nonce_size={self.nonce_size} bits, "
                f"stateful={self.stateful}, "
                f"state={self.cipher_state.state})")


class AsymmetricCipherSystem(BaseCipherSystem):
    """
    Abstract base class for asymmetric (public-key) ciphers.
    
    Extends BaseCipherSystem with asymmetric cipher specific properties.
    """
    
    def __init__(self, algorithm_name: str, key_size_bits: int, seed: Optional[int] = None):
        """
        Initialize asymmetric cipher.
        
        Args:
            algorithm_name: Name of the algorithm
            key_size_bits: Key size in bits
            seed: Optional RNG seed
        """
        super().__init__(algorithm_name, key_size_bits, seed)
    
    def generate_key_pair(self) -> Tuple[any, any]:
        """
        Generate public/private key pair.
        
        Returns:
            Tuple of (private_key, public_key)
        
        Note: This is typically called by generate_key()
        """
        raise NotImplementedError("Subclass must implement generate_key_pair()")
    
    def validate_plaintext_size(self, plaintext: bytes) -> bool:
        """
        Validate plaintext size for asymmetric encryption.
        
        Asymmetric ciphers have maximum plaintext size limits based on key size.
        
        Args:
            plaintext: Data to validate
        
        Returns:
            True if size is valid
        """
        # RSA OAEP max: (key_size_bits // 8) - 2*hash_len - 2
        # For SHA-256: (key_size_bits // 8) - 66
        # Conservative estimate:
        max_size = (self.key_size_bits // 8) - 100
        return len(plaintext) <= max_size
    
    def __repr__(self) -> str:
        """String representation."""
        has_keypair = (self.cipher_state.private_key is not None and 
                      self.cipher_state.public_key is not None)
        return (f"{self.__class__.__name__}(algorithm={self.algorithm_name}, "
                f"key_size={self.key_size_bits} bits, "
                f"has_keypair={has_keypair}, "
                f"state={self.cipher_state.state})")


class PQCKEMSystem(BaseCipherSystem):
    """
    Abstract base class for Post-Quantum Key Encapsulation Mechanisms (KEMs).
    
    Extends BaseCipherSystem with PQC KEM specific properties.
    
    Additional Attributes:
        variant: PQC variant (e.g., "ML-KEM-512", "ML-KEM-768")
    """
    
    def __init__(self, algorithm_name: str, variant: str, 
                 key_size_bits: int, seed: Optional[int] = None):
        """
        Initialize PQC KEM system.
        
        Args:
            algorithm_name: Name of the algorithm
            variant: Specific variant
            key_size_bits: Key size in bits
            seed: Optional RNG seed
        """
        super().__init__(algorithm_name, key_size_bits, seed)
        self.variant = variant
    
    def encapsulate(self) -> Tuple[bytes, bytes]:
        """
        Perform KEM encapsulation.
        
        Returns:
            Tuple of (ciphertext, shared_secret)
        """
        raise NotImplementedError("Subclass must implement encapsulate()")
    
    def decapsulate(self, ciphertext: bytes) -> bytes:
        """
        Perform KEM decapsulation.
        
        Args:
            ciphertext: KEM ciphertext
        
        Returns:
            Shared secret bytes
        """
        raise NotImplementedError("Subclass must implement decapsulate()")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}(algorithm={self.algorithm_name}, "
                f"variant={self.variant}, "
                f"key_size={self.key_size_bits} bits, "
                f"state={self.cipher_state.state})")


class PQCSignatureSystem(BaseCipherSystem):
    """
    Abstract base class for Post-Quantum Signature schemes.
    
    Extends BaseCipherSystem with PQC signature specific properties.
    
    Additional Attributes:
        variant: PQC variant (e.g., "ML-DSA-44", "Falcon-512")
        private_key: Private signing key
        public_key: Public verification key
    """
    
    def __init__(self, algorithm_name: str, variant: str,
                 key_size_bits: int, seed: Optional[int] = None):
        """
        Initialize PQC Signature system.
        
        Args:
            algorithm_name: Name of the algorithm
            variant: Specific variant
            key_size_bits: Security level in bits
            seed: Optional RNG seed
        """
        super().__init__(algorithm_name, key_size_bits, seed)
        self.variant = variant
        self.private_key = None
        self.public_key = None
    
    @abstractmethod
    def generate_keypair(self) -> None:
        """
        Generate PQC signature key pair.
        
        Must set self.private_key and self.public_key.
        Should also update cipher_state.key with public key.
        """
        raise NotImplementedError("Subclass must implement generate_keypair()")
    
    @abstractmethod
    def sign(self, data: bytes) -> bytes:
        """
        Sign data using private key.
        
        Args:
            data: Data to sign
        
        Returns:
            Signature bytes
        """
        raise NotImplementedError("Subclass must implement sign()")
    
    @abstractmethod
    def verify(self, data: bytes, signature: bytes) -> bool:
        """
        Verify signature using public key.
        
        Args:
            data: Original data
            signature: Signature to verify
        
        Returns:
            True if signature is valid, False otherwise
        """
        raise NotImplementedError("Subclass must implement verify()")
    
    def generate_key(self) -> None:
        """
        Generate key pair (calls generate_keypair for compatibility).
        """
        self.generate_keypair()
    
    def __repr__(self) -> str:
        """String representation."""
        has_keypair = (self.private_key is not None and 
                      self.public_key is not None)
        return (f"{self.__class__.__name__}(algorithm={self.algorithm_name}, "
                f"variant={self.variant}, "
                f"key_size={self.key_size_bits} bits, "
                f"has_keypair={has_keypair}, "
                f"state={self.cipher_state.state})")


# Export all classes
__all__ = [
    "BaseCipherSystem",
    "SymmetricBlockCipherSystem",
    "SymmetricStreamCipherSystem",
    "AsymmetricCipherSystem",
    "PQCKEMSystem",
    "PQCSignatureSystem",
]

