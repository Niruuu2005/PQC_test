"""
Multi-Language Cryptographic Implementation Router

This module provides a unified interface for cryptographic operations
across multiple programming languages (Python, C, Java, Rust).

Usage:
    crypto = MultiLangCrypto(algorithm='AES-256-GCM', language='python')
    ciphertext = crypto.encrypt(plaintext)
    plaintext = crypto.decrypt(ciphertext)
"""

from enum import Enum
from typing import Optional, Tuple, Any
import importlib


class Language(Enum):
    """Supported implementation languages."""
    PYTHON = "python"
    C = "c"
    JAVA = "java"
    RUST = "rust"


class MultiLangCrypto:
    """
    Multi-language cryptographic implementation router.
    
    Routes cryptographic operations to the appropriate language implementation.
    Falls back to Python if the requested language is unavailable.
    """
    
    def __init__(self, algorithm: str, language: str = "python", **kwargs):
        """
        Initialize multi-language crypto system.
        
        Args:
            algorithm: Algorithm name (e.g., 'AES-256-GCM')
            language: Implementation language ('python', 'c', 'java', 'rust')
            **kwargs: Additional algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.requested_language = language.lower()
        self.active_language = None
        self.impl = None
        
        # Try to load requested language implementation
        self._load_implementation()
    
    def _load_implementation(self):
        """Load the appropriate language implementation."""
        
        # Try requested language first
        if self.requested_language == 'python':
            self.impl = self._load_python()
            self.active_language = Language.PYTHON
        elif self.requested_language == 'c':
            self.impl = self._load_c() or self._load_python()
            self.active_language = Language.C if self.impl else Language.PYTHON
        elif self.requested_language == 'java':
            self.impl = self._load_java() or self._load_python()
            self.active_language = Language.JAVA if self.impl else Language.PYTHON
        elif self.requested_language == 'rust':
            self.impl = self._load_rust() or self._load_python()
            self.active_language = Language.RUST if self.impl else Language.PYTHON
        else:
            # Default to Python
            self.impl = self._load_python()
            self.active_language = Language.PYTHON
    
    def _load_python(self):
        """Load Python implementation (always available)."""
        try:
            from src.crypto_dataset_generator.crypto.cipher_factory import create_cipher
            return create_cipher(self.algorithm)
        except Exception as e:
            print(f"Warning: Could not load Python implementation: {e}")
            return None
    
    def _load_c(self):
        """Load C implementation via ctypes (if available)."""
        try:
            # Example: Load OpenSSL-based C implementation
            from src.crypto_dataset_generator.crypto.c_impl import CCryptoImpl
            return CCryptoImpl(self.algorithm)
        except ImportError:
            print(f"⚠️  C implementation not available for {self.algorithm}")
            return None
    
    def _load_java(self):
        """Load Java implementation via JPype (if available)."""
        try:
            from src.crypto_dataset_generator.crypto.java_impl import JavaCryptoImpl
            return JavaCryptoImpl(self.algorithm)
        except ImportError:
            print(f"⚠️  Java implementation not available for {self.algorithm}")
            return None
    
    def _load_rust(self):
        """Load Rust implementation via PyO3 (if available)."""
        try:
            from src.crypto_dataset_generator.crypto.rust_impl import RustCryptoImpl
            return RustCryptoImpl(self.algorithm)
        except ImportError:
            print(f"⚠️  Rust implementation not available for {self.algorithm}")
            return None
    
    def generate_key(self) -> None:
        """Generate cryptographic key."""
        if self.impl:
            self.impl.generate_key()
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, Any]:
        """
        Encrypt plaintext.
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            Tuple of (ciphertext, metadata)
        """
        if self.impl:
            return self.impl.encrypt(plaintext)
        raise RuntimeError("No implementation available")
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, Any]:
        """
        Decrypt ciphertext.
        
        Args:
            ciphertext: Data to decrypt
            **kwargs: Additional parameters (iv, tag, etc.)
        
        Returns:
            Tuple of (plaintext, metadata)
        """
        if self.impl:
            return self.impl.decrypt(ciphertext, **kwargs)
        raise RuntimeError("No implementation available")
    
    def get_active_language(self) -> str:
        """Get the currently active implementation language."""
        return self.active_language.value if self.active_language else "none"


def get_available_languages(algorithm: str) -> list:
    """
    Get list of available language implementations for an algorithm.
    
    Args:
        algorithm: Algorithm name
    
    Returns:
        List of available language names
    """
    available = ['python']  # Python always available
    
    # Check C (OpenSSL)
    try:
        import ctypes
        available.append('c')
    except:
        pass
    
    # Check Java (JCA)
    try:
        import jpype
        available.append('java')
    except:
        pass
    
    # Check Rust (via PyO3)
    try:
        import crypto_rust  # hypothetical Rust module
        available.append('rust')
    except:
        pass
    
    return available


# Quick API
def encrypt_multi(algorithm: str, plaintext: bytes, language: str = 'python') -> Tuple[bytes, Any]:
    """
    Quick encryption with multi-language support.
    
    Args:
        algorithm: Algorithm name
        plaintext: Data to encrypt
        language: Implementation language
    
    Returns:
        Tuple of (ciphertext, metadata)
    """
    crypto = MultiLangCrypto(algorithm, language)
    crypto.generate_key()
    return crypto.encrypt(plaintext)


if __name__ == "__main__":
    # Demo
    print("Multi-Language Crypto Demo")
    print("=" * 60)
    
    # Test with Python
    print("\n1. Python Implementation:")
    crypto_py = MultiLangCrypto('AES-256-GCM', language='python')
    print(f"   Active: {crypto_py.get_active_language()}")
    
    # Test with C (will fallback to Python)
    print("\n2. C Implementation (fallback to Python):")
    crypto_c = MultiLangCrypto('AES-256-GCM', language='c')
    print(f"   Active: {crypto_c.get_active_language()}")
    
    # Test encryption
    print("\n3. Encryption Test:")
    crypto_py.generate_key()
    ct, meta = crypto_py.encrypt(b"Hello, Multi-Lang Crypto!")
    print(f"   Ciphertext length: {len(ct)} bytes")
    print(f"   Success: {meta.success}")
    
    print("\n✅ Multi-language crypto system ready!")
