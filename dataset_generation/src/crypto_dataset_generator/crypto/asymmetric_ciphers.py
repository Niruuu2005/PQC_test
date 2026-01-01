"""
Asymmetric Cipher Implementations

This module implements concrete asymmetric (public-key) ciphers.

Version: 1.0
Date: December 30, 2025
"""

import time
from typing import Tuple, Optional
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend

from .base_cipher import AsymmetricCipherSystem
from .state_management import (
    EncryptionMetadata,
    DecryptionMetadata,
    compute_plaintext_hash,
)


class RSACipherSystem(AsymmetricCipherSystem):
    """
    RSA (Rivest-Shamir-Adleman) cipher implementation.
    
    Supports multiple padding schemes:
    - OAEP (Optimal Asymmetric Encryption Padding) - Recommended
    - PKCS1v15 (PKCS #1 v1.5) - Legacy
    - PSS (Probabilistic Signature Scheme) - For signatures (not used here)
    
    Key Sizes: 1024, 2048, 3072, 4096 bits
    """
    
    def __init__(self, key_size: int = 2048, padding_scheme: str = "OAEP", seed: Optional[int] = None):
        """
        Initialize RSA cipher.
        
        Args:
            key_size: Key size in bits (1024, 2048, 3072, or 4096)
            padding_scheme: Padding scheme ("OAEP", "PKCS1", or "PSS")
            seed: Optional RNG seed
        """
        if key_size not in [1024, 2048, 3072, 4096]:
            raise ValueError(f"Invalid RSA key size: {key_size}. Must be 1024, 2048, 3072, or 4096.")
        
        if padding_scheme.upper() not in ["OAEP", "PKCS1", "PSS"]:
            raise ValueError(f"Invalid padding scheme: {padding_scheme}. Must be OAEP, PKCS1, or PSS.")
        
        algorithm_name = f"RSA-{key_size}-{padding_scheme.upper()}"
        super().__init__(
            algorithm_name=algorithm_name,
            key_size_bits=key_size,
            seed=seed
        )
        
        self.padding_scheme = padding_scheme.upper()
        self._public_key = None
        self._private_key = None
    
    def generate_key(self) -> None:
        """Generate RSA key pair."""
        t_start = time.perf_counter()
        
        # Generate RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size_bits,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Store keys
        self._private_key = private_key
        self._public_key = public_key
        
        # For state management, we'll use the public key bytes as "key"
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Estimate entropy (not very meaningful for public keys, but included)
        from .state_management import estimate_entropy
        entropy = estimate_entropy(public_bytes)
        
        # Update cipher state
        self.cipher_state.set_key(public_bytes, entropy, generation_time_ms)
        self.cipher_state.public_key = public_key
        self.cipher_state.private_key = private_key
    
    def get_max_plaintext_size(self) -> int:
        """
        Get maximum plaintext size for RSA encryption.
        
        Returns:
            Maximum plaintext size in bytes
        """
        # RSA has plaintext size limits based on key size and padding
        key_size_bytes = self.key_size_bits // 8
        
        if self.padding_scheme == "OAEP":
            # OAEP overhead: 2 * hash_size + 2
            # Using SHA-256 (32 bytes)
            overhead = 2 * 32 + 2  # 66 bytes
            return key_size_bytes - overhead
        elif self.padding_scheme == "PKCS1":
            # PKCS1 overhead: 11 bytes minimum
            return key_size_bytes - 11
        else:  # PSS (not typically used for encryption)
            return key_size_bytes - 11
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        Encrypt plaintext using RSA.
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            Tuple of (ciphertext, metadata)
        """
        if self._public_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Check plaintext size
            max_size = self.get_max_plaintext_size()
            if len(plaintext) > max_size:
                raise ValueError(f"Plaintext too large. Maximum size: {max_size} bytes, got: {len(plaintext)} bytes")
            
            # Select padding
            if self.padding_scheme == "OAEP":
                padding_obj = padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            elif self.padding_scheme == "PKCS1":
                padding_obj = padding.PKCS1v15()
            else:  # PSS (not typically for encryption, but included)
                padding_obj = padding.PKCS1v15()
            
            # Encrypt
            ciphertext = self._public_key.encrypt(plaintext, padding_obj)
            
            t_end = time.perf_counter()
            encryption_time_ms = (t_end - t_start) * 1000
            
            metadata = EncryptionMetadata(
                algorithm=self.algorithm_name,
                plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext),
                key_size_bits=self.key_size_bits,
                ciphertext_length=len(ciphertext),
                encryption_time_ms=encryption_time_ms,
                success=True,
            )
            
            self.cipher_state.mark_encrypted()
            return ciphertext, metadata
            
        except Exception as e:
            t_end = time.perf_counter()
            encryption_time_ms = (t_end - t_start) * 1000
            
            metadata = EncryptionMetadata(
                algorithm=self.algorithm_name,
                plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext),
                key_size_bits=self.key_size_bits,
                ciphertext_length=0,
                encryption_time_ms=encryption_time_ms,
                success=False,
                error_message=str(e),
            )
            
            return b"", metadata
    
    def decrypt(self, ciphertext: bytes, iv: Optional[bytes] = None,
                tag: Optional[bytes] = None) -> Tuple[bytes, DecryptionMetadata]:
        """
        Decrypt ciphertext using RSA.
        
        Args:
            ciphertext: Data to decrypt
            iv: Not used for RSA
            tag: Not used for RSA
        
        Returns:
            Tuple of (plaintext, metadata)
        """
        if self._private_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Select padding
            if self.padding_scheme == "OAEP":
                padding_obj = padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            elif self.padding_scheme == "PKCS1":
                padding_obj = padding.PKCS1v15()
            else:  # PSS
                padding_obj = padding.PKCS1v15()
            
            # Decrypt
            plaintext = self._private_key.decrypt(ciphertext, padding_obj)
            
            t_end = time.perf_counter()
            decryption_time_ms = (t_end - t_start) * 1000
            
            metadata = DecryptionMetadata(
                success=True,
                decryption_time_ms=decryption_time_ms,
                recovered_size=len(plaintext),
            )
            
            return plaintext, metadata
            
        except Exception as e:
            t_end = time.perf_counter()
            decryption_time_ms = (t_end - t_start) * 1000
            
            metadata = DecryptionMetadata(
                success=False,
                decryption_time_ms=decryption_time_ms,
                recovered_size=0,
                error_message=str(e),
            )
            
            return b"", metadata


class ECCCipherSystem(AsymmetricCipherSystem):
    """
    ECC (Elliptic Curve Cryptography) cipher implementation using ECIES.
    
    ECIES (Elliptic Curve Integrated Encryption Scheme) provides
    encryption using elliptic curve cryptography.
    
    Supported Curves: P-256, P-384, P-521
    """
    
    def __init__(self, curve: str = "P-256", seed: Optional[int] = None):
        """
        Initialize ECC cipher.
        
        Args:
            curve: Curve name ("P-256", "P-384", or "P-521")
            seed: Optional RNG seed
        """
        curve_upper = curve.upper()
        if curve_upper not in ["P-256", "P-384", "P-521"]:
            raise ValueError(f"Invalid curve: {curve}. Must be P-256, P-384, or P-521.")
        
        # Map curve names to key sizes (approximate)
        curve_key_sizes = {
            "P-256": 256,
            "P-384": 384,
            "P-521": 521,
        }
        
        algorithm_name = f"ECC-{curve_upper}"
        super().__init__(
            algorithm_name=algorithm_name,
            key_size_bits=curve_key_sizes[curve_upper],
            seed=seed
        )
        
        self.curve_name = curve_upper
        self._public_key = None
        self._private_key = None
    
    def generate_key(self) -> None:
        """Generate ECC key pair."""
        t_start = time.perf_counter()
        
        # Map curve names to curve objects
        curve_map = {
            "P-256": ec.SECP256R1(),
            "P-384": ec.SECP384R1(),
            "P-521": ec.SECP521R1(),
        }
        
        curve_obj = curve_map[self.curve_name]
        
        # Generate ECC key pair
        private_key = ec.generate_private_key(curve_obj, default_backend())
        public_key = private_key.public_key()
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Store keys
        self._private_key = private_key
        self._public_key = public_key
        
        # Get public key bytes for state management
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        from .state_management import estimate_entropy
        entropy = estimate_entropy(public_bytes)
        
        # Update cipher state
        self.cipher_state.set_key(public_bytes, entropy, generation_time_ms)
        self.cipher_state.public_key = public_key
        self.cipher_state.private_key = private_key
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        Encrypt plaintext using ECIES.
        
        This is a simplified ECIES implementation using ECDH for key agreement
        and AES-GCM for actual encryption.
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            Tuple of (ciphertext, metadata)
        """
        if self._public_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # ECIES: Generate ephemeral key pair
            curve_map = {
                "P-256": ec.SECP256R1(),
                "P-384": ec.SECP384R1(),
                "P-521": ec.SECP521R1(),
            }
            
            curve_obj = curve_map[self.curve_name]
            ephemeral_private_key = ec.generate_private_key(curve_obj, default_backend())
            ephemeral_public_key = ephemeral_private_key.public_key()
            
            # Perform ECDH to get shared secret
            from cryptography.hazmat.primitives.asymmetric import ec as ec_mod
            shared_secret = ephemeral_private_key.exchange(
                ec_mod.ECDH(), self._public_key
            )
            
            # Derive encryption key from shared secret using HKDF
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,  # 256 bits for AES-256
                salt=None,
                info=b'ecies-encryption',
                backend=default_backend()
            ).derive(shared_secret)
            
            # Encrypt plaintext with AES-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            import os
            
            aesgcm = AESGCM(derived_key)
            nonce = os.urandom(12)  # 96-bit nonce
            
            # Associated data: ephemeral public key
            ephemeral_public_bytes = ephemeral_public_key.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint
            )
            
            ciphertext_data = aesgcm.encrypt(nonce, plaintext, ephemeral_public_bytes)
            
            # Combine: ephemeral_public_key + nonce + ciphertext
            ciphertext = ephemeral_public_bytes + nonce + ciphertext_data
            
            t_end = time.perf_counter()
            encryption_time_ms = (t_end - t_start) * 1000
            
            metadata = EncryptionMetadata(
                algorithm=self.algorithm_name,
                plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext),
                key_size_bits=self.key_size_bits,
                ciphertext_length=len(ciphertext),
                encryption_time_ms=encryption_time_ms,
                success=True,
                nonce=nonce.hex(),
            )
            
            self.cipher_state.mark_encrypted()
            return ciphertext, metadata
            
        except Exception as e:
            t_end = time.perf_counter()
            encryption_time_ms = (t_end - t_start) * 1000
            
            metadata = EncryptionMetadata(
                algorithm=self.algorithm_name,
                plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext),
                key_size_bits=self.key_size_bits,
                ciphertext_length=0,
                encryption_time_ms=encryption_time_ms,
                success=False,
                error_message=str(e),
            )
            
            return b"", metadata
    
    def decrypt(self, ciphertext: bytes, iv: Optional[bytes] = None,
                tag: Optional[bytes] = None) -> Tuple[bytes, DecryptionMetadata]:
        """
        Decrypt ciphertext using ECIES.
        
        Args:
            ciphertext: Data to decrypt (includes ephemeral public key + nonce + encrypted data)
            iv: Not used for ECIES
            tag: Not used for ECIES
        
        Returns:
            Tuple of (plaintext, metadata)
        """
        if self._private_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Parse ciphertext: ephemeral_public_key + nonce + ciphertext_data
            # Ephemeral public key size depends on curve
            curve_point_sizes = {
                "P-256": 65,  # 1 + 32 + 32 (uncompressed point)
                "P-384": 97,  # 1 + 48 + 48
                "P-521": 133,  # 1 + 66 + 66
            }
            
            ephemeral_public_size = curve_point_sizes[self.curve_name]
            nonce_size = 12
            
            if len(ciphertext) < ephemeral_public_size + nonce_size:
                raise ValueError("Ciphertext too short")
            
            ephemeral_public_bytes = ciphertext[:ephemeral_public_size]
            nonce = ciphertext[ephemeral_public_size:ephemeral_public_size + nonce_size]
            ciphertext_data = ciphertext[ephemeral_public_size + nonce_size:]
            
            # Reconstruct ephemeral public key
            curve_map = {
                "P-256": ec.SECP256R1(),
                "P-384": ec.SECP384R1(),
                "P-521": ec.SECP521R1(),
            }
            
            ephemeral_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                curve_map[self.curve_name],
                ephemeral_public_bytes
            )
            
            # Perform ECDH to get shared secret
            from cryptography.hazmat.primitives.asymmetric import ec as ec_mod
            shared_secret = self._private_key.exchange(
                ec_mod.ECDH(), ephemeral_public_key
            )
            
            # Derive decryption key
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'ecies-encryption',
                backend=default_backend()
            ).derive(shared_secret)
            
            # Decrypt with AES-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            aesgcm = AESGCM(derived_key)
            plaintext = aesgcm.decrypt(nonce, ciphertext_data, ephemeral_public_bytes)
            
            t_end = time.perf_counter()
            decryption_time_ms = (t_end - t_start) * 1000
            
            metadata = DecryptionMetadata(
                success=True,
                decryption_time_ms=decryption_time_ms,
                recovered_size=len(plaintext),
            )
            
            return plaintext, metadata
            
        except Exception as e:
            t_end = time.perf_counter()
            decryption_time_ms = (t_end - t_start) * 1000
            
            metadata = DecryptionMetadata(
                success=False,
                decryption_time_ms=decryption_time_ms,
                recovered_size=0,
                error_message=str(e),
            )
            
            return b"", metadata


# Export all cipher classes
__all__ = [
    "RSACipherSystem",
    "ECCCipherSystem",
]

