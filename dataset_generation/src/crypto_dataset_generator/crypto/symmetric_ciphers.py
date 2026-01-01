"""
Symmetric Cipher Implementations

This module implements concrete symmetric block and stream ciphers.

Version: 3.1
Date: December 30, 2025
"""

import time
from typing import Tuple, Optional
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from .base_cipher import SymmetricBlockCipherSystem, SymmetricStreamCipherSystem
from .state_management import (
    EncryptionMetadata,
    DecryptionMetadata,
    compute_plaintext_hash,
    estimate_entropy,
)

# Try to import PyCryptodome for additional algorithms
try:
    from Crypto.Cipher import Blowfish as PyCryptoBlowfish
    from Crypto.Cipher import DES3 as PyCrypto3DES
    PYCRYPTODOME_AVAILABLE = True
except ImportError:
    PYCRYPTODOME_AVAILABLE = False

# Try to import Camellia from PyCryptodome
try:
    from Crypto.Cipher import Camellia as PyCryptoCamellia
    CAMELLIA_AVAILABLE = True
except ImportError:
    CAMELLIA_AVAILABLE = False

# Try to import legacy ciphers from PyCryptodome
try:
    from Crypto.Cipher import DES, ARC2, ARC4, CAST
    LEGACY_CIPHERS_AVAILABLE = True
except ImportError:
    LEGACY_CIPHERS_AVAILABLE = False

# Try to import PyNaCl for Salsa20
try:
    import nacl.secret
    import nacl.utils
    PYNACL_AVAILABLE = True
except ImportError:
    PYNACL_AVAILABLE = False


class AESCipherSystem(SymmetricBlockCipherSystem):
    """
    AES (Advanced Encryption Standard) cipher implementation.
    
    Supports all standard AES modes:
    - ECB (Electronic Codebook) - Not recommended
    - CBC (Cipher Block Chaining)
    - CTR (Counter)
    - GCM (Galois/Counter Mode) - AEAD
    - CFB (Cipher Feedback)
    - OFB (Output Feedback)
    - XTS (XEX-based tweaked-codebook mode)
    - CCM (Counter with CBC-MAC) - AEAD
    
    Key Sizes: 128, 192, 256 bits
    Block Size: 128 bits
    """
    
    # Modes that require IV
    MODES_REQUIRING_IV = ["CBC", "CTR", "GCM", "CFB", "OFB", "XTS", "CCM"]
    
    # AEAD modes that produce authentication tags
    AEAD_MODES = ["GCM", "CCM"]
    
    # Modes that require padding
    MODES_REQUIRING_PADDING = ["ECB", "CBC"]
    
    def __init__(self, key_size: int = 256, mode: str = "GCM", seed: Optional[int] = None):
        """
        Initialize AES cipher.
        
        Args:
            key_size: Key size in bits (128, 192, or 256)
            mode: Cipher mode (ECB, CBC, CTR, GCM, etc.)
            seed: Optional RNG seed for deterministic key generation
        
        Raises:
            ValueError: If key_size is invalid
        """
        if key_size not in [128, 192, 256]:
            raise ValueError(f"Invalid AES key size: {key_size}. Must be 128, 192, or 256.")
        
        algorithm_name = f"AES-{key_size}-{mode}"
        super().__init__(
            algorithm_name=algorithm_name,
            key_size_bits=key_size,
            block_size=128,  # AES always uses 128-bit blocks
            mode=mode,
            seed=seed
        )
        
        self._cipher = None
        self._last_iv = None
        self._last_tag = None
    
    def generate_key(self) -> None:
        """Generate AES key."""
        import os
        import numpy as np
        
        t_start = time.perf_counter()
        
        key_size_bytes = self.key_size_bits // 8
        
        # Generate key (deterministic if seed provided)
        if self.cipher_state.seed is not None:
            rng = np.random.RandomState(self.cipher_state.seed)
            key = rng.bytes(key_size_bytes)
        else:
            key = os.urandom(key_size_bytes)
        
        # Estimate entropy
        entropy = estimate_entropy(key)
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Set key in state
        self.cipher_state.set_key(key, entropy, generation_time_ms)
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        Encrypt plaintext using AES.
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            Tuple of (ciphertext, metadata)
        """
        if not self.validate_key():
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Generate IV if required
            iv = None
            if self.mode in self.MODES_REQUIRING_IV:
                if self.mode == "GCM":
                    iv = self.generate_iv(12)  # GCM recommends 96-bit IV
                else:
                    iv = self.generate_iv()  # Use block size
                self._last_iv = iv
            
            # Apply padding if required
            data_to_encrypt = plaintext
            if self.mode in self.MODES_REQUIRING_PADDING:
                data_to_encrypt = self.pad_plaintext(plaintext)
            
            # Create cipher instance
            backend = default_backend()
            algorithm = algorithms.AES(self.cipher_state.key)
            
            # Select mode
            if self.mode == "ECB":
                mode_obj = modes.ECB()
            elif self.mode == "CBC":
                mode_obj = modes.CBC(iv)
            elif self.mode == "CTR":
                mode_obj = modes.CTR(iv)
            elif self.mode == "GCM":
                mode_obj = modes.GCM(iv)
            elif self.mode == "CFB":
                mode_obj = modes.CFB(iv)
            elif self.mode == "OFB":
                mode_obj = modes.OFB(iv)
            elif self.mode == "CCM":
                # CCM requires a specific nonce size (7-13 bytes, typically 12)
                if len(iv) < 7 or len(iv) > 13:
                    iv = self.generate_iv(12)  # Use 12-byte nonce for CCM
                    self._last_iv = iv
                mode_obj = modes.CCM(iv)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            
            # Create and perform encryption
            cipher = Cipher(algorithm, mode_obj, backend=backend)
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data_to_encrypt) + encryptor.finalize()
            
            # Get tag for AEAD modes
            tag = None
            if self.mode in self.AEAD_MODES:
                tag = encryptor.tag
                self._last_tag = tag
            
            t_end = time.perf_counter()
            encryption_time_ms = (t_end - t_start) * 1000
            
            # Create metadata
            metadata = EncryptionMetadata(
                algorithm=self.algorithm_name,
                plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext),
                key_size_bits=self.key_size_bits,
                ciphertext_length=len(ciphertext),
                encryption_time_ms=encryption_time_ms,
                success=True,
                iv=iv.hex() if iv else None,
                tag=tag.hex() if tag else None,
            )
            
            self.cipher_state.mark_encrypted()
            
            return ciphertext, metadata
            
        except Exception as e:
            t_end = time.perf_counter()
            encryption_time_ms = (t_end - t_start) * 1000
            
            # Return error metadata
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
        Decrypt ciphertext using AES.
        
        Args:
            ciphertext: Data to decrypt
            iv: Initialization vector (required for most modes)
            tag: Authentication tag (required for AEAD modes)
        
        Returns:
            Tuple of (plaintext, metadata)
        """
        # Check if we have a key (allow READY or ENCRYPTED state)
        if self.cipher_state.key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Use provided IV or last used IV
            if self.mode in self.MODES_REQUIRING_IV:
                if iv is None:
                    iv = self._last_iv
                if iv is None:
                    raise ValueError(f"IV required for {self.mode} mode")
            
            # Use provided tag or last tag (for AEAD modes)
            if self.mode in self.AEAD_MODES:
                if tag is None:
                    tag = self._last_tag
                if tag is None:
                    raise ValueError(f"Tag required for {self.mode} mode")
            
            # Create cipher instance
            backend = default_backend()
            algorithm = algorithms.AES(self.cipher_state.key)
            
            # Select mode
            if self.mode == "ECB":
                mode_obj = modes.ECB()
            elif self.mode == "CBC":
                mode_obj = modes.CBC(iv)
            elif self.mode == "CTR":
                mode_obj = modes.CTR(iv)
            elif self.mode == "GCM":
                mode_obj = modes.GCM(iv, tag)
            elif self.mode == "CFB":
                mode_obj = modes.CFB(iv)
            elif self.mode == "OFB":
                mode_obj = modes.OFB(iv)
            elif self.mode == "CCM":
                mode_obj = modes.CCM(iv, tag)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            
            # Create and perform decryption
            cipher = Cipher(algorithm, mode_obj, backend=backend)
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove padding if required
            if self.mode in self.MODES_REQUIRING_PADDING:
                plaintext = self.unpad_ciphertext(plaintext)
            
            t_end = time.perf_counter()
            decryption_time_ms = (t_end - t_start) * 1000
            
            # Create metadata
            metadata = DecryptionMetadata(
                success=True,
                decryption_time_ms=decryption_time_ms,
                recovered_size=len(plaintext),
            )
            
            return plaintext, metadata
            
        except Exception as e:
            t_end = time.perf_counter()
            decryption_time_ms = (t_end - t_start) * 1000
            
            # Return error metadata
            metadata = DecryptionMetadata(
                success=False,
                decryption_time_ms=decryption_time_ms,
                recovered_size=0,
                error_message=str(e),
            )
            
            return b"", metadata


class ChaCha20CipherSystem(SymmetricStreamCipherSystem):
    """
    ChaCha20 stream cipher implementation.
    
    ChaCha20 is a modern, fast, and secure stream cipher designed by Daniel J. Bernstein.
    
    Key Size: 256 bits
    Nonce Size: 128 bits (16 bytes) - required by cryptography library
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize ChaCha20 cipher.
        
        Args:
            seed: Optional RNG seed
        """
        algorithm_name = "ChaCha20"
        super().__init__(
            algorithm_name=algorithm_name,
            key_size_bits=256,
            nonce_size=128,  # cryptography library requires 128-bit nonce
            stateful=False,
            seed=seed
        )
        
        self._last_nonce = None
    
    def generate_key(self) -> None:
        """Generate ChaCha20 key."""
        import os
        import numpy as np
        
        t_start = time.perf_counter()
        
        # Generate 256-bit key
        if self.cipher_state.seed is not None:
            rng = np.random.RandomState(self.cipher_state.seed)
            key = rng.bytes(32)  # 256 bits = 32 bytes
        else:
            key = os.urandom(32)
        
        # Estimate entropy
        entropy = estimate_entropy(key)
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Set key in state
        self.cipher_state.set_key(key, entropy, generation_time_ms)
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        Encrypt plaintext using ChaCha20.
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            Tuple of (ciphertext, metadata)
        """
        if not self.validate_key():
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Generate nonce
            nonce = self.generate_nonce()
            self._last_nonce = nonce
            
            # Create cipher
            backend = default_backend()
            algorithm = algorithms.ChaCha20(self.cipher_state.key, nonce)
            cipher = Cipher(algorithm, mode=None, backend=backend)
            
            # Encrypt
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            
            t_end = time.perf_counter()
            encryption_time_ms = (t_end - t_start) * 1000
            
            # Create metadata
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
        Decrypt ciphertext using ChaCha20.
        
        Args:
            ciphertext: Data to decrypt
            iv: Nonce (used as nonce parameter)
            tag: Not used for ChaCha20
        
        Returns:
            Tuple of (plaintext, metadata)
        """
        # Check if we have a key (allow READY or ENCRYPTED state)
        if self.cipher_state.key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Use provided nonce or last nonce
            nonce = iv if iv is not None else self._last_nonce
            if nonce is None:
                raise ValueError("Nonce required for ChaCha20")
            
            # Create cipher
            backend = default_backend()
            algorithm = algorithms.ChaCha20(self.cipher_state.key, nonce)
            cipher = Cipher(algorithm, mode=None, backend=backend)
            
            # Decrypt (same operation as encrypt for stream cipher)
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            t_end = time.perf_counter()
            decryption_time_ms = (t_end - t_start) * 1000
            
            # Create metadata
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


class TripleDESCipherSystem(SymmetricBlockCipherSystem):
    """
    Triple DES (3DES) cipher implementation.
    
    3DES applies DES three times to each data block.
    NIST Status: DEPRECATED (but included for dataset generation)
    
    Key Size: 192 bits (effective 168 bits)
    Block Size: 64 bits
    """
    
    def __init__(self, mode: str = "CBC", seed: Optional[int] = None):
        """
        Initialize 3DES cipher.
        
        Args:
            mode: Cipher mode (CBC or ECB)
            seed: Optional RNG seed
        """
        if mode not in ["CBC", "ECB"]:
            raise ValueError(f"Invalid 3DES mode: {mode}. Must be CBC or ECB.")
        
        algorithm_name = f"3DES-{mode}"
        super().__init__(
            algorithm_name=algorithm_name,
            key_size_bits=192,  # 3DES uses 192-bit keys (3 * 64 bits)
            block_size=64,
            mode=mode,
            seed=seed
        )
        
        self._last_iv = None
    
    def generate_key(self) -> None:
        """Generate 3DES key."""
        import os
        import numpy as np
        
        t_start = time.perf_counter()
        
        # Generate 192-bit key (24 bytes)
        if self.cipher_state.seed is not None:
            rng = np.random.RandomState(self.cipher_state.seed)
            key = rng.bytes(24)
        else:
            key = os.urandom(24)
        
        # Estimate entropy
        entropy = estimate_entropy(key)
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Set key in state
        self.cipher_state.set_key(key, entropy, generation_time_ms)
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt plaintext using 3DES."""
        if not self.validate_key():
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Generate IV if needed
            iv = None
            if self.mode == "CBC":
                iv = self.generate_iv()  # 64-bit IV for 3DES
                self._last_iv = iv
            
            # Apply padding
            data_to_encrypt = self.pad_plaintext(plaintext)
            
            # Create cipher
            backend = default_backend()
            algorithm = algorithms.TripleDES(self.cipher_state.key)
            
            if self.mode == "ECB":
                mode_obj = modes.ECB()
            else:  # CBC
                mode_obj = modes.CBC(iv)
            
            cipher = Cipher(algorithm, mode_obj, backend=backend)
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data_to_encrypt) + encryptor.finalize()
            
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
                iv=iv.hex() if iv else None,
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
        """Decrypt ciphertext using 3DES."""
        if self.cipher_state.key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Use provided IV or last IV
            if self.mode == "CBC":
                if iv is None:
                    iv = self._last_iv
                if iv is None:
                    raise ValueError("IV required for CBC mode")
            
            # Create cipher
            backend = default_backend()
            algorithm = algorithms.TripleDES(self.cipher_state.key)
            
            if self.mode == "ECB":
                mode_obj = modes.ECB()
            else:  # CBC
                mode_obj = modes.CBC(iv)
            
            cipher = Cipher(algorithm, mode_obj, backend=backend)
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove padding
            plaintext = self.unpad_ciphertext(plaintext)
            
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


class BlowfishCipherSystem(SymmetricBlockCipherSystem):
    """
    Blowfish cipher implementation using PyCryptodome.
    
    Blowfish is a symmetric block cipher with variable key length.
    
    Key Sizes: 128, 192, 256 bits (can be 32-448 bits)
    Block Size: 64 bits
    """
    
    def __init__(self, key_size: int = 128, mode: str = "CBC", seed: Optional[int] = None):
        """
        Initialize Blowfish cipher.
        
        Args:
            key_size: Key size in bits (128, 192, or 256)
            mode: Cipher mode (CBC, ECB, or CTR)
            seed: Optional RNG seed
        """
        if not PYCRYPTODOME_AVAILABLE:
            raise ImportError("PyCryptodome is required for Blowfish. Install with: pip install pycryptodome")
        
        if key_size not in [128, 192, 256]:
            raise ValueError(f"Invalid Blowfish key size: {key_size}. Must be 128, 192, or 256.")
        
        if mode not in ["CBC", "ECB", "CTR"]:
            raise ValueError(f"Invalid Blowfish mode: {mode}. Must be CBC, ECB, or CTR.")
        
        algorithm_name = f"Blowfish-{key_size}-{mode}"
        super().__init__(
            algorithm_name=algorithm_name,
            key_size_bits=key_size,
            block_size=64,
            mode=mode,
            seed=seed
        )
        
        self._last_iv = None
    
    def generate_key(self) -> None:
        """Generate Blowfish key."""
        import os
        import numpy as np
        
        t_start = time.perf_counter()
        
        key_size_bytes = self.key_size_bits // 8
        
        if self.cipher_state.seed is not None:
            rng = np.random.RandomState(self.cipher_state.seed)
            key = rng.bytes(key_size_bytes)
        else:
            key = os.urandom(key_size_bytes)
        
        entropy = estimate_entropy(key)
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        self.cipher_state.set_key(key, entropy, generation_time_ms)
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt plaintext using Blowfish."""
        if not self.validate_key():
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Generate IV if needed
            iv = None
            if self.mode in ["CBC", "CTR"]:
                iv = self.generate_iv()
                self._last_iv = iv
            
            # Apply padding for CBC and ECB
            if self.mode in ["CBC", "ECB"]:
                data_to_encrypt = self.pad_plaintext(plaintext)
            else:
                data_to_encrypt = plaintext
            
            # Create cipher using PyCryptodome
            if self.mode == "ECB":
                cipher = PyCryptoBlowfish.new(self.cipher_state.key, PyCryptoBlowfish.MODE_ECB)
            elif self.mode == "CBC":
                cipher = PyCryptoBlowfish.new(self.cipher_state.key, PyCryptoBlowfish.MODE_CBC, iv=iv)
            elif self.mode == "CTR":
                from Crypto.Util import Counter
                ctr = Counter.new(64, initial_value=int.from_bytes(iv, 'big'))
                cipher = PyCryptoBlowfish.new(self.cipher_state.key, PyCryptoBlowfish.MODE_CTR, counter=ctr)
            
            ciphertext = cipher.encrypt(data_to_encrypt)
            
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
                iv=iv.hex() if iv else None,
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
        """Decrypt ciphertext using Blowfish."""
        if self.cipher_state.key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Use provided IV or last IV
            if self.mode in ["CBC", "CTR"]:
                if iv is None:
                    iv = self._last_iv
                if iv is None:
                    raise ValueError(f"IV required for {self.mode} mode")
            
            # Create cipher using PyCryptodome
            if self.mode == "ECB":
                cipher = PyCryptoBlowfish.new(self.cipher_state.key, PyCryptoBlowfish.MODE_ECB)
            elif self.mode == "CBC":
                cipher = PyCryptoBlowfish.new(self.cipher_state.key, PyCryptoBlowfish.MODE_CBC, iv=iv)
            elif self.mode == "CTR":
                from Crypto.Util import Counter
                ctr = Counter.new(64, initial_value=int.from_bytes(iv, 'big'))
                cipher = PyCryptoBlowfish.new(self.cipher_state.key, PyCryptoBlowfish.MODE_CTR, counter=ctr)
            
            plaintext = cipher.decrypt(ciphertext)
            
            # Remove padding for CBC and ECB
            if self.mode in ["CBC", "ECB"]:
                plaintext = self.unpad_ciphertext(plaintext)
            
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


class CamelliaCipherSystem(SymmetricBlockCipherSystem):
    """
    Camellia cipher implementation using PyCryptodome.
    
    Camellia is a symmetric block cipher developed by Mitsubishi and NTT.
    
    Key Sizes: 128, 192, 256 bits
    Block Size: 128 bits
    """
    
    def __init__(self, key_size: int = 128, mode: str = "CBC", seed: Optional[int] = None):
        """
        Initialize Camellia cipher.
        
        Args:
            key_size: Key size in bits (128, 192, or 256)
            mode: Cipher mode (CBC, ECB, or CTR)
            seed: Optional RNG seed
        """
        if not CAMELLIA_AVAILABLE:
            raise ImportError("PyCryptodome with Camellia support is required. Install with: pip install pycryptodome")
        
        if key_size not in [128, 192, 256]:
            raise ValueError(f"Invalid Camellia key size: {key_size}. Must be 128, 192, or 256.")
        
        if mode not in ["CBC", "ECB", "CTR"]:
            raise ValueError(f"Invalid Camellia mode: {mode}. Must be CBC, ECB, or CTR.")
        
        algorithm_name = f"Camellia-{key_size}-{mode}"
        super().__init__(
            algorithm_name=algorithm_name,
            key_size_bits=key_size,
            block_size=128,
            mode=mode,
            seed=seed
        )
        
        self._last_iv = None
    
    def generate_key(self) -> None:
        """Generate Camellia key."""
        import os
        import numpy as np
        
        t_start = time.perf_counter()
        
        key_size_bytes = self.key_size_bits // 8
        
        if self.cipher_state.seed is not None:
            rng = np.random.RandomState(self.cipher_state.seed)
            key = rng.bytes(key_size_bytes)
        else:
            key = os.urandom(key_size_bytes)
        
        entropy = estimate_entropy(key)
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        self.cipher_state.set_key(key, entropy, generation_time_ms)
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt plaintext using Camellia."""
        if not self.validate_key():
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Generate IV if needed
            iv = None
            if self.mode in ["CBC", "CTR"]:
                iv = self.generate_iv()
                self._last_iv = iv
            
            # Apply padding for CBC and ECB
            if self.mode in ["CBC", "ECB"]:
                data_to_encrypt = self.pad_plaintext(plaintext)
            else:
                data_to_encrypt = plaintext
            
            # Create cipher using PyCryptodome
            if self.mode == "ECB":
                cipher = PyCryptoCamellia.new(self.cipher_state.key, PyCryptoCamellia.MODE_ECB)
            elif self.mode == "CBC":
                cipher = PyCryptoCamellia.new(self.cipher_state.key, PyCryptoCamellia.MODE_CBC, iv=iv)
            elif self.mode == "CTR":
                from Crypto.Util import Counter
                ctr = Counter.new(128, initial_value=int.from_bytes(iv, 'big'))
                cipher = PyCryptoCamellia.new(self.cipher_state.key, PyCryptoCamellia.MODE_CTR, counter=ctr)
            
            ciphertext = cipher.encrypt(data_to_encrypt)
            
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
                iv=iv.hex() if iv else None,
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
        """Decrypt ciphertext using Camellia."""
        if self.cipher_state.key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Use provided IV or last IV
            if self.mode in ["CBC", "CTR"]:
                if iv is None:
                    iv = self._last_iv
                if iv is None:
                    raise ValueError(f"IV required for {self.mode} mode")
            
            # Create cipher using PyCryptodome
            if self.mode == "ECB":
                cipher = PyCryptoCamellia.new(self.cipher_state.key, PyCryptoCamellia.MODE_ECB)
            elif self.mode == "CBC":
                cipher = PyCryptoCamellia.new(self.cipher_state.key, PyCryptoCamellia.MODE_CBC, iv=iv)
            elif self.mode == "CTR":
                from Crypto.Util import Counter
                ctr = Counter.new(128, initial_value=int.from_bytes(iv, 'big'))
                cipher = PyCryptoCamellia.new(self.cipher_state.key, PyCryptoCamellia.MODE_CTR, counter=ctr)
            
            plaintext = cipher.decrypt(ciphertext)
            
            # Remove padding for CBC and ECB
            if self.mode in ["CBC", "ECB"]:
                plaintext = self.unpad_ciphertext(plaintext)
            
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


# ============================================================================
# LEGACY CIPHERS (AVOID in production - for research/analysis only)
# ============================================================================

class DESCipherSystem(SymmetricBlockCipherSystem):
    """DES - AVOID: Insecure, 56-bit key"""
    
    def __init__(self, mode: str = "CBC", seed: Optional[int] = None):
        if not LEGACY_CIPHERS_AVAILABLE:
            raise ImportError("DES requires PyCryptodome")
        super().__init__(
            algorithm_name=f"DES-{mode}",
            key_size_bits=64,  # 64-bit key (56 effective)
            block_size=64,
            mode=mode,
            seed=seed
        )
    
    def generate_key(self):
        """Generate DES key."""
        import time
        t_start = time.perf_counter()
        
        if self.cipher_state.seed is not None:
            import numpy as np
            rng = np.random.RandomState(self.cipher_state.seed)
            key = rng.bytes(8)  # 8 bytes = 64 bits
        else:
            import os
            key = os.urandom(8)
        
        entropy = estimate_entropy(key)
        t_end = time.perf_counter()
        self.cipher_state.set_key(key, entropy, (t_end - t_start) * 1000)
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        if not self.validate_key():
            raise RuntimeError("Key not generated")
        
        import time
        t_start = time.perf_counter()
        
        try:
            iv = None
            if self.mode == "CBC":
                iv = self.generate_iv()
                cipher = DES.new(self.cipher_state.key, DES.MODE_CBC, iv=iv)
            else:  # ECB
                cipher = DES.new(self.cipher_state.key, DES.MODE_ECB)
            
            data = self.pad_plaintext(plaintext)
            ciphertext = cipher.encrypt(data)
            
            t_end = time.perf_counter()
            
            metadata = EncryptionMetadata(
                algorithm=self.algorithm_name,
                plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext),
                key_size_bits=56,  # Effective key size
                ciphertext_length=len(ciphertext),
                encryption_time_ms=(t_end - t_start) * 1000,
                success=True,
                iv=iv.hex() if iv else None,
            )
            
            self._last_iv = iv
            return ciphertext, metadata
        except Exception as e:
            return b"", EncryptionMetadata(algorithm=self.algorithm_name, success=False, error_message=str(e))
    
    def decrypt(self, ciphertext: bytes, iv: Optional[bytes] = None, tag: Optional[bytes] = None) -> Tuple[bytes, DecryptionMetadata]:
        if not self.cipher_state.key:
            raise RuntimeError("Key not generated")
        
        import time
        t_start = time.perf_counter()
        
        try:
            if self.mode == "CBC":
                iv = iv or self._last_iv
                cipher = DES.new(self.cipher_state.key, DES.MODE_CBC, iv=iv)
            else:
                cipher = DES.new(self.cipher_state.key, DES.MODE_ECB)
            
            plaintext = cipher.decrypt(ciphertext)
            plaintext = self.unpad_ciphertext(plaintext)
            
            t_end = time.perf_counter()
            return plaintext, DecryptionMetadata(success=True, decryption_time_ms=(t_end - t_start) * 1000, recovered_size=len(plaintext))
        except Exception as e:
            return b"", DecryptionMetadata(success=False, error_message=str(e))


class RC2CipherSystem(SymmetricBlockCipherSystem):
    """RC2 - AVOID: Obsolete"""
    
    def __init__(self, key_size: int = 128, mode: str = "CBC", seed: Optional[int] = None):
        if not LEGACY_CIPHERS_AVAILABLE:
            raise ImportError("RC2 requires PyCryptodome")
        super().__init__(
            algorithm_name=f"RC2-{key_size}-{mode}",
            key_size_bits=key_size,
            block_size=64,
            mode=mode,
            seed=seed
        )
    
    def generate_key(self):
        import time, os
        t_start = time.perf_counter()
        key = os.urandom(self.key_size_bits // 8)
        self.cipher_state.set_key(key, estimate_entropy(key), (time.perf_counter() - t_start) * 1000)
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        if not self.validate_key():
            raise RuntimeError("Key not generated")
        
        import time
        t_start = time.perf_counter()
        
        try:
            iv = None
            if self.mode == "CBC":
                iv = self.generate_iv()
                cipher = ARC2.new(self.cipher_state.key, ARC2.MODE_CBC, iv=iv)
            else:
                cipher = ARC2.new(self.cipher_state.key, ARC2.MODE_ECB)
            
            data = self.pad_plaintext(plaintext)
            ciphertext = cipher.encrypt(data)
            
            t_end = time.perf_counter()
            self._last_iv = iv
            
            return ciphertext, EncryptionMetadata(
                algorithm=self.algorithm_name, plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext), key_size_bits=self.key_size_bits,
                ciphertext_length=len(ciphertext), encryption_time_ms=(t_end - t_start) * 1000,
                success=True, iv=iv.hex() if iv else None
            )
        except Exception as e:
            return b"", EncryptionMetadata(algorithm=self.algorithm_name, success=False, error_message=str(e))
    
    def decrypt(self, ciphertext: bytes, iv: Optional[bytes] = None, tag: Optional[bytes] = None) -> Tuple[bytes, DecryptionMetadata]:
        import time
        t_start = time.perf_counter()
        try:
            if self.mode == "CBC":
                iv = iv or self._last_iv
                cipher = ARC2.new(self.cipher_state.key, ARC2.MODE_CBC, iv=iv)
            else:
                cipher = ARC2.new(self.cipher_state.key, ARC2.MODE_ECB)
            
            plaintext = self.unpad_ciphertext(cipher.decrypt(ciphertext))
            return plaintext, DecryptionMetadata(success=True, decryption_time_ms=(time.perf_counter() - t_start) * 1000, recovered_size=len(plaintext))
        except Exception as e:
            return b"", DecryptionMetadata(success=False, error_message=str(e))


class RC4CipherSystem(SymmetricStreamCipherSystem):
    """RC4 - AVOID: Broken stream cipher"""
    
    def __init__(self, key_size: int = 128, seed: Optional[int] = None):
        if not LEGACY_CIPHERS_AVAILABLE:
            raise ImportError("RC4 requires PyCryptodome")
        super().__init__(
            algorithm_name=f"RC4-{key_size}",
            key_size_bits=key_size,
            nonce_size=0,  # RC4 doesn't use nonce
            stateful=True,
            seed=seed
        )
    
    def generate_key(self):
        import time, os
        t_start = time.perf_counter()
        key = os.urandom(self.key_size_bits // 8)
        self.cipher_state.set_key(key, estimate_entropy(key), (time.perf_counter() - t_start) * 1000)
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        if not self.validate_key():
            raise RuntimeError("Key not generated")
        
        import time
        t_start = time.perf_counter()
        
        try:
            cipher = ARC4.new(self.cipher_state.key)
            ciphertext = cipher.encrypt(plaintext)
            
            return ciphertext, EncryptionMetadata(
                algorithm=self.algorithm_name, plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext), key_size_bits=self.key_size_bits,
                ciphertext_length=len(ciphertext), encryption_time_ms=(time.perf_counter() - t_start) * 1000,
                success=True
            )
        except Exception as e:
            return b"", EncryptionMetadata(algorithm=self.algorithm_name, success=False, error_message=str(e))
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        import time
        t_start = time.perf_counter()
        try:
            cipher = ARC4.new(self.cipher_state.key)
            plaintext = cipher.decrypt(ciphertext)
            return plaintext, DecryptionMetadata(success=True, decryption_time_ms=(time.perf_counter() - t_start) * 1000, recovered_size=len(plaintext))
        except Exception as e:
            return b"", DecryptionMetadata(success=False, error_message=str(e))


class CASTCipherSystem(SymmetricBlockCipherSystem):
    """CAST-128 cipher"""
    
    def __init__(self, key_size: int = 128, mode: str = "CBC", seed: Optional[int] = None):
        if not LEGACY_CIPHERS_AVAILABLE:
            raise ImportError("CAST requires PyCryptodome")
        super().__init__(
            algorithm_name=f"CAST-{key_size}-{mode}",
            key_size_bits=key_size,
            block_size=64,
            mode=mode,
            seed=seed
        )
    
    def generate_key(self):
        import time, os
        t_start = time.perf_counter()
        key = os.urandom(self.key_size_bits // 8)
        self.cipher_state.set_key(key, estimate_entropy(key), (time.perf_counter() - t_start) * 1000)
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        if not self.validate_key():
            raise RuntimeError("Key not generated")
        
        import time
        t_start = time.perf_counter()
        
        try:
            iv = None
            if self.mode == "CBC":
                iv = self.generate_iv()
                cipher = CAST.new(self.cipher_state.key, CAST.MODE_CBC, iv=iv)
            else:
                cipher = CAST.new(self.cipher_state.key, CAST.MODE_ECB)
            
            data = self.pad_plaintext(plaintext)
            ciphertext = cipher.encrypt(data)
            self._last_iv = iv
            
            return ciphertext, EncryptionMetadata(
                algorithm=self.algorithm_name, plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext), key_size_bits=self.key_size_bits,
                ciphertext_length=len(ciphertext), encryption_time_ms=(time.perf_counter() - t_start) * 1000,
                success=True, iv=iv.hex() if iv else None
            )
        except Exception as e:
            return b"", EncryptionMetadata(algorithm=self.algorithm_name, success=False, error_message=str(e))
    
    def decrypt(self, ciphertext: bytes, iv: Optional[bytes] = None, tag: Optional[bytes] = None) -> Tuple[bytes, DecryptionMetadata]:
        import time
        t_start = time.perf_counter()
        try:
            if self.mode == "CBC":
                iv = iv or self._last_iv
                cipher = CAST.new(self.cipher_state.key, CAST.MODE_CBC, iv=iv)
            else:
                cipher = CAST.new(self.cipher_state.key, CAST.MODE_ECB)
            
            plaintext = self.unpad_ciphertext(cipher.decrypt(ciphertext))
            return plaintext, DecryptionMetadata(success=True, decryption_time_ms=(time.perf_counter() - t_start) * 1000, recovered_size=len(plaintext))
        except Exception as e:
            return b"", DecryptionMetadata(success=False, error_message=str(e))


class Salsa20CipherSystem(SymmetricStreamCipherSystem):
    """Salsa20 stream cipher"""
    
    def __init__(self, key_size: int = 256, seed: Optional[int] = None):
        if not PYNACL_AVAILABLE:
            raise ImportError("Salsa20 requires PyNaCl")
        super().__init__(
            algorithm_name=f"Salsa20-{key_size}",
            key_size_bits=key_size,
            nonce_size=192,  # Salsa20 uses 24-byte nonce
            stateful=False,
            seed=seed
        )
    
    def generate_key(self):
        import time
        t_start = time.perf_counter()
        key = nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)
        self.cipher_state.set_key(key, estimate_entropy(key), (time.perf_counter() - t_start) * 1000)
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        if not self.validate_key():
            raise RuntimeError("Key not generated")
        
        import time
        t_start = time.perf_counter()
        
        try:
            box = nacl.secret.SecretBox(self.cipher_state.key)
            ciphertext = box.encrypt(plaintext)
            
            return ciphertext, EncryptionMetadata(
                algorithm=self.algorithm_name, plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext), key_size_bits=self.key_size_bits,
                ciphertext_length=len(ciphertext), encryption_time_ms=(time.perf_counter() - t_start) * 1000,
                success=True
            )
        except Exception as e:
            return b"", EncryptionMetadata(algorithm=self.algorithm_name, success=False, error_message=str(e))
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        import time
        t_start = time.perf_counter()
        try:
            box = nacl.secret.SecretBox(self.cipher_state.key)
            plaintext = box.decrypt(ciphertext)
            return plaintext, DecryptionMetadata(success=True, decryption_time_ms=(time.perf_counter() - t_start) * 1000, recovered_size=len(plaintext))
        except Exception as e:
            return b"", DecryptionMetadata(success=False, error_message=str(e))


# Export all cipher classes
__all__ = [
    "AESCipherSystem",
    "ChaCha20CipherSystem",
    "TripleDESCipherSystem",
    "BlowfishCipherSystem",
    "CamelliaCipherSystem",
    "DESCipherSystem",
    "RC2CipherSystem",
    "RC4CipherSystem",
    "CASTCipherSystem",
    "Salsa20CipherSystem",
    "PYCRYPTODOME_AVAILABLE",
    "CAMELLIA_AVAILABLE",
    "LEGACY_CIPHERS_AVAILABLE",
    "PYNACL_AVAILABLE",
]

