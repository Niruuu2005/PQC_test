"""
Post-Quantum Cipher Implementations

This module implements post-quantum cryptographic (PQC) algorithms.

Version: 1.0
Date: December 30, 2025
"""

import time
from typing import Tuple, Optional

from .base_cipher import PQCKEMSystem, PQCSignatureSystem
from .state_management import (
    EncryptionMetadata,
    DecryptionMetadata,
    compute_plaintext_hash,
    estimate_entropy,
)

# Check for liboqs availability
try:
    import os
    # Disable auto-installer before importing
    os.environ['OQS_PYTHON_DISABLE_AUTO_INSTALL'] = '1'
    import oqs
    LIBOQS_AVAILABLE = True
    print("✅ liboqs library loaded successfully")
except (ImportError, RuntimeError, OSError) as e:
    LIBOQS_AVAILABLE = False
    print(f"⚠️  liboqs not available ({type(e).__name__}): Skipping PQC algorithms")


class MLKEMSystem(PQCKEMSystem):
    """
    ML-KEM (Module-Lattice-Based Key-Encapsulation Mechanism) implementation.
    
    ML-KEM is the NIST-approved post-quantum KEM algorithm (FIPS 203).
    Previously known as CRYSTALS-Kyber.
    
    Variants: ML-KEM-512, ML-KEM-768, ML-KEM-1024
    
    Since KEMs don't directly encrypt data, we use a hybrid approach:
    1. Use KEM to establish a shared secret
    2. Use shared secret with AES-256-GCM to encrypt actual data
    """
    
    # Variant mapping to liboqs algorithm names
    VARIANT_MAP = {
        "ML-KEM-512": "ML-KEM-512",
        "ML-KEM-768": "ML-KEM-768",
        "ML-KEM-1024": "ML-KEM-1024",
    }
    
    def __init__(self, variant: str = "ML-KEM-768", seed: Optional[int] = None):
        """
        Initialize ML-KEM cipher.
        
        Args:
            variant: ML-KEM variant ("ML-KEM-512", "ML-KEM-768", or "ML-KEM-1024")
            seed: Optional RNG seed
        """
        if not LIBOQS_AVAILABLE:
            raise ImportError("liboqs-python is required for ML-KEM. Install with: pip install liboqs-python")
        
        variant_upper = variant.upper()
        if variant_upper not in self.VARIANT_MAP:
            raise ValueError(f"Invalid ML-KEM variant: {variant}. Must be ML-KEM-512, ML-KEM-768, or ML-KEM-1024.")
        
        # Extract key size from variant name
        key_size = int(variant_upper.split("-")[-1])
        
        algorithm_name = f"ML-KEM-{key_size}"
        super().__init__(
            algorithm_name=algorithm_name,
            key_size_bits=key_size,
            variant=variant_upper,
            seed=seed
        )
        
        self._public_key = None
        self._secret_key = None
        self._kem_algorithm = self.VARIANT_MAP[variant_upper]
    
    def generate_key(self) -> None:
        """Generate ML-KEM key pair."""
        t_start = time.perf_counter()
        
        # Generate key pair using liboqs
        with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Store keys
        self._public_key = public_key
        self._secret_key = secret_key
        
        # Estimate entropy
        entropy = estimate_entropy(public_key)
        
        # Update cipher state
        self.cipher_state.set_key(public_key, entropy, generation_time_ms)
        self.cipher_state.public_key = public_key
        self.cipher_state.private_key = secret_key
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        Encrypt plaintext using hybrid ML-KEM + AES-GCM.
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            Tuple of (ciphertext, metadata) where ciphertext = kem_ciphertext + aes_ciphertext
        """
        if self._public_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Step 1: KEM Encapsulation (generate shared secret)
            with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
                # Set the public key
                kem_ciphertext, shared_secret = kem.encap_secret(self._public_key)
            
            # Step 2: Use shared secret as key for AES-256-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            import os
            
            # Derive AES key from shared secret (use first 32 bytes)
            aes_key = shared_secret[:32]
            
            # Generate nonce
            nonce = os.urandom(12)  # 96-bit nonce for GCM
            
            # Encrypt with AES-GCM
            aesgcm = AESGCM(aes_key)
            aes_ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            
            # Combine: kem_ciphertext + nonce + aes_ciphertext
            # Format: [kem_ciphertext_length(4 bytes)] + kem_ciphertext + nonce + aes_ciphertext
            import struct
            kem_ct_len = struct.pack('>I', len(kem_ciphertext))
            ciphertext = kem_ct_len + kem_ciphertext + nonce + aes_ciphertext
            
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
        Decrypt ciphertext using hybrid ML-KEM + AES-GCM.
        
        Args:
            ciphertext: Data to decrypt (format: kem_ct_len + kem_ciphertext + nonce + aes_ciphertext)
            iv: Not used
            tag: Not used
        
        Returns:
            Tuple of (plaintext, metadata)
        """
        if self._secret_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Parse ciphertext
            import struct
            
            if len(ciphertext) < 4:
                raise ValueError("Ciphertext too short")
            
            # Extract KEM ciphertext length
            kem_ct_len = struct.unpack('>I', ciphertext[:4])[0]
            
            if len(ciphertext) < 4 + kem_ct_len + 12:
                raise ValueError("Ciphertext malformed")
            
            # Extract components
            kem_ciphertext = ciphertext[4:4 + kem_ct_len]
            nonce = ciphertext[4 + kem_ct_len:4 + kem_ct_len + 12]
            aes_ciphertext = ciphertext[4 + kem_ct_len + 12:]
            
            # Step 1: KEM Decapsulation (recover shared secret)
            with oqs.KeyEncapsulation(self._kem_algorithm, secret_key=self._secret_key) as kem:
                shared_secret = kem.decap_secret(kem_ciphertext)
            
            # Step 2: Decrypt with AES-256-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            aes_key = shared_secret[:32]
            aesgcm = AESGCM(aes_key)
            plaintext = aesgcm.decrypt(nonce, aes_ciphertext, None)
            
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


class KyberCrystalsSystem(MLKEMSystem):
    """
    CRYSTALS-Kyber implementation (alias for ML-KEM).
    
    CRYSTALS-Kyber is the original name for what became ML-KEM.
    This class is provided for compatibility.
    
    Variants: CRYSTALS-Kyber-512, CRYSTALS-Kyber-768, CRYSTALS-Kyber-1024
    """
    
    VARIANT_MAP = {
        "CRYSTALS-KYBER-512": "Kyber512",
        "CRYSTALS-KYBER-768": "Kyber768",
        "CRYSTALS-KYBER-1024": "Kyber1024",
        "KYBER-512": "Kyber512",
        "KYBER-768": "Kyber768",
        "KYBER-1024": "Kyber1024",
    }
    
    def __init__(self, variant: str = "CRYSTALS-Kyber-768", seed: Optional[int] = None):
        """
        Initialize CRYSTALS-Kyber cipher.
        
        Args:
            variant: Kyber variant
            seed: Optional RNG seed
        """
        if not LIBOQS_AVAILABLE:
            raise ImportError("liboqs-python is required for CRYSTALS-Kyber. Install with: pip install liboqs-python")
        
        variant_upper = variant.upper()
        if variant_upper not in self.VARIANT_MAP:
            raise ValueError(f"Invalid Kyber variant: {variant}")
        
        # Extract key size
        if "512" in variant_upper:
            key_size = 512
        elif "768" in variant_upper:
            key_size = 768
        elif "1024" in variant_upper:
            key_size = 1024
        else:
            key_size = 768
        
        # Initialize as ML-KEM but with Kyber name
        self.key_size_bits = key_size
        self.variant = variant_upper
        self.algorithm_name = f"CRYSTALS-Kyber-{key_size}"
        self.cipher_state = None
        self._public_key = None
        self._secret_key = None
        self._kem_algorithm = self.VARIANT_MAP[variant_upper]
        
        # Initialize cipher state
        from .state_management import CipherState
        self.cipher_state = CipherState(self.algorithm_name, seed)


class HQCSystem(PQCKEMSystem):
    """
    HQC (Hamming Quasi-Cyclic) implementation.
    
    HQC is a code-based post-quantum KEM algorithm.
    
    Variants: HQC-128, HQC-192, HQC-256
    """
    
    VARIANT_MAP = {
        "HQC-128": "HQC-128",
        "HQC-192": "HQC-192",
        "HQC-256": "HQC-256",
    }
    
    def __init__(self, variant: str = "HQC-192", seed: Optional[int] = None):
        """
        Initialize HQC cipher.
        
        Args:
            variant: HQC variant ("HQC-128", "HQC-192", or "HQC-256")
            seed: Optional RNG seed
        """
        if not LIBOQS_AVAILABLE:
            raise ImportError("liboqs-python is required for HQC. Install with: pip install liboqs-python")
        
        variant_upper = variant.upper()
        if variant_upper not in self.VARIANT_MAP:
            raise ValueError(f"Invalid HQC variant: {variant}. Must be HQC-128, HQC-192, or HQC-256.")
        
        # Extract security level from variant name
        security_level = int(variant_upper.split("-")[-1])
        
        algorithm_name = f"HQC-{security_level}"
        super().__init__(
            algorithm_name=algorithm_name,
            key_size_bits=security_level,
            variant=variant_upper,
            seed=seed
        )
        
        self._public_key = None
        self._secret_key = None
        self._kem_algorithm = self.VARIANT_MAP[variant_upper]
    
    def generate_key(self) -> None:
        """Generate HQC key pair."""
        t_start = time.perf_counter()
        
        # Generate key pair using liboqs
        with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Store keys
        self._public_key = public_key
        self._secret_key = secret_key
        
        # Estimate entropy
        entropy = estimate_entropy(public_key)
        
        # Update cipher state
        self.cipher_state.set_key(public_key, entropy, generation_time_ms)
        self.cipher_state.public_key = public_key
        self.cipher_state.private_key = secret_key
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        Encrypt plaintext using hybrid HQC + AES-GCM.
        
        Same approach as ML-KEM: use KEM for key agreement, AES-GCM for encryption.
        """
        if self._public_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # KEM Encapsulation
            with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
                kem_ciphertext, shared_secret = kem.encap_secret(self._public_key)
            
            # Encrypt with AES-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            import os
            
            aes_key = shared_secret[:32]
            nonce = os.urandom(12)
            
            aesgcm = AESGCM(aes_key)
            aes_ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            
            # Combine
            import struct
            kem_ct_len = struct.pack('>I', len(kem_ciphertext))
            ciphertext = kem_ct_len + kem_ciphertext + nonce + aes_ciphertext
            
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
        Decrypt ciphertext using hybrid HQC + AES-GCM.
        """
        if self._secret_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Parse ciphertext
            import struct
            
            if len(ciphertext) < 4:
                raise ValueError("Ciphertext too short")
            
            kem_ct_len = struct.unpack('>I', ciphertext[:4])[0]
            
            if len(ciphertext) < 4 + kem_ct_len + 12:
                raise ValueError("Ciphertext malformed")
            
            kem_ciphertext = ciphertext[4:4 + kem_ct_len]
            nonce = ciphertext[4 + kem_ct_len:4 + kem_ct_len + 12]
            aes_ciphertext = ciphertext[4 + kem_ct_len + 12:]
            
            # KEM Decapsulation
            with oqs.KeyEncapsulation(self._kem_algorithm, secret_key=self._secret_key) as kem:
                shared_secret = kem.decap_secret(kem_ciphertext)
            
            # Decrypt with AES-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            aes_key = shared_secret[:32]
            aesgcm = AESGCM(aes_key)
            plaintext = aesgcm.decrypt(nonce, aes_ciphertext, None)
            
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
# NTRU (Key Encapsulation Mechanism)
# ============================================================================

class NTRUSystem(PQCKEMSystem):
    """
    NTRU Key Encapsulation Mechanism.
    
    NTRU is a lattice-based KEM that was a NIST Round 3 finalist.
    It offers fast operations and moderate key sizes.
    
    Variants:
    - NTRU-HPS-2048-509: 509-bit security
    - NTRU-HPS-2048-677: 677-bit security
    - NTRU-HPS-4096-821: 821-bit security
    - NTRU-HRSS-701: 701-bit security (HRSS variant)
    
    Uses hybrid approach: NTRU KEM + AES-256-GCM for data encryption.
    """
    
    VARIANT_MAP = {
        "NTRU-HPS-2048-509": "NTRU-HPS-2048-509",
        "NTRU-HPS-2048-677": "NTRU-HPS-2048-677",
        "NTRU-HPS-4096-821": "NTRU-HPS-4096-821",
        "NTRU-HRSS-701": "NTRU-HRSS-701",
    }
    
    def __init__(self, variant: str = "NTRU-HPS-2048-509", seed: Optional[int] = None):
        """
        Initialize NTRU KEM.
        
        Args:
            variant: NTRU variant
            seed: Optional RNG seed
        """
        if not LIBOQS_AVAILABLE:
            raise ImportError("liboqs-python is required for NTRU. Install with: pip install liboqs-python")
        
        variant_upper = variant.upper()
        if variant_upper not in self.VARIANT_MAP:
            raise ValueError(f"Invalid NTRU variant: {variant}. Must be one of {list(self.VARIANT_MAP.keys())}")
        
        # Extract security level from variant name
        if "509" in variant_upper:
            key_size = 509
        elif "677" in variant_upper:
            key_size = 677
        elif "821" in variant_upper:
            key_size = 821
        elif "701" in variant_upper:
            key_size = 701
        else:
            key_size = 509  # Default
        
        algorithm_name = variant_upper
        super().__init__(
            algorithm_name=algorithm_name,
            key_size_bits=key_size,
            variant=variant_upper,
            seed=seed
        )
        
        self._public_key = None
        self._secret_key = None
        self._kem_algorithm = self.VARIANT_MAP[variant_upper]
    
    def generate_key(self) -> None:
        """Generate NTRU key pair."""
        t_start = time.perf_counter()
        
        # Generate key pair using liboqs
        with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Store keys
        self._public_key = public_key
        self._secret_key = secret_key
        
        # Estimate entropy
        entropy = estimate_entropy(public_key)
        
        # Update cipher state
        self.cipher_state.set_key(public_key, entropy, generation_time_ms)
        self.cipher_state.public_key = public_key
        self.cipher_state.private_key = secret_key
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        Encrypt plaintext using hybrid NTRU + AES-GCM.
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            Tuple of (ciphertext, metadata) where ciphertext = kem_ciphertext + aes_ciphertext
        """
        if self._public_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Step 1: KEM Encapsulation
            with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
                kem_ciphertext, shared_secret = kem.encap_secret(self._public_key)
            
            # Step 2: Use shared secret for AES-256-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            import os
            
            aes_key = shared_secret[:32]
            nonce = os.urandom(12)
            
            aesgcm = AESGCM(aes_key)
            aes_ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            
            # Combine: [kem_ct_length] + kem_ct + nonce + aes_ct
            import struct
            kem_ct_len = struct.pack('>I', len(kem_ciphertext))
            ciphertext = kem_ct_len + kem_ciphertext + nonce + aes_ciphertext
            
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
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        """
        Decrypt ciphertext using hybrid NTRU + AES-GCM.
        
        Args:
            ciphertext: Encrypted data
        
        Returns:
            Tuple of (plaintext, metadata)
        """
        if self._secret_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            import struct
            
            # Extract kem_ciphertext length
            kem_ct_len = struct.unpack('>I', ciphertext[:4])[0]
            offset = 4
            
            # Extract kem_ciphertext
            kem_ciphertext = ciphertext[offset:offset+kem_ct_len]
            offset += kem_ct_len
            
            # Extract nonce (12 bytes)
            nonce = ciphertext[offset:offset+12]
            offset += 12
            
            # Extract AES ciphertext
            aes_ciphertext = ciphertext[offset:]
            
            # Step 1: KEM Decapsulation
            with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
                shared_secret = kem.decap_secret(kem_ciphertext, self._secret_key)
            
            # Step 2: Decrypt with AES-256-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            aes_key = shared_secret[:32]
            aesgcm = AESGCM(aes_key)
            plaintext = aesgcm.decrypt(nonce, aes_ciphertext, None)
            
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
# BIKE (Bit Flipping Key Encapsulation)
# ============================================================================

class BIKESystem(PQCKEMSystem):
    """
    BIKE (Bit Flipping Key Encapsulation) - Code-based KEM.
    
    BIKE is a code-based KEM based on QC-MDPC codes with bit flipping decoding.
    It offers small key sizes and fast operations.
    
    Security Levels:
    - BIKE-L1: 128-bit security (NIST Level 1)
    - BIKE-L3: 192-bit security (NIST Level 3)
    - BIKE-L5: 256-bit security (NIST Level 5)
    
    Uses hybrid approach: BIKE KEM + AES-256-GCM for data encryption.
    """
    
    VARIANT_MAP = {
        "BIKE-L1": "BIKE-L1",
        "BIKE-L3": "BIKE-L3",
        "BIKE-L5": "BIKE-L5",
    }
    
    SECURITY_LEVELS = {
        "BIKE-L1": 128,
        "BIKE-L3": 192,
        "BIKE-L5": 256,
    }
    
    def __init__(self, variant: str = "BIKE-L1", seed: Optional[int] = None):
        """
        Initialize BIKE KEM.
        
        Args:
            variant: BIKE security level
            seed: Optional RNG seed
        """
        if not LIBOQS_AVAILABLE:
            raise ImportError("liboqs-python is required for BIKE. Install with: pip install liboqs-python")
        
        variant_upper = variant.upper()
        if variant_upper not in self.VARIANT_MAP:
            raise ValueError(f"Invalid BIKE variant: {variant}. Must be one of {list(self.VARIANT_MAP.keys())}")
        
        key_size = self.SECURITY_LEVELS[variant_upper]
        
        algorithm_name = variant_upper
        super().__init__(
            algorithm_name=algorithm_name,
            key_size_bits=key_size,
            variant=variant_upper,
            seed=seed
        )
        
        self._public_key = None
        self._secret_key = None
        self._kem_algorithm = self.VARIANT_MAP[variant_upper]
    
    def generate_key(self) -> None:
        """Generate BIKE key pair."""
        t_start = time.perf_counter()
        
        # Generate key pair using liboqs
        with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Store keys
        self._public_key = public_key
        self._secret_key = secret_key
        
        # Estimate entropy
        entropy = estimate_entropy(public_key)
        
        # Update cipher state
        self.cipher_state.set_key(public_key, entropy, generation_time_ms)
        self.cipher_state.public_key = public_key
        self.cipher_state.private_key = secret_key
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        Encrypt plaintext using hybrid BIKE + AES-GCM.
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            Tuple of (ciphertext, metadata)
        """
        if self._public_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Step 1: KEM Encapsulation
            with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
                kem_ciphertext, shared_secret = kem.encap_secret(self._public_key)
            
            # Step 2: Use shared secret for AES-256-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            import os
            
            aes_key = shared_secret[:32]
            nonce = os.urandom(12)
            
            aesgcm = AESGCM(aes_key)
            aes_ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            
            # Combine: [kem_ct_length] + kem_ct + nonce + aes_ct
            import struct
            kem_ct_len = struct.pack('>I', len(kem_ciphertext))
            ciphertext = kem_ct_len + kem_ciphertext + nonce + aes_ciphertext
            
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
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        """
        Decrypt ciphertext using hybrid BIKE + AES-GCM.
        
        Args:
            ciphertext: Encrypted data
        
        Returns:
            Tuple of (plaintext, metadata)
        """
        if self._secret_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            import struct
            
            # Extract kem_ciphertext length
            kem_ct_len = struct.unpack('>I', ciphertext[:4])[0]
            offset = 4
            
            # Extract components
            kem_ciphertext = ciphertext[offset:offset+kem_ct_len]
            offset += kem_ct_len
            
            nonce = ciphertext[offset:offset+12]
            offset += 12
            
            aes_ciphertext = ciphertext[offset:]
            
            # Step 1: KEM Decapsulation
            with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
                shared_secret = kem.decap_secret(kem_ciphertext, self._secret_key)
            
            # Step 2: Decrypt with AES-256-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            aes_key = shared_secret[:32]
            aesgcm = AESGCM(aes_key)
            plaintext = aesgcm.decrypt(nonce, aes_ciphertext, None)
            
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
# Classic McEliece (Conservative Code-based KEM)
# ============================================================================

class ClassicMcElieceSystem(PQCKEMSystem):
    """
    Classic McEliece - Conservative code-based KEM.
    
    Classic McEliece is a code-based KEM with very conservative security margins.
    It has large public keys (100KB-1MB) but fast encapsulation/decapsulation.
    
    Variants:
    - McEliece348864: Smallest keys (~260KB public key)
    - McEliece460896: Medium keys (~520KB public key)
    - McEliece6688128: Large keys (~1MB public key)
    - McEliece6960119: Large keys (~1MB public key)
    - McEliece8192128: Largest keys (~1.3MB public key)
    
    Note: Due to very large key sizes, only key hashes are stored in cipher_state.
    
    Uses hybrid approach: McEliece KEM + AES-256-GCM for data encryption.
    """
    
    VARIANT_MAP = {
        "McEliece348864": "Classic-McEliece-348864",
        "McEliece460896": "Classic-McEliece-460896",
        "McEliece6688128": "Classic-McEliece-6688128",
        "McEliece6960119": "Classic-McEliece-6960119",
        "McEliece8192128": "Classic-McEliece-8192128",
    }
    
    SECURITY_LEVELS = {
        "McEliece348864": 128,
        "McEliece460896": 192,
        "McEliece6688128": 256,
        "McEliece6960119": 256,
        "McEliece8192128": 256,
    }
    
    def __init__(self, variant: str = "McEliece348864", seed: Optional[int] = None):
        """
        Initialize Classic McEliece KEM.
        
        Args:
            variant: McEliece variant
            seed: Optional RNG seed
        """
        if not LIBOQS_AVAILABLE:
            raise ImportError("liboqs-python is required for Classic McEliece. Install with: pip install liboqs-python")
        
        variant_normalized = variant.replace("-", "").replace("_", "")
        if variant_normalized not in self.VARIANT_MAP:
            raise ValueError(f"Invalid McEliece variant: {variant}. Must be one of {list(self.VARIANT_MAP.keys())}")
        
        key_size = self.SECURITY_LEVELS[variant_normalized]
        
        algorithm_name = variant_normalized
        super().__init__(
            algorithm_name=algorithm_name,
            key_size_bits=key_size,
            variant=variant_normalized,
            seed=seed
        )
        
        self._public_key = None
        self._secret_key = None
        self._kem_algorithm = self.VARIANT_MAP[variant_normalized]
    
    def generate_key(self) -> None:
        """Generate Classic McEliece key pair (may take several seconds)."""
        t_start = time.perf_counter()
        
        # Generate key pair using liboqs (slow for large variants)
        with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Store keys
        self._public_key = public_key
        self._secret_key = secret_key
        
        # Estimate entropy
        entropy = estimate_entropy(public_key[:1000])  # Sample first 1KB
        
        # Store only hash of public key due to large size
        import hashlib
        public_key_hash = hashlib.sha256(public_key).digest()
        
        # Update cipher state with key hash
        self.cipher_state.set_key(public_key_hash, entropy, generation_time_ms)
        self.cipher_state.public_key = public_key
        self.cipher_state.private_key = secret_key
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        Encrypt plaintext using hybrid McEliece + AES-GCM.
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            Tuple of (ciphertext, metadata)
        """
        if self._public_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            # Step 1: KEM Encapsulation
            with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
                kem_ciphertext, shared_secret = kem.encap_secret(self._public_key)
            
            # Step 2: Use shared secret for AES-256-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            import os
            
            aes_key = shared_secret[:32]
            nonce = os.urandom(12)
            
            aesgcm = AESGCM(aes_key)
            aes_ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            
            # Combine: [kem_ct_length] + kem_ct + nonce + aes_ct
            import struct
            kem_ct_len = struct.pack('>I', len(kem_ciphertext))
            ciphertext = kem_ct_len + kem_ciphertext + nonce + aes_ciphertext
            
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
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        """
        Decrypt ciphertext using hybrid McEliece + AES-GCM.
        
        Args:
            ciphertext: Encrypted data
        
        Returns:
            Tuple of (plaintext, metadata)
        """
        if self._secret_key is None:
            raise RuntimeError("Key not generated. Call generate_key() first.")
        
        t_start = time.perf_counter()
        
        try:
            import struct
            
            # Extract kem_ciphertext length
            kem_ct_len = struct.unpack('>I', ciphertext[:4])[0]
            offset = 4
            
            # Extract components
            kem_ciphertext = ciphertext[offset:offset+kem_ct_len]
            offset += kem_ct_len
            
            nonce = ciphertext[offset:offset+12]
            offset += 12
            
            aes_ciphertext = ciphertext[offset:]
            
            # Step 1: KEM Decapsulation
            with oqs.KeyEncapsulation(self._kem_algorithm) as kem:
                shared_secret = kem.decap_secret(kem_ciphertext, self._secret_key)
            
            # Step 2: Decrypt with AES-256-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            aes_key = shared_secret[:32]
            aesgcm = AESGCM(aes_key)
            plaintext = aesgcm.decrypt(nonce, aes_ciphertext, None)
            
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
# ML-DSA / Dilithium (Lattice-based Signature)
# ============================================================================

class MLDSASystem(PQCSignatureSystem):
    """
    ML-DSA (Module-Lattice-Based Digital Signature Algorithm).
    
    ML-DSA is the NIST-standardized post-quantum signature scheme (FIPS 204).
    Previously known as CRYSTALS-Dilithium.
    
    Variants:
    - ML-DSA-44 (Dilithium2): NIST Level 2 security
    - ML-DSA-65 (Dilithium3): NIST Level 3 security
    - ML-DSA-87 (Dilithium5): NIST Level 5 security
    
    For signature systems, "encrypt" means sign and "decrypt" means verify.
    """
    
    VARIANT_MAP = {
        "ML-DSA-44": "Dilithium2",
        "ML-DSA-65": "Dilithium3",
        "ML-DSA-87": "Dilithium5",
        "DILITHIUM2": "Dilithium2",
        "DILITHIUM3": "Dilithium3",
        "DILITHIUM5": "Dilithium5",
    }
    
    SECURITY_LEVELS = {
        "ML-DSA-44": 128,
        "ML-DSA-65": 192,
        "ML-DSA-87": 256,
        "DILITHIUM2": 128,
        "DILITHIUM3": 192,
        "DILITHIUM5": 256,
    }
    
    def __init__(self, variant: str = "ML-DSA-44", seed: Optional[int] = None):
        """
        Initialize ML-DSA signature system.
        
        Args:
            variant: ML-DSA variant
            seed: Optional RNG seed
        """
        if not LIBOQS_AVAILABLE:
            raise ImportError("liboqs-python is required for ML-DSA. Install with: pip install liboqs-python")
        
        variant_upper = variant.upper().replace("-", "")
        if variant_upper not in self.VARIANT_MAP and variant.upper() not in self.VARIANT_MAP:
            raise ValueError(f"Invalid ML-DSA variant: {variant}. Must be one of {list(self.VARIANT_MAP.keys())}")
        
        # Try with and without dashes
        if variant.upper() in self.VARIANT_MAP:
            variant_key = variant.upper()
        else:
            variant_key = variant_upper
        
        key_size = self.SECURITY_LEVELS[variant_key]
        
        algorithm_name = variant_key if variant_key.startswith("ML-DSA") else f"ML-DSA-{variant_key[-2:]}"
        super().__init__(
            algorithm_name=algorithm_name,
            variant=variant_key,
            key_size_bits=key_size,
            seed=seed
        )
        
        self._sig_algorithm = self.VARIANT_MAP[variant_key]
    
    def generate_keypair(self) -> None:
        """Generate ML-DSA key pair."""
        t_start = time.perf_counter()
        
        # Generate key pair using liboqs
        with oqs.Signature(self._sig_algorithm) as sig:
            public_key = sig.generate_keypair()
            secret_key = sig.export_secret_key()
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Store keys
        self.private_key = secret_key
        self.public_key = public_key
        
        # Estimate entropy
        entropy = estimate_entropy(public_key)
        
        # Update cipher state
        self.cipher_state.set_key(public_key, entropy, generation_time_ms)
        self.cipher_state.public_key = public_key
        self.cipher_state.private_key = secret_key
    
    def sign(self, data: bytes) -> bytes:
        """Sign data using ML-DSA."""
        if self.private_key is None:
            raise RuntimeError("Key not generated. Call generate_keypair() first.")
        
        with oqs.Signature(self._sig_algorithm, self.private_key) as sig:
            signature = sig.sign(data)
        
        return signature
    
    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify ML-DSA signature."""
        if self.public_key is None:
            raise RuntimeError("Key not generated. Call generate_keypair() first.")
        
        try:
            with oqs.Signature(self._sig_algorithm) as sig:
                is_valid = sig.verify(data, signature, self.public_key)
            return is_valid
        except Exception:
            return False
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        'Encrypt' for signature systems means sign.
        
        Args:
            plaintext: Data to sign
        
        Returns:
            Tuple of (signature, metadata)
        """
        if self.private_key is None:
            self.generate_keypair()
        
        t_start = time.perf_counter()
        
        try:
            signature = self.sign(plaintext)
            
            t_end = time.perf_counter()
            signing_time_ms = (t_end - t_start) * 1000
            
            metadata = EncryptionMetadata(
                algorithm=self.algorithm_name,
                plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext),
                key_size_bits=self.key_size_bits,
                ciphertext_length=len(signature),
                encryption_time_ms=signing_time_ms,
                success=True,
                tag=signature.hex()[:64]
            )
            
            return signature, metadata
            
        except Exception as e:
            t_end = time.perf_counter()
            signing_time_ms = (t_end - t_start) * 1000
            
            metadata = EncryptionMetadata(
                algorithm=self.algorithm_name,
                plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext),
                key_size_bits=self.key_size_bits,
                ciphertext_length=0,
                encryption_time_ms=signing_time_ms,
                success=False,
                error_message=str(e),
            )
            
            return b"", metadata
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        """
        'Decrypt' for signature systems means verify.
        
        Args:
            ciphertext: Signature to verify
            **kwargs: Must contain 'data' (original data)
        
        Returns:
            Tuple of (original_data if valid else b'', metadata)
        """
        data = kwargs.get('data', b'')
        
        t_start = time.perf_counter()
        
        try:
            is_valid = self.verify(data, ciphertext)
            
            t_end = time.perf_counter()
            verify_time_ms = (t_end - t_start) * 1000
            
            metadata = DecryptionMetadata(
                success=is_valid,
                decryption_time_ms=verify_time_ms,
                recovered_size=len(data) if is_valid else 0,
                error_message=None if is_valid else "Signature verification failed"
            )
            
            return data if is_valid else b'', metadata
            
        except Exception as e:
            t_end = time.perf_counter()
            verify_time_ms = (t_end - t_start) * 1000
            
            metadata = DecryptionMetadata(
                success=False,
                decryption_time_ms=verify_time_ms,
                recovered_size=0,
                error_message=str(e)
            )
            
            return b"", metadata


# ============================================================================
# FN-DSA / Falcon (Fast Lattice-based Signature)
# ============================================================================

class FNDSASystem(PQCSignatureSystem):
    """
    FN-DSA (Falcon) - Fast lattice-based signature scheme.
    
    Falcon uses NTRU lattices and offers compact signatures with fast operations.
    It was a NIST Round 3 finalist.
    
    Variants:
    - Falcon-512: NIST Level 1 security (~128-bit)
    - Falcon-1024: NIST Level 5 security (~256-bit)
    
    Features:
    - Compact signatures (~700 bytes for Falcon-512)
    - Fast signing and verification
    - Uses floating-point arithmetic
    """
    
    VARIANT_MAP = {
        "FALCON-512": "Falcon-512",
        "FALCON-1024": "Falcon-1024",
    }
    
    SECURITY_LEVELS = {
        "FALCON-512": 128,
        "FALCON-1024": 256,
    }
    
    def __init__(self, variant: str = "Falcon-512", seed: Optional[int] = None):
        """
        Initialize Falcon signature system.
        
        Args:
            variant: Falcon variant
            seed: Optional RNG seed
        """
        if not LIBOQS_AVAILABLE:
            raise ImportError("liboqs-python is required for Falcon. Install with: pip install liboqs-python")
        
        variant_upper = variant.upper()
        if variant_upper not in self.VARIANT_MAP:
            raise ValueError(f"Invalid Falcon variant: {variant}. Must be one of {list(self.VARIANT_MAP.keys())}")
        
        key_size = self.SECURITY_LEVELS[variant_upper]
        
        algorithm_name = self.VARIANT_MAP[variant_upper]
        super().__init__(
            algorithm_name=algorithm_name,
            variant=variant_upper,
            key_size_bits=key_size,
            seed=seed
        )
        
        self._sig_algorithm = self.VARIANT_MAP[variant_upper]
    
    def generate_keypair(self) -> None:
        """Generate Falcon key pair."""
        t_start = time.perf_counter()
        
        # Generate key pair using liboqs
        with oqs.Signature(self._sig_algorithm) as sig:
            public_key = sig.generate_keypair()
            secret_key = sig.export_secret_key()
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Store keys
        self.private_key = secret_key
        self.public_key = public_key
        
        # Estimate entropy
        entropy = estimate_entropy(public_key)
        
        # Update cipher state
        self.cipher_state.set_key(public_key, entropy, generation_time_ms)
        self.cipher_state.public_key = public_key
        self.cipher_state.private_key = secret_key
    
    def sign(self, data: bytes) -> bytes:
        """Sign data using Falcon."""
        if self.private_key is None:
            raise RuntimeError("Key not generated. Call generate_keypair() first.")
        
        with oqs.Signature(self._sig_algorithm, self.private_key) as sig:
            signature = sig.sign(data)
        
        return signature
    
    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify Falcon signature."""
        if self.public_key is None:
            raise RuntimeError("Key not generated. Call generate_keypair() first.")
        
        try:
            with oqs.Signature(self._sig_algorithm) as sig:
                is_valid = sig.verify(data, signature, self.public_key)
            return is_valid
        except Exception:
            return False
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        'Encrypt' for signature systems means sign.
        
        Args:
            plaintext: Data to sign
        
        Returns:
            Tuple of (signature, metadata)
        """
        if self.private_key is None:
            self.generate_keypair()
        
        t_start = time.perf_counter()
        
        try:
            signature = self.sign(plaintext)
            
            t_end = time.perf_counter()
            signing_time_ms = (t_end - t_start) * 1000
            
            metadata = EncryptionMetadata(
                algorithm=self.algorithm_name,
                plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext),
                key_size_bits=self.key_size_bits,
                ciphertext_length=len(signature),
                encryption_time_ms=signing_time_ms,
                success=True,
                tag=signature.hex()[:64]
            )
            
            return signature, metadata
            
        except Exception as e:
            t_end = time.perf_counter()
            signing_time_ms = (t_end - t_start) * 1000
            
            metadata = EncryptionMetadata(
                algorithm=self.algorithm_name,
                plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext),
                key_size_bits=self.key_size_bits,
                ciphertext_length=0,
                encryption_time_ms=signing_time_ms,
                success=False,
                error_message=str(e),
            )
            
            return b"", metadata
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        """
        'Decrypt' for signature systems means verify.
        
        Args:
            ciphertext: Signature to verify
            **kwargs: Must contain 'data' (original data)
        
        Returns:
            Tuple of (original_data if valid else b'', metadata)
        """
        data = kwargs.get('data', b'')
        
        t_start = time.perf_counter()
        
        try:
            is_valid = self.verify(data, ciphertext)
            
            t_end = time.perf_counter()
            verify_time_ms = (t_end - t_start) * 1000
            
            metadata = DecryptionMetadata(
                success=is_valid,
                decryption_time_ms=verify_time_ms,
                recovered_size=len(data) if is_valid else 0,
                error_message=None if is_valid else "Signature verification failed"
            )
            
            return data if is_valid else b'', metadata
            
        except Exception as e:
            t_end = time.perf_counter()
            verify_time_ms = (t_end - t_start) * 1000
            
            metadata = DecryptionMetadata(
                success=False,
                decryption_time_ms=verify_time_ms,
                recovered_size=0,
                error_message=str(e)
            )
            
            return b"", metadata


# ============================================================================
# SLH-DSA / SPHINCS+ (Hash-based Signature)
# ============================================================================

class SLHDSASystem(PQCSignatureSystem):
    """
    SLH-DSA (SPHINCS+) - Stateless hash-based signature scheme.
    
    SPHINCS+ is the NIST-standardized hash-based signature (FIPS 205).
    It offers conservative security based only on hash functions.
    
    Variants (fast variants with simple mode):
    - SPHINCS+-SHA2-128f-simple: SHA2, 128-bit security
    - SPHINCS+-SHA2-192f-simple: SHA2, 192-bit security
    - SPHINCS+-SHA2-256f-simple: SHA2, 256-bit security
    - SPHINCS+-SHAKE-128f-simple: SHAKE, 128-bit security
    - SPHINCS+-SHAKE-192f-simple: SHAKE, 192-bit security
    - SPHINCS+-SHAKE-256f-simple: SHAKE, 256-bit security
    
    Characteristics:
    - Large signatures (8-50KB)
    - Slow signing (~10-100ms)
    - Conservative, quantum-resistant security
    """
    
    VARIANT_MAP = {
        "SPHINCS+-SHA2-128F-SIMPLE": "SPHINCS+-SHA2-128f-simple",
        "SPHINCS+-SHA2-192F-SIMPLE": "SPHINCS+-SHA2-192f-simple",
        "SPHINCS+-SHA2-256F-SIMPLE": "SPHINCS+-SHA2-256f-simple",
        "SPHINCS+-SHAKE-128F-SIMPLE": "SPHINCS+-SHAKE-128f-simple",
        "SPHINCS+-SHAKE-192F-SIMPLE": "SPHINCS+-SHAKE-192f-simple",
        "SPHINCS+-SHAKE-256F-SIMPLE": "SPHINCS+-SHAKE-256f-simple",
    }
    
    SECURITY_LEVELS = {
        "SPHINCS+-SHA2-128F-SIMPLE": 128,
        "SPHINCS+-SHA2-192F-SIMPLE": 192,
        "SPHINCS+-SHA2-256F-SIMPLE": 256,
        "SPHINCS+-SHAKE-128F-SIMPLE": 128,
        "SPHINCS+-SHAKE-192F-SIMPLE": 192,
        "SPHINCS+-SHAKE-256F-SIMPLE": 256,
    }
    
    def __init__(self, variant: str = "SPHINCS+-SHA2-128f-simple", seed: Optional[int] = None):
        """
        Initialize SPHINCS+ signature system.
        
        Args:
            variant: SPHINCS+ variant
            seed: Optional RNG seed
        """
        if not LIBOQS_AVAILABLE:
            raise ImportError("liboqs-python is required for SPHINCS+. Install with: pip install liboqs-python")
        
        variant_upper = variant.upper()
        if variant_upper not in self.VARIANT_MAP:
            raise ValueError(f"Invalid SPHINCS+ variant: {variant}. Must be one of {list(self.VARIANT_MAP.keys())}")
        
        key_size = self.SECURITY_LEVELS[variant_upper]
        
        algorithm_name = self.VARIANT_MAP[variant_upper]
        super().__init__(
            algorithm_name=algorithm_name,
            variant=variant_upper,
            key_size_bits=key_size,
            seed=seed
        )
        
        self._sig_algorithm = self.VARIANT_MAP[variant_upper]
    
    def generate_keypair(self) -> None:
        """Generate SPHINCS+ key pair."""
        t_start = time.perf_counter()
        
        # Generate key pair using liboqs (may be slow)
        with oqs.Signature(self._sig_algorithm) as sig:
            public_key = sig.generate_keypair()
            secret_key = sig.export_secret_key()
        
        t_end = time.perf_counter()
        generation_time_ms = (t_end - t_start) * 1000
        
        # Store keys
        self.private_key = secret_key
        self.public_key = public_key
        
        # Estimate entropy
        entropy = estimate_entropy(public_key)
        
        # Update cipher state
        self.cipher_state.set_key(public_key, entropy, generation_time_ms)
        self.cipher_state.public_key = public_key
        self.cipher_state.private_key = secret_key
    
    def sign(self, data: bytes) -> bytes:
        """Sign data using SPHINCS+ (may be slow, ~10-100ms)."""
        if self.private_key is None:
            raise RuntimeError("Key not generated. Call generate_keypair() first.")
        
        with oqs.Signature(self._sig_algorithm, self.private_key) as sig:
            signature = sig.sign(data)
        
        return signature
    
    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify SPHINCS+ signature."""
        if self.public_key is None:
            raise RuntimeError("Key not generated. Call generate_keypair() first.")
        
        try:
            with oqs.Signature(self._sig_algorithm) as sig:
                is_valid = sig.verify(data, signature, self.public_key)
            return is_valid
        except Exception:
            return False
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """
        'Encrypt' for signature systems means sign.
        
        Note: SPHINCS+ signing is intentionally slow for security.
        
        Args:
            plaintext: Data to sign
        
        Returns:
            Tuple of (signature, metadata)
        """
        if self.private_key is None:
            self.generate_keypair()
        
        t_start = time.perf_counter()
        
        try:
            signature = self.sign(plaintext)
            
            t_end = time.perf_counter()
            signing_time_ms = (t_end - t_start) * 1000
            
            metadata = EncryptionMetadata(
                algorithm=self.algorithm_name,
                plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext),
                key_size_bits=self.key_size_bits,
                ciphertext_length=len(signature),
                encryption_time_ms=signing_time_ms,
                success=True,
                tag=signature.hex()[:64]  # Only first 64 hex chars due to large size
            )
            
            return signature, metadata
            
        except Exception as e:
            t_end = time.perf_counter()
            signing_time_ms = (t_end - t_start) * 1000
            
            metadata = EncryptionMetadata(
                algorithm=self.algorithm_name,
                plaintext_hash=compute_plaintext_hash(plaintext),
                plaintext_length=len(plaintext),
                key_size_bits=self.key_size_bits,
                ciphertext_length=0,
                encryption_time_ms=signing_time_ms,
                success=False,
                error_message=str(e),
            )
            
            return b"", metadata
    
    def decrypt(self, ciphertext: bytes, **kwargs) -> Tuple[bytes, DecryptionMetadata]:
        """
        'Decrypt' for signature systems means verify.
        
        Args:
            ciphertext: Signature to verify
            **kwargs: Must contain 'data' (original data)
        
        Returns:
            Tuple of (original_data if valid else b'', metadata)
        """
        data = kwargs.get('data', b'')
        
        t_start = time.perf_counter()
        
        try:
            is_valid = self.verify(data, ciphertext)
            
            t_end = time.perf_counter()
            verify_time_ms = (t_end - t_start) * 1000
            
            metadata = DecryptionMetadata(
                success=is_valid,
                decryption_time_ms=verify_time_ms,
                recovered_size=len(data) if is_valid else 0,
                error_message=None if is_valid else "Signature verification failed"
            )
            
            return data if is_valid else b'', metadata
            
        except Exception as e:
            t_end = time.perf_counter()
            verify_time_ms = (t_end - t_start) * 1000
            
            metadata = DecryptionMetadata(
                success=False,
                decryption_time_ms=verify_time_ms,
                recovered_size=0,
                error_message=str(e)
            )
            
            return b"", metadata


# Export all cipher classes
__all__ = [
    "MLKEMSystem",
    "KyberCrystalsSystem",
    "HQCSystem",
    "NTRUSystem",
    "BIKESystem",
    "ClassicMcElieceSystem",
    "MLDSASystem",
    "FNDSASystem",
    "SLHDSASystem",
    "LIBOQS_AVAILABLE",
]

