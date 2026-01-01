"""
Cipher Factory - Factory Pattern for Cipher Instantiation

This module provides factory functions to create cipher instances by algorithm name.

Version: 3.1
Date: December 30, 2025
"""

from typing import Optional, Dict, Any, List
import logging

from .base_cipher import BaseCipherSystem
from .symmetric_ciphers import (
    AESCipherSystem, 
    ChaCha20CipherSystem,
    TripleDESCipherSystem,
    BlowfishCipherSystem,
    CamelliaCipherSystem,
    DESCipherSystem,
    RC2CipherSystem,
    RC4CipherSystem,
    CASTCipherSystem,
    Salsa20CipherSystem,
    PYCRYPTODOME_AVAILABLE,
    CAMELLIA_AVAILABLE,
    LEGACY_CIPHERS_AVAILABLE,
    PYNACL_AVAILABLE,
)
from .asymmetric_ciphers import (
    RSACipherSystem,
    ECCCipherSystem,
)
from .pqc_ciphers import (
    MLKEMSystem,
    KyberCrystalsSystem,
    HQCSystem,
    NTRUSystem,
    BIKESystem,
    ClassicMcElieceSystem,
    MLDSASystem,
    FNDSASystem,
    SLHDSASystem,
    LIBOQS_AVAILABLE,
)

# Import new algorithm modules
try:
    from .hash_functions import create_hash_system, get_available_hash_algorithms
    HASH_AVAILABLE = True
except ImportError:
    HASH_AVAILABLE = False

try:
    from .mac_functions import create_mac_system, get_available_mac_algorithms
    MAC_AVAILABLE = True
except ImportError:
    MAC_AVAILABLE = False

try:
    from .kdf_functions import create_kdf_system, get_available_kdf_algorithms
    KDF_AVAILABLE = True
except ImportError:
    KDF_AVAILABLE = False

try:
    from .signature_systems import create_signature_system, get_available_signature_algorithms
    SIGNATURE_AVAILABLE = True
except ImportError:
    SIGNATURE_AVAILABLE = False

try:
    from .key_exchange import create_key_exchange_system, get_available_key_exchange_algorithms
    KEX_AVAILABLE = True
except ImportError:
    KEX_AVAILABLE = False

logger = logging.getLogger(__name__)


# Algorithm Registry
# Maps algorithm names to (cipher_class, constructor_args)
ALGORITHM_REGISTRY: Dict[str, tuple] = {
    # AES variants
    "AES-128-ECB": (AESCipherSystem, {"key_size": 128, "mode": "ECB"}),
    "AES-128-CBC": (AESCipherSystem, {"key_size": 128, "mode": "CBC"}),
    "AES-128-CTR": (AESCipherSystem, {"key_size": 128, "mode": "CTR"}),
    "AES-128-GCM": (AESCipherSystem, {"key_size": 128, "mode": "GCM"}),
    "AES-128-CFB": (AESCipherSystem, {"key_size": 128, "mode": "CFB"}),
    "AES-128-OFB": (AESCipherSystem, {"key_size": 128, "mode": "OFB"}),
    
    "AES-192-ECB": (AESCipherSystem, {"key_size": 192, "mode": "ECB"}),
    "AES-192-CBC": (AESCipherSystem, {"key_size": 192, "mode": "CBC"}),
    "AES-192-CTR": (AESCipherSystem, {"key_size": 192, "mode": "CTR"}),
    "AES-192-GCM": (AESCipherSystem, {"key_size": 192, "mode": "GCM"}),
    "AES-192-CFB": (AESCipherSystem, {"key_size": 192, "mode": "CFB"}),
    "AES-192-OFB": (AESCipherSystem, {"key_size": 192, "mode": "OFB"}),
    
    "AES-256-ECB": (AESCipherSystem, {"key_size": 256, "mode": "ECB"}),
    "AES-256-CBC": (AESCipherSystem, {"key_size": 256, "mode": "CBC"}),
    "AES-256-CTR": (AESCipherSystem, {"key_size": 256, "mode": "CTR"}),
    "AES-256-GCM": (AESCipherSystem, {"key_size": 256, "mode": "GCM"}),
    "AES-256-CFB": (AESCipherSystem, {"key_size": 256, "mode": "CFB"}),
    "AES-256-OFB": (AESCipherSystem, {"key_size": 256, "mode": "OFB"}),
    
    # ChaCha20 variants
    "CHACHA20": (ChaCha20CipherSystem, {}),
    
    # 3DES variants
    "3DES-CBC": (TripleDESCipherSystem, {"mode": "CBC"}),
    "3DES-ECB": (TripleDESCipherSystem, {"mode": "ECB"}),
    "TRIPLEDES-CBC": (TripleDESCipherSystem, {"mode": "CBC"}),
    "TRIPLEDES-ECB": (TripleDESCipherSystem, {"mode": "ECB"}),
    
    # RSA variants
    "RSA-1024-OAEP": (RSACipherSystem, {"key_size": 1024, "padding_scheme": "OAEP"}),
    "RSA-1024-PKCS1": (RSACipherSystem, {"key_size": 1024, "padding_scheme": "PKCS1"}),
    "RSA-2048-OAEP": (RSACipherSystem, {"key_size": 2048, "padding_scheme": "OAEP"}),
    "RSA-2048-PKCS1": (RSACipherSystem, {"key_size": 2048, "padding_scheme": "PKCS1"}),
    "RSA-3072-OAEP": (RSACipherSystem, {"key_size": 3072, "padding_scheme": "OAEP"}),
    "RSA-3072-PKCS1": (RSACipherSystem, {"key_size": 3072, "padding_scheme": "PKCS1"}),
    "RSA-4096-OAEP": (RSACipherSystem, {"key_size": 4096, "padding_scheme": "OAEP"}),
    "RSA-4096-PKCS1": (RSACipherSystem, {"key_size": 4096, "padding_scheme": "PKCS1"}),
    
    # ECC variants
    "ECC-P-256": (ECCCipherSystem, {"curve": "P-256"}),
    "ECC-P-384": (ECCCipherSystem, {"curve": "P-384"}),
    "ECC-P-521": (ECCCipherSystem, {"curve": "P-521"}),
}

# Add Blowfish variants if PyCryptodome is available
if PYCRYPTODOME_AVAILABLE:
    ALGORITHM_REGISTRY.update({
        "BLOWFISH-128-CBC": (BlowfishCipherSystem, {"key_size": 128, "mode": "CBC"}),
        "BLOWFISH-128-ECB": (BlowfishCipherSystem, {"key_size": 128, "mode": "ECB"}),
        "BLOWFISH-128-CTR": (BlowfishCipherSystem, {"key_size": 128, "mode": "CTR"}),
        "BLOWFISH-192-CBC": (BlowfishCipherSystem, {"key_size": 192, "mode": "CBC"}),
        "BLOWFISH-192-ECB": (BlowfishCipherSystem, {"key_size": 192, "mode": "ECB"}),
        "BLOWFISH-192-CTR": (BlowfishCipherSystem, {"key_size": 192, "mode": "CTR"}),
        "BLOWFISH-256-CBC": (BlowfishCipherSystem, {"key_size": 256, "mode": "CBC"}),
        "BLOWFISH-256-ECB": (BlowfishCipherSystem, {"key_size": 256, "mode": "ECB"}),
        "BLOWFISH-256-CTR": (BlowfishCipherSystem, {"key_size": 256, "mode": "CTR"}),
    })

# Add Camellia variants if available
if CAMELLIA_AVAILABLE:
    ALGORITHM_REGISTRY.update({
        "CAMELLIA-128-CBC": (CamelliaCipherSystem, {"key_size": 128, "mode": "CBC"}),
        "CAMELLIA-128-ECB": (CamelliaCipherSystem, {"key_size": 128, "mode": "ECB"}),
        "CAMELLIA-128-CTR": (CamelliaCipherSystem, {"key_size": 128, "mode": "CTR"}),
        "CAMELLIA-192-CBC": (CamelliaCipherSystem, {"key_size": 192, "mode": "CBC"}),
        "CAMELLIA-192-ECB": (CamelliaCipherSystem, {"key_size": 192, "mode": "ECB"}),
        "CAMELLIA-192-CTR": (CamelliaCipherSystem, {"key_size": 192, "mode": "CTR"}),
        "CAMELLIA-256-CBC": (CamelliaCipherSystem, {"key_size": 256, "mode": "CBC"}),
        "CAMELLIA-256-ECB": (CamelliaCipherSystem, {"key_size": 256, "mode": "ECB"}),
        "CAMELLIA-256-CTR": (CamelliaCipherSystem, {"key_size": 256, "mode": "CTR"}),
    })

# Add PQC algorithms if liboqs is available
if LIBOQS_AVAILABLE:
    ALGORITHM_REGISTRY.update({
        # ===== PQC KEMs =====
        
        # ML-KEM (NIST approved - FIPS 203)
        "ML-KEM-512": (MLKEMSystem, {"variant": "ML-KEM-512"}),
        "ML-KEM-768": (MLKEMSystem, {"variant": "ML-KEM-768"}),
        "ML-KEM-1024": (MLKEMSystem, {"variant": "ML-KEM-1024"}),
        
        # CRYSTALS-Kyber (alias for ML-KEM)
        "CRYSTALS-KYBER-512": (KyberCrystalsSystem, {"variant": "CRYSTALS-Kyber-512"}),
        "CRYSTALS-KYBER-768": (KyberCrystalsSystem, {"variant": "CRYSTALS-Kyber-768"}),
        "CRYSTALS-KYBER-1024": (KyberCrystalsSystem, {"variant": "CRYSTALS-Kyber-1024"}),
        "KYBER-512": (KyberCrystalsSystem, {"variant": "Kyber-512"}),
        "KYBER-768": (KyberCrystalsSystem, {"variant": "Kyber-768"}),
        "KYBER-1024": (KyberCrystalsSystem, {"variant": "Kyber-1024"}),
        
        # HQC (code-based)
        "HQC-128": (HQCSystem, {"variant": "HQC-128"}),
        "HQC-192": (HQCSystem, {"variant": "HQC-192"}),
        "HQC-256": (HQCSystem, {"variant": "HQC-256"}),
        
        # NTRU (lattice-based KEM - NIST Round 3 finalist)
        "NTRU-HPS-2048-509": (NTRUSystem, {"variant": "NTRU-HPS-2048-509"}),
        "NTRU-HPS-2048-677": (NTRUSystem, {"variant": "NTRU-HPS-2048-677"}),
        "NTRU-HPS-4096-821": (NTRUSystem, {"variant": "NTRU-HPS-4096-821"}),
        "NTRU-HRSS-701": (NTRUSystem, {"variant": "NTRU-HRSS-701"}),
        
        # BIKE (code-based KEM with QC-MDPC)
        "BIKE-L1": (BIKESystem, {"variant": "BIKE-L1"}),
        "BIKE-L3": (BIKESystem, {"variant": "BIKE-L3"}),
        "BIKE-L5": (BIKESystem, {"variant": "BIKE-L5"}),
        
        # Classic McEliece (conservative code-based KEM)
        "MCELIECE348864": (ClassicMcElieceSystem, {"variant": "McEliece348864"}),
        "MCELIECE460896": (ClassicMcElieceSystem, {"variant": "McEliece460896"}),
        "MCELIECE6688128": (ClassicMcElieceSystem, {"variant": "McEliece6688128"}),
        "MCELIECE6960119": (ClassicMcElieceSystem, {"variant": "McEliece6960119"}),
        "MCELIECE8192128": (ClassicMcElieceSystem, {"variant": "McEliece8192128"}),
        
        # ===== PQC Signatures =====
        
        # ML-DSA / Dilithium (NIST approved - FIPS 204)
        "ML-DSA-44": (MLDSASystem, {"variant": "ML-DSA-44"}),
        "ML-DSA-65": (MLDSASystem, {"variant": "ML-DSA-65"}),
        "ML-DSA-87": (MLDSASystem, {"variant": "ML-DSA-87"}),
        "DILITHIUM2": (MLDSASystem, {"variant": "DILITHIUM2"}),
        "DILITHIUM3": (MLDSASystem, {"variant": "DILITHIUM3"}),
        "DILITHIUM5": (MLDSASystem, {"variant": "DILITHIUM5"}),
        
        # FN-DSA / Falcon (lattice-based signature - NIST Round 3 finalist)
        "FALCON-512": (FNDSASystem, {"variant": "Falcon-512"}),
        "FALCON-1024": (FNDSASystem, {"variant": "Falcon-1024"}),
        
        # SLH-DSA / SPHINCS+ (hash-based signature - NIST approved - FIPS 205)
        "SPHINCS+-SHA2-128F-SIMPLE": (SLHDSASystem, {"variant": "SPHINCS+-SHA2-128f-simple"}),
        "SPHINCS+-SHA2-192F-SIMPLE": (SLHDSASystem, {"variant": "SPHINCS+-SHA2-192f-simple"}),
        "SPHINCS+-SHA2-256F-SIMPLE": (SLHDSASystem, {"variant": "SPHINCS+-SHA2-256f-simple"}),
        "SPHINCS+-SHAKE-128F-SIMPLE": (SLHDSASystem, {"variant": "SPHINCS+-SHAKE-128f-simple"}),
        "SPHINCS+-SHAKE-192F-SIMPLE": (SLHDSASystem, {"variant": "SPHINCS+-SHAKE-192f-simple"}),
        "SPHINCS+-SHAKE-256F-SIMPLE": (SLHDSASystem, {"variant": "SPHINCS+-SHAKE-256f-simple"}),
    })

# Add legacy ciphers if PyCryptodome is available (AVOID in production - for research only)
if LEGACY_CIPHERS_AVAILABLE:
    ALGORITHM_REGISTRY.update({
        "DES-CBC": (DESCipherSystem, {"mode": "CBC"}),
        "DES-ECB": (DESCipherSystem, {"mode": "ECB"}),
        "RC2-128-CBC": (RC2CipherSystem, {"key_size": 128, "mode": "CBC"}),
        "RC2-128-ECB": (RC2CipherSystem, {"key_size": 128, "mode": "ECB"}),
        "RC4-128": (RC4CipherSystem, {"key_size": 128}),
        "CAST-128-CBC": (CASTCipherSystem, {"key_size": 128, "mode": "CBC"}),
        "CAST-128-ECB": (CASTCipherSystem, {"key_size": 128, "mode": "ECB"}),
    })

# Add Salsa20 if PyNaCl is available
if PYNACL_AVAILABLE:
    ALGORITHM_REGISTRY.update({
        "SALSA20": (Salsa20CipherSystem, {}),
    })

# =============================================================================
# NOTE: Additional Algorithm Categories
# =============================================================================
# The following algorithm categories have dedicated factory functions:
#
# HASH FUNCTIONS (17 algorithms):
#   - Use: from .hash_functions import create_hash_system
#   - Algorithms: MD2, MD4, MD5, SHA-1, SHA-224/256/384/512, SHA3 variants,
#                 RIPEMD-160, Whirlpool, BLAKE2s/2b, BLAKE3
#
# MAC FUNCTIONS (7 algorithms):
#   - Use: from .mac_functions import create_mac_system
#   - Algorithms: HMAC-SHA256, HMAC-SHA512, CMAC-AES, GMAC, KMAC128/256, Poly1305
#
# KDF FUNCTIONS (9 algorithms):
#   - Use: from .kdf_functions import create_kdf_system
#   - Algorithms: PBKDF1, PBKDF2 (10K/100K), bcrypt, scrypt, Argon2 (i/d/id), HKDF
#
# SIGNATURE SYSTEMS (13 algorithms):
#   - Use: from .signature_systems import create_signature_system
#   - Algorithms: DSA (1024/2048), ECDSA (P-256/384/521), Ed25519, Ed448,
#                 RSA-PSS (2048/3072/4096)
#
# KEY EXCHANGE (8 algorithms):
#   - Use: from .key_exchange import create_key_exchange_system
#   - Algorithms: DH (1024/2048/3072), ECDH (P-256/384/521), X25519, X448
#
# TOTAL: 96+ algorithms available across all categories
# =============================================================================


def create_cipher(algorithm_name: str, seed: Optional[int] = None) -> BaseCipherSystem:
    """
    Create cipher instance for given algorithm name.
    
    Factory method to instantiate the correct cipher class based on algorithm name.
    
    Args:
        algorithm_name: Name of algorithm (e.g., "AES-256-GCM", "ChaCha20")
        seed: Optional RNG seed for deterministic key generation
    
    Returns:
        Instance of appropriate cipher class
    
    Raises:
        ValueError: If algorithm name is not recognized
    
    Examples:
        >>> cipher = create_cipher("AES-256-GCM", seed=42)
        >>> cipher.generate_key()
        >>> ciphertext, metadata = cipher.encrypt(b"Hello, World!")
    """
    # Normalize algorithm name (uppercase, remove spaces)
    normalized_name = algorithm_name.upper().replace(" ", "")
    
    # Check if algorithm exists in registry
    if normalized_name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Available algorithms: {', '.join(get_available_algorithms())}"
        )
    
    # Get cipher class and constructor arguments
    cipher_class, constructor_args = ALGORITHM_REGISTRY[normalized_name]
    
    # Add seed to constructor args
    constructor_args_with_seed = constructor_args.copy()
    constructor_args_with_seed["seed"] = seed
    
    # Instantiate cipher
    try:
        cipher = cipher_class(**constructor_args_with_seed)
        logger.debug(f"Created cipher: {algorithm_name} with seed={seed}")
        return cipher
    except Exception as e:
        logger.error(f"Failed to create cipher {algorithm_name}: {e}")
        raise ValueError(f"Failed to create cipher {algorithm_name}: {e}")


def get_available_algorithms() -> List[str]:
    """
    Get list of all available algorithms.
    
    Returns:
        List of algorithm names
    
    Examples:
        >>> algos = get_available_algorithms()
        >>> print(len(algos))
        20  # Number of currently registered algorithms
    """
    return sorted(ALGORITHM_REGISTRY.keys())


def get_algorithm_metadata(algorithm_name: str) -> Dict[str, Any]:
    """
    Get metadata about a specific algorithm.
    
    Args:
        algorithm_name: Name of algorithm
    
    Returns:
        Dictionary with algorithm properties:
        - algorithm_type: "symmetric_block", "symmetric_stream", etc.
        - key_size: Key size in bits
        - block_size: Block size in bits (for block ciphers)
        - nist_status: "APPROVED", "DEPRECATED", etc.
        - pqc_safe: Whether post-quantum safe
    
    Raises:
        ValueError: If algorithm not found
    
    Examples:
        >>> meta = get_algorithm_metadata("AES-256-GCM")
        >>> print(meta["key_size"])
        256
    """
    normalized_name = algorithm_name.upper().replace(" ", "")
    
    if normalized_name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # Create temporary instance to get metadata
    cipher = create_cipher(normalized_name)
    
    # Determine algorithm type
    if isinstance(cipher, AESCipherSystem):
        algo_type = "symmetric_block"
        if cipher.mode in ["GCM", "CCM"]:
            algo_type = "symmetric_aead"
    elif isinstance(cipher, ChaCha20CipherSystem):
        algo_type = "symmetric_stream"
    else:
        algo_type = "unknown"
    
    # Build metadata
    metadata = {
        "algorithm_name": cipher.algorithm_name,
        "algorithm_type": algo_type,
        "key_size_bits": cipher.key_size_bits,
        "nist_status": "APPROVED",  # AES and ChaCha20 are both approved
        "pqc_safe": False,  # Classical algorithms are not PQC-safe
        "package": "cryptography",
    }
    
    # Add cipher-specific metadata
    if hasattr(cipher, "block_size"):
        metadata["block_size"] = cipher.block_size
        metadata["mode"] = cipher.mode
    
    if hasattr(cipher, "nonce_size"):
        metadata["nonce_size"] = cipher.nonce_size
    
    # Cleanup
    cipher.cleanup()
    
    return metadata


def filter_algorithms(
    algorithm_type: Optional[str] = None,
    min_key_size: Optional[int] = None,
    max_key_size: Optional[int] = None,
    nist_status: Optional[str] = None,
    pqc_safe: Optional[bool] = None,
) -> List[str]:
    """
    Filter algorithms by criteria.
    
    Args:
        algorithm_type: Filter by type ("symmetric_block", "symmetric_stream", etc.)
        min_key_size: Minimum key size in bits
        max_key_size: Maximum key size in bits
        nist_status: Filter by NIST status
        pqc_safe: Filter by post-quantum safety
    
    Returns:
        List of algorithm names matching criteria
    
    Examples:
        >>> # Get all AES-GCM variants
        >>> algos = filter_algorithms(algorithm_type="symmetric_aead")
        >>> # Get 256-bit algorithms
        >>> algos = filter_algorithms(min_key_size=256, max_key_size=256)
    """
    filtered = []
    
    for algo_name in get_available_algorithms():
        try:
            metadata = get_algorithm_metadata(algo_name)
            
            # Apply filters
            if algorithm_type and metadata.get("algorithm_type") != algorithm_type:
                continue
            
            if min_key_size and metadata.get("key_size_bits", 0) < min_key_size:
                continue
            
            if max_key_size and metadata.get("key_size_bits", float('inf')) > max_key_size:
                continue
            
            if nist_status and metadata.get("nist_status") != nist_status:
                continue
            
            if pqc_safe is not None and metadata.get("pqc_safe") != pqc_safe:
                continue
            
            filtered.append(algo_name)
            
        except Exception as e:
            logger.warning(f"Error getting metadata for {algo_name}: {e}")
            continue
    
    return filtered


def validate_algorithm(algorithm_name: str) -> bool:
    """
    Check if algorithm is available and valid.
    
    Args:
        algorithm_name: Name of algorithm
    
    Returns:
        True if available, False otherwise
    
    Examples:
        >>> validate_algorithm("AES-256-GCM")
        True
        >>> validate_algorithm("InvalidAlgorithm")
        False
    """
    normalized_name = algorithm_name.upper().replace(" ", "")
    return normalized_name in ALGORITHM_REGISTRY


# Export all functions
__all__ = [
    "create_cipher",
    "get_available_algorithms",
    "get_algorithm_metadata",
    "filter_algorithms",
    "validate_algorithm",
    "ALGORITHM_REGISTRY",
]

