"""
Input Validators
=================

Input validation and type checking functions.

This module provides validators for:
- Algorithm names
- Plaintext data
- Output paths
- Configuration objects

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def validate_algorithm_name(name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate algorithm name format.
    
    Args:
        name: Algorithm name to validate
    
    Returns:
        Tuple of (valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid
    
    Example:
        >>> validate_algorithm_name("AES-256-GCM")
        (True, None)
        >>> validate_algorithm_name("")
        (False, 'Algorithm name cannot be empty')
    """
    if not name:
        return (False, "Algorithm name cannot be empty")
    
    if not isinstance(name, str):
        return (False, f"Algorithm name must be a string, got {type(name).__name__}")
    
    # Check length
    if len(name) > 100:
        return (False, f"Algorithm name too long (max 100 chars): {len(name)}")
    
    # Check for valid characters (alphanumeric, hyphens, underscores, parentheses, slashes)
    if not re.match(r'^[A-Za-z0-9\-_/().\s]+$', name):
        return (False, f"Algorithm name contains invalid characters: {name}")
    
    return (True, None)


def validate_plaintext(
    plaintext: bytes,
    min_length: int = 1,
    max_length: int = 10 * 1024 * 1024  # 10 MB
) -> Tuple[bool, Optional[str]]:
    """
    Validate plaintext constraints.
    
    Args:
        plaintext: Plaintext bytes to validate
        min_length: Minimum length in bytes (default: 1)
        max_length: Maximum length in bytes (default: 10 MB)
    
    Returns:
        Tuple of (valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid
    
    Example:
        >>> validate_plaintext(b"Hello")
        (True, None)
        >>> validate_plaintext(b"")
        (False, 'Plaintext too short (min 1 bytes): 0')
    """
    if not isinstance(plaintext, bytes):
        return (False, f"Plaintext must be bytes, got {type(plaintext).__name__}")
    
    length = len(plaintext)
    
    if length < min_length:
        return (False, f"Plaintext too short (min {min_length} bytes): {length}")
    
    if length > max_length:
        return (False, f"Plaintext too long (max {max_length} bytes): {length}")
    
    return (True, None)


def validate_output_path(
    path: str,
    writable: bool = True,
    must_exist: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate output path.
    
    Args:
        path: File or directory path to validate
        writable: Check if path is writable (default: True)
        must_exist: Path must exist (default: False)
    
    Returns:
        Tuple of (valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid
    
    Example:
        >>> validate_output_path("output.csv")
        (True, None)
        >>> validate_output_path("")
        (False, 'Output path cannot be empty')
    """
    if not path:
        return (False, "Output path cannot be empty")
    
    if not isinstance(path, str):
        return (False, f"Output path must be a string, got {type(path).__name__}")
    
    path_obj = Path(path)
    
    # Check if path must exist
    if must_exist and not path_obj.exists():
        return (False, f"Path does not exist: {path}")
    
    # Check if path is writable
    if writable:
        # Check parent directory exists or can be created
        parent = path_obj.parent
        
        if path_obj.exists():
            # Check if existing path is writable
            if not os.access(str(path_obj), os.W_OK):
                return (False, f"Path is not writable: {path}")
        else:
            # Check if parent directory is writable
            if parent.exists():
                if not os.access(str(parent), os.W_OK):
                    return (False, f"Parent directory is not writable: {parent}")
            else:
                # Try to determine if we can create the directory
                # (This is a best-effort check)
                try:
                    # Check if any ancestor exists and is writable
                    for ancestor in parent.parents:
                        if ancestor.exists():
                            if not os.access(str(ancestor), os.W_OK):
                                return (False, f"Cannot create directory (ancestor not writable): {ancestor}")
                            break
                except Exception as e:
                    return (False, f"Error validating path: {str(e)}")
    
    return (True, None)


def validate_configuration(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate full configuration object.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        Tuple of (valid, error_list)
        - (True, []) if valid
        - (False, [error1, error2, ...]) if invalid
    
    Example:
        >>> config = {"algorithms": ["AES-256-GCM"], "output_csv": "data.csv"}
        >>> validate_configuration(config)
        (True, [])
    """
    errors = []
    
    if not isinstance(config, dict):
        return (False, [f"Configuration must be a dictionary, got {type(config).__name__}"])
    
    # Validate algorithms (if present)
    if "algorithms" in config:
        algorithms = config["algorithms"]
        if algorithms is not None:
            if not isinstance(algorithms, list):
                errors.append(f"'algorithms' must be a list or None, got {type(algorithms).__name__}")
            else:
                for i, algo in enumerate(algorithms):
                    valid, error = validate_algorithm_name(algo)
                    if not valid:
                        errors.append(f"Invalid algorithm at index {i}: {error}")
    
    # Validate test_strings (if present)
    if "test_strings" in config:
        test_strings = config["test_strings"]
        if test_strings is not None:
            if not isinstance(test_strings, list):
                errors.append(f"'test_strings' must be a list or None, got {type(test_strings).__name__}")
            else:
                for i, test_str in enumerate(test_strings):
                    valid, error = validate_plaintext(test_str, min_length=1, max_length=1024)
                    if not valid:
                        errors.append(f"Invalid test string at index {i}: {error}")
    
    # Validate output_csv (required)
    if "output_csv" not in config:
        errors.append("Missing required field: 'output_csv'")
    else:
        output_csv = config["output_csv"]
        valid, error = validate_output_path(output_csv, writable=True, must_exist=False)
        if not valid:
            errors.append(f"Invalid output_csv: {error}")
    
    # Validate output_json (if present)
    if "output_json" in config:
        output_json = config["output_json"]
        if output_json is not None:
            valid, error = validate_output_path(output_json, writable=True, must_exist=False)
            if not valid:
                errors.append(f"Invalid output_json: {error}")
    
    # Validate enable_attacks (if present)
    if "enable_attacks" in config:
        enable_attacks = config["enable_attacks"]
        if not isinstance(enable_attacks, bool):
            errors.append(f"'enable_attacks' must be a boolean, got {type(enable_attacks).__name__}")
    
    # Validate rng_seed (if present)
    if "rng_seed" in config:
        rng_seed = config["rng_seed"]
        if not isinstance(rng_seed, int):
            errors.append(f"'rng_seed' must be an integer, got {type(rng_seed).__name__}")
        elif rng_seed < 0:
            errors.append(f"'rng_seed' must be non-negative, got {rng_seed}")
    
    # Validate thread_pool_size (if present)
    if "thread_pool_size" in config:
        thread_pool_size = config["thread_pool_size"]
        if not isinstance(thread_pool_size, int):
            errors.append(f"'thread_pool_size' must be an integer, got {type(thread_pool_size).__name__}")
        elif thread_pool_size < 1:
            errors.append(f"'thread_pool_size' must be at least 1, got {thread_pool_size}")
        elif thread_pool_size > 32:
            errors.append(f"'thread_pool_size' too large (max 32), got {thread_pool_size}")
    
    # Validate timeout_seconds (if present)
    if "timeout_seconds" in config:
        timeout_seconds = config["timeout_seconds"]
        if not isinstance(timeout_seconds, (int, float)):
            errors.append(f"'timeout_seconds' must be a number, got {type(timeout_seconds).__name__}")
        elif timeout_seconds <= 0:
            errors.append(f"'timeout_seconds' must be positive, got {timeout_seconds}")
    
    # Validate max_memory_mb (if present)
    if "max_memory_mb" in config:
        max_memory_mb = config["max_memory_mb"]
        if not isinstance(max_memory_mb, (int, float)):
            errors.append(f"'max_memory_mb' must be a number, got {type(max_memory_mb).__name__}")
        elif max_memory_mb <= 0:
            errors.append(f"'max_memory_mb' must be positive, got {max_memory_mb}")
    
    # Return validation result
    is_valid = len(errors) == 0
    return (is_valid, errors)


def validate_key_size(key_size_bits: int, algorithm_name: str = "") -> Tuple[bool, Optional[str]]:
    """
    Validate key size for cryptographic algorithms.
    
    Args:
        key_size_bits: Key size in bits
        algorithm_name: Algorithm name (for specific validation)
    
    Returns:
        Tuple of (valid, error_message)
    
    Example:
        >>> validate_key_size(256, "AES")
        (True, None)
        >>> validate_key_size(100, "AES")
        (False, 'Invalid key size for AES: 100 (must be 128, 192, or 256)')
    """
    if not isinstance(key_size_bits, int):
        return (False, f"Key size must be an integer, got {type(key_size_bits).__name__}")
    
    if key_size_bits <= 0:
        return (False, f"Key size must be positive, got {key_size_bits}")
    
    # Algorithm-specific validation
    algo_lower = algorithm_name.lower()
    
    if "aes" in algo_lower:
        if key_size_bits not in [128, 192, 256]:
            return (False, f"Invalid key size for AES: {key_size_bits} (must be 128, 192, or 256)")
    
    elif "des" in algo_lower and "3des" not in algo_lower and "triple" not in algo_lower:
        if key_size_bits != 56:
            return (False, f"Invalid key size for DES: {key_size_bits} (must be 56)")
    
    elif "3des" in algo_lower or "tripledes" in algo_lower:
        if key_size_bits not in [112, 168]:
            return (False, f"Invalid key size for 3DES: {key_size_bits} (must be 112 or 168)")
    
    # General constraint: reasonable range
    if key_size_bits < 40:
        return (False, f"Key size too small (min 40 bits): {key_size_bits}")
    
    if key_size_bits > 33552:  # Max from spec
        return (False, f"Key size too large (max 33552 bits): {key_size_bits}")
    
    return (True, None)


def validate_bytes_length(
    data: bytes,
    expected_length: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate bytes data length.
    
    Args:
        data: Bytes data to validate
        expected_length: Exact expected length (optional)
        min_length: Minimum length (optional)
        max_length: Maximum length (optional)
    
    Returns:
        Tuple of (valid, error_message)
    
    Example:
        >>> validate_bytes_length(b"12345", expected_length=5)
        (True, None)
        >>> validate_bytes_length(b"123", min_length=5)
        (False, 'Data too short (min 5 bytes): 3')
    """
    if not isinstance(data, bytes):
        return (False, f"Data must be bytes, got {type(data).__name__}")
    
    length = len(data)
    
    if expected_length is not None:
        if length != expected_length:
            return (False, f"Data length mismatch (expected {expected_length} bytes): {length}")
    
    if min_length is not None:
        if length < min_length:
            return (False, f"Data too short (min {min_length} bytes): {length}")
    
    if max_length is not None:
        if length > max_length:
            return (False, f"Data too long (max {max_length} bytes): {length}")
    
    return (True, None)


def validate_positive_number(value: Any, name: str = "value") -> Tuple[bool, Optional[str]]:
    """
    Validate that a value is a positive number.
    
    Args:
        value: Value to validate
        name: Name of the value (for error message)
    
    Returns:
        Tuple of (valid, error_message)
    
    Example:
        >>> validate_positive_number(5, "count")
        (True, None)
        >>> validate_positive_number(-1, "count")
        (False, "count must be positive, got -1")
    """
    if not isinstance(value, (int, float)):
        return (False, f"{name} must be a number, got {type(value).__name__}")
    
    if value <= 0:
        return (False, f"{name} must be positive, got {value}")
    
    return (True, None)


def validate_range(
    value: Any,
    min_value: float,
    max_value: float,
    name: str = "value"
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        name: Name of the value (for error message)
    
    Returns:
        Tuple of (valid, error_message)
    
    Example:
        >>> validate_range(5, 0, 10, "score")
        (True, None)
        >>> validate_range(15, 0, 10, "score")
        (False, "score out of range [0, 10]: 15")
    """
    if not isinstance(value, (int, float)):
        return (False, f"{name} must be a number, got {type(value).__name__}")
    
    if value < min_value or value > max_value:
        return (False, f"{name} out of range [{min_value}, {max_value}]: {value}")
    
    return (True, None)

