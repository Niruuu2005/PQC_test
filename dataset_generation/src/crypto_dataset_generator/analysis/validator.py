"""
Data Validation Module

This module validates ciphertext, metrics, and attack results.

Version: 1.0
Date: December 30, 2025
"""

from typing import Tuple, List, Dict


def validate_ciphertext(ciphertext: bytes, algorithm_name: str) -> Tuple[bool, List[str]]:
    """
    Validate ciphertext properties.
    
    Checks that ciphertext meets basic requirements.
    
    Args:
        ciphertext: Encrypted data to validate
        algorithm_name: Name of encryption algorithm
    
    Returns:
        Tuple of (valid, error_list)
    
    Examples:
        >>> valid, errors = validate_ciphertext(ciphertext, "AES-256-GCM")
        >>> if not valid:
        ...     print(f"Errors: {errors}")
    """
    errors = []
    
    # Check if ciphertext exists
    if ciphertext is None:
        errors.append("Ciphertext is None")
        return False, errors
    
    # Check if ciphertext is bytes
    if not isinstance(ciphertext, bytes):
        errors.append(f"Ciphertext must be bytes, got {type(ciphertext).__name__}")
        return False, errors
    
    # Check if ciphertext is not empty
    if len(ciphertext) == 0:
        errors.append("Ciphertext is empty")
        return False, errors
    
    # Check minimum length (at least 1 byte)
    if len(ciphertext) < 1:
        errors.append(f"Ciphertext too short: {len(ciphertext)} bytes")
    
    # Check maximum length (prevent memory issues)
    MAX_LENGTH = 100 * 1024 * 1024  # 100 MB
    if len(ciphertext) > MAX_LENGTH:
        errors.append(f"Ciphertext too long: {len(ciphertext)} bytes (max: {MAX_LENGTH})")
    
    # Algorithm-specific checks
    if "AES" in algorithm_name.upper():
        # AES block size is 16 bytes
        # For modes like ECB/CBC, ciphertext should be multiple of block size
        if "ECB" in algorithm_name.upper() or "CBC" in algorithm_name.upper():
            if len(ciphertext) % 16 != 0:
                errors.append(f"AES {algorithm_name}: ciphertext length ({len(ciphertext)}) not multiple of block size (16)")
    
    return len(errors) == 0, errors


def validate_metrics(metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate metric values and bounds.
    
    Args:
        metrics: Dictionary of metrics to validate
    
    Returns:
        Tuple of (valid, error_list)
    
    Examples:
        >>> metrics = {'shannon_entropy': 7.5, 'chi_square_statistic': 250}
        >>> valid, errors = validate_metrics(metrics)
        >>> print(valid)
        True
    """
    errors = []
    
    # Check if metrics dict exists
    if metrics is None:
        errors.append("Metrics dictionary is None")
        return False, errors
    
    if not isinstance(metrics, dict):
        errors.append(f"Metrics must be a dictionary, got {type(metrics).__name__}")
        return False, errors
    
    # Define expected bounds for each metric
    bounds = {
        'shannon_entropy': (0.0, 8.0),
        'renyi_entropy': (0.0, 8.0),
        'min_entropy': (0.0, 8.0),
        'chi_square_statistic': (0.0, float('inf')),
        'frequency_bias': (0.0, 1.0),
        'byte_frequency_variance': (0.0, float('inf')),
        'ciphertext_bias': (0.0, 1.0),
        'randomness_score': (0.0, 1.0),
        'compression_ratio': (0.0, float('inf')),
        'hamming_distance_normalized': (0.0, 1.0),
        'avalanche_percentage': (0.0, 100.0),
        'avalanche_normalized': (0.0, 1.0),
        'sac_score': (0.0, 1.0),
        'autocorrelation': (-1.0, 1.0),
        'serial_correlation': (-1.0, 1.0),
        'plaintext_correlation': (-1.0, 1.0),
    }
    
    # Validate each metric
    for metric_name, value in metrics.items():
        # Check if value is numeric
        if not isinstance(value, (int, float, bool)):
            errors.append(f"{metric_name}: value must be numeric, got {type(value).__name__}")
            continue
        
        # Skip boolean values
        if isinstance(value, bool):
            continue
        
        # Check if NaN or Inf
        if isinstance(value, float):
            import math
            if math.isnan(value):
                errors.append(f"{metric_name}: value is NaN")
                continue
            if math.isinf(value) and metric_name in bounds:
                min_val, max_val = bounds[metric_name]
                if max_val != float('inf'):
                    errors.append(f"{metric_name}: value is infinite")
                    continue
        
        # Check bounds
        if metric_name in bounds:
            min_val, max_val = bounds[metric_name]
            if not (min_val <= value <= max_val):
                errors.append(f"{metric_name}: value {value} outside bounds [{min_val}, {max_val}]")
    
    return len(errors) == 0, errors


def validate_attack_results(results: Dict) -> Tuple[bool, List[str]]:
    """
    Validate attack result format.
    
    Args:
        results: Attack results dictionary
    
    Returns:
        Tuple of (valid, error_list)
    
    Examples:
        >>> results = {'attack_name': 'brute_force', 'success': True}
        >>> valid, errors = validate_attack_results(results)
        >>> print(valid)
        True
    """
    errors = []
    
    # Check if results dict exists
    if results is None:
        errors.append("Results dictionary is None")
        return False, errors
    
    if not isinstance(results, dict):
        errors.append(f"Results must be a dictionary, got {type(results).__name__}")
        return False, errors
    
    # Check required fields
    required_fields = ['attack_name', 'success']
    for field in required_fields:
        if field not in results:
            errors.append(f"Missing required field: {field}")
    
    # Validate field types
    if 'attack_name' in results and not isinstance(results['attack_name'], str):
        errors.append(f"attack_name must be string, got {type(results['attack_name']).__name__}")
    
    if 'success' in results and not isinstance(results['success'], bool):
        errors.append(f"success must be boolean, got {type(results['success']).__name__}")
    
    return len(errors) == 0, errors


def validate_csv_record(record: Dict) -> Tuple[bool, List[str]]:
    """
    Validate complete CSV record.
    
    Checks that a record has all required fields and valid values.
    
    Args:
        record: CSV record dictionary
    
    Returns:
        Tuple of (valid, error_list)
    
    Examples:
        >>> record = {
        ...     'algorithm': 'AES-256-GCM',
        ...     'plaintext': b'test',
        ...     'ciphertext': b'encrypted',
        ... }
        >>> valid, errors = validate_csv_record(record)
    """
    errors = []
    
    # Check if record dict exists
    if record is None:
        errors.append("Record dictionary is None")
        return False, errors
    
    if not isinstance(record, dict):
        errors.append(f"Record must be a dictionary, got {type(record).__name__}")
        return False, errors
    
    # Check required fields
    required_fields = ['algorithm', 'plaintext', 'ciphertext']
    for field in required_fields:
        if field not in record:
            errors.append(f"Missing required field: {field}")
    
    # Validate algorithm
    if 'algorithm' in record:
        if not isinstance(record['algorithm'], str):
            errors.append(f"algorithm must be string, got {type(record['algorithm']).__name__}")
        elif not record['algorithm']:
            errors.append("algorithm is empty string")
    
    # Validate plaintext
    if 'plaintext' in record:
        if not isinstance(record['plaintext'], (bytes, str)):
            errors.append(f"plaintext must be bytes or string, got {type(record['plaintext']).__name__}")
    
    # Validate ciphertext
    if 'ciphertext' in record:
        if not isinstance(record['ciphertext'], bytes):
            errors.append(f"ciphertext must be bytes, got {type(record['ciphertext']).__name__}")
        elif 'algorithm' in record:
            # Validate ciphertext for specific algorithm
            valid_ct, ct_errors = validate_ciphertext(record['ciphertext'], record['algorithm'])
            if not valid_ct:
                errors.extend(ct_errors)
    
    # Validate metrics (if present)
    if 'metrics' in record and isinstance(record['metrics'], dict):
        valid_metrics, metric_errors = validate_metrics(record['metrics'])
        if not valid_metrics:
            errors.extend(metric_errors)
    
    return len(errors) == 0, errors


# Export all functions
__all__ = [
    'validate_ciphertext',
    'validate_metrics',
    'validate_attack_results',
    'validate_csv_record',
]

