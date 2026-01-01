"""
Core Cryptographic Metrics

This module implements core cryptographic metrics for ciphertext analysis.

Version: 1.0
Date: December 30, 2025
"""

import math
import zlib
from typing import Dict, Tuple, List
from collections import Counter


def shannon_entropy(data: bytes) -> float:
    """
    Calculate Shannon entropy of data.
    
    Shannon entropy H(X) = -Σ p(x) * log2(p(x))
    
    Args:
        data: Byte sequence to analyze
    
    Returns:
        Entropy in bits per byte [0.0, 8.0]
    
    Examples:
        >>> shannon_entropy(b'\\x00' * 100)  # Low entropy
        0.0
        >>> shannon_entropy(bytes(range(256)))  # High entropy
        8.0
    """
    if len(data) == 0:
        return 0.0
    
    # Special case: if all bytes are the same, entropy is 0
    if len(set(data)) == 1:
        return 0.0
    
    # Count byte frequencies
    byte_counts = Counter(data)
    total_bytes = len(data)
    
    # Calculate entropy
    entropy = 0.0
    for count in byte_counts.values():
        if count > 0:
            probability = count / total_bytes
            entropy -= probability * math.log2(probability)
    
    return entropy


def chi_square_statistic(data: bytes) -> float:
    """
    Calculate chi-square statistic for uniformity test.
    
    Tests if byte distribution is uniform. Lower values indicate
    more uniform distribution.
    
    χ² = Σ ((observed - expected)² / expected)
    
    Args:
        data: Byte sequence to analyze
    
    Returns:
        Chi-square statistic (≥ 0)
    
    Examples:
        >>> chi_square_statistic(bytes(range(256)) * 10)  # Uniform
        0.0
    """
    if len(data) == 0:
        return 0.0
    
    # Count byte frequencies
    byte_counts = [0] * 256
    for byte in data:
        byte_counts[byte] += 1
    
    # Expected frequency (uniform distribution)
    expected = len(data) / 256.0
    
    # Calculate chi-square
    chi_square = 0.0
    for observed in byte_counts:
        chi_square += ((observed - expected) ** 2) / expected
    
    return chi_square


def frequency_bias(data: bytes) -> float:
    """
    Calculate maximum byte frequency bias.
    
    Measures the bias towards the most frequent byte.
    
    Args:
        data: Byte sequence to analyze
    
    Returns:
        Bias [0.0, 1.0], where 0 = uniform, 1 = single byte only
    
    Examples:
        >>> frequency_bias(b'\\x00' * 100)  # High bias
        1.0
        >>> frequency_bias(bytes(range(256)))  # Low bias
        ~0.004
    """
    if len(data) == 0:
        return 0.0
    
    # Count byte frequencies
    byte_counts = Counter(data)
    
    # Find max frequency
    max_frequency = max(byte_counts.values()) if byte_counts else 0
    
    # Calculate bias
    # For uniform distribution over N unique values, expected frequency is 1/N
    # We normalize by comparing to uniform distribution
    num_unique = len(byte_counts)
    if num_unique == 0:
        return 0.0
    
    expected_frequency = len(data) / num_unique
    
    # Bias is how much the max exceeds uniform
    # Normalize so that all-same = 1.0, uniform = 0.0
    bias = (max_frequency - expected_frequency) / (len(data) - expected_frequency) if len(data) > expected_frequency else 0.0
    
    return min(1.0, max(0.0, bias))


def byte_frequency_variance(data: bytes) -> float:
    """
    Calculate variance of byte frequencies.
    
    Measures how much byte frequencies deviate from mean.
    Lower variance indicates more uniform distribution.
    
    Args:
        data: Byte sequence to analyze
    
    Returns:
        Variance of byte frequencies (≥ 0)
    
    Examples:
        >>> byte_frequency_variance(bytes(range(256)) * 10)  # Low variance
        0.0
    """
    if len(data) == 0:
        return 0.0
    
    # Count byte frequencies
    byte_counts = [0] * 256
    for byte in data:
        byte_counts[byte] += 1
    
    # Calculate mean
    mean = sum(byte_counts) / 256.0
    
    # Calculate variance
    variance = sum((count - mean) ** 2 for count in byte_counts) / 256.0
    
    return variance


def ciphertext_bias(data: bytes) -> float:
    """
    Calculate bias towards maximum byte value.
    
    Measures if ciphertext is biased towards high byte values (0xFF).
    
    Args:
        data: Byte sequence to analyze
    
    Returns:
        Bias [0.0, 1.0], where 0 = biased towards 0x00, 1 = biased towards 0xFF, 0.5 = no bias
    
    Examples:
        >>> ciphertext_bias(b'\\xff' * 100)  # High bias
        1.0
        >>> ciphertext_bias(b'\\x00' * 100)  # Low bias
        0.0
    """
    if len(data) == 0:
        return 0.5  # No bias
    
    # Calculate mean byte value
    mean_value = sum(data) / len(data)
    
    # Normalize to [0, 1]
    bias = mean_value / 255.0
    
    return bias


def randomness_score(data: bytes) -> float:
    """
    Calculate composite randomness score.
    
    Combines multiple metrics to produce overall randomness score.
    Higher score indicates better randomness.
    
    Args:
        data: Byte sequence to analyze
    
    Returns:
        Randomness score [0.0, 1.0], where 1.0 = perfect randomness
    
    Examples:
        >>> randomness_score(bytes(range(256)) * 10)  # High randomness
        ~0.9
        >>> randomness_score(b'\\x00' * 1000)  # Low randomness
        ~0.0
    """
    if len(data) == 0:
        return 0.0
    
    # Component 1: Entropy (weight 0.5)
    entropy = shannon_entropy(data)
    entropy_score = entropy / 8.0
    
    # Component 2: Frequency uniformity (weight 0.3)
    bias = frequency_bias(data)
    uniformity_score = 1.0 - bias
    
    # Component 3: Chi-square test (weight 0.2)
    chi2 = chi_square_statistic(data)
    # Expected chi-square for uniform distribution is 255
    # Values close to 255 are good
    chi2_score = max(0.0, 1.0 - abs(chi2 - 255) / 255.0)
    
    # Weighted combination
    score = (
        0.5 * entropy_score +
        0.3 * uniformity_score +
        0.2 * chi2_score
    )
    
    return min(1.0, max(0.0, score))


def compression_ratio(plaintext: bytes, ciphertext: bytes) -> float:
    """
    Calculate compression ratio.
    
    Good encryption should not compress well (ratio close to 1.0).
    
    Args:
        plaintext: Original plaintext
        ciphertext: Encrypted ciphertext
    
    Returns:
        Compression ratio, where ratio < 1.0 means data compressed
    
    Examples:
        >>> compression_ratio(b'A' * 1000, bytes(range(256)) * 4)  # High entropy, no compression
        ~1.0
    """
    if len(ciphertext) == 0:
        return 0.0
    
    # Compress ciphertext using zlib
    try:
        compressed = zlib.compress(ciphertext, level=9)
        ratio = len(compressed) / len(ciphertext)
    except Exception:
        ratio = 1.0
    
    return ratio


def hamming_distance_normalized(plaintext: bytes, ciphertext: bytes) -> float:
    """
    Calculate normalized Hamming distance between plaintext and ciphertext.
    
    Measures how many bits differ between plaintext and ciphertext.
    Good encryption should have ~50% bit changes.
    
    Args:
        plaintext: Original plaintext
        ciphertext: Encrypted ciphertext
    
    Returns:
        Normalized Hamming distance [0.0, 1.0], where 0.5 is ideal
    
    Examples:
        >>> hamming_distance_normalized(b'\\x00', b'\\xff')  # All bits flipped
        1.0
        >>> hamming_distance_normalized(b'\\x00', b'\\x00')  # No bits flipped
        0.0
    """
    if len(plaintext) == 0 or len(ciphertext) == 0:
        return 0.0
    
    # Ensure same length (truncate to shorter)
    min_len = min(len(plaintext), len(ciphertext))
    plaintext = plaintext[:min_len]
    ciphertext = ciphertext[:min_len]
    
    # Count bit differences
    bit_differences = 0
    
    for p_byte, c_byte in zip(plaintext, ciphertext):
        # XOR to find differing bits
        xor_result = p_byte ^ c_byte
        # Count set bits using Brian Kernighan's algorithm
        while xor_result:
            bit_differences += 1
            xor_result &= xor_result - 1
    
    # Total possible bits
    total_bits = min_len * 8
    
    # Normalize to [0, 1]
    normalized = bit_differences / total_bits if total_bits > 0 else 0.0
    
    return normalized


def compute_all_metrics(plaintext: bytes, ciphertext: bytes) -> Dict[str, float]:
    """
    Compute all cryptographic metrics in one call.
    
    This is more efficient than calling each metric individually.
    
    Args:
        plaintext: Original plaintext
        ciphertext: Encrypted ciphertext
    
    Returns:
        Dictionary mapping metric_name -> value
    
    Examples:
        >>> metrics = compute_all_metrics(b"Hello", ciphertext)
        >>> print(metrics['shannon_entropy'])
        7.2
    """
    metrics = {}
    
    # Entropy and distribution metrics
    metrics['shannon_entropy'] = shannon_entropy(ciphertext)
    metrics['chi_square_statistic'] = chi_square_statistic(ciphertext)
    metrics['frequency_bias'] = frequency_bias(ciphertext)
    metrics['byte_frequency_variance'] = byte_frequency_variance(ciphertext)
    metrics['ciphertext_bias'] = ciphertext_bias(ciphertext)
    metrics['randomness_score'] = randomness_score(ciphertext)
    
    # Plaintext vs ciphertext metrics
    if plaintext and ciphertext:
        metrics['compression_ratio'] = compression_ratio(plaintext, ciphertext)
        metrics['hamming_distance_normalized'] = hamming_distance_normalized(plaintext, ciphertext)
    else:
        metrics['compression_ratio'] = 0.0
        metrics['hamming_distance_normalized'] = 0.0
    
    return metrics


def validate_metrics(metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate all metrics are within expected bounds.
    
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
    
    # Define bounds for each metric
    bounds = {
        'shannon_entropy': (0.0, 8.0),
        'chi_square_statistic': (0.0, float('inf')),
        'frequency_bias': (0.0, 1.0),
        'byte_frequency_variance': (0.0, float('inf')),
        'ciphertext_bias': (0.0, 1.0),
        'randomness_score': (0.0, 1.0),
        'compression_ratio': (0.0, float('inf')),
        'hamming_distance_normalized': (0.0, 1.0),
    }
    
    # Validate each metric
    for metric_name, (min_val, max_val) in bounds.items():
        if metric_name in metrics:
            value = metrics[metric_name]
            if not (min_val <= value <= max_val):
                errors.append(f"{metric_name} = {value} is outside bounds [{min_val}, {max_val}]")
    
    return len(errors) == 0, errors


# Export all functions
__all__ = [
    'shannon_entropy',
    'chi_square_statistic',
    'frequency_bias',
    'byte_frequency_variance',
    'ciphertext_bias',
    'randomness_score',
    'compression_ratio',
    'hamming_distance_normalized',
    'compute_all_metrics',
    'validate_metrics',
]

