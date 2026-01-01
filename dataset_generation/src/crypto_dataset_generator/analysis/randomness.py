"""
Randomness Tests Module

This module implements NIST-style randomness tests for cryptographic analysis.

Version: 1.0
Date: December 30, 2025
"""

from typing import Tuple, Dict, Any
import math
from collections import Counter


def runs_test(data: bytes) -> Tuple[int, float, float]:
    """
    NIST Runs Test for randomness.
    
    A run is a sequence of consecutive identical bits.
    Tests if the number of runs is as expected for random data.
    
    Args:
        data: Byte sequence to test
    
    Returns:
        Tuple of (runs_count, expected_runs, z_score)
        - |z_score| < 2.0 suggests randomness
    
    Examples:
        >>> runs, expected, z = runs_test(ciphertext)
        >>> print(f"Z-score: {z:.2f} {'(random)' if abs(z) < 2.0 else '(not random)'}")
    """
    if len(data) == 0:
        return 0, 0.0, 0.0
    
    # Convert bytes to bits
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    
    n = len(bits)
    if n < 2:
        return 0, 0.0, 0.0
    
    # Count 1s and 0s
    n1 = sum(bits)
    n0 = n - n1
    
    if n1 == 0 or n0 == 0:
        return 0, 0.0, 0.0
    
    # Count runs
    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i - 1]:
            runs += 1
    
    # Expected number of runs
    pi = n1 / n
    expected_runs = 2 * n * pi * (1 - pi) + 1
    
    # Standard deviation
    std_dev = math.sqrt(2 * n * pi * (1 - pi) * (2 * n * pi * (1 - pi) - 1))
    
    # Z-score
    if std_dev > 0:
        z_score = (runs - expected_runs) / std_dev
    else:
        z_score = 0.0
    
    return runs, expected_runs, z_score


def monobit_test(data: bytes) -> Tuple[int, int, float]:
    """
    NIST Monobit (Frequency) Test.
    
    Tests if the number of 0s and 1s are approximately equal.
    
    Args:
        data: Byte sequence to test
    
    Returns:
        Tuple of (zeros_count, ones_count, chi2_statistic)
        - Lower chi2 indicates better randomness
    
    Examples:
        >>> zeros, ones, chi2 = monobit_test(ciphertext)
        >>> print(f"0s: {zeros}, 1s: {ones}, χ² = {chi2:.2f}")
    """
    if len(data) == 0:
        return 0, 0, 0.0
    
    # Convert bytes to bits and count
    zeros = 0
    ones = 0
    
    for byte in data:
        for i in range(8):
            if (byte >> i) & 1:
                ones += 1
            else:
                zeros += 1
    
    total = zeros + ones
    expected = total / 2.0
    
    # Chi-square statistic
    chi2 = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected
    
    return zeros, ones, chi2


def block_frequency_test(data: bytes, block_size: int = 128) -> Tuple[float, float]:
    """
    NIST Block Frequency Test.
    
    Divides data into blocks and tests if the proportion of 1s
    in each block is approximately 1/2.
    
    Args:
        data: Byte sequence to test
        block_size: Size of blocks in bits (default: 128)
    
    Returns:
        Tuple of (chi2_statistic, p_value)
    
    Examples:
        >>> chi2, p = block_frequency_test(ciphertext, block_size=128)
        >>> print(f"χ² = {chi2:.2f}, p = {p:.4f}")
    """
    if len(data) == 0 or block_size == 0:
        return 0.0, 1.0
    
    # Convert bytes to bits
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    
    n = len(bits)
    num_blocks = n // block_size
    
    if num_blocks == 0:
        return 0.0, 1.0
    
    # Calculate proportion of 1s in each block
    chi2 = 0.0
    
    for i in range(num_blocks):
        block_start = i * block_size
        block_end = block_start + block_size
        block = bits[block_start:block_end]
        
        # Proportion of 1s
        pi = sum(block) / block_size
        
        # Chi-square contribution
        chi2 += (pi - 0.5) ** 2
    
    chi2 = 4.0 * block_size * chi2
    
    # Calculate p-value (approximation)
    from .statistical import chi_square_pvalue
    p_value = chi_square_pvalue(chi2, df=num_blocks - 1)
    
    return chi2, p_value


def longest_run_test(data: bytes) -> Tuple[int, float]:
    """
    Test for longest run of identical bits.
    
    Finds the longest sequence of consecutive 1s or 0s.
    Very long runs suggest non-randomness.
    
    Args:
        data: Byte sequence to test
    
    Returns:
        Tuple of (longest_run_length, expected_length)
    
    Examples:
        >>> longest, expected = longest_run_test(ciphertext)
        >>> print(f"Longest run: {longest}, Expected: {expected:.1f}")
    """
    if len(data) == 0:
        return 0, 0.0
    
    # Convert bytes to bits
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    
    n = len(bits)
    if n == 0:
        return 0, 0.0
    
    # Find longest run
    longest_run = 0
    current_run = 1
    
    for i in range(1, n):
        if bits[i] == bits[i - 1]:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    
    # Expected longest run (approximate)
    expected = math.log2(n) if n > 0 else 0.0
    
    return longest_run, expected


def byte_pattern_test(data: bytes) -> Dict[str, float]:
    """
    Test for repeating byte patterns.
    
    Checks for common patterns that shouldn't appear in random data.
    
    Args:
        data: Byte sequence to test
    
    Returns:
        Dictionary with pattern statistics
    
    Examples:
        >>> result = byte_pattern_test(ciphertext)
        >>> print(result['max_consecutive_same'])
        2  # Good - no long sequences of same byte
    """
    if len(data) == 0:
        return {
            'max_consecutive_same': 0,
            'num_unique_bytes': 0,
            'most_common_byte_count': 0,
        }
    
    result = {}
    
    # Max consecutive identical bytes
    max_consecutive = 1
    current_consecutive = 1
    
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    
    result['max_consecutive_same'] = max_consecutive
    
    # Number of unique bytes
    result['num_unique_bytes'] = len(set(data))
    
    # Most common byte count
    byte_counts = Counter(data)
    result['most_common_byte_count'] = max(byte_counts.values()) if byte_counts else 0
    
    return result


def test_randomness(data: bytes) -> Dict[str, Any]:
    """
    Execute multiple randomness tests and aggregate results.
    
    Args:
        data: Byte sequence to test
    
    Returns:
        Dictionary with all test results and overall assessment
    
    Examples:
        >>> results = test_randomness(ciphertext)
        >>> print(f"Overall: {results['overall_random']}")
        True
    """
    results = {}
    
    # Monobit test
    zeros, ones, monobit_chi2 = monobit_test(data)
    results['monobit'] = {
        'zeros': zeros,
        'ones': ones,
        'chi2': monobit_chi2,
        'passed': monobit_chi2 < 3.84,  # 5% significance level, df=1
    }
    
    # Runs test
    runs, expected_runs, z_score = runs_test(data)
    results['runs'] = {
        'runs_count': runs,
        'expected_runs': expected_runs,
        'z_score': z_score,
        'passed': abs(z_score) < 2.0,  # Within 2 standard deviations
    }
    
    # Block frequency test
    chi2, p_value = block_frequency_test(data, block_size=128)
    results['block_frequency'] = {
        'chi2': chi2,
        'p_value': p_value,
        'passed': p_value > 0.01,  # 1% significance level
    }
    
    # Longest run test
    longest, expected = longest_run_test(data)
    results['longest_run'] = {
        'longest_run': longest,
        'expected': expected,
        'ratio': longest / expected if expected > 0 else 0.0,
        'passed': longest < expected * 2,  # Heuristic: no more than 2x expected
    }
    
    # Byte pattern test
    pattern_stats = byte_pattern_test(data)
    results['patterns'] = pattern_stats
    results['patterns']['passed'] = (
        pattern_stats['max_consecutive_same'] < len(data) / 10 and
        pattern_stats['num_unique_bytes'] > 200  # Expect most bytes represented
    )
    
    # Overall assessment
    passed_tests = sum(1 for test_result in [
        results['monobit']['passed'],
        results['runs']['passed'],
        results['block_frequency']['passed'],
        results['longest_run']['passed'],
        results['patterns']['passed'],
    ] if test_result)
    
    results['tests_passed'] = passed_tests
    results['tests_total'] = 5
    results['overall_random'] = passed_tests >= 4  # At least 4 out of 5 tests pass
    
    return results


# Export all functions
__all__ = [
    'runs_test',
    'monobit_test',
    'block_frequency_test',
    'longest_run_test',
    'byte_pattern_test',
    'test_randomness',
]

