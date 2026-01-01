"""
Statistical Tests Module

This module provides statistical tests for cryptographic analysis.

Version: 1.0
Date: December 30, 2025
"""

from typing import Tuple, List
import math


def chi_square_test(data: bytes) -> Tuple[float, float]:
    """
    Perform chi-square test for uniformity.
    
    Tests the null hypothesis that the byte distribution is uniform.
    
    Args:
        data: Byte sequence to test
    
    Returns:
        Tuple of (chi2_statistic, p_value)
        - chi2_statistic: Chi-square test statistic
        - p_value: Probability value (higher = more uniform)
    
    Examples:
        >>> chi2, p = chi_square_test(bytes(range(256)) * 10)
        >>> print(f"χ² = {chi2:.2f}, p = {p:.4f}")
        χ² = 0.00, p = 1.0000
    """
    if len(data) == 0:
        return 0.0, 1.0
    
    # Count byte frequencies
    observed = [0] * 256
    for byte in data:
        observed[byte] += 1
    
    # Expected frequency (uniform)
    expected = len(data) / 256.0
    
    # Calculate chi-square statistic
    chi2 = 0.0
    for obs in observed:
        chi2 += ((obs - expected) ** 2) / expected
    
    # Calculate p-value (approximation using chi-square distribution)
    # Degrees of freedom = 255
    p_value = chi_square_pvalue(chi2, df=255)
    
    return chi2, p_value


def chi_square_pvalue(chi2: float, df: int) -> float:
    """
    Calculate p-value for chi-square statistic.
    
    Uses approximation formula for large df.
    
    Args:
        chi2: Chi-square statistic
        df: Degrees of freedom
    
    Returns:
        p-value [0.0, 1.0]
    """
    if chi2 <= 0:
        return 1.0
    
    # Wilson-Hilferty approximation for large df
    # z = ((chi2/df)^(1/3) - (1 - 2/(9*df))) / sqrt(2/(9*df))
    try:
        z = ((chi2 / df) ** (1/3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        
        # Convert to p-value using standard normal CDF approximation
        p = 0.5 * (1 + math.erf(-z / math.sqrt(2)))
        
        return max(0.0, min(1.0, p))
    except (ValueError, ZeroDivisionError):
        return 0.5


def ks_test(data: bytes) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test for uniform distribution.
    
    Tests if the byte values follow a uniform distribution.
    
    Args:
        data: Byte sequence to test
    
    Returns:
        Tuple of (statistic, p_value)
        - statistic: KS test statistic
        - p_value: Probability value
    
    Examples:
        >>> ks_stat, p = ks_test(ciphertext)
        >>> print(f"KS = {ks_stat:.4f}, p = {p:.4f}")
    """
    if len(data) == 0:
        return 0.0, 1.0
    
    # Sort data
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Calculate empirical CDF vs uniform CDF
    max_diff = 0.0
    
    for i, value in enumerate(sorted_data):
        # Empirical CDF
        empirical_cdf = (i + 1) / n
        
        # Theoretical uniform CDF
        uniform_cdf = (value + 1) / 256.0
        
        # Calculate difference
        diff = abs(empirical_cdf - uniform_cdf)
        max_diff = max(max_diff, diff)
    
    # KS statistic
    ks_statistic = max_diff
    
    # Calculate p-value (approximation)
    p_value = ks_pvalue(ks_statistic, n)
    
    return ks_statistic, p_value


def ks_pvalue(ks_stat: float, n: int) -> float:
    """
    Calculate p-value for KS statistic.
    
    Uses asymptotic approximation.
    
    Args:
        ks_stat: KS test statistic
        n: Sample size
    
    Returns:
        p-value [0.0, 1.0]
    """
    if ks_stat <= 0:
        return 1.0
    
    # Asymptotic formula: P ≈ 2 * exp(-2 * n * D^2)
    try:
        lambda_val = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * ks_stat
        p = 2 * math.exp(-2 * lambda_val ** 2)
        
        return max(0.0, min(1.0, p))
    except (ValueError, OverflowError):
        return 0.0


def entropy_test(data: bytes, expected_entropy: float = 7.5) -> bool:
    """
    Test if entropy exceeds threshold.
    
    Args:
        data: Byte sequence to test
        expected_entropy: Minimum acceptable entropy (default: 7.5 bits)
    
    Returns:
        True if entropy >= expected_entropy, False otherwise
    
    Examples:
        >>> entropy_test(good_ciphertext, expected_entropy=7.5)
        True
        >>> entropy_test(weak_ciphertext, expected_entropy=7.5)
        False
    """
    if len(data) == 0:
        return False
    
    # Calculate Shannon entropy
    from collections import Counter
    
    byte_counts = Counter(data)
    total_bytes = len(data)
    
    entropy = 0.0
    for count in byte_counts.values():
        if count > 0:
            probability = count / total_bytes
            entropy -= probability * math.log2(probability)
    
    return entropy >= expected_entropy


def runs_test_statistic(data: bytes) -> Tuple[int, float, float]:
    """
    Calculate runs test statistic for randomness.
    
    A run is a sequence of consecutive identical bits.
    
    Args:
        data: Byte sequence to test
    
    Returns:
        Tuple of (actual_runs, expected_runs, z_score)
    
    Examples:
        >>> actual, expected, z = runs_test_statistic(ciphertext)
        >>> print(f"Runs: {actual}, Expected: {expected:.1f}, Z: {z:.2f}")
    """
    if len(data) == 0:
        return 0, 0.0, 0.0
    
    # Convert bytes to bits
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    
    n = len(bits)
    
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
    
    # Expected runs
    expected_runs = ((2 * n0 * n1) / n) + 1
    
    # Variance of runs
    variance = ((2 * n0 * n1) * (2 * n0 * n1 - n)) / ((n ** 2) * (n - 1))
    
    # Z-score
    if variance > 0:
        z_score = (runs - expected_runs) / math.sqrt(variance)
    else:
        z_score = 0.0
    
    return runs, expected_runs, z_score


def anderson_darling_test(data: bytes) -> Tuple[float, List[float]]:
    """
    Perform Anderson-Darling test for uniformity.
    
    The Anderson-Darling test is more sensitive to deviations in the tails
    of the distribution compared to KS test.
    
    Args:
        data: Byte sequence to test
    
    Returns:
        Tuple of (statistic, critical_values)
        - statistic: AD test statistic
        - critical_values: [15%, 10%, 5%, 2.5%, 1%] significance levels
    
    Examples:
        >>> ad_stat, critical = anderson_darling_test(ciphertext)
        >>> print(f"AD = {ad_stat:.4f}")
    """
    if len(data) == 0:
        return 0.0, [0.576, 0.656, 0.787, 0.918, 1.092]
    
    # Sort data
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Calculate AD statistic
    ad_statistic = 0.0
    
    for i, value in enumerate(sorted_data):
        # Uniform CDF value
        F = (value + 1) / 256.0
        
        # Avoid log(0)
        if F <= 0:
            F = 1e-10
        if F >= 1:
            F = 1 - 1e-10
        
        # AD formula
        ad_statistic += (2 * (i + 1) - 1) * (math.log(F) + math.log(1 - sorted_data[n - i - 1] / 256.0))
    
    ad_statistic = -n - (ad_statistic / n)
    
    # Critical values for significance levels [15%, 10%, 5%, 2.5%, 1%]
    critical_values = [0.576, 0.656, 0.787, 0.918, 1.092]
    
    return ad_statistic, critical_values


# Export all functions
__all__ = [
    'chi_square_test',
    'ks_test',
    'entropy_test',
    'runs_test_statistic',
    'anderson_darling_test',
]

