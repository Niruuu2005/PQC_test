"""
Correlation Analysis Module

This module provides correlation analysis for cryptographic data.

Version: 1.0
Date: December 30, 2025
"""

import math
from typing import List, Tuple


def autocorrelation(data: bytes, lag: int = 1) -> float:
    """
    Calculate autocorrelation at specified lag.
    
    Autocorrelation measures the similarity of a signal with a
    delayed version of itself. Good encryption should have low
    autocorrelation (close to 0).
    
    Args:
        data: Byte sequence to analyze
        lag: Lag value (default: 1)
    
    Returns:
        Autocorrelation coefficient [-1.0, 1.0]
    
    Examples:
        >>> autocorrelation(bytes(range(256)), lag=1)
        ≈ 0.0  # Random data has low autocorrelation
        >>> autocorrelation(b'ABAB' * 100, lag=2)
        ≈ 1.0  # Periodic data has high autocorrelation
    """
    if len(data) <= lag:
        return 0.0
    
    # Convert bytes to numeric values
    values = list(data)
    n = len(values)
    
    # Calculate mean
    mean = sum(values) / n
    
    # Calculate variance
    variance = sum((x - mean) ** 2 for x in values) / n
    
    if variance == 0:
        return 0.0
    
    # Calculate autocorrelation
    autocorr = 0.0
    for i in range(n - lag):
        autocorr += (values[i] - mean) * (values[i + lag] - mean)
    
    autocorr = autocorr / (n * variance)
    
    return autocorr


def autocorrelation_function(data: bytes, max_lag: int = 10) -> List[float]:
    """
    Calculate autocorrelation function for multiple lags.
    
    Args:
        data: Byte sequence to analyze
        max_lag: Maximum lag to calculate (default: 10)
    
    Returns:
        List of autocorrelation values for lags 1 to max_lag
    
    Examples:
        >>> acf = autocorrelation_function(ciphertext, max_lag=5)
        >>> print(acf)
        [0.01, 0.02, -0.01, 0.00, -0.02]  # Low correlations = good
    """
    if len(data) == 0:
        return []
    
    acf = []
    for lag in range(1, max_lag + 1):
        if len(data) <= lag:
            break
        acf.append(autocorrelation(data, lag))
    
    return acf


def cross_correlation(data1: bytes, data2: bytes) -> float:
    """
    Calculate cross-correlation between two byte sequences.
    
    Cross-correlation measures the similarity between two signals.
    Low cross-correlation between plaintext and ciphertext is desirable.
    
    Args:
        data1: First byte sequence
        data2: Second byte sequence
    
    Returns:
        Cross-correlation coefficient [-1.0, 1.0]
    
    Examples:
        >>> cross_correlation(plaintext, ciphertext)
        ≈ 0.0  # Good encryption has low cross-correlation
        >>> cross_correlation(data, data)
        1.0  # Perfect correlation with self
    """
    if len(data1) == 0 or len(data2) == 0:
        return 0.0
    
    # Use shorter length
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]
    
    # Convert to numeric values
    values1 = list(data1)
    values2 = list(data2)
    
    # Calculate means
    mean1 = sum(values1) / min_len
    mean2 = sum(values2) / min_len
    
    # Calculate standard deviations
    std1 = math.sqrt(sum((x - mean1) ** 2 for x in values1) / min_len)
    std2 = math.sqrt(sum((x - mean2) ** 2 for x in values2) / min_len)
    
    if std1 == 0 or std2 == 0:
        return 0.0
    
    # Calculate cross-correlation
    cross_corr = 0.0
    for v1, v2 in zip(values1, values2):
        cross_corr += (v1 - mean1) * (v2 - mean2)
    
    cross_corr = cross_corr / (min_len * std1 * std2)
    
    return cross_corr


def calculate_correlation_coefficient(data: bytes, plaintext: bytes) -> float:
    """
    Calculate Pearson correlation coefficient between ciphertext and plaintext.
    
    This is essentially cross-correlation between plaintext and ciphertext.
    Good encryption should have correlation close to 0.
    
    Args:
        data: Ciphertext
        plaintext: Original plaintext
    
    Returns:
        Correlation coefficient [-1.0, 1.0]
    
    Examples:
        >>> calculate_correlation_coefficient(ciphertext, plaintext)
        ≈ 0.0  # Good encryption
    """
    return cross_correlation(data, plaintext)


def detect_periodic_behavior(data: bytes, max_period: int = 256) -> Tuple[bool, int]:
    """
    Detect periodic behavior in byte sequence.
    
    Searches for repeating patterns that might indicate weakness in encryption.
    
    Args:
        data: Byte sequence to analyze
        max_period: Maximum period length to search (default: 256)
    
    Returns:
        Tuple of (has_period, period_length)
        If no period detected, returns (False, 0)
    
    Examples:
        >>> detect_periodic_behavior(b'ABCABC' * 100)
        (True, 3)
        >>> detect_periodic_behavior(random_bytes)
        (False, 0)
    """
    if len(data) < 2:
        return False, 0
    
    # Search for periods using autocorrelation
    best_period = 0
    best_correlation = 0.3  # Threshold for detecting periodicity
    
    for period in range(1, min(max_period, len(data) // 2)):
        corr = autocorrelation(data, lag=period)
        
        if corr > best_correlation:
            best_correlation = corr
            best_period = period
    
    has_period = best_period > 0
    
    return has_period, best_period


def calculate_serial_correlation(data: bytes) -> float:
    """
    Calculate serial correlation (lag-1 autocorrelation).
    
    This is a specific case of autocorrelation with lag=1,
    measuring correlation between consecutive bytes.
    
    Args:
        data: Byte sequence to analyze
    
    Returns:
        Serial correlation coefficient [-1.0, 1.0]
    
    Examples:
        >>> calculate_serial_correlation(ciphertext)
        ≈ 0.0  # Good encryption has low serial correlation
    """
    return autocorrelation(data, lag=1)


def analyze_correlation_detailed(plaintext: bytes, ciphertext: bytes) -> dict:
    """
    Perform detailed correlation analysis.
    
    Args:
        plaintext: Original plaintext
        ciphertext: Encrypted ciphertext
    
    Returns:
        Dictionary with detailed correlation metrics
    
    Examples:
        >>> result = analyze_correlation_detailed(plaintext, ciphertext)
        >>> print(result['serial_correlation'])
        0.01
        >>> print(result['plaintext_correlation'])
        0.02
    """
    result = {}
    
    # Ciphertext autocorrelation (lag 1)
    result['serial_correlation'] = calculate_serial_correlation(ciphertext)
    
    # Autocorrelation function (first 10 lags)
    result['autocorrelation_function'] = autocorrelation_function(ciphertext, max_lag=10)
    
    # Cross-correlation with plaintext
    result['plaintext_correlation'] = cross_correlation(plaintext, ciphertext)
    
    # Periodic behavior detection
    has_period, period = detect_periodic_behavior(ciphertext)
    result['has_periodic_behavior'] = has_period
    result['period_length'] = period
    
    # Average absolute autocorrelation
    acf = result['autocorrelation_function']
    result['mean_abs_autocorrelation'] = sum(abs(x) for x in acf) / len(acf) if acf else 0.0
    
    # Max absolute autocorrelation
    result['max_abs_autocorrelation'] = max((abs(x) for x in acf), default=0.0)
    
    return result


# Export all functions
__all__ = [
    'autocorrelation',
    'autocorrelation_function',
    'cross_correlation',
    'calculate_correlation_coefficient',
    'detect_periodic_behavior',
    'calculate_serial_correlation',
    'analyze_correlation_detailed',
]

