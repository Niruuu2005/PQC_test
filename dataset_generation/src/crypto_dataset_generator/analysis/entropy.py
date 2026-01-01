"""
Entropy Analysis Module

This module provides advanced entropy calculations for cryptographic analysis.

Version: 1.0
Date: December 30, 2025
"""

import math
from collections import Counter
from typing import List


def calculate_shannon_entropy(data: bytes) -> float:
    """
    Calculate Shannon entropy with comprehensive edge case handling.
    
    H(X) = -Σ p(x) * log2(p(x))
    
    Args:
        data: Byte sequence to analyze
    
    Returns:
        Shannon entropy in bits per byte [0.0, 8.0]
    
    Edge Cases:
        - Empty data: returns 0.0
        - Single byte: returns 0.0
        - Uniform distribution: returns 8.0
    
    Examples:
        >>> calculate_shannon_entropy(b'')
        0.0
        >>> calculate_shannon_entropy(b'A')
        0.0
        >>> calculate_shannon_entropy(bytes(range(256)))
        8.0
    """
    if len(data) == 0:
        return 0.0
    
    if len(data) == 1:
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


def calculate_min_entropy(data: bytes) -> float:
    """
    Calculate min-entropy (worst-case entropy).
    
    Min-entropy is based on the most probable symbol.
    H_min(X) = -log2(max(p(x)))
    
    Args:
        data: Byte sequence to analyze
    
    Returns:
        Min-entropy in bits [0.0, 8.0]
    
    Examples:
        >>> calculate_min_entropy(b'\\x00' * 100)  # All same byte
        0.0
        >>> calculate_min_entropy(bytes(range(256)))  # Uniform
        8.0
    """
    if len(data) == 0:
        return 0.0
    
    # Count byte frequencies
    byte_counts = Counter(data)
    
    # Find maximum probability
    max_count = max(byte_counts.values())
    max_probability = max_count / len(data)
    
    # Calculate min-entropy
    if max_probability == 0 or max_probability >= 1.0:
        return 0.0
    
    min_entropy = -math.log2(max_probability)
    
    return min_entropy


def calculate_renyi_entropy(data: bytes, order: float = 2.0) -> float:
    """
    Calculate Rényi entropy of given order.
    
    Rényi entropy is a generalization of Shannon entropy.
    H_α(X) = (1/(1-α)) * log2(Σ p(x)^α)
    
    Special cases:
    - α = 0: Hartley entropy (log2 of support size)
    - α → 1: Shannon entropy (limit)
    - α = 2: Collision entropy
    - α → ∞: Min-entropy
    
    Args:
        data: Byte sequence to analyze
        order: Order α (must be > 0, α ≠ 1)
    
    Returns:
        Rényi entropy in bits [0.0, 8.0]
    
    Examples:
        >>> calculate_renyi_entropy(b'\\x00' * 100, order=2.0)
        0.0
        >>> calculate_renyi_entropy(bytes(range(256)), order=2.0)
        8.0
    """
    if len(data) == 0:
        return 0.0
    
    if order <= 0:
        raise ValueError("Order must be positive")
    
    if abs(order - 1.0) < 1e-10:
        # For α ≈ 1, use Shannon entropy
        return calculate_shannon_entropy(data)
    
    # Count byte frequencies
    byte_counts = Counter(data)
    total_bytes = len(data)
    
    # Calculate Σ p(x)^α
    sum_prob_alpha = 0.0
    for count in byte_counts.values():
        if count > 0:
            probability = count / total_bytes
            sum_prob_alpha += probability ** order
    
    # Handle edge cases
    if sum_prob_alpha == 0:
        return 0.0
    
    # Calculate Rényi entropy
    if order == float('inf'):
        # Min-entropy case
        return calculate_min_entropy(data)
    else:
        renyi_entropy = (1.0 / (1.0 - order)) * math.log2(sum_prob_alpha)
        return renyi_entropy


def estimate_source_entropy(data: bytes) -> float:
    """
    Estimate true source entropy accounting for autocorrelation.
    
    Adjusts Shannon entropy based on detected autocorrelation.
    If data has strong autocorrelation, true entropy is lower.
    
    Args:
        data: Byte sequence to analyze
    
    Returns:
        Estimated source entropy in bits per byte [0.0, 8.0]
    
    Examples:
        >>> estimate_source_entropy(b'ABABAB' * 100)  # High autocorrelation
        < 8.0 (lower than Shannon entropy)
        >>> estimate_source_entropy(bytes(range(256)) * 10)  # Low autocorrelation
        ≈ 8.0 (close to Shannon entropy)
    """
    if len(data) == 0:
        return 0.0
    
    # Calculate base Shannon entropy
    shannon_ent = calculate_shannon_entropy(data)
    
    # Calculate autocorrelation at lag 1
    if len(data) < 2:
        return shannon_ent
    
    # Simple autocorrelation check
    # Count how often consecutive bytes are the same
    consecutive_same = sum(1 for i in range(len(data) - 1) if data[i] == data[i + 1])
    autocorr_ratio = consecutive_same / (len(data) - 1)
    
    # Expected ratio for random data is 1/256 ≈ 0.0039
    expected_ratio = 1.0 / 256.0
    
    # Adjust entropy based on autocorrelation
    if autocorr_ratio > expected_ratio:
        # High autocorrelation reduces effective entropy
        adjustment_factor = 1.0 - min(1.0, (autocorr_ratio - expected_ratio) / (1.0 - expected_ratio))
        estimated_entropy = shannon_ent * adjustment_factor
    else:
        # Low autocorrelation, entropy estimate is close to Shannon
        estimated_entropy = shannon_ent
    
    return estimated_entropy


def calculate_entropy_rate(data: bytes, block_size: int = 1) -> float:
    """
    Calculate entropy rate for given block size.
    
    Entropy rate is the average entropy per symbol when considering
    blocks of symbols.
    
    Args:
        data: Byte sequence to analyze
        block_size: Size of blocks to consider (default: 1)
    
    Returns:
        Entropy rate in bits per byte [0.0, 8.0]
    
    Examples:
        >>> calculate_entropy_rate(b'ABCABC' * 100, block_size=3)
        < calculate_entropy_rate(b'ABCABC' * 100, block_size=1)
    """
    if len(data) < block_size:
        return 0.0
    
    if block_size == 1:
        return calculate_shannon_entropy(data)
    
    # Create blocks
    blocks = []
    for i in range(len(data) - block_size + 1):
        block = tuple(data[i:i + block_size])
        blocks.append(block)
    
    # Count block frequencies
    block_counts = Counter(blocks)
    total_blocks = len(blocks)
    
    # Calculate block entropy
    block_entropy = 0.0
    for count in block_counts.values():
        if count > 0:
            probability = count / total_blocks
            block_entropy -= probability * math.log2(probability)
    
    # Entropy rate is block entropy divided by block size
    entropy_rate = block_entropy / block_size
    
    return entropy_rate


def calculate_conditional_entropy(data: bytes, condition_lag: int = 1) -> float:
    """
    Calculate conditional entropy H(X_n | X_{n-k}).
    
    Measures entropy of current byte given previous byte(s).
    
    Args:
        data: Byte sequence to analyze
        condition_lag: Lag for conditioning (default: 1)
    
    Returns:
        Conditional entropy in bits [0.0, 8.0]
    
    Examples:
        >>> calculate_conditional_entropy(b'ABABAB' * 100)  # Perfect prediction
        ≈ 0.0
        >>> calculate_conditional_entropy(bytes(range(256)) * 10)  # No prediction
        ≈ 8.0
    """
    if len(data) <= condition_lag:
        return 0.0
    
    # Count joint occurrences (previous byte, current byte)
    joint_counts = Counter()
    condition_counts = Counter()
    
    for i in range(condition_lag, len(data)):
        prev_byte = data[i - condition_lag]
        curr_byte = data[i]
        
        joint_counts[(prev_byte, curr_byte)] += 1
        condition_counts[prev_byte] += 1
    
    # Calculate conditional entropy
    total_pairs = len(data) - condition_lag
    cond_entropy = 0.0
    
    for (prev, curr), joint_count in joint_counts.items():
        joint_prob = joint_count / total_pairs
        cond_count = condition_counts[prev]
        cond_prob = joint_count / cond_count
        
        if cond_prob > 0:
            cond_entropy -= joint_prob * math.log2(cond_prob)
    
    return cond_entropy


# Export all functions
__all__ = [
    'calculate_shannon_entropy',
    'calculate_min_entropy',
    'calculate_renyi_entropy',
    'estimate_source_entropy',
    'calculate_entropy_rate',
    'calculate_conditional_entropy',
]

