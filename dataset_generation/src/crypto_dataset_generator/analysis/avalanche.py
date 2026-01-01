"""
Avalanche Effect Analysis Module

This module analyzes the avalanche effect in cryptographic algorithms.

Version: 1.0
Date: December 30, 2025
"""

from typing import Dict, List, Tuple


def calculate_avalanche_effect(plaintext: bytes, ciphertext: bytes) -> float:
    """
    Calculate avalanche effect percentage.
    
    The avalanche effect measures how much the ciphertext changes when
    a single bit in the plaintext is flipped. Good encryption should
    have ~50% bit changes.
    
    Args:
        plaintext: Original plaintext
        ciphertext: Encrypted ciphertext
    
    Returns:
        Avalanche effect as percentage [0.0, 100.0]
    
    Examples:
        >>> calculate_avalanche_effect(b'\\x00', b'\\xff')  # All bits changed
        100.0
        >>> calculate_avalanche_effect(b'\\x00', b'\\x80')  # 1 bit changed
        12.5
    """
    if len(plaintext) == 0 or len(ciphertext) == 0:
        return 0.0
    
    # Ensure same length
    min_len = min(len(plaintext), len(ciphertext))
    plaintext = plaintext[:min_len]
    ciphertext = ciphertext[:min_len]
    
    # Count bit differences
    bit_differences = 0
    total_bits = min_len * 8
    
    for p_byte, c_byte in zip(plaintext, ciphertext):
        xor_result = p_byte ^ c_byte
        # Count set bits
        while xor_result:
            bit_differences += 1
            xor_result &= xor_result - 1
    
    # Calculate percentage
    percentage = (bit_differences / total_bits * 100.0) if total_bits > 0 else 0.0
    
    return percentage


def test_strict_avalanche_criterion(plaintext: bytes, ciphertext: bytes, 
                                    threshold: float = 0.45) -> bool:
    """
    Test if encryption satisfies Strict Avalanche Criterion (SAC).
    
    SAC requires that flipping any single input bit should flip each
    output bit with probability ~0.5.
    
    Args:
        plaintext: Original plaintext
        ciphertext: Encrypted ciphertext
        threshold: Minimum acceptable avalanche percentage (default: 45%)
    
    Returns:
        True if SAC is satisfied, False otherwise
    
    Examples:
        >>> test_strict_avalanche_criterion(b'test', good_ciphertext)
        True
        >>> test_strict_avalanche_criterion(b'test', weak_ciphertext)
        False
    """
    avalanche_pct = calculate_avalanche_effect(plaintext, ciphertext)
    
    # SAC requires avalanche effect close to 50%
    # We use a threshold (default 45-55% is acceptable)
    return threshold <= (avalanche_pct / 100.0) <= (1.0 - threshold)


def bit_change_distribution(plaintext: bytes, ciphertext: bytes) -> Dict[int, int]:
    """
    Calculate distribution of bit position changes.
    
    Shows which bit positions change between plaintext and ciphertext.
    Good encryption should have uniform distribution across all bit positions.
    
    Args:
        plaintext: Original plaintext
        ciphertext: Encrypted ciphertext
    
    Returns:
        Dictionary mapping bit_position -> change_count
    
    Examples:
        >>> dist = bit_change_distribution(b'\\x00', b'\\xff')
        >>> print(dist)  # All 8 bits changed
        {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
    """
    if len(plaintext) == 0 or len(ciphertext) == 0:
        return {}
    
    # Ensure same length
    min_len = min(len(plaintext), len(ciphertext))
    plaintext = plaintext[:min_len]
    ciphertext = ciphertext[:min_len]
    
    # Initialize distribution
    distribution = {}
    
    # Count changes at each bit position
    for p_byte, c_byte in zip(plaintext, ciphertext):
        xor_result = p_byte ^ c_byte
        
        # Check each bit position (0-7)
        for bit_pos in range(8):
            if xor_result & (1 << bit_pos):
                distribution[bit_pos] = distribution.get(bit_pos, 0) + 1
    
    return distribution


def calculate_sac_score(distribution: Dict[int, int]) -> float:
    """
    Calculate SAC score based on bit change distribution.
    
    A good SAC score indicates uniform bit flipping across all positions.
    
    Args:
        distribution: Dictionary from bit_change_distribution()
    
    Returns:
        SAC score [0.0, 1.0], where 1.0 = perfect SAC
    
    Examples:
        >>> dist = {i: 100 for i in range(8)}  # Uniform distribution
        >>> calculate_sac_score(dist)
        1.0
        >>> dist = {0: 800, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}  # Biased
        >>> calculate_sac_score(dist)
        < 0.5
    """
    if not distribution:
        return 0.0
    
    # Get counts for all 8 bit positions
    counts = [distribution.get(i, 0) for i in range(8)]
    
    if sum(counts) == 0:
        return 0.0
    
    # Calculate mean
    mean = sum(counts) / 8.0
    
    # Calculate variance
    variance = sum((count - mean) ** 2 for count in counts) / 8.0
    
    # Calculate coefficient of variation (CV)
    # Lower CV = more uniform = better SAC
    cv = (variance ** 0.5) / mean if mean > 0 else float('inf')
    
    # Convert to score [0, 1]
    # CV of 0 = perfect score of 1.0
    # CV of 1 or higher = score of 0.0
    score = max(0.0, 1.0 - cv)
    
    return score


def analyze_avalanche_detailed(plaintext: bytes, ciphertext: bytes) -> Dict[str, float]:
    """
    Perform detailed avalanche analysis.
    
    Args:
        plaintext: Original plaintext
        ciphertext: Encrypted ciphertext
    
    Returns:
        Dictionary with detailed avalanche metrics
    
    Examples:
        >>> result = analyze_avalanche_detailed(plaintext, ciphertext)
        >>> print(result['avalanche_percentage'])
        49.5
        >>> print(result['sac_satisfied'])
        True
    """
    result = {}
    
    # Calculate avalanche effect
    result['avalanche_percentage'] = calculate_avalanche_effect(plaintext, ciphertext)
    result['avalanche_normalized'] = result['avalanche_percentage'] / 100.0
    
    # Test SAC
    result['sac_satisfied'] = test_strict_avalanche_criterion(plaintext, ciphertext)
    
    # Bit change distribution
    distribution = bit_change_distribution(plaintext, ciphertext)
    result['bit_change_distribution'] = distribution
    
    # SAC score
    result['sac_score'] = calculate_sac_score(distribution)
    
    # Additional metrics
    if distribution:
        counts = [distribution.get(i, 0) for i in range(8)]
        result['min_bit_changes'] = min(counts) if counts else 0
        result['max_bit_changes'] = max(counts) if counts else 0
        result['mean_bit_changes'] = sum(counts) / 8.0 if counts else 0.0
    else:
        result['min_bit_changes'] = 0
        result['max_bit_changes'] = 0
        result['mean_bit_changes'] = 0.0
    
    return result


# Export all functions
__all__ = [
    'calculate_avalanche_effect',
    'test_strict_avalanche_criterion',
    'bit_change_distribution',
    'calculate_sac_score',
    'analyze_avalanche_detailed',
]

