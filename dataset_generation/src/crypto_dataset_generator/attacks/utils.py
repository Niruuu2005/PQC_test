"""
Attack Utility Functions

Common utility functions for attack implementations.

Version: 1.0
Date: December 30, 2025
"""

import math
import random
from typing import List, Tuple, Optional, Dict
from collections import Counter


def hamming_weight(data: bytes) -> int:
    """
    Calculate Hamming weight (number of 1 bits).
    
    Args:
        data: Byte sequence
    
    Returns:
        Number of 1 bits
    
    Examples:
        >>> hamming_weight(b'\\xff')  # 8 bits set
        8
        >>> hamming_weight(b'\\x00')  # 0 bits set
        0
    """
    weight = 0
    for byte in data:
        while byte:
            weight += 1
            byte &= byte - 1
    return weight


def hamming_distance(data1: bytes, data2: bytes) -> int:
    """
    Calculate Hamming distance between two byte sequences.
    
    Args:
        data1: First byte sequence
        data2: Second byte sequence
    
    Returns:
        Number of differing bits
    
    Examples:
        >>> hamming_distance(b'\\x00', b'\\xff')
        8
        >>> hamming_distance(b'\\x00', b'\\x00')
        0
    """
    min_len = min(len(data1), len(data2))
    distance = 0
    
    for i in range(min_len):
        xor = data1[i] ^ data2[i]
        distance += hamming_weight(bytes([xor]))
    
    # Account for length difference
    distance += abs(len(data1) - len(data2)) * 8
    
    return distance


def xor_bytes(data1: bytes, data2: bytes) -> bytes:
    """
    XOR two byte sequences.
    
    Args:
        data1: First byte sequence
        data2: Second byte sequence
    
    Returns:
        XOR result (length of shorter sequence)
    
    Examples:
        >>> xor_bytes(b'\\xff', b'\\x00')
        b'\\xff'
        >>> xor_bytes(b'\\xaa', b'\\x55')
        b'\\xff'
    """
    min_len = min(len(data1), len(data2))
    return bytes([data1[i] ^ data2[i] for i in range(min_len)])


def generate_random_key(key_size_bits: int, seed: Optional[int] = None) -> bytes:
    """
    Generate a random key for brute-force testing.
    
    Args:
        key_size_bits: Key size in bits
        seed: Random seed (optional)
    
    Returns:
        Random key as bytes
    
    Examples:
        >>> key = generate_random_key(128, seed=42)
        >>> len(key)
        16
    """
    if seed is not None:
        random.seed(seed)
    
    key_size_bytes = (key_size_bits + 7) // 8
    return bytes([random.randint(0, 255) for _ in range(key_size_bytes)])


def calculate_entropy_bits(data: bytes) -> float:
    """
    Calculate Shannon entropy of data.
    
    Args:
        data: Byte sequence
    
    Returns:
        Entropy in bits per byte
    
    Examples:
        >>> calculate_entropy_bits(bytes(range(256)))
        8.0
        >>> calculate_entropy_bits(b'\\x00' * 100)
        0.0
    """
    if len(data) == 0:
        return 0.0
    
    if len(set(data)) == 1:
        return 0.0
    
    byte_counts = Counter(data)
    total = len(data)
    
    entropy = 0.0
    for count in byte_counts.values():
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob)
    
    return entropy


def frequency_analysis(data: bytes, top_n: int = 10) -> List[Tuple[int, int, float]]:
    """
    Perform frequency analysis on data.
    
    Args:
        data: Byte sequence to analyze
        top_n: Number of top bytes to return
    
    Returns:
        List of (byte_value, count, frequency) tuples
    
    Examples:
        >>> result = frequency_analysis(b'AAABBC')
        >>> result[0]  # Most frequent
        (65, 3, 0.5)  # 'A' appears 3 times (50%)
    """
    if len(data) == 0:
        return []
    
    byte_counts = Counter(data)
    total = len(data)
    
    # Get top N most common
    top_bytes = byte_counts.most_common(top_n)
    
    result = []
    for byte_val, count in top_bytes:
        frequency = count / total
        result.append((byte_val, count, frequency))
    
    return result


def detect_patterns(data: bytes, pattern_length: int = 4) -> List[Tuple[bytes, int]]:
    """
    Detect repeating patterns in data.
    
    Args:
        data: Byte sequence to analyze
        pattern_length: Length of patterns to search for
    
    Returns:
        List of (pattern, count) tuples for patterns appearing > once
    
    Examples:
        >>> detect_patterns(b'ABCABCABC', pattern_length=3)
        [(b'ABC', 3)]
    """
    if len(data) < pattern_length:
        return []
    
    patterns = Counter()
    
    for i in range(len(data) - pattern_length + 1):
        pattern = data[i:i + pattern_length]
        patterns[pattern] += 1
    
    # Return patterns that appear more than once
    result = [(pattern, count) for pattern, count in patterns.items() if count > 1]
    result.sort(key=lambda x: x[1], reverse=True)
    
    return result


def estimate_key_size_kasiski(data: bytes, max_key_size: int = 30) -> Optional[int]:
    """
    Estimate key size using Kasiski examination.
    
    Args:
        data: Ciphertext to analyze
        max_key_size: Maximum key size to consider
    
    Returns:
        Estimated key size, or None if not found
    
    Examples:
        >>> estimate_key_size_kasiski(periodic_ciphertext)
        8  # Estimated key size
    """
    if len(data) < 20:
        return None
    
    # Find repeated trigrams
    trigram_positions = {}
    
    for i in range(len(data) - 2):
        trigram = data[i:i + 3]
        if trigram not in trigram_positions:
            trigram_positions[trigram] = []
        trigram_positions[trigram].append(i)
    
    # Calculate distances between repetitions
    distances = []
    for positions in trigram_positions.values():
        if len(positions) > 1:
            for j in range(1, len(positions)):
                distances.append(positions[j] - positions[j-1])
    
    if not distances:
        return None
    
    # Find GCD of distances
    from math import gcd
    from functools import reduce
    
    common_divisor = reduce(gcd, distances)
    
    if 2 <= common_divisor <= max_key_size:
        return common_divisor
    
    return None


def calculate_index_of_coincidence(data: bytes) -> float:
    """
    Calculate Index of Coincidence for frequency analysis.
    
    Args:
        data: Byte sequence
    
    Returns:
        Index of Coincidence value
    
    Examples:
        >>> calculate_index_of_coincidence(random_data)
        ~0.0039  # Close to 1/256 for random
    """
    if len(data) < 2:
        return 0.0
    
    byte_counts = Counter(data)
    n = len(data)
    
    sum_freq = sum(count * (count - 1) for count in byte_counts.values())
    
    if n * (n - 1) == 0:
        return 0.0
    
    ioc = sum_freq / (n * (n - 1))
    
    return ioc


def find_repeating_key_xor(ciphertext: bytes, key_size: int) -> Optional[bytes]:
    """
    Attempt to find repeating-key XOR key.
    
    Args:
        ciphertext: XOR-encrypted data
        key_size: Suspected key size
    
    Returns:
        Recovered key (or best guess)
    
    Examples:
        >>> find_repeating_key_xor(xor_ciphertext, 8)
        b'secretky'
    """
    if len(ciphertext) < key_size:
        return None
    
    # Split into blocks
    blocks = [[] for _ in range(key_size)]
    
    for i, byte in enumerate(ciphertext):
        blocks[i % key_size].append(byte)
    
    # Find most likely key byte for each position
    key = []
    
    for block in blocks:
        if not block:
            key.append(0)
            continue
        
        # Try each possible key byte
        best_score = 0
        best_byte = 0
        
        for key_byte in range(256):
            # Decrypt with this key byte
            decrypted = [b ^ key_byte for b in block]
            
            # Score based on ASCII printable characters
            score = sum(1 for b in decrypted if 32 <= b <= 126)
            
            if score > best_score:
                best_score = score
                best_byte = key_byte
        
        key.append(best_byte)
    
    return bytes(key)


def simulate_timing_variations(base_time_ms: float, key_dependent: bool = False, 
                               key_byte: int = 0) -> float:
    """
    Simulate timing attack variations.
    
    Args:
        base_time_ms: Base execution time in milliseconds
        key_dependent: Whether timing depends on key
        key_byte: Key byte value (affects timing if key_dependent=True)
    
    Returns:
        Simulated execution time
    
    Examples:
        >>> simulate_timing_variations(10.0, key_dependent=True, key_byte=0xff)
        10.08  # Slightly higher due to key dependency
    """
    # Add random noise
    noise = random.gauss(0, base_time_ms * 0.01)
    
    # Add key-dependent timing leak if applicable
    if key_dependent:
        # Hamming weight affects timing
        hw = bin(key_byte).count('1')
        timing_leak = hw * base_time_ms * 0.001  # 0.1% per bit
    else:
        timing_leak = 0
    
    return base_time_ms + noise + timing_leak


# Export all functions
__all__ = [
    'hamming_weight',
    'hamming_distance',
    'xor_bytes',
    'generate_random_key',
    'calculate_entropy_bits',
    'frequency_analysis',
    'detect_patterns',
    'estimate_key_size_kasiski',
    'calculate_index_of_coincidence',
    'find_repeating_key_xor',
    'simulate_timing_variations',
]

