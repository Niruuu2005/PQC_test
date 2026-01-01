"""
Statistical Attacks

Implements 8 statistical cryptanalytic attacks.

Version: 1.0
Date: December 30, 2025
"""

import time
from typing import Dict, Any
from .base_attack import StatisticalAttackBase, AttackResult
from .utils import (frequency_analysis, calculate_entropy_bits, 
                    calculate_index_of_coincidence, detect_patterns, 
                    estimate_key_size_kasiski, find_repeating_key_xor)


class FrequencyAnalysisAttack(StatisticalAttackBase):
    """Attack 1/8: Frequency Analysis - Analyzes byte frequency distribution."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.description = "Character/byte frequency analysis"
    
    def execute(self, ciphertext: bytes, **kwargs) -> AttackResult:
        start = time.time()
        
        # Perform frequency analysis
        freq_result = frequency_analysis(ciphertext, top_n=10)
        
        # Determine success based on frequency distribution
        if freq_result and freq_result[0][2] > 0.1:  # Top byte >10% frequency
            success = True
            confidence = min(freq_result[0][2] * 2, 1.0)
        else:
            success = False
            confidence = 0.0
        
        return AttackResult(
            attack_name=self.attack_name,
            target_algorithm=self.target_algorithm,
            success=success,
            confidence=confidence,
            time_taken=time.time() - start,
            iterations=len(ciphertext),
            metadata={'frequency_distribution': freq_result[:5]}
        )


class IndexOfCoincidenceAttack(StatisticalAttackBase):
    """Attack 2/8: Index of Coincidence - Statistical measure for detecting patterns."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.description = "Index of Coincidence statistical measure"
    
    def execute(self, ciphertext: bytes, **kwargs) -> AttackResult:
        start = time.time()
        
        ioc = calculate_index_of_coincidence(ciphertext)
        
        # IOC for random data ~= 1/256 ≈ 0.0039
        # IOC for natural language ≈ 0.065
        if ioc > 0.045:  # Suggests non-random structure
            success = True
            confidence = min(ioc / 0.065, 1.0)
        else:
            success = False
            confidence = 0.0
        
        return AttackResult(
            attack_name=self.attack_name,
            target_algorithm=self.target_algorithm,
            success=success,
            confidence=confidence,
            time_taken=time.time() - start,
            iterations=len(ciphertext),
            metadata={'ioc': ioc, 'random_ioc': 1/256}
        )


class ChiSquareAttack(StatisticalAttackBase):
    """Attack 3/8: Chi-Square Attack - Distribution testing."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.description = "Chi-square distribution testing"
    
    def execute(self, ciphertext: bytes, **kwargs) -> AttackResult:
        start = time.time()
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in ciphertext:
            byte_counts[byte] += 1
        
        # Calculate chi-square
        expected = len(ciphertext) / 256.0
        chi2 = sum((obs - expected) ** 2 / expected for obs in byte_counts)
        
        # Chi-square for 255 df: significant if > 293
        if chi2 > 350:  # Significant deviation from uniform
            success = True
            confidence = min((chi2 - 255) / 500, 1.0)
        else:
            success = False
            confidence = 0.0
        
        return AttackResult(
            attack_name=self.attack_name,
            target_algorithm=self.target_algorithm,
            success=success,
            confidence=confidence,
            time_taken=time.time() - start,
            iterations=256,
            metadata={'chi_square': chi2, 'expected': 255}
        )


class EntropyAnalysisAttack(StatisticalAttackBase):
    """Attack 4/8: Entropy Analysis - Low entropy detection."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.description = "Low entropy detection attack"
    
    def execute(self, ciphertext: bytes, **kwargs) -> AttackResult:
        start = time.time()
        
        entropy = calculate_entropy_bits(ciphertext)
        
        # Good encryption should have entropy close to 8.0
        if entropy < 7.0:  # Low entropy suggests weakness
            success = True
            confidence = (8.0 - entropy) / 3.0  # Normalize to [0, 1]
        else:
            success = False
            confidence = 0.0
        
        return AttackResult(
            attack_name=self.attack_name,
            target_algorithm=self.target_algorithm,
            success=success,
            confidence=confidence,
            time_taken=time.time() - start,
            iterations=len(ciphertext),
            metadata={'entropy': entropy, 'max_entropy': 8.0}
        )


class PatternRecognitionAttack(StatisticalAttackBase):
    """Attack 5/8: Pattern Recognition - Detects repeated patterns."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.description = "Repeated pattern detection"
    
    def execute(self, ciphertext: bytes, pattern_length: int = 4, **kwargs) -> AttackResult:
        start = time.time()
        
        patterns = detect_patterns(ciphertext, pattern_length)
        
        if patterns and len(patterns) > 0:
            success = True
            confidence = min(len(patterns) / 10, 1.0)
        else:
            success = False
            confidence = 0.0
        
        return AttackResult(
            attack_name=self.attack_name,
            target_algorithm=self.target_algorithm,
            success=success,
            confidence=confidence,
            time_taken=time.time() - start,
            iterations=len(ciphertext),
            metadata={'patterns_found': len(patterns), 'top_pattern': patterns[0] if patterns else None}
        )


class NgramAnalysisAttack(StatisticalAttackBase):
    """Attack 6/8: N-gram Analysis - Multi-byte pattern analysis."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.description = "N-gram statistical analysis"
    
    def execute(self, ciphertext: bytes, n: int = 3, **kwargs) -> AttackResult:
        start = time.time()
        
        # Analyze n-grams
        ngrams = {}
        for i in range(len(ciphertext) - n + 1):
            ngram = ciphertext[i:i+n]
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        
        # Check for non-uniform distribution
        if ngrams:
            max_freq = max(ngrams.values())
            avg_freq = len(ciphertext) / len(ngrams) if ngrams else 0
            
            if max_freq > avg_freq * 2:  # Significant deviation
                success = True
                confidence = min(max_freq / (avg_freq * 5), 1.0)
            else:
                success = False
                confidence = 0.0
        else:
            success = False
            confidence = 0.0
        
        return AttackResult(
            attack_name=self.attack_name,
            target_algorithm=self.target_algorithm,
            success=success,
            confidence=confidence,
            time_taken=time.time() - start,
            iterations=len(ciphertext),
            metadata={'n': n, 'unique_ngrams': len(ngrams)}
        )


class KasiskiExaminationAttack(StatisticalAttackBase):
    """Attack 7/8: Kasiski Examination - Period detection."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.description = "Kasiski examination for period detection"
    
    def execute(self, ciphertext: bytes, **kwargs) -> AttackResult:
        start = time.time()
        
        key_size = estimate_key_size_kasiski(ciphertext)
        
        if key_size:
            success = True
            confidence = 0.7
            recovered = find_repeating_key_xor(ciphertext, key_size)
        else:
            success = False
            confidence = 0.0
            recovered = None
        
        return AttackResult(
            attack_name=self.attack_name,
            target_algorithm=self.target_algorithm,
            success=success,
            confidence=confidence,
            recovered_data=recovered,
            time_taken=time.time() - start,
            iterations=len(ciphertext),
            metadata={'estimated_key_size': key_size}
        )


class AutocorrelationAttack(StatisticalAttackBase):
    """Attack 8/8: Autocorrelation Attack - Self-similarity detection."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.description = "Autocorrelation self-similarity analysis"
    
    def execute(self, ciphertext: bytes, max_lag: int = 256, **kwargs) -> AttackResult:
        start = time.time()
        
        # Simple autocorrelation check
        autocorr_scores = []
        for lag in range(1, min(max_lag, len(ciphertext) // 2)):
            matches = sum(1 for i in range(len(ciphertext) - lag) 
                         if ciphertext[i] == ciphertext[i + lag])
            autocorr_scores.append((lag, matches / (len(ciphertext) - lag)))
        
        # Find significant autocorrelation
        if autocorr_scores:
            max_autocorr = max(autocorr_scores, key=lambda x: x[1])
            if max_autocorr[1] > 0.1:  # >10% correlation
                success = True
                confidence = min(max_autocorr[1] * 5, 1.0)
            else:
                success = False
                confidence = 0.0
        else:
            success = False
            confidence = 0.0
        
        return AttackResult(
            attack_name=self.attack_name,
            target_algorithm=self.target_algorithm,
            success=success,
            confidence=confidence,
            time_taken=time.time() - start,
            iterations=len(autocorr_scores),
            metadata={'max_autocorrelation': max_autocorr if autocorr_scores else None}
        )


__all__ = [
    'FrequencyAnalysisAttack',
    'IndexOfCoincidenceAttack',
    'ChiSquareAttack',
    'EntropyAnalysisAttack',
    'PatternRecognitionAttack',
    'NgramAnalysisAttack',
    'KasiskiExaminationAttack',
    'AutocorrelationAttack',
]

