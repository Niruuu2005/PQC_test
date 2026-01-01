"""
Base Attack Classes

This module defines the abstract base class for all cryptanalytic attacks.

Version: 1.0
Date: December 30, 2025
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time


@dataclass
class AttackResult:
    """
    Result of a cryptanalytic attack execution.
    
    Enhanced for comprehensive dataset generation with multi-run support.
    
    Attributes:
        attack_name: Name of the attack
        target_algorithm: Algorithm that was attacked
        success: Whether the attack succeeded
        confidence: Confidence level of the result [0.0, 1.0]
        recovered_data: Data recovered by the attack (if any)
        time_taken: Time taken to execute attack (seconds)
        iterations: Number of iterations performed
        memory_used_mb: Peak memory usage in MB
        cpu_usage_percent: Average CPU utilization (0-100)
        attack_language: Implementation language (Python/C++/Rust)
        parameter_set: Which parameter variation (baseline/aggressive/stress)
        metadata: Additional attack-specific information
        metrics: Structured attack-specific metrics (up to 15 key-value pairs)
        vulnerability_detected: Whether a vulnerability was found
        vulnerability_type: Type of vulnerability if detected
        severity_score: Severity rating (0-10)
        recommendation: Remediation advice
        error_message: Error message if attack failed
    """
    attack_name: str
    target_algorithm: str
    success: bool
    confidence: float  # 0.0 to 1.0
    recovered_data: Optional[bytes] = None
    time_taken: float = 0.0
    iterations: int = 0
    memory_used_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    attack_language: str = "Python"
    parameter_set: str = "baseline"
    metadata: Dict[str, Any] = None
    metrics: Dict[str, float] = None
    vulnerability_detected: bool = False
    vulnerability_type: str = ""
    severity_score: float = 0.0
    recommendation: str = ""
    error_message: str = ""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.metrics is None:
            self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            'attack_name': self.attack_name,
            'target_algorithm': self.target_algorithm,
            'success': self.success,
            'confidence': self.confidence,
            'recovered_data_hex': self.recovered_data.hex() if self.recovered_data else '',
            'time_taken': self.time_taken,
            'iterations': self.iterations,
            'memory_used_mb': self.memory_used_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'attack_language': self.attack_language,
            'parameter_set': self.parameter_set,
            'vulnerability_detected': self.vulnerability_detected,
            'vulnerability_type': self.vulnerability_type,
            'severity_score': self.severity_score,
            'recommendation': self.recommendation,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'metrics': self.metrics,
        }
    
    def to_csv_row(self, attack_id: str = "") -> Dict[str, Any]:
        """Convert to flat CSV row with metric columns."""
        row = {
            'attack_id': attack_id,
            'attack_name': self.attack_name,
            'target_algorithm': self.target_algorithm,
            'attack_language': self.attack_language,
            'parameter_set': self.parameter_set,
            'execution_time_ms': self.time_taken * 1000,  # Convert to ms
            'memory_used_mb': self.memory_used_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'iterations_performed': self.iterations,
            'attack_success': self.success,
            'confidence_score': self.confidence,
            'recovered_data_hex': self.recovered_data.hex() if self.recovered_data else '',
            'error_message': self.error_message,
            'vulnerability_detected': self.vulnerability_detected,
            'vulnerability_type': self.vulnerability_type,
            'severity_score': self.severity_score,
            'recommendation': self.recommendation,
        }
        
        # Add up to 10 metric pairs
        metric_items = list(self.metrics.items())
        for i in range(10):
            if i < len(metric_items):
                name, value = metric_items[i]
                row[f'metric_{i+1}_name'] = name
                row[f'metric_{i+1}_value'] = value
            else:
                row[f'metric_{i+1}_name'] = ''
                row[f'metric_{i+1}_value'] = 0.0
        
        return row


class BaseAttack(ABC):
    """
    Abstract base class for all cryptanalytic attacks.
    
    All attack implementations must inherit from this class and implement
    the execute() method.
    """
    
    def __init__(self, target_algorithm: str):
        """
        Initialize attack.
        
        Args:
            target_algorithm: Name of the algorithm to attack
        """
        self.target_algorithm = target_algorithm
        self.attack_name = self.__class__.__name__
        self.category = "unknown"
        self.description = ""
        self.complexity = "unknown"  # theoretical, practical, etc.
    
    @abstractmethod
    def execute(self, ciphertext: bytes, **kwargs) -> AttackResult:
        """
        Execute the attack.
        
        Args:
            ciphertext: Ciphertext to attack
            **kwargs: Additional attack-specific parameters
        
        Returns:
            AttackResult with attack outcome
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    def get_parameter_variations(self) -> List[Dict[str, Any]]:
        """
        Get 3 parameter variations for this attack.
        
        Returns:
            List of 3 parameter dictionaries: [baseline, aggressive, stress]
        
        Examples:
            >>> attack = ExhaustiveKeySearch("AES-128")
            >>> params = attack.get_parameter_variations()
            >>> len(params)
            3
            >>> params[0]['name']
            'baseline'
        """
        # Default parameter variations - subclasses should override
        return [
            {
                'name': 'baseline',
                'timeout_ms': 1000,
                'max_iterations': 10000,
                'sample_size': 256,
            },
            {
                'name': 'aggressive',
                'timeout_ms': 10000,
                'max_iterations': 100000,
                'sample_size': 1024,
            },
            {
                'name': 'stress',
                'timeout_ms': 60000,
                'max_iterations': 1000000,
                'sample_size': 4096,
            },
        ]
    
    def execute_with_params(self, ciphertext: bytes, params: Dict[str, Any]) -> AttackResult:
        """
        Execute attack with specific parameter set.
        
        Args:
            ciphertext: Ciphertext to attack
            params: Parameter dictionary from get_parameter_variations()
        
        Returns:
            AttackResult with parameter_set field populated
        
        Examples:
            >>> attack = FrequencyAnalysisAttack("AES-128")
            >>> params = attack.get_parameter_variations()[0]  # baseline
            >>> result = attack.execute_with_params(ciphertext, params)
            >>> result.parameter_set
            'baseline'
        """
        import psutil
        import os
        
        # Track resource usage
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu_time = process.cpu_percent(interval=None)
        
        # Execute attack with parameters
        param_name = params.get('name', 'baseline')
        kwargs = {k: v for k, v in params.items() if k != 'name'}
        
        result = self.execute(ciphertext, **kwargs)
        
        # Update result with parameter info and resource usage
        result.parameter_set = param_name
        
        # Calculate resource usage
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        result.memory_used_mb = max(end_memory - start_memory, 0.1)  # Minimum 0.1 MB
        result.cpu_usage_percent = process.cpu_percent(interval=None) or 1.0
        
        return result
    
    def execute_all_variations(self, ciphertext: bytes) -> List[AttackResult]:
        """
        Execute attack with all 3 parameter variations.
        
        Args:
            ciphertext: Ciphertext to attack
        
        Returns:
            List of 3 AttackResults (baseline, aggressive, stress)
        
        Examples:
            >>> attack = ChiSquareAttack("AES-256-CBC")
            >>> results = attack.execute_all_variations(ciphertext)
            >>> len(results)
            3
            >>> results[0].parameter_set
            'baseline'
            >>> results[1].parameter_set
            'aggressive'
            >>> results[2].parameter_set
            'stress'
        """
        variations = self.get_parameter_variations()
        results = []
        
        for params in variations:
            try:
                result = self.execute_with_params(ciphertext, params)
                results.append(result)
            except Exception as e:
                # Create failed result
                result = AttackResult(
                    attack_name=self.attack_name,
                    target_algorithm=self.target_algorithm,
                    success=False,
                    confidence=0.0,
                    parameter_set=params.get('name', 'unknown'),
                    error_message=str(e),
                )
                results.append(result)
        
        return results
    
    def is_applicable(self, algorithm: str) -> bool:
        """
        Check if this attack is applicable to the given algorithm.
        
        Args:
            algorithm: Algorithm name to check
        
        Returns:
            True if attack can be applied, False otherwise
        
        Examples:
            >>> attack = SomeAttack("AES-256-GCM")
            >>> attack.is_applicable("AES-256-GCM")
            True
        """
        # Default: attack is applicable to target algorithm
        return algorithm == self.target_algorithm or self.target_algorithm == "ANY"
    
    def estimate_complexity(self, key_size: int) -> float:
        """
        Estimate computational complexity of the attack.
        
        Args:
            key_size: Key size in bits
        
        Returns:
            Estimated number of operations (log scale)
        
        Examples:
            >>> attack = BruteForceAttack("AES-128")
            >>> attack.estimate_complexity(128)
            128.0  # 2^128 operations
        """
        # Default: exponential in key size
        return float(key_size)
    
    def estimate_success_probability(self, **kwargs) -> float:
        """
        Estimate probability of attack success.
        
        Args:
            **kwargs: Attack-specific parameters
        
        Returns:
            Probability of success [0.0, 1.0]
        
        Examples:
            >>> attack = StatisticalAttack("Weak-Cipher")
            >>> attack.estimate_success_probability(sample_size=1000)
            0.85
        """
        # Default: low success probability for strong crypto
        return 0.1
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.attack_name} on {self.target_algorithm}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<{self.__class__.__name__}(target={self.target_algorithm}, category={self.category})>"


class BruteForceAttackBase(BaseAttack):
    """Base class for brute-force attacks."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.category = "brute_force"
    
    def estimate_complexity(self, key_size: int) -> float:
        """Brute force has exponential complexity."""
        return float(key_size)  # 2^key_size operations


class StatisticalAttackBase(BaseAttack):
    """Base class for statistical attacks."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.category = "statistical"
    
    def estimate_complexity(self, key_size: int) -> float:
        """Statistical attacks typically polynomial."""
        return key_size ** 2  # O(n^2) operations


class CryptanalysisAttackBase(BaseAttack):
    """Base class for cryptanalysis attacks (linear, differential)."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.category = "cryptanalysis"
    
    def estimate_complexity(self, key_size: int) -> float:
        """Cryptanalysis attacks are sub-exponential."""
        return key_size * 2 ** (key_size / 2)  # Birthday attack complexity


class AlgebraicAttackBase(BaseAttack):
    """Base class for algebraic attacks."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.category = "algebraic"


class SideChannelAttackBase(BaseAttack):
    """Base class for side-channel attacks."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.category = "side_channel"
    
    def estimate_success_probability(self, **kwargs) -> float:
        """Side-channel attacks can have high success if implementation is weak."""
        # Depends on implementation, not algorithm strength
        return kwargs.get('implementation_quality', 0.5)


class QuantumAttackBase(BaseAttack):
    """Base class for quantum attacks (simulated)."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.category = "quantum"
        self.complexity = "theoretical"
    
    def estimate_complexity(self, key_size: int) -> float:
        """Quantum attacks (Grover) have square root speedup."""
        return key_size / 2  # 2^(key_size/2) operations


class LatticeAttackBase(BaseAttack):
    """Base class for lattice-based attacks."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.category = "lattice"


class HashCollisionAttackBase(BaseAttack):
    """Base class for hash collision attacks."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.category = "hash_collision"
    
    def estimate_complexity(self, key_size: int) -> float:
        """Birthday attack has square root complexity."""
        return key_size / 2  # 2^(n/2) for n-bit hash


class ImplementationFlawAttackBase(BaseAttack):
    """Base class for implementation flaw attacks."""
    
    def __init__(self, target_algorithm: str):
        super().__init__(target_algorithm)
        self.category = "implementation_flaw"
    
    def estimate_success_probability(self, **kwargs) -> float:
        """Success depends on implementation flaws."""
        return kwargs.get('has_flaw', False) * 0.9 + 0.05


# Export all classes
__all__ = [
    'AttackResult',
    'BaseAttack',
    'BruteForceAttackBase',
    'StatisticalAttackBase',
    'CryptanalysisAttackBase',
    'AlgebraicAttackBase',
    'SideChannelAttackBase',
    'QuantumAttackBase',
    'LatticeAttackBase',
    'HashCollisionAttackBase',
    'ImplementationFlawAttackBase',
]

