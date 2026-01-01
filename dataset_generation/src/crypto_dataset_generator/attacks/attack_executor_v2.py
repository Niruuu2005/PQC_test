"""
Attack Executor V2 - Multi-Run Execution Engine

Handles execution of attacks with:
- 3 parameter variations per attack
- Resource monitoring (CPU, memory, time)
- Timeout enforcement
- Error handling and recovery

Version: 2.0
Date: December 31, 2025
"""

import time
import psutil
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass

from .base_attack import BaseAttack, AttackResult
from .attack_factory import create_attack, get_available_attacks

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context for attack execution"""
    encryption_row_id: int
    algorithm_name: str
    key_hex: str
    key_size_bits: int
    plaintext_hex: str
    ciphertext_hex: bytes
    plaintext_length: int
    ciphertext_length: int
    encryption_time_ms: float
    original_entropy: float
    original_chi_square: float
    original_avalanche: float


class ResourceMonitor:
    """Monitor system resource usage during attack execution"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None
        self.start_cpu_time = None
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu_time = self.process.cpu_times()
    
    def get_stats(self) -> Dict[str, float]:
        """Get current resource statistics"""
        if self.start_time is None:
            return {'time_ms': 0.0, 'memory_mb': 0.0, 'cpu_percent': 0.0}
        
        elapsed_time = (time.time() - self.start_time) * 1000  # ms
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_used = max(current_memory - self.start_memory, 0.1)
        
        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
        except:
            cpu_percent = 0.0
        
        return {
            'time_ms': elapsed_time,
            'memory_mb': memory_used,
            'cpu_percent': cpu_percent or 1.0,
        }


class AttackExecutorV2:
    """
    Enhanced attack executor with multi-run support and resource monitoring.
    
    Features:
    - Execute attacks with 3 parameter variations
    - Monitor CPU, memory, execution time
    - Enforce timeouts
    - Handle errors gracefully
    - Parallel execution support
    """
    
    def __init__(self, timeout_per_attack: int = 120, max_workers: int = 1):
        """
        Initialize executor.
        
        Args:
            timeout_per_attack: Maximum seconds per attack execution
            max_workers: Number of parallel workers (1 = sequential)
        """
        self.timeout_per_attack = timeout_per_attack
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
    
    def execute_single_attack(
        self,
        attack_name: str,
        context: ExecutionContext,
        params: Dict[str, Any]
    ) -> AttackResult:
        """
        Execute a single attack with specific parameters.
        
        Args:
            attack_name: Name of attack to execute
            context: Encryption context
            params: Attack parameters
        
        Returns:
            AttackResult with resource usage populated
        """
        monitor = ResourceMonitor()
        monitor.start()
        
        try:
            # Create attack instance
            attack = create_attack(attack_name, context.algorithm_name)
            
            # Execute with timeout
            if self.executor:
                future = self.executor.submit(
                    attack.execute_with_params,
                    context.ciphertext_hex,
                    params
                )
                try:
                    result = future.result(timeout=self.timeout_per_attack)
                except FuturesTimeoutError:
                    result = AttackResult(
                        attack_name=attack_name,
                        target_algorithm=context.algorithm_name,
                        success=False,
                        confidence=0.0,
                        parameter_set=params.get('name', 'unknown'),
                        error_message=f"Timeout after {self.timeout_per_attack}s",
                    )
            else:
                result = attack.execute_with_params(context.ciphertext_hex, params)
            
            # Update resource usage
            stats = monitor.get_stats()
            result.time_taken = stats['time_ms'] / 1000.0  # Convert to seconds
            result.memory_used_mb = stats['memory_mb']
            result.cpu_usage_percent = stats['cpu_percent']
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing {attack_name}: {e}")
            stats = monitor.get_stats()
            return AttackResult(
                attack_name=attack_name,
                target_algorithm=context.algorithm_name,
                success=False,
                confidence=0.0,
                time_taken=stats['time_ms'] / 1000.0,
                memory_used_mb=stats['memory_mb'],
                cpu_usage_percent=stats['cpu_percent'],
                parameter_set=params.get('name', 'unknown'),
                error_message=str(e),
            )
    
    def execute_attack_all_variations(
        self,
        attack_name: str,
        context: ExecutionContext
    ) -> List[AttackResult]:
        """
        Execute attack with all 3 parameter variations.
        
        Args:
            attack_name: Name of attack to execute
            context: Encryption context
        
        Returns:
            List of 3 AttackResults (baseline, aggressive, stress)
        """
        try:
            attack = create_attack(attack_name, context.algorithm_name)
            variations = attack.get_parameter_variations()
        except Exception as e:
            logger.error(f"Error creating attack {attack_name}: {e}")
            # Return 3 failed results
            return [
                AttackResult(
                    attack_name=attack_name,
                    target_algorithm=context.algorithm_name,
                    success=False,
                    confidence=0.0,
                    parameter_set=param_set,
                    error_message=str(e),
                )
                for param_set in ['baseline', 'aggressive', 'stress']
            ]
        
        results = []
        for params in variations:
            result = self.execute_single_attack(attack_name, context, params)
            results.append(result)
        
        return results
    
    def execute_all_attacks(
        self,
        context: ExecutionContext,
        attack_names: Optional[List[str]] = None
    ) -> List[AttackResult]:
        """
        Execute all attacks (or specified subset) with all variations.
        
        Args:
            context: Encryption context
            attack_names: List of attack names to execute (None = all)
        
        Returns:
            List of AttackResults (3 per attack)
        """
        if attack_names is None:
            attack_names = get_available_attacks()
        
        all_results = []
        total_attacks = len(attack_names)
        
        for idx, attack_name in enumerate(attack_names, 1):
            logger.info(f"Executing {attack_name} ({idx}/{total_attacks})...")
            
            results = self.execute_attack_all_variations(attack_name, context)
            all_results.extend(results)
            
            # Log progress
            success_count = sum(1 for r in results if r.success)
            logger.debug(f"{attack_name}: {success_count}/3 runs successful")
        
        return all_results
    
    def execute_encryption_row(
        self,
        context: ExecutionContext
    ) -> List[AttackResult]:
        """
        Execute all 90 attacks × 3 variations = 270 executions for one encryption.
        
        Args:
            context: Encryption context
        
        Returns:
            List of 270 AttackResults
        """
        logger.info(
            f"Processing encryption row {context.encryption_row_id}: "
            f"{context.algorithm_name}"
        )
        
        start_time = time.time()
        results = self.execute_all_attacks(context)
        elapsed = time.time() - start_time
        
        logger.info(
            f"Completed {context.encryption_row_id} in {elapsed:.2f}s: "
            f"{len(results)} attack executions"
        )
        
        return results
    
    def shutdown(self):
        """Shutdown executor and cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)


# Utility functions

def create_execution_context_from_csv_row(row: Dict[str, Any]) -> ExecutionContext:
    """
    Create ExecutionContext from crypto_dataset.csv row.
    
    Args:
        row: Dictionary representing a CSV row
    
    Returns:
        ExecutionContext instance
    """
    return ExecutionContext(
        encryption_row_id=int(row['row_id']),
        algorithm_name=row['algorithm_name'],
        key_hex=row['key_hex'],
        key_size_bits=int(row['key_size_bits']),
        plaintext_hex=row['plaintext_hex'],
        ciphertext_hex=bytes.fromhex(row['ciphertext_hex']),
        plaintext_length=int(row['plaintext_length']),
        ciphertext_length=int(row['ciphertext_length']),
        encryption_time_ms=float(row['encryption_time_ms']),
        original_entropy=float(row['shannon_entropy']),
        original_chi_square=float(row['chi_square_statistic']),
        original_avalanche=float(row['avalanche_effect']),
    )


def results_to_csv_rows(
    results: List[AttackResult],
    context: ExecutionContext,
    base_execution_id: int = 1
) -> List[Dict[str, Any]]:
    """
    Convert AttackResults to CSV rows with full schema.
    
    Args:
        results: List of AttackResults
        context: Encryption context
        base_execution_id: Starting execution ID
    
    Returns:
        List of dictionaries ready for CSV export
    """
    csv_rows = []
    
    for idx, result in enumerate(results):
        # Determine run number from parameter_set
        run_map = {'baseline': 1, 'aggressive': 2, 'stress': 3}
        run_number = run_map.get(result.parameter_set, 1)
        
        # Build base row
        row = {
            'attack_execution_id': base_execution_id + idx,
            'encryption_row_id': context.encryption_row_id,
            'algorithm_name': context.algorithm_name,
            'attack_id': '',  # Will be filled by lookup
            'attack_name': result.attack_name,
            'attack_category': '',  # Will be filled by lookup
            'run_number': run_number,
            'timestamp': datetime.now().isoformat(),
            
            # Encryption context
            'key_hex': context.key_hex,
            'key_size_bits': context.key_size_bits,
            'plaintext_hex': context.plaintext_hex,
            'ciphertext_hex': context.ciphertext_hex.hex(),
            'plaintext_length': context.plaintext_length,
            'ciphertext_length': context.ciphertext_length,
            'encryption_time_ms': context.encryption_time_ms,
            'original_entropy': context.original_entropy,
            'original_chi_square': context.original_chi_square,
            'original_avalanche': context.original_avalanche,
            
            # Attack execution
            'attack_language': result.attack_language,
            'attack_implementation': 'standard',
            'parameter_set': result.parameter_set,
            'timeout_ms': 120000,  # Default timeout
            'execution_time_ms': result.time_taken * 1000,
            'memory_used_mb': result.memory_used_mb,
            'cpu_usage_percent': result.cpu_usage_percent,
            'iterations_performed': result.iterations,
            'attack_success': result.success,
            'confidence_score': result.confidence,
            'recovered_data_hex': result.recovered_data.hex() if result.recovered_data else '',
            'error_message': result.error_message,
        }
        
        # Add metrics (up to 10 pairs)
        metric_items = list(result.metrics.items()) if result.metrics else []
        for i in range(10):
            if i < len(metric_items):
                name, value = metric_items[i]
                row[f'metric_{i+1}_name'] = name
                row[f'metric_{i+1}_value'] = value
            else:
                row[f'metric_{i+1}_name'] = ''
                row[f'metric_{i+1}_value'] = 0.0
        
        # Analysis results
        row['vulnerability_detected'] = result.vulnerability_detected
        row['vulnerability_type'] = result.vulnerability_type
        row['severity_score'] = result.severity_score
        row['recommendation'] = result.recommendation
        row['notes'] = result.error_message if result.error_message else ''
        
        csv_rows.append(row)
    
    return csv_rows


__all__ = [
    'AttackExecutorV2',
    'ExecutionContext',
    'ResourceMonitor',
    'create_execution_context_from_csv_row',
    'results_to_csv_rows',
]

