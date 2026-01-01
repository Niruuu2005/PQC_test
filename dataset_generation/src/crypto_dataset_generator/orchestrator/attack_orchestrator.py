"""
Attack Orchestrator - Main coordination engine

Orchestrates execution of 90 attacks × 3 variations on 510 encryptions
= 137,700 attack executions with comprehensive monitoring.

Version: 1.0
Date: December 31, 2025
"""

import csv
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..attacks.attack_executor_v2 import (
    AttackExecutorV2,
    ExecutionContext,
    create_execution_context_from_csv_row,
    results_to_csv_rows,
)
from .language_bridge import LanguageBridge
from .result_aggregator import ResultAggregator

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    """Configuration for attack orchestration"""
    input_csv: str
    output_csv: str
    workers: int = 1
    checkpoint_interval: int = 10
    timeout_per_attack: int = 120
    resume_from_checkpoint: bool = True
    log_level: str = 'INFO'


class AttackOrchestrator:
    """
    Main orchestration engine for cryptanalysis dataset generation.
    
    Responsibilities:
    - Load encryption data from crypto_dataset.csv
    - Coordinate execution of 90 attacks × 3 variations per encryption
    - Manage checkpoints for resumable execution
    - Aggregate results into attack_dataset.csv
    - Generate summary statistics
    
    Features:
    - Parallel execution with process pool
    - Checkpoint/resume capability
    - Progress tracking with ETA
    - Comprehensive error handling
    - Multi-language bridge (Python/C++/Rust)
    """
    
    def __init__(self, config: OrchestrationConfig):
        """
        Initialize orchestrator.
        
        Args:
            config: Orchestration configuration
        """
        self.config = config
        self.input_path = Path(config.input_csv)
        self.output_path = Path(config.output_csv)
        self.checkpoint_path = self.output_path.with_suffix('.checkpoint.json')
        
        self.language_bridge = LanguageBridge()
        self.result_aggregator = ResultAggregator()
        self.processed_row_ids = set()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if it exists"""
        if self.config.resume_from_checkpoint and self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                self.processed_row_ids = set(checkpoint.get('processed_row_ids', []))
                logger.info(f"Resumed from checkpoint: {len(self.processed_row_ids)} rows already processed")
                return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return {}
    
    def save_checkpoint(self):
        """Save current progress to checkpoint"""
        checkpoint = {
            'processed_row_ids': list(self.processed_row_ids),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'total_processed': len(self.processed_row_ids),
        }
        try:
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.debug(f"Checkpoint saved: {len(self.processed_row_ids)} rows")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_encryptions(self) -> List[Dict[str, Any]]:
        """
        Load encryption records from input CSV.
        
        Returns:
            List of encryption record dictionaries
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        encryptions = []
        with open(self.input_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_id = int(row['row_id'])
                if row_id not in self.processed_row_ids:
                    encryptions.append(row)
        
        logger.info(f"Loaded {len(encryptions)} encryption records (skipped {len(self.processed_row_ids)} already processed)")
        return encryptions
    
    def execute_encryption_row(self, row: Dict[str, Any], executor: AttackExecutorV2) -> List[Dict[str, Any]]:
        """
        Execute all attacks on a single encryption row.
        
        Args:
            row: Encryption record from CSV
            executor: Attack executor instance
        
        Returns:
            List of CSV rows (attack results)
        """
        row_id = int(row['row_id'])
        logger.info(f"Processing encryption row {row_id}: {row['algorithm_name']}")
        
        try:
            # Create execution context
            context = create_execution_context_from_csv_row(row)
            
            # Execute all attacks (90 attacks × 3 variations = 270 executions)
            results = executor.execute_encryption_row(context)
            
            # Convert to CSV rows
            base_execution_id = row_id * 1000  # Space IDs: row 1 = 1000-1269, row 2 = 2000-2269, etc.
            csv_rows = results_to_csv_rows(results, context, base_execution_id)
            
            # Mark as processed
            self.processed_row_ids.add(row_id)
            
            # Aggregate statistics
            self.result_aggregator.add_results(results)
            
            logger.info(f"Completed row {row_id}: {len(csv_rows)} attack executions")
            return csv_rows
            
        except Exception as e:
            logger.error(f"Error processing row {row_id}: {e}", exc_info=True)
            return []
    
    def execute_all(self):
        """
        Execute all attacks on all encryption rows.
        
        Main entry point for dataset generation.
        """
        logger.info("=" * 80)
        logger.info("ATTACK DATASET GENERATION STARTED")
        logger.info("=" * 80)
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Workers: {self.config.workers}")
        logger.info(f"Timeout per attack: {self.config.timeout_per_attack}s")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Load checkpoint
        self.load_checkpoint()
        
        # Load encryptions
        encryptions = self.load_encryptions()
        total_encryptions = len(encryptions)
        
        if total_encryptions == 0:
            logger.info("No encryptions to process (all done or input empty)")
            return
        
        # Estimate time
        estimated_seconds = total_encryptions * 90 * 3 * 0.5  # 0.5s per attack average
        estimated_hours = estimated_seconds / 3600
        logger.info(f"Estimated time: {estimated_hours:.2f} hours ({estimated_seconds:.0f}s)")
        
        # Initialize CSV output
        csv_written = self._initialize_output_csv()
        
        # Create executor
        executor = AttackExecutorV2(
            timeout_per_attack=self.config.timeout_per_attack,
            max_workers=1,  # Per-row executor is sequential
        )
        
        # Process encryptions
        if self.config.workers > 1:
            # Parallel execution across encryption rows
            self._execute_parallel(encryptions, executor, csv_written)
        else:
            # Sequential execution
            self._execute_sequential(encryptions, executor, csv_written)
        
        # Finalize
        executor.shutdown()
        
        # Generate summary
        elapsed = time.time() - start_time
        self._generate_summary(elapsed)
        
        # Cleanup checkpoint
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        
        logger.info("=" * 80)
        logger.info("ATTACK DATASET GENERATION COMPLETED")
        logger.info(f"Total time: {elapsed:.2f}s ({elapsed/3600:.2f}h)")
        logger.info(f"Output saved to: {self.output_path}")
        logger.info("=" * 80)
    
    def _initialize_output_csv(self) -> bool:
        """Initialize output CSV with header"""
        csv_exists = self.output_path.exists()
        
        if not csv_exists or not self.config.resume_from_checkpoint:
            # Write header
            header = [
                'attack_execution_id', 'encryption_row_id', 'algorithm_name',
                'attack_id', 'attack_name', 'attack_category', 'run_number', 'timestamp',
                'key_hex', 'key_size_bits', 'plaintext_hex', 'ciphertext_hex',
                'plaintext_length', 'ciphertext_length', 'encryption_time_ms',
                'original_entropy', 'original_chi_square', 'original_avalanche',
                'attack_language', 'attack_implementation', 'parameter_set',
                'timeout_ms', 'execution_time_ms', 'memory_used_mb', 'cpu_usage_percent',
                'iterations_performed', 'attack_success', 'confidence_score',
                'recovered_data_hex', 'error_message',
            ]
            
            # Add metric columns
            for i in range(1, 11):
                header.extend([f'metric_{i}_name', f'metric_{i}_value'])
            
            # Add analysis columns
            header.extend([
                'vulnerability_detected', 'vulnerability_type', 'severity_score',
                'recommendation', 'notes'
            ])
            
            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
            
            return False
        
        return True
    
    def _execute_sequential(self, encryptions: List[Dict[str, Any]], executor: AttackExecutorV2, csv_written: bool):
        """Sequential execution of encryptions"""
        total = len(encryptions)
        
        with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
            # Get header from first write or existing file
            if csv_written:
                # Read header from existing file
                with open(self.output_path, 'r', newline='', encoding='utf-8') as rf:
                    reader = csv.DictReader(rf)
                    fieldnames = reader.fieldnames
            else:
                # Will be set after first row
                fieldnames = None
            
            writer = None
            
            for idx, encryption_row in enumerate(encryptions, 1):
                # Execute attacks
                csv_rows = self.execute_encryption_row(encryption_row, executor)
                
                # Write results
                if csv_rows:
                    if writer is None:
                        fieldnames = list(csv_rows[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    for csv_row in csv_rows:
                        writer.writerow(csv_row)
                    f.flush()
                
                # Checkpoint
                if idx % self.config.checkpoint_interval == 0:
                    self.save_checkpoint()
                    logger.info(f"Progress: {idx}/{total} ({100*idx/total:.1f}%)")
    
    def _execute_parallel(self, encryptions: List[Dict[str, Any]], executor: AttackExecutorV2, csv_written: bool):
        """Parallel execution of encryptions (not yet implemented for simplicity)"""
        logger.warning("Parallel execution not yet fully implemented, falling back to sequential")
        self._execute_sequential(encryptions, executor, csv_written)
    
    def _generate_summary(self, elapsed_time: float):
        """Generate summary statistics"""
        summary_path = self.output_path.with_suffix('.summary.json')
        
        stats = self.result_aggregator.get_statistics()
        stats['total_time_seconds'] = elapsed_time
        stats['total_time_hours'] = elapsed_time / 3600
        stats['encryptions_processed'] = len(self.processed_row_ids)
        
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_path}")
        logger.info(f"Total attack executions: {stats.get('total_attacks', 0)}")
        logger.info(f"Successful attacks: {stats.get('successful_attacks', 0)}")
        logger.info(f"Success rate: {stats.get('success_rate_percent', 0):.2f}%")


__all__ = ['AttackOrchestrator', 'OrchestrationConfig']

