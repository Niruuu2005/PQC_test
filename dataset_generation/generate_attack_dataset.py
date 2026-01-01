#!/usr/bin/env python
"""
Attack Dataset Generation - Main Script

Generates comprehensive cryptanalysis attack dataset by executing 90 attacks
on 510 encryptions with 3 parameter variations each.

Usage:
    python generate_attack_dataset.py --input crypto_dataset.csv --output attack_dataset.csv

Version: 1.0
Date: December 31, 2025
"""

import argparse
import logging
import sys
from pathlib import Path

from src.crypto_dataset_generator.orchestrator.attack_orchestrator import (
    AttackOrchestrator,
    OrchestrationConfig,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate cryptanalysis attack dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full dataset (sequential)
  python generate_attack_dataset.py

  # Generate with 4 parallel workers
  python generate_attack_dataset.py --workers 4

  # Resume from checkpoint
  python generate_attack_dataset.py --resume

  # Custom input/output
  python generate_attack_dataset.py --input my_crypto.csv --output my_attacks.csv

Output Files:
  attack_dataset.csv          - Main dataset (~110 MB, 137,700 rows)
  attack_dataset.checkpoint.json - Checkpoint for resume
  attack_dataset.summary.json    - Statistics and summary

Dataset Structure:
  510 encryptions × 90 attacks × 3 runs = 137,700 attack executions
  Estimated time: 4-24 hours depending on hardware
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='crypto_dataset.csv',
        help='Input CSV file with encryptions (default: crypto_dataset.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='attack_dataset.csv',
        help='Output CSV file for attack results (default: attack_dataset.csv)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1, sequential)'
    )
    
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10,
        help='Save checkpoint every N encryptions (default: 10)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Timeout per attack in seconds (default: 120)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )
    
    parser.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        help='Start fresh, ignore checkpoint'
    )
    
    parser.set_defaults(resume=True)
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without executing'
    )
    
    return parser.parse_args()


def validate_configuration(config: OrchestrationConfig) -> bool:
    """
    Validate configuration before execution.
    
    Args:
        config: Orchestration configuration
    
    Returns:
        True if valid, False otherwise
    """
    errors = []
    
    # Check input file exists
    input_path = Path(config.input_csv)
    if not input_path.exists():
        errors.append(f"Input file not found: {config.input_csv}")
    
    # Check output directory is writable
    output_path = Path(config.output_csv)
    output_dir = output_path.parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")
    
    # Validate workers
    if config.workers < 1:
        errors.append("Workers must be >= 1")
    
    # Validate timeout
    if config.timeout_per_attack < 1:
        errors.append("Timeout must be >= 1 second")
    
    # Print errors
    if errors:
        logging.error("Configuration validation failed:")
        for error in errors:
            logging.error(f"  - {error}")
        return False
    
    return True


def print_configuration(config: OrchestrationConfig):
    """Print configuration summary"""
    print("=" * 80)
    print("ATTACK DATASET GENERATION CONFIGURATION")
    print("=" * 80)
    print(f"Input File:         {config.input_csv}")
    print(f"Output File:        {config.output_csv}")
    print(f"Workers:            {config.workers}")
    print(f"Checkpoint Every:   {config.checkpoint_interval} rows")
    print(f"Attack Timeout:     {config.timeout_per_attack}s")
    print(f"Resume from Checkpoint: {config.resume_from_checkpoint}")
    print(f"Log Level:          {config.log_level}")
    print("=" * 80)
    print()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Create configuration
    config = OrchestrationConfig(
        input_csv=args.input,
        output_csv=args.output,
        workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        timeout_per_attack=args.timeout,
        resume_from_checkpoint=args.resume,
        log_level=args.log_level,
    )
    
    # Print configuration
    print_configuration(config)
    
    # Validate
    if not validate_configuration(config):
        sys.exit(1)
    
    # Dry run
    if args.dry_run:
        print("Dry run mode: Configuration valid, exiting.")
        sys.exit(0)
    
    try:
        # Create orchestrator
        orchestrator = AttackOrchestrator(config)
        
        # Execute
        orchestrator.execute_all()
        
        print()
        print("=" * 80)
        print("SUCCESS: Attack dataset generation completed!")
        print(f"Output: {config.output_csv}")
        print(f"Summary: {config.output_csv.replace('.csv', '.summary.json')}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved to checkpoint.")
        print("Run with --resume to continue from checkpoint.")
        sys.exit(130)
    
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        print("See logs for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()

