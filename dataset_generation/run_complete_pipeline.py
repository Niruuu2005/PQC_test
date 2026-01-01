#!/usr/bin/env python
"""
AIRAWAT - Complete Dataset Generation Pipeline
Orchestrates all three phases with detailed logging and error handling.

Phase 1: Generate Cryptographic Dataset
Phase 2: Generate Attack Dataset  
Phase 3: Generate Cryptanalysis Summary

Usage:
    python run_complete_pipeline.py [--samples N] [--workers N]
"""

import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Setup detailed logging
def setup_logging():
    """Configure comprehensive logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"complete_pipeline_{timestamp}.log"
    
    # Configure logging with UTF-8 encoding for file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()


def print_banner(phase_name):
    """Print phase banner."""
    print("\n" + "="*80)
    print(f"  {phase_name}")
    print("="*80 + "\n")


def retry_on_error(func, max_retries=3, retry_delay=5):
    """
    Retry a function on failure.
    
    Args:
        func: Function to execute
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        Result of successful function execution
    
    Raises:
        Exception if all retries fail
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries}")
            result = func()
            logger.info(f"[OK] Success on attempt {attempt}")
            return result
        except Exception as e:
            logger.error(f"[FAIL] Attempt {attempt} failed: {e}")
            
            if attempt < max_retries:
                logger.info(f"[RETRY] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"[ERROR] All {max_retries} attempts failed")
                raise
    
    raise RuntimeError("Retry logic failed")


def phase_1_crypto_dataset(samples=10):
    """
    Phase 1: Generate Cryptographic Dataset
    
    Creates encryption samples for all available algorithms.
    """
    print_banner("PHASE 1: Cryptographic Dataset Generation")
    
    logger.info(f"Starting crypto dataset generation with {samples} samples per algorithm")
    
    def generate():
        import subprocess
        
        # Setup environment with oqs.dll PATH
        env = os.environ.copy()
        oqs_bin_path = r"C:\Users\npati\_oqs\bin"
        if oqs_bin_path not in env.get('PATH', ''):
            env['PATH'] = oqs_bin_path + os.pathsep + env.get('PATH', '')
        
        cmd = [
            sys.executable,
            "main.py",
            "--samples", str(samples),
            "--output", "crypto_dataset.csv"
        ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
        
        logger.info("Crypto dataset generation output:")
        for line in result.stdout.split('\n'):
            if line.strip():
                logger.info(f"  {line}")
        
        # Verify output
        if not Path("crypto_dataset.csv").exists():
            raise FileNotFoundError("crypto_dataset.csv not generated")
        
        file_size = Path("crypto_dataset.csv").stat().st_size
        logger.info(f"[OK] Generated crypto_dataset.csv ({file_size:,} bytes)")
        
        return True
    
    return retry_on_error(generate, max_retries=3)


def phase_2_attack_dataset(workers=1):
    """
    Phase 2: Generate Attack Dataset with Enhanced Logging
    
    Runs all attacks on all encryptions with detailed progress tracking.
    """
    print_banner("PHASE 2: Attack Dataset Generation")
    
    logger.info(f"Starting attack dataset generation with {workers} workers")
    logger.info("This will execute 90 attacks × 510 encryptions × 3 runs = 137,700 executions")
    logger.info("Expected duration: 3-6 hours (depends on system)")
    
    def generate():
        import subprocess
        
        # Setup environment with oqs.dll PATH
        env = os.environ.copy()
        oqs_bin_path = r"C:\Users\npati\_oqs\bin"
        if oqs_bin_path not in env.get('PATH', ''):
            env['PATH'] = oqs_bin_path + os.pathsep + env.get('PATH', '')
        
        cmd = [
            sys.executable,
            "generate_attack_dataset.py",
            "--workers", str(workers),
            "--resume"  # Enable checkpoint resume
        ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        logger.info("[INFO] Detailed progress will be logged below...")
        
        # Run with real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Stream output in real-time
        encryption_count = 0
        attack_count = 0
        
        for line in process.stdout:
            line = line.strip()
            if line:
                logger.info(f"  {line}")
                
                # Track progress
                if "Encryption" in line and "90" in line:
                    encryption_count += 1
                    if encryption_count % 10 == 0:
                        logger.info(f"[MILESTONE] {encryption_count} encryptions completed")
                
                if "attacks for encryption" in line.lower():
                    attack_count += 90
                    logger.info(f"[OK] Batch completed: Total {attack_count:,} attacks executed")
        
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"Attack generation failed with code {process.returncode}")
        
        # Verify output
        if not Path("attack_dataset.csv").exists():
            raise FileNotFoundError("attack_dataset.csv not generated")
        
        file_size = Path("attack_dataset.csv").stat().st_size
        logger.info(f"[OK] Generated attack_dataset.csv ({file_size:,} bytes)")
        
        return True
    
    return retry_on_error(generate, max_retries=3, retry_delay=10)


def phase_3_cryptanalysis_summary():
    """
    Phase 3: Generate Cryptanalysis Summary
    
    Creates final summary CSV with security recommendations.
    """
    print_banner("PHASE 3: Cryptanalysis Summary Generation")
    
    logger.info("Generating cryptanalysis summary from attack results")
    
    def generate():
        import pandas as pd
        import numpy as np
        
        logger.info("Loading attack dataset...")
        df_attacks = pd.read_csv("attack_dataset.csv")
        logger.info(f"  Loaded {len(df_attacks):,} attack records")
        
        logger.info("Loading crypto dataset...")
        df_crypto = pd.read_csv("crypto_dataset.csv")
        logger.info(f"  Loaded {len(df_crypto):,} encryption records")
        
        # Analyze by algorithm
        logger.info("Analyzing attack success rates by algorithm...")
        
        summary_data = []
        
        for algo in df_crypto['algorithm_name'].unique():
            logger.info(f"  Analyzing {algo}...")
            
            # Get attacks for this algorithm
            algo_attacks = df_attacks[df_attacks['algorithm_name'] == algo]
            
            if len(algo_attacks) == 0:
                logger.warning(f"    No attacks found for {algo}")
                continue
            
            total_attacks = len(algo_attacks)
            successful_attacks = algo_attacks['attack_success'].sum()
            success_rate = (successful_attacks / total_attacks * 100) if total_attacks > 0 else 0
            
            # Get crypto metrics
            algo_crypto = df_crypto[df_crypto['algorithm_name'] == algo].iloc[0]
            
            # Determine recommendation
            if success_rate >= 80:
                recommendation = "AVOID"
            elif success_rate >= 40:
                recommendation = "PHASE_OUT"
            elif success_rate >= 10:
                recommendation = "CAUTION"
            else:
                recommendation = "RECOMMENDED"
            
            summary_data.append({
                'Algorithm': algo,
                'Key_Size_Bits': algo_crypto.get('key_size_bits', 0),
                'Block_Size_Bits': algo_crypto.get('block_size', 0),
                'Total_Attacks_Tested': total_attacks,
                'Successful_Attacks': successful_attacks,
                'Success_Rate_Percent': round(success_rate, 2),
                'Avg_Encryption_Time_ms': algo_crypto.get('encryption_time_ms', 0),
                'Recommendation': recommendation
            })
            
            logger.info(f"    Attacks: {total_attacks}, Success: {successful_attacks} ({success_rate:.1f}%) → {recommendation}")
        
        # Create summary DataFrame
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Success_Rate_Percent', ascending=False)
        
        # Save
        output_file = "Cryptographic_Algorithm_Summary.csv"
        df_summary.to_csv(output_file, index=False)
        logger.info(f"[OK] Generated {output_file} ({len(df_summary)} algorithms)")
        
        # Print summary statistics
        logger.info("\n[SUMMARY] Summary Statistics:")
        logger.info(f"  RECOMMENDED: {len(df_summary[df_summary['Recommendation'] == 'RECOMMENDED'])} algorithms")
        logger.info(f"  CAUTION: {len(df_summary[df_summary['Recommendation'] == 'CAUTION'])} algorithms")
        logger.info(f"  PHASE_OUT: {len(df_summary[df_summary['Recommendation'] == 'PHASE_OUT'])} algorithms")
        logger.info(f"  AVOID: {len(df_summary[df_summary['Recommendation'] == 'AVOID'])} algorithms")
        
        return True
    
    return retry_on_error(generate, max_retries=3)


def main():
    """Main pipeline orchestrator."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("  AIRAWAT - COMPLETE DATASET GENERATION PIPELINE")
    print("="*80)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    logger.info("Starting complete pipeline execution")
    
    try:
        # Phase 1: Crypto Dataset
        logger.info("\n" + "="*60)
        phase_1_crypto_dataset(samples=10)
        logger.info("[OK] Phase 1 Complete\n")
        
        # Phase 2: Attack Dataset
        logger.info("\n" + "="*60)
        phase_2_attack_dataset(workers=1)
        logger.info("[OK] Phase 2 Complete\n")
        
        # Phase 3: Cryptanalysis Summary
        logger.info("\n" + "="*60)
        phase_3_cryptanalysis_summary()
        logger.info("[OK] Phase 3 Complete\n")
        
        # Final summary
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        print("\n" + "="*80)
        print("  [SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"  Duration: {hours}h {minutes}m {seconds}s")
        print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print("\nGenerated Files:")
        print("  [OK] crypto_dataset.csv")
        print("  [OK] attack_dataset.csv")
        print("  [OK] Cryptographic_Algorithm_Summary.csv")
        print("="*80 + "\n")
        
        logger.info(f"[SUCCESS] Complete pipeline finished in {hours}h {minutes}m {seconds}s")
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        
        print("\n" + "="*80)
        print("  [FAILED] PIPELINE FAILED")
        print("="*80)
        print(f"  Error: {e}")
        print("  Check logs for details")
        print("="*80 + "\n")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
