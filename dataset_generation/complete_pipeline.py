#!/usr/bin/env python
"""
Complete Pipeline Automation
Executes Phases 3-5 automatically after Phase 2 completes.

Usage:
    python complete_pipeline.py

This script will:
1. Wait for attack_dataset.csv to reach 137,700 rows
2. Run Phase 3: Post-generation validation
3. Run Phase 4: Statistical analysis
4. Run Phase 5: Generate final reports

Version: 1.0
Date: December 31, 2025
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime


def count_csv_rows(filepath):
    """Count rows in CSV file (excluding header)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1
    except:
        return 0


def wait_for_completion():
    """Wait for attack dataset generation to complete"""
    target_rows = 137700
    attack_dataset = Path('attack_dataset.csv')
    
    print("=" * 80)
    print("WAITING FOR ATTACK DATASET GENERATION TO COMPLETE")
    print("=" * 80)
    print(f"Target: {target_rows:,} rows")
    print()
    
    while True:
        if not attack_dataset.exists():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for attack_dataset.csv to be created...")
            time.sleep(60)
            continue
        
        current_rows = count_csv_rows(attack_dataset)
        progress_pct = (current_rows / target_rows) * 100
        
        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
              f"Progress: {current_rows:,}/{target_rows:,} ({progress_pct:.1f}%)", 
              end='', flush=True)
        
        if current_rows >= target_rows:
            print(f"\n\n[COMPLETE] Attack dataset generation finished!")
            print(f"Final row count: {current_rows:,}")
            break
        
        time.sleep(60)  # Check every minute
    
    # Wait a bit more to ensure file is fully written
    print("\nWaiting 30 seconds to ensure file is fully written...")
    time.sleep(30)


def run_phase_3():
    """Run Phase 3: Post-generation validation"""
    print("\n" + "=" * 80)
    print("PHASE 3: POST-GENERATION VALIDATION")
    print("=" * 80)
    print()
    
    cmd = [
        sys.executable,
        'validate_attack_dataset.py',
        '--input', 'attack_dataset.csv',
        '--crypto-dataset', 'crypto_dataset.csv',
        '--output', 'validation_report_post.json',
        '--quality-report', 'data_quality_report.json'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n[SUCCESS] Phase 3 completed successfully")
        return True
    else:
        print(f"\n[ERROR] Phase 3 failed with exit code {result.returncode}")
        return False


def run_phase_4():
    """Run Phase 4: Statistical analysis"""
    print("\n" + "=" * 80)
    print("PHASE 4: STATISTICAL ANALYSIS")
    print("=" * 80)
    print()
    
    cmd = [
        sys.executable,
        'analyze_attacks.py',
        '--input', 'attack_dataset.csv',
        '--output-dir', 'analysis_results',
        '--visualize'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n[SUCCESS] Phase 4 completed successfully")
        return True
    else:
        print(f"\n[ERROR] Phase 4 failed with exit code {result.returncode}")
        return False


def run_phase_5():
    """Run Phase 5: Generate final reports"""
    print("\n" + "=" * 80)
    print("PHASE 5: FINAL REPORTS")
    print("=" * 80)
    print()
    
    # Update IMPLEMENTATION_PROGRESS.md with final metrics
    print("Updating IMPLEMENTATION_PROGRESS.md...")
    update_progress_document()
    
    # Generate FINAL_RESULTS_SUMMARY.md from template
    print("Generating FINAL_RESULTS_SUMMARY.md...")
    generate_final_summary()
    
    # Create data files inventory
    print("Creating data files inventory...")
    create_inventory()
    
    print("\n[SUCCESS] Phase 5 completed successfully")
    return True


def update_progress_document():
    """Update IMPLEMENTATION_PROGRESS.md with final metrics"""
    # Read current progress document
    progress_file = Path('IMPLEMENTATION_PROGRESS.md')
    if not progress_file.exists():
        print("[WARN] IMPLEMENTATION_PROGRESS.md not found")
        return
    
    # Add completion timestamp and final metrics
    with open(progress_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n---\n\n")
        f.write(f"## Final Completion\n\n")
        f.write(f"**Completion Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Status:** ALL PHASES COMPLETE\n\n")
        
        # Add file sizes
        files_to_check = [
            'attack_dataset.csv',
            'validation_report_post.json',
            'data_quality_report.json',
            'analysis_results/vulnerability_report.json',
            'analysis_results/algorithm_rankings.csv',
            'analysis_results/security_scores.csv'
        ]
        
        f.write("### Generated Files:\n\n")
        for filepath in files_to_check:
            path = Path(filepath)
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                f.write(f"- `{filepath}`: {size_mb:.2f} MB\n")
    
    print("[OK] IMPLEMENTATION_PROGRESS.md updated")


def generate_final_summary():
    """Generate FINAL_RESULTS_SUMMARY.md from template and analysis results"""
    template_file = Path('FINAL_RESULTS_SUMMARY_TEMPLATE.md')
    output_file = Path('FINAL_RESULTS_SUMMARY.md')
    
    if not template_file.exists():
        print("[WARN] Template not found, skipping")
        return
    
    # Read template
    with open(template_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Load analysis results
    try:
        # Load attack statistics
        stats_file = Path('analysis_results/attack_statistics.json')
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            # Replace placeholders
            content = content.replace('[TO BE FILLED]', str(stats.get('total_attacks', 'N/A')))
            content = content.replace('[ROWS]', f"{stats.get('total_attacks', 0):,}")
            content = content.replace('[ALGORITHMS]', str(stats.get('total_algorithms', 51)))
            content = content.replace('[ATTACKS]', str(stats.get('unique_attacks', 90)))
    except Exception as e:
        print(f"[WARN] Could not load analysis results: {e}")
    
    # Add completion date
    content = content.replace('[DATE]', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Write final summary
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] {output_file} generated")


def create_inventory():
    """Create inventory of all generated files"""
    inventory_file = Path('DATA_FILES_INVENTORY.md')
    
    files_to_inventory = [
        'crypto_dataset.csv',
        'attack_dataset.csv',
        'attack_dataset.summary.json',
        'validation_report_pre.json',
        'validation_report_post.json',
        'data_quality_report.json',
        'analysis_results/vulnerability_report.json',
        'analysis_results/algorithm_rankings.csv',
        'analysis_results/security_scores.csv',
        'analysis_results/attack_statistics.json',
        'analysis_results/correlation_matrix.csv',
        'analysis_results/analysis_summary.html',
        'IMPLEMENTATION_PROGRESS.md',
        'FINAL_RESULTS_SUMMARY.md'
    ]
    
    with open(inventory_file, 'w', encoding='utf-8') as f:
        f.write("# AIRAWAT Data Files Inventory\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Files\n\n")
        f.write("| File | Size (MB) | Rows | Status |\n")
        f.write("|------|-----------|------|--------|\n")
        
        total_size = 0
        for filepath in files_to_inventory:
            path = Path(filepath)
            if path.exists():
                size_bytes = path.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                total_size += size_mb
                
                # Count rows for CSV files
                rows = "N/A"
                if filepath.endswith('.csv'):
                    rows = f"{count_csv_rows(filepath):,}"
                
                f.write(f"| `{filepath}` | {size_mb:.2f} | {rows} | ✅ |\n")
            else:
                f.write(f"| `{filepath}` | - | - | ❌ Missing |\n")
        
        f.write(f"\n**Total Size:** {total_size:.2f} MB\n")
    
    print(f"[OK] {inventory_file} generated")


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("AIRAWAT COMPLETE PIPELINE AUTOMATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    try:
        # Wait for Phase 2 to complete
        wait_for_completion()
        
        # Run Phase 3
        if not run_phase_3():
            print("\n[ERROR] Phase 3 failed, stopping pipeline")
            sys.exit(1)
        
        # Run Phase 4
        if not run_phase_4():
            print("\n[ERROR] Phase 4 failed, stopping pipeline")
            sys.exit(1)
        
        # Run Phase 5
        if not run_phase_5():
            print("\n[ERROR] Phase 5 failed, stopping pipeline")
            sys.exit(1)
        
        # Success!
        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print("ALL PHASES COMPLETE!")
        print("=" * 80)
        print(f"Total time: {elapsed/3600:.2f} hours")
        print()
        print("Generated files:")
        print("  - attack_dataset.csv")
        print("  - validation_report_post.json")
        print("  - data_quality_report.json")
        print("  - analysis_results/ (6 files)")
        print("  - IMPLEMENTATION_PROGRESS.md (updated)")
        print("  - FINAL_RESULTS_SUMMARY.md")
        print("  - DATA_FILES_INVENTORY.md")
        print()
        print("Next steps:")
        print("  1. Review FINAL_RESULTS_SUMMARY.md for key findings")
        print("  2. Open analysis_results/analysis_summary.html for interactive dashboard")
        print("  3. Check algorithm_rankings.csv for security scores")
        print("=" * 80)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

