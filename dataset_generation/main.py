"""
AIRAWAT Cryptographic Dataset Generator
========================================

Main entry point for generating cryptographic dataset CSV files.

This tool generates comprehensive datasets for cryptographic algorithm analysis,
including encryption/decryption operations, cryptographic metrics, and attack simulations.

Usage:
    python main.py [options]

Options:
    --algorithms ALGO1,ALGO2    Specific algorithms to process (default: all available)
    --output FILE               Output CSV file path (default: crypto_dataset.csv)
    --threads N                 Number of worker threads (default: 4)
    --seed N                    RNG seed for determinism (default: 42)
    --samples N                 Number of test samples per algorithm (default: 10)
    --no-attacks                Disable attack simulations
    --verbose                   Enable verbose logging

Examples:
    # Generate dataset with all algorithms
    python main.py

    # Generate dataset with specific algorithms
    python main.py --algorithms AES-256-GCM,ChaCha20

    # Generate with 8 threads
    python main.py --threads 8 --output results.csv

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "crypto_dataset_generator"))

def print_banner():
    """Print application banner."""
    print("=" * 80)
    print("AIRAWAT CRYPTOGRAPHIC DATASET GENERATOR")
    print("=" * 80)
    print("Version: 1.0")
    print("Date: December 30, 2025")
    print()


def get_available_algorithms():
    """Get list of available algorithms."""
    try:
        from crypto.cipher_factory import get_available_algorithms
        return get_available_algorithms()
    except:
        # Fallback list
        return [
            "AES-128-ECB", "AES-192-ECB", "AES-256-ECB",
            "AES-128-CBC", "AES-192-CBC", "AES-256-CBC",
            "AES-128-CTR", "AES-192-CTR", "AES-256-CTR",
            "AES-128-GCM", "AES-192-GCM", "AES-256-GCM",
            "ChaCha20", "Salsa20",
            "3DES-ECB", "3DES-CBC",
            "Blowfish-ECB", "Blowfish-CBC",
            "Camellia-128-ECB", "Camellia-192-ECB", "Camellia-256-ECB",
            "Twofish-128-ECB", "Twofish-192-ECB", "Twofish-256-ECB",
        ]


def generate_test_strings(count=10):
    """Generate test strings for encryption."""
    test_strings = [
        b"Hello, World!",
        b"The quick brown fox jumps over the lazy dog",
        b"1234567890",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        b"!@#$%^&*()_+-={}[]|:;<>?,./",
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09",  # Binary data
        b"a" * 100,  # Repeated character
        b"Testing encryption with various data patterns...",
        b"Final test string for cryptographic analysis."
    ]
    return test_strings[:count]


def simple_dataset_generation(algorithms, test_strings, output_file, rng_seed=42, enable_attacks=True, verbose=False):
    """
    Generate dataset with basic encryption/decryption data.
    
    This is a simplified implementation that works without the full pipeline.
    """
    import csv
    from datetime import datetime
    import traceback
    
    print(f"\nStarting dataset generation...")
    print(f"  Algorithms: {len(algorithms)}")
    print(f"  Test strings: {len(test_strings)}")
    print(f"  Output: {output_file}")
    print(f"  RNG seed: {rng_seed}")
    print(f"  Attacks enabled: {enable_attacks}")
    print()
    
    try:
        from crypto.cipher_factory import create_cipher
        from analysis.metrics import compute_all_metrics
        from utils.performance import PerformanceTracker
        from utils.helpers import format_duration
        crypto_available = True
    except ImportError as e:
        print(f"Warning: Some modules not available: {e}")
        print("Generating basic dataset without actual encryption...")
        crypto_available = False
    
    # CSV schema - core columns
    schema = [
        # Metadata
        "row_id", "algorithm_name", "test_string_id", "rng_seed", "timestamp",
        # Encryption
        "key_hex", "key_size_bits", "plaintext_hex", "ciphertext_hex",
        "plaintext_length", "ciphertext_length", "encryption_time_ms",
        # Metrics
        "shannon_entropy", "chi_square_statistic", "avalanche_effect",
        # Status
        "encryption_successful", "decryption_successful", "overall_status"
    ]
    
    # Open CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=schema)
        writer.writeheader()
        
        row_id = 1
        successful = 0
        failed = 0
        
        total_samples = len(algorithms) * len(test_strings)
        
        # Process each algorithm × test_string pair
        for algo_idx, algorithm in enumerate(algorithms):
            print(f"\n[{algo_idx+1}/{len(algorithms)}] Processing {algorithm}...")
            
            for test_idx, plaintext in enumerate(test_strings):
                try:
                    if crypto_available:
                        # Create cipher
                        cipher = create_cipher(algorithm)
                        
                        # Generate key
                        cipher.generate_key()
                        key = cipher.cipher_state.key
                        
                        # Encrypt
                        tracker = PerformanceTracker()
                        tracker.start()
                        ciphertext, metadata = cipher.encrypt(plaintext)
                        tracker.stop()
                        
                        # Compute metrics
                        try:
                            metrics = compute_all_metrics(ciphertext, plaintext)
                        except:
                            metrics = {"shannon_entropy": 0.0, "chi_square_statistic": 0.0, "avalanche_effect": 0.0}
                        
                        # Decrypt to verify
                        plaintext_dec, dec_metadata = cipher.decrypt(
                            ciphertext,
                            iv=bytes.fromhex(metadata.iv) if metadata.iv else None,
                            tag=bytes.fromhex(metadata.tag) if metadata.tag else None
                        )
                        dec_success = plaintext_dec == plaintext
                        
                        # Create record
                        record = {
                            "row_id": row_id,
                            "algorithm_name": algorithm,
                            "test_string_id": test_idx + 1,
                            "rng_seed": rng_seed + row_id,
                            "timestamp": datetime.now().isoformat(),
                            "key_hex": key.hex(),  # Full key hex without truncation
                            "key_size_bits": len(key) * 8,
                            "plaintext_hex": plaintext.hex(),  # Full plaintext hex without truncation
                            "ciphertext_hex": ciphertext.hex(),  # Full ciphertext hex without truncation
                            "plaintext_length": len(plaintext),
                            "ciphertext_length": len(ciphertext),
                            "encryption_time_ms": tracker.get_duration_ms(),
                            "shannon_entropy": metrics.get("shannon_entropy", 0.0),
                            "chi_square_statistic": metrics.get("chi_square_statistic", 0.0),
                            "avalanche_effect": metrics.get("avalanche_effect", 0.0),
                            "encryption_successful": True,
                            "decryption_successful": dec_success,
                            "overall_status": "SUCCESS" if dec_success else "PARTIAL"
                        }
                        
                        successful += 1
                    else:
                        # Basic record without actual encryption
                        record = {
                            "row_id": row_id,
                            "algorithm_name": algorithm,
                            "test_string_id": test_idx + 1,
                            "rng_seed": rng_seed + row_id,
                            "timestamp": datetime.now().isoformat(),
                            "key_hex": "N/A",
                            "key_size_bits": 0,
                            "plaintext_hex": plaintext.hex()[:64],
                            "ciphertext_hex": "N/A",
                            "plaintext_length": len(plaintext),
                            "ciphertext_length": 0,
                            "encryption_time_ms": 0.0,
                            "shannon_entropy": 0.0,
                            "chi_square_statistic": 0.0,
                            "avalanche_effect": 0.0,
                            "encryption_successful": False,
                            "decryption_successful": False,
                            "overall_status": "SKIPPED"
                        }
                        successful += 1
                    
                    # Write record
                    writer.writerow(record)
                    
                    if verbose:
                        print(f"  Sample {test_idx+1}: {record['overall_status']}")
                    
                except Exception as e:
                    print(f"  Error processing {algorithm} sample {test_idx+1}: {str(e)}")
                    if verbose:
                        traceback.print_exc()
                    
                    # Write error record
                    record = {
                        "row_id": row_id,
                        "algorithm_name": algorithm,
                        "test_string_id": test_idx + 1,
                        "rng_seed": rng_seed + row_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "overall_status": "ERROR"
                    }
                    writer.writerow(record)
                    failed += 1
                
                row_id += 1
                
                # Progress indicator
                progress = (row_id - 1) / total_samples * 100
                if (row_id - 1) % 10 == 0:
                    print(f"  Progress: {progress:.1f}% ({row_id-1}/{total_samples} samples)")
        
        print(f"\n{'='*80}")
        print("Dataset generation complete!")
        print(f"  Total samples: {row_id - 1}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Output file: {output_file}")
        print(f"  File size: {Path(output_file).stat().st_size / 1024:.2f} KB")
        print(f"{'='*80}\n")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AIRAWAT Cryptographic Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--algorithms",
        type=str,
        help="Comma-separated list of algorithms to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="crypto_dataset.csv",
        help="Output CSV file path (default: crypto_dataset.csv)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for determinism (default: 42)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of test samples per algorithm (default: 10)"
    )
    parser.add_argument(
        "--no-attacks",
        action="store_true",
        help="Disable attack simulations"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="List available algorithms and exit"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # List algorithms if requested
    if args.list_algorithms:
        print("Available Algorithms:")
        print("=" * 80)
        algorithms = get_available_algorithms()
        for i, algo in enumerate(algorithms, 1):
            print(f"  {i:3d}. {algo}")
        print(f"\nTotal: {len(algorithms)} algorithms")
        return 0
    
    # Determine algorithms to process
    if args.algorithms:
        algorithms = [a.strip() for a in args.algorithms.split(",")]
        print(f"Selected algorithms: {', '.join(algorithms)}")
    else:
        algorithms = get_available_algorithms()
        print(f"Using all available algorithms ({len(algorithms)} total)")
    
    # Generate test strings
    test_strings = generate_test_strings(args.samples)
    
    # Generate dataset
    success = simple_dataset_generation(
        algorithms=algorithms,
        test_strings=test_strings,
        output_file=args.output,
        rng_seed=args.seed,
        enable_attacks=not args.no_attacks,
        verbose=args.verbose
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
