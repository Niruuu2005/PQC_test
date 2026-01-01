"""
Pipelines Subsystem
===================

Main orchestration layer for cryptographic dataset generation.

This subsystem provides:
- Centralized logging
- Configuration management
- Thread pool execution
- Thread-safe CSV writing
- Record assembly
- Sample processing
- Data processing and statistics
- Main pipeline orchestration

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

# Logger
from .logger import (
    setup_logger,
    get_logger,
    LoggerManager,
)

# Configuration
from .config import (
    Configuration,
    AlgorithmConfig,
    ExecutionConfig,
    OutputConfig,
    load_config,
    validate_config,
    get_default_config,
)

# Execution Controller
try:
    from .executor import ExecutionController
except ImportError:
    ExecutionController = None

# CSV Writer
try:
    from .csv_writer import CSVWriter
except ImportError:
    CSVWriter = None

# Record Assembler
try:
    from .record_assembler import RecordAssembler
except ImportError:
    RecordAssembler = None

# Sample Processor
try:
    from .sample_processor import SampleProcessor
except ImportError:
    SampleProcessor = None

# Data Processor
try:
    from .data_processor import DataProcessor
except ImportError:
    DataProcessor = None

# Main Pipeline
try:
    from .main_pipeline import CryptographicDatasetGenerator, run_full_crypto_dataset
except ImportError:
    CryptographicDatasetGenerator = None
    run_full_crypto_dataset = None

__all__ = [
    # Logger
    'setup_logger',
    'get_logger',
    'LoggerManager',
    # Configuration
    'Configuration',
    'AlgorithmConfig',
    'ExecutionConfig',
    'OutputConfig',
    'load_config',
    'validate_config',
    'get_default_config',
    # Execution
    'ExecutionController',
    # CSV Writing
    'CSVWriter',
    # Record Assembly
    'RecordAssembler',
    # Sample Processing
    'SampleProcessor',
    # Data Processing
    'DataProcessor',
    # Main Pipeline
    'CryptographicDatasetGenerator',
    'run_full_crypto_dataset',
]

