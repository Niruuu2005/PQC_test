"""
Utils Subsystem
===============

Infrastructure and utility functions for the cryptographic dataset generator.

This subsystem provides:
- Custom exception classes
- Helper functions (formatting, file operations)
- Input validation
- Data formatting
- Timeout management
- Memory monitoring
- Performance tracking

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

# Custom Exceptions
from .errors import (
    CipherError,
    AttackError,
    MetricsError,
    PipelineError,
    ConfigurationError,
    ValidationError,
)

# Helper Functions
from .helpers import (
    ensure_directory,
    format_size,
    format_duration,
    merge_dictionaries,
    safe_divide,
    slugify,
    generate_uuid,
)

# Validators
from .validators import (
    validate_algorithm_name,
    validate_plaintext,
    validate_output_path,
    validate_configuration,
)

# Formatters
from .formatters import (
    format_bytes,
    format_timestamp,
    format_percentage,
    format_list,
)

# Timeout Management
from .timeout import (
    TimeoutManager,
    timeout_decorator,
    run_with_timeout,
)

# Memory Management
from .memory import (
    MemoryMonitor,
    get_memory_usage,
    get_memory_usage_percent,
    check_memory_limit,
    get_peak_memory_usage,
    trigger_garbage_collection,
)

# Performance Tracking
from .performance import (
    PerformanceTracker,
    measure_time,
    measure_memory,
    profile_function,
)

# Constants (import module, not individual constants)
from . import constants

__all__ = [
    # Exceptions
    'CipherError',
    'AttackError',
    'MetricsError',
    'PipelineError',
    'ConfigurationError',
    'ValidationError',
    # Helpers
    'ensure_directory',
    'format_size',
    'format_duration',
    'merge_dictionaries',
    'safe_divide',
    'slugify',
    'generate_uuid',
    # Validators
    'validate_algorithm_name',
    'validate_plaintext',
    'validate_output_path',
    'validate_configuration',
    # Formatters
    'format_bytes',
    'format_timestamp',
    'format_percentage',
    'format_list',
    # Timeout
    'TimeoutManager',
    'timeout_decorator',
    'run_with_timeout',
    # Memory
    'MemoryMonitor',
    'get_memory_usage',
    'get_memory_usage_percent',
    'check_memory_limit',
    'get_peak_memory_usage',
    'trigger_garbage_collection',
    # Performance
    'PerformanceTracker',
    'measure_time',
    'measure_memory',
    'profile_function',
    # Constants module
    'constants',
]

