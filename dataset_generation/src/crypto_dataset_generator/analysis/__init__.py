"""
Cryptographic Dataset Generator - Analysis Subsystem

This module handles all cryptographic analysis and metrics computation.

Version: 1.0
Date: December 30, 2025
"""

__version__ = "1.0.0"

# Import all submodules
from . import metrics
from . import entropy
from . import statistical
from . import randomness
from . import avalanche
from . import correlation
from . import aggregator
from . import reporter
from . import validator

# Import commonly used functions
from .metrics import (
    shannon_entropy,
    chi_square_statistic,
    frequency_bias,
    randomness_score,
    hamming_distance_normalized,
    compute_all_metrics,
    validate_metrics,
)

from .entropy import (
    calculate_shannon_entropy,
    calculate_min_entropy,
    calculate_renyi_entropy,
    estimate_source_entropy,
)

from .avalanche import (
    calculate_avalanche_effect,
    test_strict_avalanche_criterion,
    analyze_avalanche_detailed,
)

from .correlation import (
    autocorrelation,
    cross_correlation,
    analyze_correlation_detailed,
)

from .statistical import (
    chi_square_test,
    ks_test,
    entropy_test,
)

from .randomness import (
    runs_test,
    monobit_test,
    test_randomness,
)

from .aggregator import (
    aggregate_sample_metrics,
    aggregate_algorithm_results,
    compute_percentiles,
)

from .reporter import (
    generate_json_summary,
    generate_text_report,
    generate_markdown_report,
)

from .validator import (
    validate_ciphertext,
    validate_metrics,
    validate_csv_record,
)

__all__ = [
    # Submodules
    'metrics',
    'entropy',
    'statistical',
    'randomness',
    'avalanche',
    'correlation',
    'aggregator',
    'reporter',
    'validator',
    # Core functions
    'shannon_entropy',
    'chi_square_statistic',
    'frequency_bias',
    'randomness_score',
    'hamming_distance_normalized',
    'compute_all_metrics',
    'validate_metrics',
    'calculate_shannon_entropy',
    'calculate_min_entropy',
    'calculate_renyi_entropy',
    'estimate_source_entropy',
    'calculate_avalanche_effect',
    'test_strict_avalanche_criterion',
    'analyze_avalanche_detailed',
    'autocorrelation',
    'cross_correlation',
    'analyze_correlation_detailed',
    'chi_square_test',
    'ks_test',
    'entropy_test',
    'runs_test',
    'monobit_test',
    'test_randomness',
    'aggregate_sample_metrics',
    'aggregate_algorithm_results',
    'compute_percentiles',
    'generate_json_summary',
    'generate_text_report',
    'generate_markdown_report',
    'validate_ciphertext',
    'validate_csv_record',
]

