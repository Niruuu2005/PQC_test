"""
Global Constants
================

Global constants and configuration values for the utils subsystem.

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

# ============================================================================
# DEFAULT VALUES
# ============================================================================

# Random Number Generator
DEFAULT_SEED = 42
DEFAULT_RNG_ALGORITHM = "MT19937"  # Mersenne Twister

# Threading
DEFAULT_THREAD_POOL_SIZE = 4
MAX_THREAD_POOL_SIZE = 32
MIN_THREAD_POOL_SIZE = 1

# Timeouts (seconds)
DEFAULT_TIMEOUT_SECONDS = 30
ENCRYPTION_TIMEOUT_SECONDS = 30
DECRYPTION_TIMEOUT_SECONDS = 30
ATTACK_TIMEOUT_SECONDS = 5
GLOBAL_TIMEOUT_SECONDS = 3600 * 12  # 12 hours

# Memory Management (MB)
DEFAULT_MAX_MEMORY_MB = 8192  # 8 GB
MAX_MEMORY_MB = 8192
MIN_MEMORY_MB = 256

# ============================================================================
# SCHEMA CONFIGURATION
# ============================================================================

# CSV Schema
CSV_SCHEMA_VERSION = "2.0.0"
CSV_COLUMN_COUNT = 127  # Extended from 121
CSV_DELIMITER = ","
CSV_QUOTE_CHAR = '"'
CSV_ENCODING = "utf-8"

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# Test Strings
TEST_STRING_COUNT = 10
DEFAULT_TEST_STRING_ENCODING = "utf-8"

# Algorithm Targets
ALGORITHM_COUNT_TARGET = 180
ALGORITHM_COUNT_MINIMUM = 100

# Attack Targets
ATTACK_COUNT_TARGET = 90
ATTACK_COUNT_MINIMUM = 50

# ============================================================================
# DATA LIMITS
# ============================================================================

# Plaintext/Ciphertext
MAX_PLAINTEXT_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_CIPHERTEXT_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_PLAINTEXT_BYTES = 1
DEFAULT_PLAINTEXT_BLOCK_SIZE = 16  # bytes

# Key Sizes (bits)
MAX_KEY_SIZE_BITS = 33552  # NTRU highest
MIN_KEY_SIZE_BITS = 40  # Minimum secure (historically)
DEFAULT_SYMMETRIC_KEY_SIZE = 256  # AES-256
DEFAULT_ASYMMETRIC_KEY_SIZE = 2048  # RSA-2048

# ============================================================================
# FILE PATHS
# ============================================================================

# Output Directories
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_CSV_DIR = "output/csv"
DEFAULT_JSON_DIR = "output/json"
DEFAULT_LOG_DIR = "logs"

# File Names
DEFAULT_CSV_FILENAME = "crypto_dataset.csv"
DEFAULT_JSON_FILENAME = "crypto_dataset.json"
DEFAULT_LOG_FILENAME = "crypto_dataset.log"
DEFAULT_ERROR_LOG_FILENAME = "errors.log"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log Levels
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"

DEFAULT_LOG_LEVEL = LOG_LEVEL_INFO

# Log Format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

# ============================================================================
# PERFORMANCE THRESHOLDS
# ============================================================================

# Execution Time Thresholds (milliseconds)
FAST_OPERATION_MS = 10
NORMAL_OPERATION_MS = 100
SLOW_OPERATION_MS = 1000
VERY_SLOW_OPERATION_MS = 5000

# Memory Thresholds (MB)
LOW_MEMORY_MB = 100
MEDIUM_MEMORY_MB = 500
HIGH_MEMORY_MB = 1000
CRITICAL_MEMORY_MB = 2000

# Throughput Targets
TARGET_SAMPLES_PER_SECOND = 10
TARGET_ENCRYPTIONS_PER_SECOND = 100
TARGET_ATTACKS_PER_SECOND = 1000

# ============================================================================
# FORMATTING CONSTANTS
# ============================================================================

# Display Formatting
MAX_DISPLAY_BYTES = 64  # Max bytes to show in hex
MAX_DISPLAY_LIST_ITEMS = 10  # Max list items to show
DEFAULT_DECIMAL_PRECISION = 2

# String Formatting
MAX_STRING_LENGTH = 100
TRUNCATION_SUFFIX = "..."

# ============================================================================
# ALGORITHM CATEGORIES
# ============================================================================

SYMMETRIC_BLOCK_CIPHERS = [
    "AES", "DES", "3DES", "Blowfish", "Twofish", "Camellia",
    "CAST5", "CAST128", "IDEA", "RC2", "RC5", "RC6",
    "SEED", "GOST", "SM4"
]

SYMMETRIC_STREAM_CIPHERS = [
    "ChaCha20", "Salsa20", "RC4", "Trivium", "Grain",
    "Rabbit", "HC-256", "A5/1", "A5/2"
]

ASYMMETRIC_CIPHERS = [
    "RSA", "ElGamal", "ECC"
]

POST_QUANTUM_CIPHERS = [
    "Kyber", "NTRU", "Saber", "FrodoKEM", "Classic-McEliece",
    "HQC", "BIKE"
]

# ============================================================================
# ATTACK CATEGORIES
# ============================================================================

ATTACK_CATEGORIES = [
    "statistical",
    "linear_cryptanalysis",
    "differential_cryptanalysis",
    "algebraic",
    "brute_force",
    "side_channel",
    "lattice",
    "hash_collision",
    "implementation_flaw",
    "quantum"
]

# ============================================================================
# ERROR CODES
# ============================================================================

ERROR_CODE_SUCCESS = 0
ERROR_CODE_GENERAL = 1
ERROR_CODE_CIPHER_ERROR = 10
ERROR_CODE_ATTACK_ERROR = 20
ERROR_CODE_METRICS_ERROR = 30
ERROR_CODE_PIPELINE_ERROR = 40
ERROR_CODE_CONFIG_ERROR = 50
ERROR_CODE_VALIDATION_ERROR = 60
ERROR_CODE_TIMEOUT = 70
ERROR_CODE_MEMORY_LIMIT = 80

# ============================================================================
# SECURITY LEVELS
# ============================================================================

SECURITY_LEVEL_BROKEN = "broken"
SECURITY_LEVEL_WEAK = "weak"
SECURITY_LEVEL_MODERATE = "moderate"
SECURITY_LEVEL_STRONG = "strong"
SECURITY_LEVEL_VERY_STRONG = "very_strong"
SECURITY_LEVEL_UNKNOWN = "unknown"

# Security Thresholds (bits of security)
SECURITY_BITS_BROKEN = 0
SECURITY_BITS_WEAK = 64
SECURITY_BITS_MODERATE = 80
SECURITY_BITS_STRONG = 128
SECURITY_BITS_VERY_STRONG = 192

# ============================================================================
# METRIC RANGES
# ============================================================================

# Entropy (bits per byte)
ENTROPY_MIN = 0.0
ENTROPY_MAX = 8.0
ENTROPY_IDEAL = 8.0

# Chi-Square (depends on degrees of freedom)
CHI_SQUARE_UNIFORM_THRESHOLD = 293.0  # For 255 degrees of freedom, 95% confidence

# Avalanche Effect (percentage)
AVALANCHE_MIN = 0.0
AVALANCHE_MAX = 100.0
AVALANCHE_IDEAL = 50.0
AVALANCHE_GOOD_THRESHOLD = 45.0

# Correlation Coefficient
CORRELATION_MIN = -1.0
CORRELATION_MAX = 1.0
CORRELATION_IDEAL = 0.0
CORRELATION_WEAK_THRESHOLD = 0.3

# ============================================================================
# MISCELLANEOUS
# ============================================================================

# Progress Reporting
PROGRESS_REPORT_INTERVAL = 10  # Report every N samples
PROGRESS_BAR_WIDTH = 50

# Batch Processing
DEFAULT_BATCH_SIZE = 100
MAX_BATCH_SIZE = 1000

# Retry Configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 1

# Cache Configuration
ENABLE_CACHING = True
MAX_CACHE_SIZE = 1000

# Version Information
UTILS_VERSION = "1.0.0"
UTILS_VERSION_DATE = "2025-12-30"

# ============================================================================
# ENVIRONMENT VARIABLES (optional overrides)
# ============================================================================

ENV_VAR_THREAD_POOL_SIZE = "CRYPTO_THREAD_POOL_SIZE"
ENV_VAR_TIMEOUT_SECONDS = "CRYPTO_TIMEOUT_SECONDS"
ENV_VAR_MAX_MEMORY_MB = "CRYPTO_MAX_MEMORY_MB"
ENV_VAR_LOG_LEVEL = "CRYPTO_LOG_LEVEL"
ENV_VAR_OUTPUT_DIR = "CRYPTO_OUTPUT_DIR"

