"""
Configuration Management
========================

Pipeline configuration and settings management.

This module provides:
- Configuration dataclasses
- Configuration loading (YAML/JSON)
- Configuration validation
- Default configurations
- Configuration merging

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Import utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.validators import validate_configuration, validate_algorithm_name
from utils.errors import ConfigurationError
from utils import constants


@dataclass
class AlgorithmConfig:
    """
    Algorithm selection configuration.
    
    Attributes:
        algorithm_list: Specific list of algorithms (None = all available)
        filter_by_type: Filter by type (symmetric, asymmetric, pqc, etc.)
        exclude_algorithms: Algorithms to exclude
    """
    algorithm_list: Optional[List[str]] = None
    filter_by_type: Optional[List[str]] = None
    exclude_algorithms: Optional[List[str]] = None


@dataclass
class ExecutionConfig:
    """
    Execution parameters configuration.
    
    Attributes:
        thread_pool_size: Number of worker threads
        timeout_seconds: Global timeout for operations
        rng_seed: Random number generator seed
        max_memory_mb: Maximum memory usage limit
        enable_attacks: Whether to execute attacks
        progress_interval: Progress report interval (samples)
    """
    thread_pool_size: int = constants.DEFAULT_THREAD_POOL_SIZE
    timeout_seconds: int = constants.DEFAULT_TIMEOUT_SECONDS
    rng_seed: int = constants.DEFAULT_SEED
    max_memory_mb: int = constants.MAX_MEMORY_MB
    enable_attacks: bool = True
    progress_interval: int = constants.PROGRESS_REPORT_INTERVAL


@dataclass
class OutputConfig:
    """
    Output configuration.
    
    Attributes:
        csv_path: Path to output CSV file
        json_path: Path to output JSON summary (optional)
        log_directory: Directory for log files
        visualization_output: Path for visualization output (optional)
        append_mode: Whether to append to existing CSV
    """
    csv_path: str = "crypto_dataset.csv"
    json_path: Optional[str] = None
    log_directory: str = "logs"
    visualization_output: Optional[str] = None
    append_mode: bool = False


@dataclass
class Configuration:
    """
    Complete pipeline configuration.
    
    This is the main configuration class that combines all
    configuration aspects.
    
    Attributes:
        algorithms: List of algorithms to process (None = all)
        test_strings: List of test strings (None = default test strings)
        algorithm_config: Algorithm selection configuration
        execution_config: Execution parameters
        output_config: Output configuration
        verbose: Enable verbose logging
    """
    # Algorithm selection
    algorithms: Optional[List[str]] = None
    test_strings: Optional[List[bytes]] = None
    
    # Sub-configurations
    algorithm_config: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    output_config: OutputConfig = field(default_factory=OutputConfig)
    
    # General settings
    verbose: bool = True
    
    # Convenience properties
    @property
    def output_csv(self) -> str:
        """Get output CSV path."""
        return self.output_config.csv_path
    
    @property
    def output_json(self) -> Optional[str]:
        """Get output JSON path."""
        return self.output_config.json_path
    
    @property
    def enable_attacks(self) -> bool:
        """Get enable attacks flag."""
        return self.execution_config.enable_attacks
    
    @property
    def rng_seed(self) -> int:
        """Get RNG seed."""
        return self.execution_config.rng_seed
    
    @property
    def thread_pool_size(self) -> int:
        """Get thread pool size."""
        return self.execution_config.thread_pool_size
    
    @property
    def timeout_seconds(self) -> int:
        """Get timeout."""
        return self.execution_config.timeout_seconds
    
    @property
    def max_memory_mb(self) -> int:
        """Get max memory."""
        return self.execution_config.max_memory_mb
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


def get_default_config() -> Configuration:
    """
    Get default configuration.
    
    Returns:
        Configuration with default values
    
    Example:
        >>> config = get_default_config()
        >>> config.thread_pool_size
        4
    """
    return Configuration()


def load_config(config_file: str) -> Configuration:
    """
    Load configuration from file.
    
    Supports JSON and YAML formats (if PyYAML is available).
    
    Args:
        config_file: Path to configuration file
    
    Returns:
        Loaded configuration
    
    Raises:
        ConfigurationError: If file cannot be loaded or is invalid
    
    Example:
        >>> config = load_config("config.json")
        >>> config.thread_pool_size
        8
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_file}",
            config_key="config_file",
            error_type="file_not_found"
        )
    
    # Load based on extension
    suffix = config_path.suffix.lower()
    
    try:
        if suffix == '.json':
            with open(config_path, 'r') as f:
                data = json.load(f)
        elif suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ConfigurationError(
                    "YAML support requires PyYAML: pip install pyyaml",
                    config_key="format",
                    error_type="missing_dependency"
                )
        else:
            raise ConfigurationError(
                f"Unsupported config file format: {suffix}",
                config_key="format",
                error_type="unsupported_format"
            )
        
        # Convert to Configuration object
        return _dict_to_config(data)
        
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Error loading configuration: {str(e)}",
            config_key="config_file",
            error_type="load_error",
            original_error=e
        )


def _dict_to_config(data: Dict[str, Any]) -> Configuration:
    """
    Convert dictionary to Configuration object.
    
    Args:
        data: Configuration dictionary
    
    Returns:
        Configuration object
    """
    # Extract sub-configurations
    algo_config_data = data.get('algorithm_config', {})
    exec_config_data = data.get('execution_config', {})
    output_config_data = data.get('output_config', {})
    
    # Create sub-configuration objects
    algo_config = AlgorithmConfig(**algo_config_data) if algo_config_data else AlgorithmConfig()
    exec_config = ExecutionConfig(**exec_config_data) if exec_config_data else ExecutionConfig()
    output_config = OutputConfig(**output_config_data) if output_config_data else OutputConfig()
    
    # Create main configuration
    config = Configuration(
        algorithms=data.get('algorithms'),
        test_strings=data.get('test_strings'),
        algorithm_config=algo_config,
        execution_config=exec_config,
        output_config=output_config,
        verbose=data.get('verbose', True)
    )
    
    return config


def validate_config(config: Configuration) -> Tuple[bool, List[str]]:
    """
    Validate configuration.
    
    Args:
        config: Configuration to validate
    
    Returns:
        Tuple of (valid, error_list)
    
    Example:
        >>> config = get_default_config()
        >>> valid, errors = validate_config(config)
        >>> if not valid:
        ...     print("Errors:", errors)
    """
    errors = []
    
    # Validate algorithms if specified
    if config.algorithms:
        for algo in config.algorithms:
            valid, error = validate_algorithm_name(algo)
            if not valid:
                errors.append(f"Invalid algorithm '{algo}': {error}")
    
    # Validate test strings if specified
    if config.test_strings:
        for i, test_str in enumerate(config.test_strings):
            if not isinstance(test_str, bytes):
                errors.append(f"Test string {i} must be bytes, got {type(test_str).__name__}")
    
    # Validate thread pool size
    if config.thread_pool_size < 1:
        errors.append(f"Thread pool size must be >= 1, got {config.thread_pool_size}")
    if config.thread_pool_size > constants.MAX_THREAD_POOL_SIZE:
        errors.append(f"Thread pool size too large (max {constants.MAX_THREAD_POOL_SIZE}), got {config.thread_pool_size}")
    
    # Validate timeout
    if config.timeout_seconds <= 0:
        errors.append(f"Timeout must be positive, got {config.timeout_seconds}")
    
    # Validate seed
    if config.rng_seed < 0:
        errors.append(f"RNG seed must be non-negative, got {config.rng_seed}")
    
    # Validate memory limit
    if config.max_memory_mb <= 0:
        errors.append(f"Max memory must be positive, got {config.max_memory_mb}")
    
    # Validate output paths
    if not config.output_csv:
        errors.append("Output CSV path cannot be empty")
    
    # Validate progress interval
    if config.execution_config.progress_interval < 1:
        errors.append(f"Progress interval must be >= 1, got {config.execution_config.progress_interval}")
    
    return (len(errors) == 0, errors)


def merge_configs(
    base: Configuration,
    overrides: Configuration
) -> Configuration:
    """
    Merge two configurations (overrides take precedence).
    
    Args:
        base: Base configuration
        overrides: Override configuration
    
    Returns:
        Merged configuration
    
    Example:
        >>> base = get_default_config()
        >>> override = Configuration(thread_pool_size=8)
        >>> merged = merge_configs(base, override)
        >>> merged.thread_pool_size
        8
    """
    # Convert to dicts
    base_dict = base.to_dict()
    override_dict = overrides.to_dict()
    
    # Merge (override wins)
    merged_dict = {**base_dict, **override_dict}
    
    # Convert back to Configuration
    return _dict_to_config(merged_dict)


def save_config(config: Configuration, output_file: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        output_file: Output file path (JSON or YAML)
    
    Raises:
        ConfigurationError: If save fails
    
    Example:
        >>> config = get_default_config()
        >>> save_config(config, "config.json")
    """
    output_path = Path(output_file)
    suffix = output_path.suffix.lower()
    
    try:
        data = config.to_dict()
        
        # Convert bytes to hex strings for JSON/YAML serialization
        if data.get('test_strings'):
            data['test_strings'] = [
                ts.hex() if isinstance(ts, bytes) else ts
                for ts in data['test_strings']
            ]
        
        if suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(output_path, 'w') as f:
                    yaml.safe_dump(data, f, default_flow_style=False)
            except ImportError:
                raise ConfigurationError(
                    "YAML support requires PyYAML: pip install pyyaml",
                    config_key="format",
                    error_type="missing_dependency"
                )
        else:
            raise ConfigurationError(
                f"Unsupported config file format: {suffix}",
                config_key="format",
                error_type="unsupported_format"
            )
    
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Error saving configuration: {str(e)}",
            config_key="output_file",
            error_type="save_error",
            original_error=e
        )


def create_config_from_args(**kwargs) -> Configuration:
    """
    Create configuration from keyword arguments.
    
    Args:
        **kwargs: Configuration parameters
    
    Returns:
        Configuration object
    
    Example:
        >>> config = create_config_from_args(
        ...     algorithms=["AES-256-GCM", "ChaCha20"],
        ...     thread_pool_size=8,
        ...     output_csv="data.csv"
        ... )
    """
    # Start with defaults
    config = get_default_config()
    
    # Update with provided arguments
    if 'algorithms' in kwargs:
        config.algorithms = kwargs['algorithms']
    if 'test_strings' in kwargs:
        config.test_strings = kwargs['test_strings']
    if 'thread_pool_size' in kwargs:
        config.execution_config.thread_pool_size = kwargs['thread_pool_size']
    if 'timeout_seconds' in kwargs:
        config.execution_config.timeout_seconds = kwargs['timeout_seconds']
    if 'rng_seed' in kwargs:
        config.execution_config.rng_seed = kwargs['rng_seed']
    if 'max_memory_mb' in kwargs:
        config.execution_config.max_memory_mb = kwargs['max_memory_mb']
    if 'enable_attacks' in kwargs:
        config.execution_config.enable_attacks = kwargs['enable_attacks']
    if 'output_csv' in kwargs:
        config.output_config.csv_path = kwargs['output_csv']
    if 'output_json' in kwargs:
        config.output_config.json_path = kwargs['output_json']
    if 'log_directory' in kwargs:
        config.output_config.log_directory = kwargs['log_directory']
    if 'verbose' in kwargs:
        config.verbose = kwargs['verbose']
    
    return config

