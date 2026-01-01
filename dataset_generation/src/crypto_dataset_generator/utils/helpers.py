"""
Helper Functions
================

General utility functions for common operations.

This module provides:
- Directory management
- Size formatting
- Duration formatting
- Dictionary operations
- Safe math operations
- String manipulation
- UUID generation

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict


def ensure_directory(path: str) -> str:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    
    Returns:
        The path string (normalized)
    
    Raises:
        OSError: If directory creation fails
    
    Example:
        >>> ensure_directory("output/results")
        'output/results'
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return str(path_obj)


def format_size(size_bytes: int) -> str:
    """
    Convert bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB", "2.3 GB")
    
    Example:
        >>> format_size(1024)
        '1.0 KB'
        >>> format_size(1536)
        '1.5 KB'
        >>> format_size(1048576)
        '1.0 MB'
    """
    if size_bytes < 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    
    # Format with appropriate precision
    if unit_index == 0:  # Bytes
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def format_duration(milliseconds: float) -> str:
    """
    Convert milliseconds to human-readable format.
    
    Args:
        milliseconds: Duration in milliseconds
    
    Returns:
        Formatted string (e.g., "123ms", "1.23s", "5m 30s", "1h 23m")
    
    Example:
        >>> format_duration(123)
        '123ms'
        >>> format_duration(1234)
        '1.23s'
        >>> format_duration(65000)
        '1m 5s'
        >>> format_duration(3723000)
        '1h 2m'
    """
    if milliseconds < 0:
        return "0ms"
    
    # Less than 1 second - show milliseconds
    if milliseconds < 1000:
        return f"{int(milliseconds)}ms"
    
    # Less than 1 minute - show seconds
    seconds = milliseconds / 1000
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    # Less than 1 hour - show minutes and seconds
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    if minutes < 60:
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds}s"
        return f"{minutes}m"
    
    # 1 hour or more - show hours and minutes
    hours = int(minutes // 60)
    remaining_minutes = int(minutes % 60)
    if remaining_minutes > 0:
        return f"{hours}h {remaining_minutes}m"
    return f"{hours}h"


def merge_dictionaries(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    strategy: str = "override"
) -> Dict[str, Any]:
    """
    Merge two dictionaries using specified strategy.
    
    Args:
        dict1: First dictionary (base)
        dict2: Second dictionary (overrides)
        strategy: Merge strategy - one of:
            - "override": dict2 values override dict1 (default)
            - "merge": Recursively merge nested dicts
            - "error_on_conflict": Raise error if keys conflict
    
    Returns:
        Merged dictionary
    
    Raises:
        ValueError: If strategy is "error_on_conflict" and keys conflict
        ValueError: If strategy is invalid
    
    Example:
        >>> merge_dictionaries({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
        {'a': 1, 'b': 3, 'c': 4}
    """
    if strategy not in ["override", "merge", "error_on_conflict"]:
        raise ValueError(f"Invalid merge strategy: {strategy}")
    
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key not in result:
            result[key] = value
        else:
            # Key exists in both dictionaries
            if strategy == "error_on_conflict":
                raise ValueError(f"Key conflict detected: {key}")
            elif strategy == "override":
                result[key] = value
            elif strategy == "merge":
                # Recursive merge for nested dictionaries
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dictionaries(result[key], value, strategy="merge")
                else:
                    result[key] = value
    
    return result


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0
) -> float:
    """
    Safe division with default value for zero denominator.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero (default: 0.0)
    
    Returns:
        Result of division or default value
    
    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, default=float('inf'))
        inf
    """
    if denominator == 0:
        return default
    return numerator / denominator


def slugify(text: str) -> str:
    """
    Convert text to slug format (lowercase, hyphens, alphanumeric).
    
    Args:
        text: Input text
    
    Returns:
        Normalized slug string
    
    Example:
        >>> slugify("AES-256-GCM")
        'aes-256-gcm'
        >>> slugify("Hello World!")
        'hello-world'
        >>> slugify("Test__Multiple---Separators")
        'test-multiple-separators'
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces and underscores with hyphens
    text = re.sub(r'[\s_]+', '-', text)
    
    # Remove non-alphanumeric characters except hyphens
    text = re.sub(r'[^a-z0-9-]', '', text)
    
    # Replace multiple consecutive hyphens with single hyphen
    text = re.sub(r'-+', '-', text)
    
    # Remove leading/trailing hyphens
    text = text.strip('-')
    
    return text


def generate_uuid() -> str:
    """
    Generate a UUID (Universally Unique Identifier).
    
    Returns:
        UUID string (format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    
    Example:
        >>> uuid_str = generate_uuid()
        >>> len(uuid_str)
        36
        >>> uuid_str.count('-')
        4
    """
    return str(uuid.uuid4())


def truncate_string(text: str, max_length: int = 64, suffix: str = "...") -> str:
    """
    Truncate string to maximum length with suffix.
    
    Args:
        text: Input text
        max_length: Maximum length (including suffix)
        suffix: Suffix to append if truncated (default: "...")
    
    Returns:
        Truncated string
    
    Example:
        >>> truncate_string("Hello World", 8)
        'Hello...'
        >>> truncate_string("Short", 10)
        'Short'
    """
    if len(text) <= max_length:
        return text
    
    # Account for suffix length
    truncate_at = max(0, max_length - len(suffix))
    return text[:truncate_at] + suffix


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_value: Minimum bound
        max_value: Maximum bound
    
    Returns:
        Clamped value
    
    Example:
        >>> clamp(5, 0, 10)
        5
        >>> clamp(-5, 0, 10)
        0
        >>> clamp(15, 0, 10)
        10
    """
    return max(min_value, min(max_value, value))


def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of two.
    
    Args:
        n: Number to check
    
    Returns:
        True if n is a power of two, False otherwise
    
    Example:
        >>> is_power_of_two(8)
        True
        >>> is_power_of_two(10)
        False
        >>> is_power_of_two(1024)
        True
    """
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Filename or path
    
    Returns:
        File extension (without dot) or empty string
    
    Example:
        >>> get_file_extension("data.csv")
        'csv'
        >>> get_file_extension("/path/to/file.json")
        'json'
        >>> get_file_extension("noextension")
        ''
    """
    path = Path(filename)
    return path.suffix.lstrip('.')


def normalize_path(path: str) -> str:
    """
    Normalize file path (resolve relative paths, expand user, etc.).
    
    Args:
        path: File path
    
    Returns:
        Normalized absolute path
    
    Example:
        >>> normalize_path("./data/../output")
        # Returns absolute path to 'output' directory
    """
    path_obj = Path(path).expanduser().resolve()
    return str(path_obj)


def get_relative_path(path: str, base_path: str) -> str:
    """
    Get relative path from base path.
    
    Args:
        path: Target path
        base_path: Base path
    
    Returns:
        Relative path from base to target
    
    Example:
        >>> get_relative_path("/home/user/project/data", "/home/user")
        'project/data'
    """
    try:
        path_obj = Path(path).resolve()
        base_obj = Path(base_path).resolve()
        return str(path_obj.relative_to(base_obj))
    except ValueError:
        # Paths don't have common base
        return str(path)


def round_to_precision(value: float, precision: int = 2) -> float:
    """
    Round value to specified decimal precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places (default: 2)
    
    Returns:
        Rounded value
    
    Example:
        >>> round_to_precision(3.14159, 2)
        3.14
        >>> round_to_precision(3.14159, 4)
        3.1416
    """
    return round(value, precision)


def percentage(part: float, total: float, precision: int = 2) -> float:
    """
    Calculate percentage safely.
    
    Args:
        part: Part value
        total: Total value
        precision: Decimal precision (default: 2)
    
    Returns:
        Percentage value (0.0 if total is zero)
    
    Example:
        >>> percentage(25, 100)
        25.0
        >>> percentage(1, 3)
        33.33
        >>> percentage(10, 0)
        0.0
    """
    if total == 0:
        return 0.0
    return round_to_precision((part / total) * 100, precision)

