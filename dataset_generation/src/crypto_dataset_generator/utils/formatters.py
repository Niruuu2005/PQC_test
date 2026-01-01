"""
Data Formatters
===============

Data formatting functions for display and logging.

This module provides formatters for:
- Bytes data (hex representation)
- Timestamps (human-readable format)
- Percentages
- Lists (with truncation)

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

from datetime import datetime
from typing import Any, List


def format_bytes(data: bytes, max_length: int = 64) -> str:
    """
    Format bytes for display (hex string with truncation).
    
    Args:
        data: Bytes data to format
        max_length: Maximum characters to display (default: 64)
    
    Returns:
        Hex string with truncation indicator if needed
    
    Example:
        >>> format_bytes(b"Hello")
        '48656c6c6f'
        >>> format_bytes(b"A" * 100, max_length=20)
        '414141414141... (truncated, 100 bytes)'
    """
    if not data:
        return "(empty)"
    
    hex_str = data.hex()
    
    if len(hex_str) <= max_length:
        return hex_str
    
    # Truncate and add indicator
    truncated = hex_str[:max_length]
    return f"{truncated}... (truncated, {len(data)} bytes)"


def format_timestamp(timestamp: str = None) -> str:
    """
    Format ISO 8601 timestamp to human-readable format.
    
    Args:
        timestamp: ISO 8601 timestamp string (default: current time)
    
    Returns:
        Human-readable timestamp
    
    Example:
        >>> format_timestamp("2025-12-30T10:35:00.123456")
        '2025-12-30 10:35:00'
    """
    if timestamp is None:
        # Use current time
        dt = datetime.now()
    else:
        # Parse ISO 8601 timestamp
        try:
            # Handle various ISO 8601 formats
            if 'T' in timestamp:
                # Full ISO format
                if '.' in timestamp:
                    # With microseconds
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    # Without microseconds
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                # Simple date format
                dt = datetime.fromisoformat(timestamp)
        except (ValueError, AttributeError):
            return timestamp  # Return as-is if parsing fails
    
    # Format as YYYY-MM-DD HH:MM:SS
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage value.
    
    Args:
        value: Percentage value (0-100)
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Formatted percentage string
    
    Example:
        >>> format_percentage(45.6789)
        '45.68%'
        >>> format_percentage(100.0, decimals=0)
        '100%'
    """
    if decimals == 0:
        return f"{int(value)}%"
    else:
        return f"{value:.{decimals}f}%"


def format_list(items: List[str], max_items: int = 10, separator: str = ", ") -> str:
    """
    Format list with truncation.
    
    Args:
        items: List of items to format
        max_items: Maximum items to display (default: 10)
        separator: Separator between items (default: ", ")
    
    Returns:
        Formatted list string
    
    Example:
        >>> format_list(["a", "b", "c"])
        'a, b, c'
        >>> format_list([str(i) for i in range(15)], max_items=5)
        '0, 1, 2, 3, 4, ... (10 more)'
    """
    if not items:
        return "(empty)"
    
    if len(items) <= max_items:
        return separator.join(items)
    
    # Show first max_items and indicate how many more
    shown = separator.join(items[:max_items])
    remaining = len(items) - max_items
    return f"{shown}{separator}... ({remaining} more)"


def format_number(value: float, precision: int = 2) -> str:
    """
    Format number with specified precision.
    
    Args:
        value: Number to format
        precision: Decimal precision (default: 2)
    
    Returns:
        Formatted number string
    
    Example:
        >>> format_number(3.14159)
        '3.14'
        >>> format_number(1000.5, precision=1)
        '1000.5'
    """
    if precision == 0:
        return str(int(value))
    else:
        return f"{value:.{precision}f}"


def format_scientific(value: float, precision: int = 2) -> str:
    """
    Format number in scientific notation.
    
    Args:
        value: Number to format
        precision: Decimal precision (default: 2)
    
    Returns:
        Scientific notation string
    
    Example:
        >>> format_scientific(0.00012345)
        '1.23e-04'
        >>> format_scientific(123456789)
        '1.23e+08'
    """
    return f"{value:.{precision}e}"


def format_boolean(value: bool, true_str: str = "Yes", false_str: str = "No") -> str:
    """
    Format boolean value as string.
    
    Args:
        value: Boolean value
        true_str: String for True (default: "Yes")
        false_str: String for False (default: "No")
    
    Returns:
        Formatted string
    
    Example:
        >>> format_boolean(True)
        'Yes'
        >>> format_boolean(False, "ON", "OFF")
        'OFF'
    """
    return true_str if value else false_str


def format_dict(data: dict, indent: int = 2, max_depth: int = 3) -> str:
    """
    Format dictionary for display.
    
    Args:
        data: Dictionary to format
        indent: Indentation spaces (default: 2)
        max_depth: Maximum nesting depth (default: 3)
    
    Returns:
        Formatted dictionary string
    
    Example:
        >>> format_dict({"key": "value", "num": 42})
        'key: value\\nnum: 42'
    """
    def _format_value(value: Any, depth: int) -> str:
        if depth >= max_depth:
            return str(value)
        
        if isinstance(value, dict):
            lines = []
            for k, v in value.items():
                lines.append(f"{' ' * (indent * depth)}{k}: {_format_value(v, depth + 1)}")
            return '\n'.join(lines)
        elif isinstance(value, list):
            if len(value) <= 5:
                return ', '.join(str(v) for v in value)
            else:
                shown = ', '.join(str(v) for v in value[:5])
                return f"{shown}, ... ({len(value) - 5} more)"
        else:
            return str(value)
    
    return _format_value(data, 0)


def format_hex(value: int, min_width: int = 0) -> str:
    """
    Format integer as hexadecimal string.
    
    Args:
        value: Integer value
        min_width: Minimum width (zero-padded, default: 0)
    
    Returns:
        Hex string (without 0x prefix)
    
    Example:
        >>> format_hex(255)
        'ff'
        >>> format_hex(15, min_width=4)
        '000f'
    """
    if min_width > 0:
        return f"{value:0{min_width}x}"
    else:
        return f"{value:x}"


def format_binary(value: int, min_width: int = 8) -> str:
    """
    Format integer as binary string.
    
    Args:
        value: Integer value
        min_width: Minimum width (zero-padded, default: 8)
    
    Returns:
        Binary string (without 0b prefix)
    
    Example:
        >>> format_binary(5)
        '00000101'
        >>> format_binary(255, min_width=16)
        '0000000011111111'
    """
    return f"{value:0{min_width}b}"


def format_table_row(columns: List[str], widths: List[int], separator: str = " | ") -> str:
    """
    Format a table row with fixed column widths.
    
    Args:
        columns: Column values
        widths: Column widths
        separator: Column separator (default: " | ")
    
    Returns:
        Formatted row string
    
    Example:
        >>> format_table_row(["Name", "Age"], [20, 5])
        'Name                 | Age  '
    """
    formatted_cols = []
    for col, width in zip(columns, widths):
        # Truncate or pad to width
        col_str = str(col)
        if len(col_str) > width:
            col_str = col_str[:width-3] + "..."
        formatted_cols.append(col_str.ljust(width))
    
    return separator.join(formatted_cols)


def format_key_value(key: str, value: Any, key_width: int = 30) -> str:
    """
    Format key-value pair for display.
    
    Args:
        key: Key name
        value: Value
        key_width: Width for key column (default: 30)
    
    Returns:
        Formatted string
    
    Example:
        >>> format_key_value("Algorithm", "AES-256-GCM")
        'Algorithm:                     AES-256-GCM'
    """
    key_str = f"{key}:".ljust(key_width)
    return f"{key_str} {value}"


def format_multiline(text: str, indent: int = 4, width: int = 80) -> str:
    """
    Format text with word wrapping and indentation.
    
    Args:
        text: Text to format
        indent: Indentation spaces (default: 4)
        width: Maximum line width (default: 80)
    
    Returns:
        Formatted multi-line string
    
    Example:
        >>> format_multiline("This is a long text that needs wrapping", indent=2, width=20)
        '  This is a long\\n  text that needs\\n  wrapping'
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = indent
    
    for word in words:
        word_length = len(word)
        
        if current_length + word_length + 1 > width:
            # Start new line
            if current_line:
                lines.append(' ' * indent + ' '.join(current_line))
            current_line = [word]
            current_length = indent + word_length
        else:
            current_line.append(word)
            current_length += word_length + 1
    
    # Add last line
    if current_line:
        lines.append(' ' * indent + ' '.join(current_line))
    
    return '\n'.join(lines)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    (Alias for helpers.format_size for consistency)
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    
    Example:
        >>> format_file_size(1024)
        '1.0 KB'
    """
    from .helpers import format_size
    return format_size(size_bytes)


def format_duration_ms(milliseconds: float) -> str:
    """
    Format duration in milliseconds.
    (Alias for helpers.format_duration for consistency)
    
    Args:
        milliseconds: Duration in milliseconds
    
    Returns:
        Formatted duration string
    
    Example:
        >>> format_duration_ms(1234)
        '1.23s'
    """
    from .helpers import format_duration
    return format_duration(milliseconds)

