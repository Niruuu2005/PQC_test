"""
Logging Infrastructure
======================

Centralized logging system with file rotation and context-aware logging.

This module provides:
- Singleton logger manager
- Console and file handlers
- Log rotation (size and time-based)
- Context fields for structured logging
- Multiple log levels

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any

# Import utils
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helpers import ensure_directory
from utils import constants


class LoggerManager:
    """
    Singleton logger manager for centralized logging.
    
    This class ensures that all loggers in the system are configured
    consistently and can be accessed from anywhere.
    """
    
    _instance: Optional['LoggerManager'] = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger by name.
        
        Args:
            name: Logger name (typically module name)
        
        Returns:
            Configured logger instance
        
        Example:
            >>> logger = LoggerManager.get_logger("crypto_module")
            >>> logger.info("Processing started")
        """
        instance = cls()
        
        if name not in instance._loggers:
            logger = logging.getLogger(name)
            instance._loggers[name] = logger
        
        return instance._loggers[name]
    
    @classmethod
    def reset(cls):
        """Reset logger manager (useful for testing)."""
        cls._instance = None
        cls._loggers = {}


def setup_logger(
    name: str,
    level: str = constants.DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    rotation_mode: str = "size",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    when: str = "midnight",
    interval: int = 1
) -> logging.Logger:
    """
    Setup and configure a logger.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Optional custom log format
        rotation_mode: "size" or "time" for rotation strategy
        max_bytes: Max file size before rotation (for size-based)
        backup_count: Number of backup files to keep
        when: When to rotate (for time-based: 'S', 'M', 'H', 'D', 'midnight', 'W0'-'W6')
        interval: Rotation interval (for time-based)
    
    Returns:
        Configured logger
    
    Example:
        >>> logger = setup_logger("my_module", level="DEBUG", log_file="logs/app.log")
        >>> logger.info("Application started")
    """
    # Get or create logger
    logger = LoggerManager.get_logger(name)
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Use default format if not provided
    if log_format is None:
        log_format = constants.DEFAULT_LOG_FORMAT
    
    formatter = logging.Formatter(log_format)
    
    # Add console handler
    console_handler = configure_console_handler(log_level, formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file provided
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        ensure_directory(str(log_path.parent))
        
        if rotation_mode == "size":
            file_handler = configure_rotating_file_handler(
                log_file, log_level, formatter, max_bytes, backup_count
            )
        elif rotation_mode == "time":
            file_handler = configure_timed_rotating_file_handler(
                log_file, log_level, formatter, when, interval, backup_count
            )
        else:
            # Simple file handler without rotation
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def configure_console_handler(
    level: int,
    formatter: logging.Formatter
) -> logging.StreamHandler:
    """
    Configure console (stdout) handler.
    
    Args:
        level: Log level
        formatter: Log formatter
    
    Returns:
        Configured console handler
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def configure_rotating_file_handler(
    log_file: str,
    level: int,
    formatter: logging.Formatter,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> RotatingFileHandler:
    """
    Configure rotating file handler (rotation by size).
    
    Args:
        log_file: Log file path
        level: Log level
        formatter: Log formatter
        max_bytes: Max file size before rotation
        backup_count: Number of backup files
    
    Returns:
        Configured rotating file handler
    """
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def configure_timed_rotating_file_handler(
    log_file: str,
    level: int,
    formatter: logging.Formatter,
    when: str = "midnight",
    interval: int = 1,
    backup_count: int = 7
) -> TimedRotatingFileHandler:
    """
    Configure timed rotating file handler (rotation by time).
    
    Args:
        log_file: Log file path
        level: Log level
        formatter: Log formatter
        when: When to rotate ('S', 'M', 'H', 'D', 'midnight', 'W0'-'W6')
        interval: Rotation interval
        backup_count: Number of backup files
    
    Returns:
        Configured timed rotating file handler
    """
    handler = TimedRotatingFileHandler(
        log_file,
        when=when,
        interval=interval,
        backupCount=backup_count
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def get_logger(name: str) -> logging.Logger:
    """
    Get logger by name (convenience function).
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger("crypto_module")
        >>> logger.info("Processing...")
    """
    return LoggerManager.get_logger(name)


def set_log_level(logger: logging.Logger, level: str) -> None:
    """
    Set log level for a logger.
    
    Args:
        logger: Logger instance
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Example:
        >>> logger = get_logger("my_module")
        >>> set_log_level(logger, "DEBUG")
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(log_level)


class ContextLogger:
    """
    Logger wrapper that adds context fields to all log messages.
    
    This is useful for adding common context like algorithm name,
    sample ID, etc. to all log messages in a specific scope.
    
    Example:
        >>> logger = get_logger("pipeline")
        >>> context_logger = ContextLogger(logger, algorithm="AES-256-GCM", sample_id=42)
        >>> context_logger.info("Processing...")
        >>> # Output: [INFO] [AES-256-GCM] [sample_id=42] Processing...
    """
    
    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize context logger.
        
        Args:
            logger: Base logger
            **context: Context fields to add
        """
        self.logger = logger
        self.context = context
    
    def _format_message(self, message: str) -> str:
        """Format message with context."""
        context_str = " ".join(f"[{k}={v}]" for k, v in self.context.items())
        return f"{context_str} {message}" if context_str else message
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message with context."""
        self.logger.debug(self._format_message(message), *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message with context."""
        self.logger.info(self._format_message(message), *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message with context."""
        self.logger.warning(self._format_message(message), *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message with context."""
        self.logger.error(self._format_message(message), *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message with context."""
        self.logger.critical(self._format_message(message), *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with context."""
        self.logger.exception(self._format_message(message), *args, **kwargs)


def create_context_logger(
    logger: logging.Logger,
    **context: Any
) -> ContextLogger:
    """
    Create a context logger.
    
    Args:
        logger: Base logger
        **context: Context fields
    
    Returns:
        Context logger
    
    Example:
        >>> logger = get_logger("pipeline")
        >>> ctx_logger = create_context_logger(logger, algorithm="AES", run_id="abc123")
        >>> ctx_logger.info("Started")
    """
    return ContextLogger(logger, **context)


# Module-level convenience function
_default_logger: Optional[logging.Logger] = None


def get_default_logger() -> logging.Logger:
    """
    Get default logger for the module.
    
    Returns:
        Default logger
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger(
            "crypto_dataset_generator",
            level=constants.DEFAULT_LOG_LEVEL
        )
    return _default_logger

