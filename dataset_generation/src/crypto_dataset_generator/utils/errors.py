"""
Custom Exception Classes
=========================

Custom exception classes for better error handling and reporting.

This module defines exception classes for different subsystems:
- CipherError: Cryptographic operation errors
- AttackError: Attack execution errors
- MetricsError: Metric computation errors
- PipelineError: Pipeline execution errors
- ConfigurationError: Configuration validation errors
- ValidationError: Input validation errors

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

from typing import Any, Dict, Optional


class CipherError(Exception):
    """
    Exception raised for cipher-related errors.
    
    Attributes:
        algorithm_name: Name of the cipher algorithm
        error_type: Type of error (e.g., "encryption_failed", "decryption_failed")
        details: Additional error details
        original_error: Original exception if wrapped
    """
    
    def __init__(
        self,
        message: str,
        algorithm_name: Optional[str] = None,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize CipherError.
        
        Args:
            message: Error message
            algorithm_name: Name of the cipher algorithm
            error_type: Type of error
            details: Additional error details
            original_error: Original exception
        """
        self.algorithm_name = algorithm_name
        self.error_type = error_type
        self.details = details or {}
        self.original_error = original_error
        
        # Build comprehensive error message
        full_message = f"Cipher Error: {message}"
        if algorithm_name:
            full_message += f" (Algorithm: {algorithm_name})"
        if error_type:
            full_message += f" [Type: {error_type}]"
        if original_error:
            full_message += f" | Original: {str(original_error)}"
        
        super().__init__(full_message)


class AttackError(Exception):
    """
    Exception raised for attack execution errors.
    
    Attributes:
        attack_name: Name of the attack
        error_type: Type of error (e.g., "timeout", "not_applicable")
        details: Additional error details
        original_error: Original exception if wrapped
    """
    
    def __init__(
        self,
        message: str,
        attack_name: Optional[str] = None,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize AttackError.
        
        Args:
            message: Error message
            attack_name: Name of the attack
            error_type: Type of error
            details: Additional error details
            original_error: Original exception
        """
        self.attack_name = attack_name
        self.error_type = error_type
        self.details = details or {}
        self.original_error = original_error
        
        # Build comprehensive error message
        full_message = f"Attack Error: {message}"
        if attack_name:
            full_message += f" (Attack: {attack_name})"
        if error_type:
            full_message += f" [Type: {error_type}]"
        if original_error:
            full_message += f" | Original: {str(original_error)}"
        
        super().__init__(full_message)


class MetricsError(Exception):
    """
    Exception raised for metric computation errors.
    
    Attributes:
        metric_name: Name of the metric
        error_type: Type of error (e.g., "computation_failed", "invalid_input")
        details: Additional error details
        original_error: Original exception if wrapped
    """
    
    def __init__(
        self,
        message: str,
        metric_name: Optional[str] = None,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize MetricsError.
        
        Args:
            message: Error message
            metric_name: Name of the metric
            error_type: Type of error
            details: Additional error details
            original_error: Original exception
        """
        self.metric_name = metric_name
        self.error_type = error_type
        self.details = details or {}
        self.original_error = original_error
        
        # Build comprehensive error message
        full_message = f"Metrics Error: {message}"
        if metric_name:
            full_message += f" (Metric: {metric_name})"
        if error_type:
            full_message += f" [Type: {error_type}]"
        if original_error:
            full_message += f" | Original: {str(original_error)}"
        
        super().__init__(full_message)


class PipelineError(Exception):
    """
    Exception raised for pipeline execution errors.
    
    Attributes:
        stage: Pipeline stage where error occurred
        error_type: Type of error (e.g., "initialization_failed", "execution_failed")
        details: Additional error details
        original_error: Original exception if wrapped
    """
    
    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize PipelineError.
        
        Args:
            message: Error message
            stage: Pipeline stage
            error_type: Type of error
            details: Additional error details
            original_error: Original exception
        """
        self.stage = stage
        self.error_type = error_type
        self.details = details or {}
        self.original_error = original_error
        
        # Build comprehensive error message
        full_message = f"Pipeline Error: {message}"
        if stage:
            full_message += f" (Stage: {stage})"
        if error_type:
            full_message += f" [Type: {error_type}]"
        if original_error:
            full_message += f" | Original: {str(original_error)}"
        
        super().__init__(full_message)


class ConfigurationError(Exception):
    """
    Exception raised for configuration errors.
    
    Attributes:
        config_key: Configuration key that caused the error
        error_type: Type of error (e.g., "missing_key", "invalid_value")
        details: Additional error details
        original_error: Original exception if wrapped
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize ConfigurationError.
        
        Args:
            message: Error message
            config_key: Configuration key
            error_type: Type of error
            details: Additional error details
            original_error: Original exception
        """
        self.config_key = config_key
        self.error_type = error_type
        self.details = details or {}
        self.original_error = original_error
        
        # Build comprehensive error message
        full_message = f"Configuration Error: {message}"
        if config_key:
            full_message += f" (Key: {config_key})"
        if error_type:
            full_message += f" [Type: {error_type}]"
        if original_error:
            full_message += f" | Original: {str(original_error)}"
        
        super().__init__(full_message)


class ValidationError(Exception):
    """
    Exception raised for validation errors.
    
    Attributes:
        field_name: Name of the field that failed validation
        error_message: Detailed validation error message
        expected_value: Expected value or format
        actual_value: Actual value that failed validation
    """
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None
    ):
        """
        Initialize ValidationError.
        
        Args:
            message: Error message
            field_name: Name of the field
            expected_value: Expected value or format
            actual_value: Actual value
        """
        self.field_name = field_name
        self.error_message = message
        self.expected_value = expected_value
        self.actual_value = actual_value
        
        # Build comprehensive error message
        full_message = f"Validation Error: {message}"
        if field_name:
            full_message += f" (Field: {field_name})"
        if expected_value is not None:
            full_message += f" | Expected: {expected_value}"
        if actual_value is not None:
            full_message += f" | Actual: {actual_value}"
        
        super().__init__(full_message)


# Convenience functions for raising errors

def raise_cipher_error(
    message: str,
    algorithm_name: Optional[str] = None,
    error_type: Optional[str] = None,
    **kwargs
) -> None:
    """
    Convenience function to raise a CipherError.
    
    Args:
        message: Error message
        algorithm_name: Name of the cipher algorithm
        error_type: Type of error
        **kwargs: Additional error details
    
    Raises:
        CipherError: Always raises
    """
    raise CipherError(message, algorithm_name, error_type, details=kwargs)


def raise_attack_error(
    message: str,
    attack_name: Optional[str] = None,
    error_type: Optional[str] = None,
    **kwargs
) -> None:
    """
    Convenience function to raise an AttackError.
    
    Args:
        message: Error message
        attack_name: Name of the attack
        error_type: Type of error
        **kwargs: Additional error details
    
    Raises:
        AttackError: Always raises
    """
    raise AttackError(message, attack_name, error_type, details=kwargs)


def raise_metrics_error(
    message: str,
    metric_name: Optional[str] = None,
    error_type: Optional[str] = None,
    **kwargs
) -> None:
    """
    Convenience function to raise a MetricsError.
    
    Args:
        message: Error message
        metric_name: Name of the metric
        error_type: Type of error
        **kwargs: Additional error details
    
    Raises:
        MetricsError: Always raises
    """
    raise MetricsError(message, metric_name, error_type, details=kwargs)


def raise_pipeline_error(
    message: str,
    stage: Optional[str] = None,
    error_type: Optional[str] = None,
    **kwargs
) -> None:
    """
    Convenience function to raise a PipelineError.
    
    Args:
        message: Error message
        stage: Pipeline stage
        error_type: Type of error
        **kwargs: Additional error details
    
    Raises:
        PipelineError: Always raises
    """
    raise PipelineError(message, stage, error_type, details=kwargs)


def raise_configuration_error(
    message: str,
    config_key: Optional[str] = None,
    error_type: Optional[str] = None,
    **kwargs
) -> None:
    """
    Convenience function to raise a ConfigurationError.
    
    Args:
        message: Error message
        config_key: Configuration key
        error_type: Type of error
        **kwargs: Additional error details
    
    Raises:
        ConfigurationError: Always raises
    """
    raise ConfigurationError(message, config_key, error_type, details=kwargs)


def raise_validation_error(
    message: str,
    field_name: Optional[str] = None,
    expected_value: Optional[Any] = None,
    actual_value: Optional[Any] = None
) -> None:
    """
    Convenience function to raise a ValidationError.
    
    Args:
        message: Error message
        field_name: Name of the field
        expected_value: Expected value or format
        actual_value: Actual value
    
    Raises:
        ValidationError: Always raises
    """
    raise ValidationError(message, field_name, expected_value, actual_value)

