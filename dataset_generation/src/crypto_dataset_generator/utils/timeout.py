"""
Timeout Management
==================

Timeout protection for long-running operations.

This module provides:
- TimeoutManager class for managing operation timeouts
- Decorator for timeout protection
- Utility functions for running operations with timeout

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

import functools
import threading
import time
from typing import Any, Callable, Optional, Tuple


class TimeoutError(Exception):
    """Exception raised when an operation times out."""
    pass


class TimeoutManager:
    """
    Manages operation timeouts using threading.
    
    Attributes:
        timeout_seconds: Timeout duration in seconds
        thread: Thread executing the operation
        result: Operation result (if completed)
        exception: Exception raised (if any)
        timed_out: Whether operation timed out
    """
    
    def __init__(self, timeout_seconds: int):
        """
        Initialize TimeoutManager.
        
        Args:
            timeout_seconds: Timeout duration in seconds
        """
        self.timeout_seconds = timeout_seconds
        self.thread: Optional[threading.Thread] = None
        self.result: Any = None
        self.exception: Optional[Exception] = None
        self.timed_out: bool = False
        self._completed: threading.Event = threading.Event()
    
    def _run_function(self, function: Callable, args: Tuple, kwargs: dict) -> None:
        """
        Internal method to run function and capture result/exception.
        
        Args:
            function: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
        """
        try:
            self.result = function(*args, **kwargs)
        except Exception as e:
            self.exception = e
        finally:
            self._completed.set()
    
    def start_timeout(
        self,
        operation: Callable,
        args: Tuple = (),
        kwargs: Optional[dict] = None
    ) -> Any:
        """
        Start operation with timeout.
        
        Args:
            operation: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
        
        Returns:
            Operation result
        
        Raises:
            TimeoutError: If operation times out
            Exception: Any exception raised by operation
        """
        if kwargs is None:
            kwargs = {}
        
        # Start thread
        self.thread = threading.Thread(
            target=self._run_function,
            args=(operation, args, kwargs)
        )
        self.thread.daemon = True
        self.thread.start()
        
        # Wait for completion or timeout
        completed = self._completed.wait(timeout=self.timeout_seconds)
        
        if not completed:
            # Timeout occurred
            self.timed_out = True
            raise TimeoutError(
                f"Operation timed out after {self.timeout_seconds} seconds"
            )
        
        # Check for exception
        if self.exception is not None:
            raise self.exception
        
        return self.result
    
    def cancel_timeout(self) -> None:
        """
        Cancel timeout (note: cannot forcibly stop thread in Python).
        
        This sets a flag but doesn't forcibly terminate the thread.
        The thread will continue to run in the background.
        """
        self.timed_out = True


def timeout_decorator(seconds: int) -> Callable:
    """
    Decorator for timeout protection.
    
    Args:
        seconds: Timeout duration in seconds
    
    Returns:
        Decorator function
    
    Raises:
        TimeoutError: If decorated function times out
    
    Example:
        >>> @timeout_decorator(5)
        ... def long_operation():
        ...     time.sleep(10)
        ...     return "done"
        >>> long_operation()  # Raises TimeoutError after 5 seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return run_with_timeout(func, args, kwargs, seconds)
        return wrapper
    return decorator


def run_with_timeout(
    function: Callable,
    args: Tuple = (),
    kwargs: Optional[dict] = None,
    timeout_seconds: int = 30
) -> Any:
    """
    Execute function with timeout.
    
    Args:
        function: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        timeout_seconds: Timeout duration in seconds (default: 30)
    
    Returns:
        Function result
    
    Raises:
        TimeoutError: If function times out
        Exception: Any exception raised by function
    
    Example:
        >>> def slow_function(x):
        ...     time.sleep(2)
        ...     return x * 2
        >>> result = run_with_timeout(slow_function, (5,), timeout_seconds=5)
        >>> result
        10
    """
    if kwargs is None:
        kwargs = {}
    
    manager = TimeoutManager(timeout_seconds)
    return manager.start_timeout(function, args, kwargs)


def create_timeout_thread(
    function: Callable,
    args: Tuple = (),
    kwargs: Optional[dict] = None,
    timeout_seconds: int = 30
) -> threading.Thread:
    """
    Create a thread for timeout-protected execution.
    
    Args:
        function: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        timeout_seconds: Timeout duration in seconds
    
    Returns:
        Thread object (not started)
    
    Example:
        >>> thread = create_timeout_thread(my_function, (arg1, arg2))
        >>> thread.start()
        >>> thread.join(timeout=10)
    """
    if kwargs is None:
        kwargs = {}
    
    def wrapper():
        try:
            function(*args, **kwargs)
        except Exception:
            pass  # Silently ignore exceptions in thread
    
    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    return thread


class TimeoutContext:
    """
    Context manager for timeout protection.
    
    Example:
        >>> with TimeoutContext(5) as ctx:
        ...     # Your long-running operation here
        ...     time.sleep(2)
        ...     print("Completed")
    """
    
    def __init__(self, timeout_seconds: int):
        """
        Initialize timeout context.
        
        Args:
            timeout_seconds: Timeout duration in seconds
        """
        self.timeout_seconds = timeout_seconds
        self.start_time: Optional[float] = None
        self.timed_out: bool = False
    
    def __enter__(self):
        """Enter context (start timer)."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context (check timeout)."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                self.timed_out = True
        return False  # Don't suppress exceptions
    
    def check_timeout(self) -> None:
        """
        Check if timeout has been exceeded.
        
        Raises:
            TimeoutError: If timeout exceeded
        """
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                self.timed_out = True
                raise TimeoutError(
                    f"Operation timed out after {elapsed:.2f} seconds "
                    f"(limit: {self.timeout_seconds} seconds)"
                )
    
    def get_remaining_time(self) -> float:
        """
        Get remaining time before timeout.
        
        Returns:
            Remaining seconds (0 if timed out)
        """
        if self.start_time:
            elapsed = time.time() - self.start_time
            remaining = self.timeout_seconds - elapsed
            return max(0.0, remaining)
        return self.timeout_seconds

