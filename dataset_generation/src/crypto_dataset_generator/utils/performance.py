"""
Performance Tracking
====================

Performance tracking and profiling utilities.

This module provides:
- PerformanceTracker class for tracking execution metrics
- Functions for measuring time and memory
- Profiling utilities

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

import functools
import time
from typing import Any, Callable, Dict, Optional

from .memory import get_memory_usage


class PerformanceTracker:
    """
    Tracks execution performance (time and memory).
    
    Attributes:
        start_time: Start timestamp
        end_time: End timestamp
        memory_start: Memory at start (MB)
        memory_end: Memory at end (MB)
        name: Tracker name (optional)
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize performance tracker.
        
        Args:
            name: Tracker name (optional)
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.memory_start: Optional[float] = None
        self.memory_end: Optional[float] = None
    
    def start(self) -> None:
        """Start tracking."""
        self.start_time = time.perf_counter()
        self.memory_start = get_memory_usage()
    
    def stop(self) -> None:
        """Stop tracking."""
        self.end_time = time.perf_counter()
        self.memory_end = get_memory_usage()
    
    def get_duration(self) -> float:
        """
        Get duration in seconds.
        
        Returns:
            Duration in seconds (0 if not completed)
        """
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def get_duration_ms(self) -> float:
        """
        Get duration in milliseconds.
        
        Returns:
            Duration in milliseconds
        """
        return self.get_duration() * 1000
    
    def get_memory_delta(self) -> float:
        """
        Get memory delta in MB.
        
        Returns:
            Memory increase in MB (0 if not completed)
        """
        if self.memory_start is None or self.memory_end is None:
            return 0.0
        return self.memory_end - self.memory_start
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'name': self.name,
            'duration_seconds': self.get_duration(),
            'duration_ms': self.get_duration_ms(),
            'memory_start_mb': self.memory_start or 0.0,
            'memory_end_mb': self.memory_end or 0.0,
            'memory_delta_mb': self.get_memory_delta(),
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def measure_time(operation: Callable) -> float:
    """
    Measure operation execution time.
    
    Args:
        operation: Callable to execute
    
    Returns:
        Duration in seconds
    
    Example:
        >>> duration = measure_time(lambda: some_function())
        >>> print(f"Took {duration:.3f} seconds")
    """
    start_time = time.perf_counter()
    operation()
    end_time = time.perf_counter()
    return end_time - start_time


def measure_memory(operation: Callable) -> float:
    """
    Measure memory usage increase.
    
    Args:
        operation: Callable to execute
    
    Returns:
        Memory increase in MB
    
    Example:
        >>> memory_delta = measure_memory(lambda: process_large_data())
        >>> print(f"Used {memory_delta:.1f} MB")
    """
    memory_start = get_memory_usage()
    operation()
    memory_end = get_memory_usage()
    return memory_end - memory_start


def profile_function(function: Callable) -> Dict[str, Any]:
    """
    Profile function execution (time and memory).
    
    Args:
        function: Callable to profile
    
    Returns:
        Profile dictionary with metrics
    
    Example:
        >>> profile = profile_function(lambda: compute_metrics(data))
        >>> print(f"Time: {profile['duration_seconds']:.3f}s")
    """
    tracker = PerformanceTracker(name=function.__name__)
    tracker.start()
    
    try:
        result = function()
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
    
    tracker.stop()
    
    stats = tracker.get_stats()
    stats['success'] = success
    stats['error'] = error
    stats['result'] = result
    
    return stats


def time_decorator(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    
    Example:
        >>> @time_decorator
        ... def slow_function():
        ...     time.sleep(1)
        >>> slow_function()  # Prints execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"{func.__name__} took {duration:.4f} seconds")
        return result
    return wrapper


def profile_decorator(func: Callable) -> Callable:
    """
    Decorator to profile function execution.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    
    Example:
        >>> @profile_decorator
        ... def process_data():
        ...     # Heavy processing
        ...     pass
        >>> process_data()  # Prints time and memory stats
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracker = PerformanceTracker(name=func.__name__)
        tracker.start()
        
        try:
            result = func(*args, **kwargs)
        finally:
            tracker.stop()
            stats = tracker.get_stats()
            print(f"{func.__name__}:")
            print(f"  Time: {stats['duration_ms']:.2f} ms")
            print(f"  Memory: {stats['memory_delta_mb']:+.2f} MB")
        
        return result
    return wrapper


class Timer:
    """
    Simple timer for timing code blocks.
    
    Example:
        >>> timer = Timer()
        >>> # ... do some work ...
        >>> elapsed = timer.elapsed()
        >>> print(f"Elapsed: {elapsed:.3f}s")
    """
    
    def __init__(self):
        """Initialize timer."""
        self.start_time = time.perf_counter()
    
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        return time.perf_counter() - self.start_time
    
    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        return self.elapsed() * 1000
    
    def reset(self) -> float:
        """
        Reset timer and return elapsed time.
        
        Returns:
            Elapsed time since last reset
        """
        elapsed = self.elapsed()
        self.start_time = time.perf_counter()
        return elapsed


class Stopwatch:
    """
    Stopwatch with lap timing capability.
    
    Example:
        >>> sw = Stopwatch()
        >>> sw.start()
        >>> # ... work ...
        >>> lap1 = sw.lap()
        >>> # ... more work ...
        >>> lap2 = sw.lap()
        >>> sw.stop()
    """
    
    def __init__(self):
        """Initialize stopwatch."""
        self.start_time: Optional[float] = None
        self.laps: list = []
        self.is_running: bool = False
    
    def start(self) -> None:
        """Start stopwatch."""
        self.start_time = time.perf_counter()
        self.is_running = True
        self.laps = []
    
    def lap(self) -> float:
        """
        Record lap time.
        
        Returns:
            Lap duration in seconds
        """
        if not self.is_running:
            return 0.0
        
        current_time = time.perf_counter()
        if self.laps:
            lap_duration = current_time - self.laps[-1]['time']
        else:
            lap_duration = current_time - self.start_time
        
        self.laps.append({
            'time': current_time,
            'duration': lap_duration,
            'total': current_time - self.start_time
        })
        
        return lap_duration
    
    def stop(self) -> float:
        """
        Stop stopwatch.
        
        Returns:
            Total duration in seconds
        """
        if not self.is_running:
            return 0.0
        
        self.is_running = False
        return time.perf_counter() - self.start_time
    
    def get_total_time(self) -> float:
        """
        Get total elapsed time.
        
        Returns:
            Total time in seconds
        """
        if not self.start_time:
            return 0.0
        
        if self.is_running:
            return time.perf_counter() - self.start_time
        elif self.laps:
            return self.laps[-1]['total']
        else:
            return 0.0
    
    def get_laps(self) -> list:
        """
        Get all lap times.
        
        Returns:
            List of lap dictionaries
        """
        return self.laps.copy()


def benchmark(func: Callable, iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark function with multiple iterations.
    
    Args:
        func: Function to benchmark
        iterations: Number of iterations
    
    Returns:
        Dictionary with benchmark results
    
    Example:
        >>> results = benchmark(lambda: expensive_operation(), iterations=10)
        >>> print(f"Average: {results['avg_ms']:.2f} ms")
    """
    durations = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        durations.append(end_time - start_time)
    
    durations.sort()
    
    return {
        'iterations': iterations,
        'total_seconds': sum(durations),
        'avg_seconds': sum(durations) / iterations,
        'avg_ms': (sum(durations) / iterations) * 1000,
        'min_seconds': min(durations),
        'max_seconds': max(durations),
        'median_seconds': durations[len(durations) // 2],
    }


def format_performance_stats(stats: Dict[str, Any]) -> str:
    """
    Format performance statistics as string.
    
    Args:
        stats: Performance statistics dictionary
    
    Returns:
        Formatted string
    """
    lines = []
    
    if 'name' in stats and stats['name']:
        lines.append(f"Performance: {stats['name']}")
    
    if 'duration_ms' in stats:
        lines.append(f"Time: {stats['duration_ms']:.2f} ms")
    
    if 'memory_delta_mb' in stats:
        delta = stats['memory_delta_mb']
        sign = "+" if delta >= 0 else ""
        lines.append(f"Memory: {sign}{delta:.2f} MB")
    
    if 'success' in stats:
        status = "✓" if stats['success'] else "✗"
        lines.append(f"Status: {status}")
    
    return " | ".join(lines)

