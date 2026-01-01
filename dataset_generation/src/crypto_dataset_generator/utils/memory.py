"""
Memory Management
=================

Memory monitoring and management utilities.

This module provides:
- MemoryMonitor class for real-time memory tracking
- Functions for getting memory usage
- Memory limit checking
- Garbage collection triggers

Author: AIRAWAT Development Team
Version: 1.0
Date: December 30, 2025
"""

import gc
import sys
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    import warnings
    warnings.warn("psutil not available, memory monitoring will be limited")


class MemoryMonitor:
    """
    Real-time memory tracking.
    
    Attributes:
        peak_memory_mb: Peak memory usage in MB
        current_memory_mb: Current memory usage in MB
    """
    
    def __init__(self):
        """Initialize memory monitor."""
        self.peak_memory_mb: float = 0.0
        self.current_memory_mb: float = 0.0
        self._baseline_mb: float = get_memory_usage()
    
    def get_current_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Current memory usage in MB
        """
        self.current_memory_mb = get_memory_usage()
        
        # Update peak if current exceeds it
        if self.current_memory_mb > self.peak_memory_mb:
            self.peak_memory_mb = self.current_memory_mb
        
        return self.current_memory_mb
    
    def get_peak_usage(self) -> float:
        """
        Get peak memory usage in MB.
        
        Returns:
            Peak memory usage in MB
        """
        # Update current first to ensure peak is accurate
        self.get_current_usage()
        return self.peak_memory_mb
    
    def check_limit(self, limit_mb: int) -> bool:
        """
        Check if memory usage exceeds limit.
        
        Args:
            limit_mb: Memory limit in MB
        
        Returns:
            True if limit exceeded, False otherwise
        """
        current = self.get_current_usage()
        return current > limit_mb
    
    def reset_peak(self) -> None:
        """Reset peak memory usage to current."""
        self.get_current_usage()
        self.peak_memory_mb = self.current_memory_mb
    
    def get_delta_from_baseline(self) -> float:
        """
        Get memory delta from baseline.
        
        Returns:
            Memory increase in MB since initialization
        """
        current = self.get_current_usage()
        return current - self._baseline_mb


def get_memory_usage() -> float:
    """
    Get current process memory usage in MB.
    
    Returns:
        Memory usage in MB
    
    Example:
        >>> memory_mb = get_memory_usage()
        >>> print(f"Using {memory_mb:.2f} MB")
    """
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            # Return RSS (Resident Set Size) in MB
            return mem_info.rss / (1024 * 1024)
        except Exception:
            pass
    
    # Fallback: use sys module (less accurate)
    try:
        # This gives object count and size, not actual memory
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in kilobytes on Unix, bytes on Windows
        if sys.platform == 'darwin' or sys.platform.startswith('linux'):
            return usage.ru_maxrss / 1024  # KB to MB
        else:
            return usage.ru_maxrss / (1024 * 1024)  # Bytes to MB
    except (ImportError, AttributeError):
        # Last resort: estimate from object count
        return len(gc.get_objects()) * 0.001  # Rough estimate


def get_memory_usage_percent() -> float:
    """
    Get memory usage as percentage of system memory.
    
    Returns:
        Percentage [0.0, 100.0]
    
    Example:
        >>> percent = get_memory_usage_percent()
        >>> print(f"Using {percent:.1f}% of system memory")
    """
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            mem_percent = process.memory_percent()
            return mem_percent
        except Exception:
            pass
    
    # Fallback: cannot determine without psutil
    return 0.0


def check_memory_limit(limit_mb: int) -> bool:
    """
    Check if memory usage exceeds limit.
    
    Args:
        limit_mb: Memory limit in MB
    
    Returns:
        True if limit exceeded, False otherwise
    
    Example:
        >>> if check_memory_limit(1000):
        ...     print("Memory limit exceeded!")
    """
    current_mb = get_memory_usage()
    return current_mb > limit_mb


def get_peak_memory_usage() -> float:
    """
    Get peak memory usage since process start.
    
    Returns:
        Peak memory usage in MB
    
    Note:
        This requires psutil. Without it, returns current usage.
    
    Example:
        >>> peak_mb = get_peak_memory_usage()
        >>> print(f"Peak memory: {peak_mb:.2f} MB")
    """
    if PSUTIL_AVAILABLE:
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == 'darwin' or sys.platform.startswith('linux'):
                return usage.ru_maxrss / 1024  # KB to MB
            else:
                return usage.ru_maxrss / (1024 * 1024)  # Bytes to MB
        except (ImportError, AttributeError):
            pass
    
    # Fallback: return current usage
    return get_memory_usage()


def trigger_garbage_collection() -> int:
    """
    Force garbage collection.
    
    Returns:
        Number of objects collected
    
    Example:
        >>> collected = trigger_garbage_collection()
        >>> print(f"Collected {collected} objects")
    """
    collected = gc.collect()
    return collected


def get_memory_info() -> dict:
    """
    Get detailed memory information.
    
    Returns:
        Dictionary with memory statistics
    
    Example:
        >>> info = get_memory_info()
        >>> print(f"RSS: {info['rss_mb']:.2f} MB")
    """
    info = {
        'rss_mb': 0.0,
        'vms_mb': 0.0,
        'percent': 0.0,
        'available_mb': 0.0,
        'total_mb': 0.0,
    }
    
    if PSUTIL_AVAILABLE:
        try:
            # Process memory
            process = psutil.Process()
            mem_info = process.memory_info()
            info['rss_mb'] = mem_info.rss / (1024 * 1024)
            info['vms_mb'] = mem_info.vms / (1024 * 1024)
            info['percent'] = process.memory_percent()
            
            # System memory
            sys_mem = psutil.virtual_memory()
            info['available_mb'] = sys_mem.available / (1024 * 1024)
            info['total_mb'] = sys_mem.total / (1024 * 1024)
        except Exception:
            pass
    else:
        # Fallback
        info['rss_mb'] = get_memory_usage()
    
    return info


def format_memory_info(info: Optional[dict] = None) -> str:
    """
    Format memory information as string.
    
    Args:
        info: Memory info dictionary (default: get_memory_info())
    
    Returns:
        Formatted string
    
    Example:
        >>> print(format_memory_info())
        Memory Usage: 125.3 MB (RSS), 0.8% of system
    """
    if info is None:
        info = get_memory_info()
    
    lines = []
    lines.append(f"RSS: {info['rss_mb']:.1f} MB")
    
    if info['vms_mb'] > 0:
        lines.append(f"VMS: {info['vms_mb']:.1f} MB")
    
    if info['percent'] > 0:
        lines.append(f"System: {info['percent']:.1f}%")
    
    if info['available_mb'] > 0:
        lines.append(f"Available: {info['available_mb']:.1f} MB")
    
    return " | ".join(lines)


class MemoryLimitExceededError(Exception):
    """Exception raised when memory limit is exceeded."""
    pass


def enforce_memory_limit(limit_mb: int) -> None:
    """
    Enforce memory limit, raise exception if exceeded.
    
    Args:
        limit_mb: Memory limit in MB
    
    Raises:
        MemoryLimitExceededError: If limit exceeded
    
    Example:
        >>> enforce_memory_limit(1000)  # Raises if using >1000 MB
    """
    current_mb = get_memory_usage()
    if current_mb > limit_mb:
        raise MemoryLimitExceededError(
            f"Memory limit exceeded: {current_mb:.1f} MB > {limit_mb} MB"
        )

