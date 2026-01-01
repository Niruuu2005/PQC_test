"""
Resource Monitor - System resource tracking

Monitors CPU, memory, and execution time during attack execution.

Version: 1.0
Date: December 31, 2025
"""

import psutil
import time
import os
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time"""
    timestamp: float
    memory_mb: float
    cpu_percent: float
    cpu_count: int
    memory_available_mb: float
    memory_percent: float


@dataclass
class ResourceStats:
    """Aggregated resource statistics"""
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    total_time_seconds: float = 0.0
    snapshots: list = field(default_factory=list)


class ResourceMonitor:
    """
    Monitor system resource usage.
    
    Tracks:
    - Memory usage (process and system)
    - CPU utilization
    - Execution time
    """
    
    def __init__(self, interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.snapshots = []
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.snapshots = []
        self._take_snapshot()
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resources"""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent(interval=None)
            
            vm = psutil.virtual_memory()
            
            snapshot = ResourceSnapshot(
                timestamp=time.time(),
                memory_mb=memory_info.rss / 1024 / 1024,
                cpu_percent=cpu_percent,
                cpu_count=psutil.cpu_count(),
                memory_available_mb=vm.available / 1024 / 1024,
                memory_percent=vm.percent,
            )
            
            self.snapshots.append(snapshot)
            return snapshot
            
        except Exception as e:
            # Return dummy snapshot on error
            return ResourceSnapshot(
                timestamp=time.time(),
                memory_mb=0.0,
                cpu_percent=0.0,
                cpu_count=1,
                memory_available_mb=0.0,
                memory_percent=0.0,
            )
    
    def update(self):
        """Take a new snapshot"""
        self._take_snapshot()
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current resource statistics"""
        if not self.snapshots:
            return {
                'memory_mb': 0.0,
                'cpu_percent': 0.0,
                'elapsed_seconds': 0.0,
            }
        
        current = self._take_snapshot()
        elapsed = current.timestamp - self.start_time if self.start_time else 0.0
        
        return {
            'memory_mb': current.memory_mb,
            'cpu_percent': current.cpu_percent,
            'elapsed_seconds': elapsed,
            'memory_available_mb': current.memory_available_mb,
            'memory_percent': current.memory_percent,
        }
    
    def get_statistics(self) -> ResourceStats:
        """Get aggregated resource statistics"""
        if not self.snapshots:
            return ResourceStats()
        
        memory_values = [s.memory_mb for s in self.snapshots]
        cpu_values = [s.cpu_percent for s in self.snapshots]
        
        elapsed = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        
        return ResourceStats(
            peak_memory_mb=max(memory_values) if memory_values else 0.0,
            avg_memory_mb=sum(memory_values) / len(memory_values) if memory_values else 0.0,
            peak_cpu_percent=max(cpu_values) if cpu_values else 0.0,
            avg_cpu_percent=sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
            total_time_seconds=elapsed,
            snapshots=self.snapshots,
        )
    
    def stop(self) -> ResourceStats:
        """Stop monitoring and return statistics"""
        self._take_snapshot()
        return self.get_statistics()


__all__ = ['ResourceMonitor', 'ResourceSnapshot', 'ResourceStats']

