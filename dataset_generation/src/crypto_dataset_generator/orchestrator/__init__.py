"""
Attack Orchestration System

Coordinates execution of 90 attacks across multiple languages (Python, C++, Rust)
with comprehensive resource monitoring and dataset generation.

Version: 1.0
Date: December 31, 2025
"""

from .attack_orchestrator import AttackOrchestrator
from .language_bridge import LanguageBridge
from .resource_monitor import ResourceMonitor
from .result_aggregator import ResultAggregator

__all__ = [
    'AttackOrchestrator',
    'LanguageBridge',
    'ResourceMonitor',
    'ResultAggregator',
]

__version__ = '1.0.0'

