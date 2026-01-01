"""
Result Aggregator - Statistics and analysis

Aggregates attack results and generates comprehensive statistics.

Version: 1.0
Date: December 31, 2025
"""

import logging
from typing import List, Dict, Any
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregate attack results and compute statistics.
    
    Tracks:
    - Success rates by attack and category
    - Performance metrics
    - Vulnerability detection rates
    - Language performance comparison
    """
    
    def __init__(self):
        """Initialize aggregator"""
        self.results = []
        self.stats = defaultdict(lambda: defaultdict(int))
    
    def add_result(self, result: Any):
        """Add a single attack result"""
        self.results.append(result)
        self._update_stats(result)
    
    def add_results(self, results: List[Any]):
        """Add multiple attack results"""
        for result in results:
            self.add_result(result)
    
    def _update_stats(self, result: Any):
        """Update running statistics"""
        attack_name = getattr(result, 'attack_name', 'Unknown')
        success = getattr(result, 'success', False)
        language = getattr(result, 'attack_language', 'Python')
        
        self.stats['total']['count'] += 1
        
        if success:
            self.stats['total']['successes'] += 1
            self.stats['by_attack'][attack_name] += 1
            self.stats['by_language'][language] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics.
        
        Returns:
            Dictionary with aggregated statistics
        """
        if not self.results:
            return {
                'total_attacks': 0,
                'successful_attacks': 0,
                'success_rate_percent': 0.0,
            }
        
        total = self.stats['total']['count']
        successes = self.stats['total']['successes']
        
        # Success rate
        success_rate = (successes / total * 100) if total > 0 else 0.0
        
        # By attack
        attack_success_rates = {}
        attack_counts = Counter()
        for result in self.results:
            attack_name = getattr(result, 'attack_name', 'Unknown')
            attack_counts[attack_name] += 1
        
        attack_successes = self.stats['by_attack']
        for attack_name, count in attack_counts.items():
            successes_for_attack = attack_successes.get(attack_name, 0)
            attack_success_rates[attack_name] = (successes_for_attack / count * 100) if count > 0 else 0.0
        
        # By language
        language_stats = {}
        language_counts = Counter()
        for result in self.results:
            language = getattr(result, 'attack_language', 'Python')
            language_counts[language] += 1
        
        for language, count in language_counts.items():
            successes_for_lang = self.stats['by_language'].get(language, 0)
            language_stats[language] = {
                'total': count,
                'successes': successes_for_lang,
                'success_rate': (successes_for_lang / count * 100) if count > 0 else 0.0,
            }
        
        # Performance metrics
        execution_times = [getattr(r, 'time_taken', 0.0) for r in self.results]
        memory_usage = [getattr(r, 'memory_used_mb', 0.0) for r in self.results]
        
        # Vulnerability detection
        vulnerabilities_detected = sum(
            1 for r in self.results
            if getattr(r, 'vulnerability_detected', False)
        )
        
        return {
            'total_attacks': total,
            'successful_attacks': successes,
            'failed_attacks': total - successes,
            'success_rate_percent': success_rate,
            
            'attack_success_rates': attack_success_rates,
            'language_statistics': language_stats,
            
            'performance': {
                'total_execution_time_seconds': sum(execution_times),
                'avg_execution_time_seconds': sum(execution_times) / len(execution_times) if execution_times else 0.0,
                'max_execution_time_seconds': max(execution_times) if execution_times else 0.0,
                'avg_memory_mb': sum(memory_usage) / len(memory_usage) if memory_usage else 0.0,
                'max_memory_mb': max(memory_usage) if memory_usage else 0.0,
            },
            
            'vulnerabilities': {
                'total_detected': vulnerabilities_detected,
                'detection_rate_percent': (vulnerabilities_detected / total * 100) if total > 0 else 0.0,
            },
            
            'top_successful_attacks': sorted(
                attack_success_rates.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
        }
    
    def print_summary(self):
        """Print statistics summary to logger"""
        stats = self.get_statistics()
        
        logger.info("=" * 80)
        logger.info("ATTACK RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Attacks: {stats['total_attacks']}")
        logger.info(f"Successful: {stats['successful_attacks']}")
        logger.info(f"Failed: {stats['failed_attacks']}")
        logger.info(f"Success Rate: {stats['success_rate_percent']:.2f}%")
        logger.info("")
        
        logger.info("Language Statistics:")
        for lang, lang_stats in stats['language_statistics'].items():
            logger.info(f"  {lang}: {lang_stats['successes']}/{lang_stats['total']} "
                       f"({lang_stats['success_rate']:.2f}%)")
        logger.info("")
        
        logger.info("Top 10 Most Successful Attacks:")
        for attack, rate in stats['top_successful_attacks']:
            logger.info(f"  {attack}: {rate:.2f}%")
        logger.info("")
        
        logger.info("Performance:")
        perf = stats['performance']
        logger.info(f"  Total Time: {perf['total_execution_time_seconds']:.2f}s")
        logger.info(f"  Avg Time/Attack: {perf['avg_execution_time_seconds']:.4f}s")
        logger.info(f"  Avg Memory: {perf['avg_memory_mb']:.2f} MB")
        logger.info("")
        
        logger.info("Vulnerabilities:")
        vuln = stats['vulnerabilities']
        logger.info(f"  Detected: {vuln['total_detected']}")
        logger.info(f"  Detection Rate: {vuln['detection_rate_percent']:.2f}%")
        logger.info("=" * 80)


__all__ = ['ResultAggregator']

