"""Algorithm Comparator - Compare and Rank Algorithms"""

import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class AlgorithmComparator:
    """Compares algorithms and generates rankings"""
    
    def __init__(self, attack_data: pd.DataFrame):
        self.attack_data = attack_data
        logger.info("Initialized AlgorithmComparator")
        
    def rank_all_algorithms(self, metric: str = "security_score") -> pd.DataFrame:
        """Rank all algorithms by specified metric"""
        algo_stats = self.attack_data.groupby('algorithm_name').agg({
            'attack_success': ['count', 'sum', 'mean'],
            'execution_time_ms': 'mean',
        }).reset_index()
        
        algo_stats.columns = ['algorithm', 'total_attacks', 'successful_attacks', 
                              'success_rate', 'avg_attack_time_ms']
        
        # Calculate security score (inverse of success rate)
        algo_stats['security_score'] = (1 - algo_stats['success_rate']) * 100
        algo_stats['attack_resistance'] = 1 - algo_stats['success_rate']
        
        # Rank by security score (descending)
        algo_stats = algo_stats.sort_values('security_score', ascending=False)
        algo_stats.insert(0, 'rank', range(1, len(algo_stats) + 1))
        
        return algo_stats
        
    def compare_two_algorithms(self, algo1: str, algo2: str) -> Dict:
        """Direct comparison of two algorithms"""
        data1 = self.attack_data[self.attack_data['algorithm_name'] == algo1]
        data2 = self.attack_data[self.attack_data['algorithm_name'] == algo2]
        
        return {
            'comparison': f"{algo1} vs {algo2}",
            algo1: {
                'success_rate': float(data1['attack_success'].mean()),
                'avg_time_ms': float(data1['execution_time_ms'].mean()),
            },
            algo2: {
                'success_rate': float(data2['attack_success'].mean()),
                'avg_time_ms': float(data2['execution_time_ms'].mean()),
            }
        }

