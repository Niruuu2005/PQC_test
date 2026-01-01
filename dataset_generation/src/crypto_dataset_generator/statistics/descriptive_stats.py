"""Descriptive Statistics Calculator"""

import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DescriptiveStatistics:
    """Calculate descriptive statistics"""
    
    def __init__(self, attack_data: pd.DataFrame):
        self.attack_data = attack_data
        logger.info("Initialized DescriptiveStatistics")
        
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """Overall summary statistics"""
        return {
            'total_attacks': len(self.attack_data),
            'successful_attacks': int(self.attack_data['attack_success'].sum()),
            'overall_success_rate': float(self.attack_data['attack_success'].mean()),
            'execution_time': {
                'mean_ms': float(self.attack_data['execution_time_ms'].mean()),
                'median_ms': float(self.attack_data['execution_time_ms'].median()),
                'std_dev_ms': float(self.attack_data['execution_time_ms'].std()),
                'min_ms': float(self.attack_data['execution_time_ms'].min()),
                'max_ms': float(self.attack_data['execution_time_ms'].max()),
            }
        }
        
    def calculate_per_algorithm_stats(self) -> pd.DataFrame:
        """Statistics grouped by algorithm"""
        return self.attack_data.groupby('algorithm_name').agg({
            'attack_success': ['count', 'sum', 'mean'],
            'execution_time_ms': ['mean', 'std'],
            'confidence_score': 'mean',
        }).reset_index()

