"""Weakness Detector - Identify Common Patterns"""

import pandas as pd
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class WeaknessDetector:
    """Detects common weakness patterns across algorithms"""
    
    def __init__(self, attack_data: pd.DataFrame):
        self.attack_data = attack_data
        logger.info("Initialized WeaknessDetector")
        
    def detect_common_weaknesses(self) -> List[Dict]:
        """Find weaknesses shared across multiple algorithms"""
        # Group by attack to find attacks that succeed frequently
        attack_success = self.attack_data.groupby('attack_name').agg({
            'attack_success': ['sum', 'count', 'mean']
        }).reset_index()
        
        attack_success.columns = ['attack', 'successes', 'total', 'success_rate']
        common_weaknesses = attack_success[attack_success['success_rate'] > 0.5]
        
        return common_weaknesses.to_dict('records')
        
    def generate_weakness_report(self) -> Dict:
        """Comprehensive weakness analysis"""
        return {
            'common_weaknesses': self.detect_common_weaknesses(),
            'total_patterns': len(self.detect_common_weaknesses()),
        }

