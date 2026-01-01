"""Distribution Analysis"""

import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DistributionAnalyzer:
    """Analyze distributions of success rates and metrics"""
    
    def __init__(self, attack_data: pd.DataFrame):
        self.attack_data = attack_data
        logger.info("Initialized DistributionAnalyzer")
        
    def analyze_success_rate_distribution(self) -> Dict[str, Any]:
        """Distribution of success rates"""
        success_rates = self.attack_data['attack_success']
        
        return {
            'mean': float(success_rates.mean()),
            'median': float(success_rates.median()),
            'std_dev': float(success_rates.std()),
            'min': float(success_rates.min()),
            'max': float(success_rates.max()),
        }

