"""Correlation Analysis"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyze correlations between attacks and algorithms"""
    
    def __init__(self, attack_data: pd.DataFrame):
        self.attack_data = attack_data
        logger.info("Initialized CorrelationAnalyzer")
        
    def calculate_attack_correlations(self) -> pd.DataFrame:
        """Correlation matrix: attacks vs algorithms"""
        # Create pivot table: algorithms as rows, attacks as columns
        pivot = self.attack_data.pivot_table(
            index='algorithm_name',
            columns='attack_name',
            values='attack_success',
            aggfunc='mean'
        )
        
        # Calculate correlation
        return pivot.T  # Transpose to have attacks as rows

