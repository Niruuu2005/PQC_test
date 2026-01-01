"""Attack Success Analyzer - Calculate Attack Effectiveness"""

import pandas as pd
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class AttackSuccessAnalyzer:
    """Analyzer for calculating attack success rates and effectiveness"""
    
    def __init__(self, attack_data: pd.DataFrame):
        self.attack_data = attack_data
        logger.info(f"Initialized AttackSuccessAnalyzer with {len(attack_data)} records")
        
    def calculate_success_rates(self) -> pd.DataFrame:
        """Calculate success rates per attack per algorithm"""
        grouped = self.attack_data.groupby(['algorithm_name', 'attack_name']).agg({
            'attack_success': ['sum', 'count', 'mean']
        }).reset_index()
        
        grouped.columns = ['algorithm', 'attack', 'successful', 'total', 'success_rate']
        return grouped
        
    def analyze_by_category(self) -> Dict[str, Dict]:
        """Group analysis by attack category"""
        results = {}
        
        for category in self.attack_data['attack_category'].unique():
            cat_data = self.attack_data[self.attack_data['attack_category'] == category]
            results[category] = {
                'executed': len(cat_data),
                'successful': int(cat_data['attack_success'].sum()),
                'success_rate': float(cat_data['attack_success'].mean()),
                'avg_time_ms': float(cat_data['execution_time_ms'].mean()),
            }
            
        return results
        
    def identify_most_effective_attacks(self, top_n: int = 10) -> List[Dict]:
        """Find most effective attacks overall"""
        attack_stats = self.attack_data.groupby('attack_name').agg({
            'attack_success': 'mean',
            'confidence_score': 'mean',
        }).reset_index()
        
        attack_stats = attack_stats.sort_values('attack_success', ascending=False).head(top_n)
        
        return attack_stats.to_dict('records')

