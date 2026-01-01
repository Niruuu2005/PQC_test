"""Ranking Generator"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RankingGenerator:
    """Generate algorithm security rankings"""
    
    def __init__(self, comparison_data: pd.DataFrame):
        self.comparison_data = comparison_data
        logger.info("Initialized RankingGenerator")
        
    def generate_overall_rankings(self) -> pd.DataFrame:
        """Rank all algorithms by security score"""
        # Ensure sorted by security score (already done in comparator)
        return self.comparison_data
        
    def export_rankings_csv(self, output_path: str):
        """Export rankings to CSV"""
        rankings = self.generate_overall_rankings()
        rankings.to_csv(output_path, index=False)
        logger.info(f"Rankings exported to {output_path}")

