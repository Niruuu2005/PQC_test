"""
Report Generation Module

This module generates various report formats for analysis results.

Version: 1.0
Date: December 30, 2025
"""

import json
from typing import Dict, Any
from datetime import datetime


def generate_json_summary(statistics: Dict) -> str:
    """
    Generate JSON summary report.
    
    Args:
        statistics: Dictionary of aggregated statistics
    
    Returns:
        JSON-formatted string
    
    Examples:
        >>> json_report = generate_json_summary(stats)
        >>> print(json_report[:100])
    """
    # Add metadata
    report = {
        'report_type': 'cryptographic_analysis_summary',
        'generated_at': datetime.now().isoformat(),
        'statistics': statistics,
    }
    
    return json.dumps(report, indent=2)


def generate_text_report(statistics: Dict) -> str:
    """
    Generate plain text report.
    
    Args:
        statistics: Dictionary of aggregated statistics
    
    Returns:
        Formatted text report
    
    Examples:
        >>> text_report = generate_text_report(stats)
        >>> print(text_report)
    """
    lines = []
    lines.append("=" * 70)
    lines.append("CRYPTOGRAPHIC ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Overall statistics
    if 'overall' in statistics:
        lines.append("OVERALL STATISTICS")
        lines.append("-" * 70)
        overall = statistics['overall']
        for key, value in overall.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")
    
    # Per-metric statistics
    if 'metrics' in statistics:
        lines.append("PER-METRIC STATISTICS")
        lines.append("-" * 70)
        metrics = statistics['metrics']
        for metric_name, metric_stats in metrics.items():
            lines.append(f"\n{metric_name}:")
            if isinstance(metric_stats, dict):
                for stat_name, stat_value in metric_stats.items():
                    if isinstance(stat_value, float):
                        lines.append(f"  {stat_name}: {stat_value:.4f}")
                    else:
                        lines.append(f"  {stat_name}: {stat_value}")
            else:
                lines.append(f"  value: {metric_stats}")
        lines.append("")
    
    # Algorithm comparisons
    if 'algorithms' in statistics:
        lines.append("ALGORITHM COMPARISON")
        lines.append("-" * 70)
        algorithms = statistics['algorithms']
        
        # Sort by quality if possible
        if isinstance(algorithms, list):
            for algo in algorithms:
                lines.append(f"\n{algo.get('algorithm', 'unknown')}:")
                for key, value in algo.items():
                    if key != 'algorithm':
                        if isinstance(value, float):
                            lines.append(f"  {key}: {value:.4f}")
                        else:
                            lines.append(f"  {key}: {value}")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def generate_markdown_report(statistics: Dict) -> str:
    """
    Generate Markdown report.
    
    Args:
        statistics: Dictionary of aggregated statistics
    
    Returns:
        Markdown-formatted report
    
    Examples:
        >>> md_report = generate_markdown_report(stats)
        >>> print(md_report)
    """
    lines = []
    lines.append("# Cryptographic Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Overall statistics
    if 'overall' in statistics:
        lines.append("## Overall Statistics")
        lines.append("")
        overall = statistics['overall']
        for key, value in overall.items():
            if isinstance(value, float):
                lines.append(f"- **{key}**: {value:.4f}")
            else:
                lines.append(f"- **{key}**: {value}")
        lines.append("")
    
    # Per-metric statistics
    if 'metrics' in statistics:
        lines.append("## Per-Metric Statistics")
        lines.append("")
        metrics = statistics['metrics']
        for metric_name, metric_stats in metrics.items():
            lines.append(f"### {metric_name}")
            lines.append("")
            if isinstance(metric_stats, dict):
                for stat_name, stat_value in metric_stats.items():
                    if isinstance(stat_value, float):
                        lines.append(f"- **{stat_name}**: {stat_value:.4f}")
                    else:
                        lines.append(f"- **{stat_name}**: {stat_value}")
            else:
                lines.append(f"- **value**: {metric_stats}")
            lines.append("")
    
    # Algorithm comparisons
    if 'algorithms' in statistics:
        lines.append("## Algorithm Comparison")
        lines.append("")
        algorithms = statistics['algorithms']
        
        # Create table if algorithms is a list
        if isinstance(algorithms, list) and algorithms:
            # Table header
            lines.append("| Algorithm | Samples | Entropy | Randomness | Score |")
            lines.append("|-----------|---------|---------|------------|-------|")
            
            # Table rows
            for algo in algorithms:
                algorithm_name = algo.get('algorithm', 'unknown')
                sample_count = algo.get('sample_count', 0)
                entropy = algo.get('mean_entropy', 0.0)
                randomness = algo.get('mean_randomness', 0.0)
                score = algo.get('composite_score', 0.0)
                
                lines.append(f"| {algorithm_name} | {sample_count} | {entropy:.4f} | {randomness:.4f} | {score:.4f} |")
            
            lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*End of Report*")
    
    return "\n".join(lines)


def generate_executive_summary(statistics: Dict) -> str:
    """
    Generate high-level executive summary.
    
    Args:
        statistics: Dictionary of aggregated statistics
    
    Returns:
        Executive summary text
    
    Examples:
        >>> summary = generate_executive_summary(stats)
        >>> print(summary)
    """
    lines = []
    lines.append("EXECUTIVE SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    
    # Key findings
    lines.append("KEY FINDINGS:")
    lines.append("")
    
    # Overall quality assessment
    if 'overall' in statistics:
        overall = statistics['overall']
        
        # Extract key metrics
        if 'mean_entropy' in overall:
            entropy = overall['mean_entropy']
            lines.append(f"- Average Entropy: {entropy:.2f} bits/byte")
            
            if entropy >= 7.5:
                lines.append("  Status: EXCELLENT - Very high entropy indicates strong encryption")
            elif entropy >= 7.0:
                lines.append("  Status: GOOD - High entropy suggests good encryption quality")
            else:
                lines.append("  Status: NEEDS IMPROVEMENT - Low entropy may indicate weakness")
        
        if 'mean_randomness' in overall:
            randomness = overall['mean_randomness']
            lines.append(f"- Average Randomness Score: {randomness:.2f}")
            
            if randomness >= 0.9:
                lines.append("  Status: EXCELLENT - Ciphertext exhibits strong randomness")
            elif randomness >= 0.8:
                lines.append("  Status: GOOD - Acceptable randomness properties")
            else:
                lines.append("  Status: NEEDS IMPROVEMENT - Randomness below recommended threshold")
    
    lines.append("")
    
    # Recommendations
    lines.append("RECOMMENDATIONS:")
    lines.append("")
    
    if 'algorithms' in statistics and isinstance(statistics['algorithms'], list):
        algorithms = statistics['algorithms']
        
        if algorithms:
            # Best algorithm
            best = algorithms[0]
            lines.append(f"- Recommended Algorithm: {best.get('algorithm', 'unknown')}")
            lines.append(f"  Score: {best.get('composite_score', 0.0):.4f}")
            
            # Worst algorithm (if multiple)
            if len(algorithms) > 1:
                worst = algorithms[-1]
                lines.append(f"- Needs Review: {worst.get('algorithm', 'unknown')}")
                lines.append(f"  Score: {worst.get('composite_score', 0.0):.4f}")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# Export all functions
__all__ = [
    'generate_json_summary',
    'generate_text_report',
    'generate_markdown_report',
    'generate_executive_summary',
]

