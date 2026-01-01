"""
Result Aggregation Module

This module aggregates metrics and results across samples.

Version: 1.0
Date: December 30, 2025
"""

from typing import List, Dict, Any
import statistics


def aggregate_sample_metrics(samples: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple samples.
    
    Calculates mean, median, std dev, min, max for each metric.
    
    Args:
        samples: List of metric dictionaries
    
    Returns:
        Dictionary with aggregated statistics per metric
    
    Examples:
        >>> samples = [
        ...     {'shannon_entropy': 7.5, 'randomness_score': 0.9},
        ...     {'shannon_entropy': 7.8, 'randomness_score': 0.92},
        ... ]
        >>> result = aggregate_sample_metrics(samples)
        >>> print(result['shannon_entropy']['mean'])
        7.65
    """
    if not samples:
        return {}
    
    # Get all metric names
    metric_names = set()
    for sample in samples:
        metric_names.update(sample.keys())
    
    aggregated = {}
    
    # Aggregate each metric
    for metric_name in metric_names:
        # Collect values for this metric
        values = [sample.get(metric_name, 0.0) for sample in samples if metric_name in sample]
        
        if not values:
            continue
        
        # Calculate statistics
        aggregated[metric_name] = {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'count': len(values),
        }
        
        # Standard deviation (if enough samples)
        if len(values) >= 2:
            aggregated[metric_name]['std'] = statistics.stdev(values)
        else:
            aggregated[metric_name]['std'] = 0.0
        
        # Variance
        if len(values) >= 2:
            aggregated[metric_name]['variance'] = statistics.variance(values)
        else:
            aggregated[metric_name]['variance'] = 0.0
    
    return aggregated


def aggregate_algorithm_results(results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate results per algorithm.
    
    Groups results by algorithm and calculates statistics.
    
    Args:
        results: List of result dictionaries (with 'algorithm' key)
    
    Returns:
        Dictionary mapping algorithm_name -> statistics
    
    Examples:
        >>> results = [
        ...     {'algorithm': 'AES-256-GCM', 'entropy': 7.8},
        ...     {'algorithm': 'AES-256-GCM', 'entropy': 7.9},
        ... ]
        >>> agg = aggregate_algorithm_results(results)
        >>> print(agg['AES-256-GCM']['entropy']['mean'])
        7.85
    """
    if not results:
        return {}
    
    # Group by algorithm
    by_algorithm = {}
    
    for result in results:
        algorithm = result.get('algorithm', 'unknown')
        
        if algorithm not in by_algorithm:
            by_algorithm[algorithm] = []
        
        by_algorithm[algorithm].append(result)
    
    # Aggregate each algorithm
    aggregated = {}
    
    for algorithm, algorithm_results in by_algorithm.items():
        # Extract metrics (exclude 'algorithm' key)
        metrics_only = [
            {k: v for k, v in r.items() if k != 'algorithm' and isinstance(v, (int, float))}
            for r in algorithm_results
        ]
        
        aggregated[algorithm] = aggregate_sample_metrics(metrics_only)
        aggregated[algorithm]['sample_count'] = len(algorithm_results)
    
    return aggregated


def compute_percentiles(values: List[float], percentiles: List[int] = [25, 50, 75, 90, 95, 99]) -> Dict[int, float]:
    """
    Calculate percentiles for a list of values.
    
    Args:
        values: List of numeric values
        percentiles: List of percentile values to calculate (default: [25, 50, 75, 90, 95, 99])
    
    Returns:
        Dictionary mapping percentile -> value
    
    Examples:
        >>> values = list(range(100))
        >>> pct = compute_percentiles(values)
        >>> print(pct[50])  # Median
        49.5
    """
    if not values:
        return {p: 0.0 for p in percentiles}
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    result = {}
    
    for p in percentiles:
        # Calculate index
        index = (p / 100.0) * (n - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, n - 1)
        
        # Interpolate
        fraction = index - lower_index
        value = sorted_values[lower_index] + fraction * (sorted_values[upper_index] - sorted_values[lower_index])
        
        result[p] = value
    
    return result


def summarize_metrics(metrics_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Summarize a single metrics dictionary.
    
    Args:
        metrics_dict: Dictionary of metrics
    
    Returns:
        Summary dictionary with categorization and overall score
    
    Examples:
        >>> metrics = {'shannon_entropy': 7.8, 'randomness_score': 0.9}
        >>> summary = summarize_metrics(metrics)
        >>> print(summary['overall_quality'])
        'excellent'
    """
    summary = {
        'total_metrics': len(metrics_dict),
        'metrics': metrics_dict,
    }
    
    # Categorize by quality
    entropy = metrics_dict.get('shannon_entropy', 0.0)
    randomness = metrics_dict.get('randomness_score', 0.0)
    
    # Overall quality assessment
    if entropy >= 7.5 and randomness >= 0.9:
        summary['overall_quality'] = 'excellent'
    elif entropy >= 7.0 and randomness >= 0.8:
        summary['overall_quality'] = 'good'
    elif entropy >= 6.0 and randomness >= 0.6:
        summary['overall_quality'] = 'fair'
    else:
        summary['overall_quality'] = 'poor'
    
    # Entropy category
    if entropy >= 7.9:
        summary['entropy_category'] = 'very_high'
    elif entropy >= 7.5:
        summary['entropy_category'] = 'high'
    elif entropy >= 7.0:
        summary['entropy_category'] = 'medium'
    else:
        summary['entropy_category'] = 'low'
    
    # Randomness category
    if randomness >= 0.95:
        summary['randomness_category'] = 'excellent'
    elif randomness >= 0.85:
        summary['randomness_category'] = 'good'
    elif randomness >= 0.7:
        summary['randomness_category'] = 'fair'
    else:
        summary['randomness_category'] = 'poor'
    
    return summary


def compare_algorithms(algorithm_stats: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """
    Compare algorithms based on their aggregated statistics.
    
    Args:
        algorithm_stats: Output from aggregate_algorithm_results()
    
    Returns:
        List of algorithm comparisons, sorted by quality
    
    Examples:
        >>> stats = aggregate_algorithm_results(results)
        >>> comparison = compare_algorithms(stats)
        >>> print(comparison[0]['algorithm'])  # Best algorithm
        'AES-256-GCM'
    """
    comparisons = []
    
    for algorithm, stats in algorithm_stats.items():
        comparison = {
            'algorithm': algorithm,
            'sample_count': stats.get('sample_count', 0),
        }
        
        # Extract mean values
        if 'shannon_entropy' in stats:
            comparison['mean_entropy'] = stats['shannon_entropy'].get('mean', 0.0)
        else:
            comparison['mean_entropy'] = 0.0
        
        if 'randomness_score' in stats:
            comparison['mean_randomness'] = stats['randomness_score'].get('mean', 0.0)
        else:
            comparison['mean_randomness'] = 0.0
        
        # Calculate composite score
        comparison['composite_score'] = (
            comparison['mean_entropy'] / 8.0 * 0.5 +
            comparison['mean_randomness'] * 0.5
        )
        
        comparisons.append(comparison)
    
    # Sort by composite score (descending)
    comparisons.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return comparisons


# Export all functions
__all__ = [
    'aggregate_sample_metrics',
    'aggregate_algorithm_results',
    'compute_percentiles',
    'summarize_metrics',
    'compare_algorithms',
]

