"""
Attack Factory and Registry

Factory pattern for creating attack instances and managing attack registry.

Version: 1.0
Date: December 30, 2025
"""

from typing import Dict, List, Optional, Type, Tuple
import logging

from .base_attack import BaseAttack, AttackResult

logger = logging.getLogger(__name__)


# Global attack registry
ATTACK_REGISTRY: Dict[str, Dict] = {}


def register_attack(attack_class: Type[BaseAttack], 
                   category: str,
                   description: str = "",
                   applicable_to: List[str] = None) -> None:
    """
    Register an attack in the global registry.
    
    Args:
        attack_class: Attack class to register
        category: Attack category (e.g., 'brute_force', 'statistical')
        description: Attack description
        applicable_to: List of algorithm patterns this attack applies to
    
    Examples:
        >>> register_attack(BruteForceKeySearch, 'brute_force', 
        ...                 'Exhaustive key search', applicable_to=['*'])
    """
    attack_name = attack_class.__name__
    
    if attack_name in ATTACK_REGISTRY:
        logger.warning(f"Attack {attack_name} already registered, overwriting")
    
    ATTACK_REGISTRY[attack_name] = {
        'class': attack_class,
        'category': category,
        'description': description,
        'applicable_to': applicable_to or ['*'],
    }
    
    logger.debug(f"Registered attack: {attack_name} (category: {category})")


def create_attack(attack_name: str, target_algorithm: str) -> BaseAttack:
    """
    Create an attack instance using the factory.
    
    Args:
        attack_name: Name of the attack to create
        target_algorithm: Target algorithm name
    
    Returns:
        Attack instance
    
    Raises:
        ValueError: If attack name not found in registry
    
    Examples:
        >>> attack = create_attack('BruteForceKeySearch', 'AES-128-CBC')
        >>> isinstance(attack, BaseAttack)
        True
    """
    # Normalize attack name
    attack_name_normalized = attack_name.replace('-', '').replace('_', '').upper()
    
    # Try exact match first
    if attack_name in ATTACK_REGISTRY:
        attack_info = ATTACK_REGISTRY[attack_name]
        return attack_info['class'](target_algorithm)
    
    # Try normalized match
    for registered_name, attack_info in ATTACK_REGISTRY.items():
        if registered_name.replace('-', '').replace('_', '').upper() == attack_name_normalized:
            return attack_info['class'](target_algorithm)
    
    # Not found
    available = ', '.join(list(ATTACK_REGISTRY.keys())[:10])
    raise ValueError(
        f"Unknown attack: {attack_name}. "
        f"Available attacks: {available}... ({len(ATTACK_REGISTRY)} total)"
    )


def get_available_attacks() -> List[str]:
    """
    Get list of all available attack names.
    
    Returns:
        List of attack names
    
    Examples:
        >>> attacks = get_available_attacks()
        >>> 'BruteForceKeySearch' in attacks
        True
    """
    return sorted(ATTACK_REGISTRY.keys())


def get_attacks_by_category(category: str) -> List[str]:
    """
    Get attacks filtered by category.
    
    Args:
        category: Category name (e.g., 'brute_force', 'statistical')
    
    Returns:
        List of attack names in that category
    
    Examples:
        >>> attacks = get_attacks_by_category('brute_force')
        >>> len(attacks)
        8
    """
    return [
        name for name, info in ATTACK_REGISTRY.items()
        if info['category'] == category
    ]


def get_applicable_attacks(algorithm: str) -> List[str]:
    """
    Get attacks applicable to a specific algorithm.
    
    Args:
        algorithm: Algorithm name
    
    Returns:
        List of applicable attack names
    
    Examples:
        >>> attacks = get_applicable_attacks('AES-256-GCM')
        >>> 'BruteForceKeySearch' in attacks
        True
    """
    applicable = []
    
    for attack_name, attack_info in ATTACK_REGISTRY.items():
        patterns = attack_info['applicable_to']
        
        # Check if attack applies to this algorithm
        if '*' in patterns:
            applicable.append(attack_name)
        elif any(pattern in algorithm for pattern in patterns):
            applicable.append(attack_name)
    
    return applicable


def get_attack_metadata(attack_name: str) -> Dict:
    """
    Get metadata for a specific attack.
    
    Args:
        attack_name: Attack name
    
    Returns:
        Attack metadata dictionary
    
    Raises:
        ValueError: If attack not found
    
    Examples:
        >>> metadata = get_attack_metadata('BruteForceKeySearch')
        >>> metadata['category']
        'brute_force'
    """
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Attack not found: {attack_name}")
    
    info = ATTACK_REGISTRY[attack_name].copy()
    # Remove class object from metadata
    info.pop('class', None)
    
    return info


def get_attack_categories() -> List[str]:
    """
    Get all attack categories.
    
    Returns:
        List of unique category names
    
    Examples:
        >>> categories = get_attack_categories()
        >>> 'brute_force' in categories
        True
    """
    categories = set()
    for info in ATTACK_REGISTRY.values():
        categories.add(info['category'])
    
    return sorted(categories)


def get_attack_statistics() -> Dict[str, int]:
    """
    Get statistics about registered attacks.
    
    Returns:
        Dictionary with attack counts per category
    
    Examples:
        >>> stats = get_attack_statistics()
        >>> stats['total']
        83
    """
    stats = {'total': len(ATTACK_REGISTRY)}
    
    for category in get_attack_categories():
        count = len(get_attacks_by_category(category))
        stats[category] = count
    
    return stats


def execute_attack(attack_name: str, target_algorithm: str, 
                  ciphertext: bytes, **kwargs) -> AttackResult:
    """
    Convenience function to create and execute an attack in one call.
    
    Args:
        attack_name: Name of attack to execute
        target_algorithm: Target algorithm
        ciphertext: Ciphertext to attack
        **kwargs: Additional attack parameters
    
    Returns:
        AttackResult
    
    Examples:
        >>> result = execute_attack('FrequencyAnalysis', 'AES-128-CBC', 
        ...                         ciphertext, sample_size=1000)
        >>> result.success
        False
    """
    attack = create_attack(attack_name, target_algorithm)
    return attack.execute(ciphertext, **kwargs)


def execute_multiple_attacks(attack_names: List[str], target_algorithm: str,
                            ciphertext: bytes, **kwargs) -> Dict[str, AttackResult]:
    """
    Execute multiple attacks and return results.
    
    Args:
        attack_names: List of attack names to execute
        target_algorithm: Target algorithm
        ciphertext: Ciphertext to attack
        **kwargs: Additional parameters for all attacks
    
    Returns:
        Dictionary mapping attack_name -> AttackResult
    
    Examples:
        >>> results = execute_multiple_attacks(
        ...     ['FrequencyAnalysis', 'EntropyAnalysis'],
        ...     'Weak-Cipher',
        ...     ciphertext
        ... )
        >>> len(results)
        2
    """
    results = {}
    
    for attack_name in attack_names:
        try:
            result = execute_attack(attack_name, target_algorithm, ciphertext, **kwargs)
            results[attack_name] = result
        except Exception as e:
            logger.error(f"Error executing {attack_name}: {e}")
            # Create failed result
            results[attack_name] = AttackResult(
                attack_name=attack_name,
                target_algorithm=target_algorithm,
                success=False,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    return results


def execute_all_applicable_attacks(target_algorithm: str, ciphertext: bytes,
                                  **kwargs) -> Dict[str, AttackResult]:
    """
    Execute all applicable attacks for an algorithm.
    
    Args:
        target_algorithm: Target algorithm
        ciphertext: Ciphertext to attack
        **kwargs: Additional parameters
    
    Returns:
        Dictionary mapping attack_name -> AttackResult
    
    Examples:
        >>> results = execute_all_applicable_attacks('AES-128-CBC', ciphertext)
        >>> len(results) > 0
        True
    """
    applicable = get_applicable_attacks(target_algorithm)
    return execute_multiple_attacks(applicable, target_algorithm, ciphertext, **kwargs)


# Export all functions
__all__ = [
    'ATTACK_REGISTRY',
    'register_attack',
    'create_attack',
    'get_available_attacks',
    'get_attacks_by_category',
    'get_applicable_attacks',
    'get_attack_metadata',
    'get_attack_categories',
    'get_attack_statistics',
    'execute_attack',
    'execute_multiple_attacks',
    'execute_all_applicable_attacks',
]

