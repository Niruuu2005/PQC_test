"""
Cryptographic Dataset Generator - Attacks Subsystem

This module implements 77+ cryptanalytic attacks across 9 categories.

Version: 1.0
Date: December 30, 2025
"""

__version__ = "1.0.0"

# Import base classes
from .base_attack import (
    BaseAttack, AttackResult,
    BruteForceAttackBase, StatisticalAttackBase,
    CryptanalysisAttackBase, AlgebraicAttackBase,
    SideChannelAttackBase, QuantumAttackBase,
    LatticeAttackBase, HashCollisionAttackBase,
    ImplementationFlawAttackBase
)

# Import factory
from .attack_factory import (
    register_attack, create_attack,
    get_available_attacks, get_attacks_by_category,
    get_applicable_attacks, get_attack_statistics,
    execute_attack
)

# Import all attack implementations
from . import brute_force
from . import statistical
from . import advanced_attacks

# Auto-register all attacks
def _register_all_attacks():
    """Auto-register all implemented attacks."""
    # Brute-force attacks
    for attack_class in [
        brute_force.ExhaustiveKeySearch, brute_force.DictionaryAttack,
        brute_force.RainbowTableAttack, brute_force.MeetInTheMiddleAttack,
        brute_force.BirthdayAttack, brute_force.TimeMemoryTradeoff,
        brute_force.ParallelBruteForce, brute_force.HybridAttack
    ]:
        register_attack(attack_class, 'brute_force', applicable_to=['*'])
    
    # Statistical attacks
    for attack_class in [
        statistical.FrequencyAnalysisAttack, statistical.IndexOfCoincidenceAttack,
        statistical.ChiSquareAttack, statistical.EntropyAnalysisAttack,
        statistical.PatternRecognitionAttack, statistical.NgramAnalysisAttack,
        statistical.KasiskiExaminationAttack, statistical.AutocorrelationAttack
    ]:
        register_attack(attack_class, 'statistical', applicable_to=['*'])
    
    # Advanced attacks (61 total across 7 categories)
    # Register all from advanced_attacks module
    import inspect
    for name, obj in inspect.getmembers(advanced_attacks):
        if inspect.isclass(obj) and issubclass(obj, BaseAttack) and obj is not BaseAttack:
            if 'Linear' in name or 'Differential' in name:
                category = 'cryptanalysis'
            elif 'Algebraic' in name or 'SAT' in name or 'Cube' in name:
                category = 'algebraic'
            elif 'Timing' in name or 'Power' in name or 'Cache' in name or 'Fault' in name:
                category = 'side_channel'
            elif 'Quantum' in name or 'Shor' in name or 'Grover' in name:
                category = 'quantum'
            elif 'Lattice' in name or 'LLL' in name or 'BKZ' in name:
                category = 'lattice'
            elif 'Hash' in name or 'Collision' in name or 'Preimage' in name:
                category = 'hash_collision'
            elif 'Padding' in name or 'Oracle' in name or 'BEAST' in name:
                category = 'implementation_flaw'
            else:
                category = 'other'
            
            register_attack(obj, category, applicable_to=['*'])

# Register on import
_register_all_attacks()

__all__ = [
    # Base classes
    'BaseAttack', 'AttackResult',
    # Factory functions
    'create_attack', 'get_available_attacks', 'get_attacks_by_category',
    'get_applicable_attacks', 'get_attack_statistics', 'execute_attack',
]

