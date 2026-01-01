"""
Language Bridge - FFI to C++ and Rust

Provides seamless integration between Python, C++, and Rust attack implementations.

Version: 1.0
Date: December 31, 2025
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LanguageBridge:
    """
    Bridge for calling attacks implemented in different languages.
    
    Handles:
    - C++ attacks via pybind11
    - Rust attacks via PyO3
    - Automatic fallback to Python if native libraries unavailable
    """
    
    def __init__(self):
        """Initialize language bridge and load native libraries"""
        self.cpp_lib = self._load_cpp_library()
        self.rust_lib = self._load_rust_library()
        
        logger.info(f"Language Bridge initialized:")
        logger.info(f"  C++ library: {'Available' if self.cpp_lib else 'Not available'}")
        logger.info(f"  Rust library: {'Available' if self.rust_lib else 'Not available'}")
    
    def _load_cpp_library(self) -> Optional[Any]:
        """Load C++ attacks library"""
        try:
            import attacks_cpp
            logger.info("C++ attacks library loaded successfully")
            return attacks_cpp
        except ImportError as e:
            logger.warning(f"C++ attacks library not available: {e}")
            logger.warning("C++ attacks will fall back to Python implementations")
            return None
    
    def _load_rust_library(self) -> Optional[Any]:
        """Load Rust attacks library"""
        try:
            import attacks_rust
            logger.info("Rust attacks library loaded successfully")
            return attacks_rust
        except ImportError as e:
            logger.warning(f"Rust attacks library not available: {e}")
            logger.warning("Rust attacks will fall back to Python implementations")
            return None
    
    def call_cpp_attack(self, attack_name: str, ciphertext: bytes, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a C++ attack implementation.
        
        Args:
            attack_name: Name of attack
            ciphertext: Ciphertext to attack
            params: Attack parameters
        
        Returns:
            Attack result dictionary
        """
        if self.cpp_lib is None:
            return self._fallback_to_python(attack_name, ciphertext, params)
        
        try:
            result_dict = self.cpp_lib.execute_attack(attack_name, ciphertext, params)
            result_dict['attack_language'] = 'C++'
            return result_dict
        except Exception as e:
            logger.error(f"C++ attack {attack_name} failed: {e}")
            return self._fallback_to_python(attack_name, ciphertext, params)
    
    def call_rust_attack(self, attack_name: str, ciphertext: bytes, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a Rust attack implementation.
        
        Args:
            attack_name: Name of attack
            ciphertext: Ciphertext to attack
            params: Attack parameters
        
        Returns:
            Attack result dictionary
        """
        if self.rust_lib is None:
            return self._fallback_to_python(attack_name, ciphertext, params)
        
        try:
            result_dict = self.rust_lib.execute_attack(attack_name, ciphertext, params)
            result_dict['attack_language'] = 'Rust'
            return result_dict
        except Exception as e:
            logger.error(f"Rust attack {attack_name} failed: {e}")
            return self._fallback_to_python(attack_name, ciphertext, params)
    
    def _fallback_to_python(self, attack_name: str, ciphertext: bytes, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to Python implementation"""
        logger.debug(f"Falling back to Python for {attack_name}")
        
        try:
            from ..attacks.attack_factory import create_attack
            attack = create_attack(attack_name, "UNKNOWN")
            result = attack.execute(ciphertext, **params)
            return result.to_dict()
        except Exception as e:
            logger.error(f"Python fallback also failed for {attack_name}: {e}")
            return {
                'attack_name': attack_name,
                'success': False,
                'confidence': 0.0,
                'attack_language': 'Python',
                'error_message': f"All implementations failed: {e}",
            }
    
    def get_available_cpp_attacks(self) -> list:
        """Get list of available C++ attacks"""
        if self.cpp_lib is None:
            return []
        try:
            return self.cpp_lib.get_available_attacks()
        except:
            return []
    
    def get_available_rust_attacks(self) -> list:
        """Get list of available Rust attacks"""
        if self.rust_lib is None:
            return []
        try:
            return self.rust_lib.get_available_attacks()
        except:
            return []
    
    def is_cpp_available(self) -> bool:
        """Check if C++ library is available"""
        return self.cpp_lib is not None
    
    def is_rust_available(self) -> bool:
        """Check if Rust library is available"""
        return self.rust_lib is not None


__all__ = ['LanguageBridge']

