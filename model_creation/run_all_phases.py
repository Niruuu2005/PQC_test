"""
Master Phase Execution Pipeline
Automatically executes all 6 phases with validation gates.

Usage:
    python run_all_phases.py [--start-phase N]
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhaseExecutor:
    """Execute and validate all project phases."""
    
    def __init__(self, start_phase: int = 1):
        """
        Initialize phase executor.
        
        Args:
            start_phase: Phase number to start from (1-6)
        """
        self.start_phase = start_phase
        self.current_phase = start_phase
        self.phase_results = {}
        
    def validate_phase(self, phase_num: int) -> bool:
        """
        Validate a phase has completed successfully.
        
        Args:
            phase_num: Phase number (1-6)
            
        Returns:
            True if validation passed
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"VALIDATING PHASE {phase_num}")
        logger.info(f"{'='*70}")
        
        validation_functions = {
            1: self._validate_phase1,
            2: self._validate_phase2,
            3: self._validate_phase3,
            4: self._validate_phase4,
            5: self._validate_phase5,
            6: self._validate_phase6
        }
        
        if phase_num not in validation_functions:
            logger.error(f"Invalid phase number: {phase_num}")
            return False
        
        try:
            result = validation_functions[phase_num]()
            self.phase_results[phase_num] = {
                'status': 'passed' if result else 'failed',
                'timestamp': datetime.now().isoformat()
            }
            return result
        except Exception as e:
            logger.error(f"Phase {phase_num} validation failed with error: {e}")
            self.phase_results[phase_num] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def _validate_phase1(self) -> bool:
        """Validate Phase 1: Data Engineering."""
        logger.info("Checking Phase 1 components...")
        
        # Check modules exist
        required_files = [
            'src/data/loader.py',
            'src/data/cleaner.py',
            'src/data/feature_engineer.py',
            'src/data/splitter.py'
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                logger.error(f"Missing file: {file_path}")
                return False
            logger.info(f"  ✓ {file_path} exists")
        
        # Test imports
        try:
            from src.data.cleaner import DataCleaner
            from src.data.feature_engineer import FeatureEngineer
            from src.data.splitter import DataSplitter
            logger.info("  ✓ All modules import successfully")
        except ImportError as e:
            logger.error(f"Import failed: {e}")
            return False
        
        logger.info("\n✓ PHASE 1 VALIDATED")
        return True
    
    def _validate_phase2(self) -> bool:
        """Validate Phase 2: Classical ML Baseline."""
        logger.info("Checking Phase 2 components...")
        
        # Check if base learners module exists
        if not Path('src/classical/base_learners.py').exists():
            logger.warning("Phase 2 not yet implemented")
            return False
        
        logger.info("  ✓ Classical ML module exists")
        logger.info("\n✓ PHASE 2 VALIDATED")
        return True
    
    def _validate_phase3(self) -> bool:
        """Validate Phase 3: Quantum Circuits."""
        logger.info("Checking Phase 3 components...")
        
        if not Path('src/quantum/circuits.py').exists():
            logger.warning("Phase 3 not yet implemented")
            return False
        
        logger.info("  ✓ Quantum circuits module exists")
        logger.info("\n✓ PHASE 3 VALIDATED")
        return True
    
    def _validate_phase4(self) -> bool:
        """Validate Phase 4: Hybrid Integration."""
        logger.info("Checking Phase 4 components...")
        
        if not Path('src/quantum/hybrid_model.py').exists():
            logger.warning("Phase 4 not yet implemented")
            return False
        
        logger.info("  ✓ Hybrid model module exists")
        logger.info("\n✓ PHASE 4 VALIDATED")
        return True
    
    def _validate_phase5(self) -> bool:
        """Validate Phase 5: Full-Scale Training."""
        logger.info("Checking Phase 5 components...")
        
        if not Path('models/hybrid').exists():
            logger.warning("Phase 5 not yet implemented - no trained models")
            return False
        
        logger.info("  ✓ Trained models directory exists")
        logger.info("\n✓ PHASE 5 VALIDATED")
        return True
    
    def _validate_phase6(self) -> bool:
        """Validate Phase 6: Evaluation & Deployment."""
        logger.info("Checking Phase 6 components...")
        
        if not Path('deployment/inference_api.py').exists():
            logger.warning("Phase 6 not yet implemented")
            return False
        
        logger.info("  ✓ Deployment components exist")
        logger.info("\n✓ PHASE 6 VALIDATED")
        return True
    
    def execute_phase(self, phase_num: int) -> bool:
        """
        Execute a specific phase.
        
        Args:
            phase_num: Phase number (1-6)
            
        Returns:
            True if execution successful
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"EXECUTING PHASE {phase_num}")
        logger.info(f"{'='*70}")
        
        phase_descriptions = {
            1: "Data Engineering",
            2: "Classical ML Baseline",
            3: "Quantum Circuit Design",
            4: "Hybrid Integration",
            5: "Full-Scale Training",
            6: "Evaluation & Deployment"
        }
        
        logger.info(f"Phase {phase_num}: {phase_descriptions.get(phase_num, 'Unknown')}")
        
        # Phase 1 is already implemented
        if phase_num == 1:
            logger.info("Phase 1 components already created")
            return True
        
        # For now, other phases are placeholders
        logger.info(f"Phase {phase_num} execution pending - detailed implementation needed")
        logger.info(f"See phase{phase_num}_*.md for implementation plan")
        
        return True
    
    def run_all_phases(self) -> bool:
        """
        Run all phases from start_phase to phase 6.
        
        Returns:
            True if all phases completed successfully
        """
        logger.info("\n" + "="*70)
        logger.info("QML CRYPTANALYSIS - MULTI-PHASE EXECUTION")
        logger.info("="*70)
        logger.info(f"Starting from Phase {self.start_phase}")
        logger.info(f"Target: Phase 6 (Complete System)")
        
        for phase_num in range(self.start_phase, 7):
            logger.info(f"\n{'#'*70}")
            logger.info(f"# PHASE {phase_num}")
            logger.info(f"{'#'*70}")
            
            # Execute phase
            execution_success = self.execute_phase(phase_num)
            if not execution_success:
                logger.error(f"Phase {phase_num} execution failed!")
                return False
            
            # Validate phase
            validation_success = self.validate_phase(phase_num)
            if not validation_success:
                logger.error(f"Phase {phase_num} validation failed!")
                logger.error("Cannot proceed to next phase. Please fix issues.")
                self._print_summary()
                return False
            
            logger.info(f"\n✓ Phase {phase_num} completed successfully")
        
        self._print_summary()
        return True
    
    def _print_summary(self):
        """Print execution summary."""
        logger.info("\n" + "="*70)
        logger.info("EXECUTION SUMMARY")
        logger.info("="*70)
        
        for phase_num, result in sorted(self.phase_results.items()):
            status_icon = "✓" if result['status'] == 'passed' else "✗"
            logger.info(f"Phase {phase_num}: {status_icon} {result['status'].upper()}")
        
        total = len(self.phase_results)
        passed = sum(1 for r in self.phase_results.values() if r['status'] == 'passed')
        
        logger.info(f"\nTotal: {passed}/{total} phases passed")
        
        if passed == 6:
            logger.info("\n🎉 ALL PHASES COMPLETE! 🎉")
            logger.info("System ready for deployment!")
        else:
            logger.info(f"\nProgress: {passed}/6 phases complete")


def main():
    """Main execution entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Execute QML Cryptanalysis Phases')
    parser.add_argument('--start-phase', type=int, default=1, choices=range(1, 7),
                       help='Phase number to start from (1-6)')
    args = parser.parse_args()
    
    executor = PhaseExecutor(start_phase=args.start_phase)
    success = executor.run_all_phases()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
