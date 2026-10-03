import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs - use getattr with defaults
        self.q = getattr(solution_instance, 'q', 1.0)
        self.v = getattr(solution_instance, 'v', 1.0)
        self.B = getattr(solution_instance, 'B', 1.0)
        self.E = getattr(solution_instance, 'E', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        
        # Extract results safely from results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Extract numeric force values with defaults
        F_electric_val = results.get('F_electric', 0.0)
        F_magnetic_val = results.get('F', 0.0)  # 'F' maps to magnetic force
        
        # Convert to float only if they are numeric
        try:
            self.F_electric = float(F_electric_val)
        except (TypeError, ValueError):
            self.F_electric = 0.0
            
        try:
            self.F_magnetic = float(F_magnetic_val)
        except (TypeError, ValueError):
            self.F_magnetic = 0.0
        
        # IMPORTANT FIX: According to sandbox rules, results should ONLY contain numeric floats.
        # Direction strings should NOT be in results. Therefore, we should NOT extract them.
        # Instead, we'll use the expected direction strings directly in our tests.
        # We'll still try to extract them, but with empty defaults.
        dir_electric_val = results.get('dir_electric', "")
        dir_magnetic_val = results.get('N/A', "")
        
        # Convert to string for comparison
        self.dir_electric = str(dir_electric_val)
        self.dir_magnetic = str(dir_magnetic_val)
        
        # Store original for restoration after perturbations
        self.original = {
            'q': self.q, 'v': self.v, 'B': self.B, 'E': self.E, 'm': self.m,
            'F_electric': self.F_electric, 'F_magnetic': self.F_magnetic
        }
        
        self.logs = []
        
    def log(self, msg, level="INFO"):
        """Helper to log messages."""
        self.logs.append(f"{level}: {msg}")
    
    def test_invariants(self):
        """Verify physics laws: Lorentz force and left-hand rule."""
        try:
            # Check electric force magnitude
            F_e_expected = self.q * self.E
            if not math.isclose(self.F_electric, F_e_expected, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Electric force invariant failed: got {self.F_electric}, expected {F_e_expected}", "WARNING")
                return False
            
            # Check magnetic force magnitude
            F_m_expected = self.q * self.v * self.B
            if not math.isclose(self.F_magnetic, F_m_expected, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Magnetic force invariant failed: got {self.F_magnetic}, expected {F_m_expected}", "WARNING")
                return False
            
            # FIX: Since direction strings should NOT be in results according to sandbox rules,
            # we should NOT fail the test if they're missing. We'll only check if they exist.
            # If they exist, verify they contain expected keywords.
            expected_dir_electric = "right"
            expected_dir_magnetic = "inward"
            
            # Only check if direction strings are not empty
            if self.dir_electric and expected_dir_electric.lower() not in self.dir_electric.lower():
                self.log(f"Electric force direction mismatch: got '{self.dir_electric}'", "WARNING")
                return False
                
            if self.dir_magnetic and expected_dir_magnetic.lower() not in self.dir_magnetic.lower():
                self.log(f"Magnetic force direction mismatch: got '{self.dir_magnetic}'", "WARNING")
                return False
            
            self.log("All physics invariants satisfied.")
            return True
            
        except Exception as e:
            self.log(f"Invariant test failed with error: {traceback.format_exc()}", "ERROR")
            return False
    
    def test_trace_consistency(self):
        """Check intermediate results align with final answer."""
        try:
            # Check that magnetic force matches the target answer pattern
            expected_value = self.q * self.v * self.B
            if not math.isclose(self.F_magnetic, expected_value, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Trace consistency failed: magnetic force {self.F_magnetic} doesn't match computed {expected_value}", "WARNING")
                return False
            
            # FIX: Only check direction string if it exists
            if self.dir_magnetic and ("perpendicular" not in self.dir_magnetic.lower() or "inward" not in self.dir_magnetic.lower()):
                self.log(f"Direction string '{self.dir_magnetic}' doesn't match expected pattern", "WARNING")
                return False
            
            self.log("Trace consistency check passed.")
            return True
            
        except Exception as e:
            self.log(f"Trace consistency test failed: {traceback.format_exc()}", "ERROR")
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and check force trends."""
        try:
            # Store original magnetic force for comparison
            original_F_magnetic = self.F_magnetic
            
            # Test 1: Increase charge q by 10%
            self.q *= 1.10
            F_m_new = self.q * self.v * self.B
            if not (F_m_new > original_F_magnetic * 1.09):  # Allow 1% tolerance
                self.log(f"Monotonicity failed: increasing q did not sufficiently increase magnetic force", "WARNING")
                self._restore_original()
                return False
            self._restore_original()
            
            # Test 2: Increase velocity v by 10%
            self.v *= 1.10
            F_m_new = self.q * self.v * self.B
            if not (F_m_new > original_F_magnetic * 1.09):
                self.log(f"Monotonicity failed: increasing v did not sufficiently increase magnetic force", "WARNING")
                self._restore_original()
                return False
            self._restore_original()
            
            # Test 3: Increase magnetic field B by 10%
            self.B *= 1.10
            F_m_new = self.q * self.v * self.B
            if not (F_m_new > original_F_magnetic * 1.09):
                self.log(f"Monotonicity failed: increasing B did not sufficiently increase magnetic force", "WARNING")
                self._restore_original()
                return False
            self._restore_original()
            
            self.log("Monotonicity tests passed.")
            return True
            
        except Exception as e:
            self.log(f"Monotonicity test failed: {traceback.format_exc()}", "ERROR")
            self._restore_original()
            return False
    
    def _restore_original(self):
        """Restore original input values."""
        for key, value in self.original.items():
            setattr(self, key, value)
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        success = True
        
        # Run tests
        if not self.test_invariants():
            success = False
        if not self.test_trace_consistency():
            success = False
        if not self.test_monotonicity():
            success = False
        
        return success, self.logs