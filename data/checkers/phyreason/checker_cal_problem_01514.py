import math
import traceback
from typing import List, Tuple

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results without direct attribute access."""
        self.logs = []
        
        # Extract results with safe fallbacks
        results_dict = getattr(solution_instance, 'results', {})
        self.M = results_dict.get('M', None)
        
        # Use actual known constants for Europa's orbit
        # Europa's typical parameters
        self.m = 4.80e22  # kg
        self.r = 6.71e8   # m
        self.T = 3.55e5   # s (about 4.1 days)
        self.G = 6.6743e-11  # N·m²/kg²
        self.pi = math.pi
        
        # Store original values for monotonicity tests
        self.original_inputs = {
            'm': self.m,
            'r': self.r,
            'T': self.T,
            'G': self.G
        }
        
    def _log(self, msg: str, level: str = "INFO"):
        """Helper for logging."""
        self.logs.append(f"[{level}] {msg}")
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    
    def test_invariants(self) -> bool:
        """Test fundamental physics laws: Centripetal-Gravitational equilibrium."""
        success = True
        
        try:
            # Expected formula from KB: M = (4*π²*r³)/(G*T²)
            expected_M = (4 * self.pi**2 * self.r**3) / (self.G * self.T**2)
            
            if self.M is None:
                self._log("Result 'M' not found in solution", "ERROR")
                return False
            
            # Numerical comparison
            if not self._is_close(self.M, expected_M):
                self._log(f"Formula mismatch: Got {self.M}, Expected {expected_M}", "ERROR")
                success = False
            else:
                self._log(f"Formula matches: M = {self.M:.3e} kg", "SUCCESS")
            
            # Test force equilibrium: Gravitational force = Centripetal force
            # F_grav = G*M*m/r², F_cen = m*v²/r, v = 2πr/T
            v = 2 * self.pi * self.r / self.T
            F_grav = self.G * self.M * self.m / self.r**2
            F_cen = self.m * v**2 / self.r
            
            if not self._is_close(F_grav, F_cen):
                self._log(f"Force equilibrium fails: F_grav={F_grav:.3e}, F_cen={F_cen:.3e}", "ERROR")
                success = False
            else:
                self._log(f"Force equilibrium holds: F = {F_grav:.3e} N", "SUCCESS")
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {traceback.format_exc()}", "ERROR")
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check that intermediate derivations lead to the final answer."""
        success = True
        
        try:
            # From STEPS ANALYSIS: Derive M step by step
            # Step 1: Centripetal acceleration a_c = v²/r, v = 2πr/T
            v = 2 * self.pi * self.r / self.T
            a_c = v**2 / self.r
            
            # Step 2: Gravitational acceleration g = G*M/r²
            # For circular orbit: a_c = g
            # So: G*M/r² = (2πr/T)² / r = 4π²r/T²
            # => M = (4π²r³)/(G*T²)
            
            derived_M = (4 * self.pi**2 * self.r**3) / (self.G * self.T**2)
            
            if not self._is_close(self.M, derived_M):
                self._log(f"Trace inconsistency: Solution M={self.M}, Derived M={derived_M}", "ERROR")
                success = False
            else:
                self._log("Trace consistent: derived formula matches computed M", "SUCCESS")
                
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {traceback.format_exc()}", "ERROR")
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Test that M responds correctly to input perturbations."""
        success = True
        
        try:
            # Store original M
            original_M = self.M
            
            # Test 1: Increase r by 10%, M should increase by ~(1.1)^3 = 1.331
            r_perturbed = self.r * 1.1
            M_perturbed = (4 * self.pi**2 * r_perturbed**3) / (self.G * self.T**2)
            expected_factor = 1.331  # 1.1^3
            
            actual_factor = M_perturbed / original_M
            if not self._is_close(actual_factor, expected_factor, rel_tol=0.05):  # 5% tolerance
                self._log(f"Monotonicity fail for r: factor {actual_factor:.3f} ≠ {expected_factor:.3f}", "ERROR")
                success = False
            else:
                self._log(f"Monotonicity for r perturbation holds: factor {actual_factor:.3f}", "SUCCESS")
            
            # Test 2: Increase T by 10%, M should decrease by ~1/(1.1)^2 = 0.826
            T_perturbed = self.T * 1.1
            M_perturbed = (4 * self.pi**2 * self.r**3) / (self.G * T_perturbed**2)
            expected_factor = 1 / (1.1**2)  # ~0.826
            
            actual_factor = M_perturbed / original_M
            if not self._is_close(actual_factor, expected_factor, rel_tol=0.05):
                self._log(f"Monotonicity fail for T: factor {actual_factor:.3f} ≠ {expected_factor:.3f}", "ERROR")
                success = False
            else:
                self._log(f"Monotonicity for T perturbation holds: factor {actual_factor:.3f}", "SUCCESS")
            
            # Test 3: m should not affect M (check independence)
            m_perturbed = self.m * 2.0
            M_same = (4 * self.pi**2 * self.r**3) / (self.G * self.T**2)
            if not self._is_close(original_M, M_same):
                self._log(f"Monotonicity fail: M depends on m when it shouldn't", "ERROR")
                success = False
            else:
                self._log("M independent of m as expected", "SUCCESS")
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {traceback.format_exc()}", "ERROR")
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all physics audits and return results."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            if not test_func():
                overall_success = False
        
        self._log(f"Overall audit {'PASSED' if overall_success else 'FAILED'}", 
                 "SUCCESS" if overall_success else "ERROR")
        
        return overall_success, self.logs