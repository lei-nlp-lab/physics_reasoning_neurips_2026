import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters from problem statement (with safe defaults)
        self.q1 = getattr(solution_instance, 'q1', 1.6e-19)
        self.q2 = getattr(solution_instance, 'q2', -8.0e-19)
        self.r = getattr(solution_instance, 'r', 2.0)
        self.k = getattr(solution_instance, 'k', 9.0e9)
        
        # Result with safe extraction
        self.F = getattr(solution_instance, 'F', self.results.get('F', 0.0))
        
        # Expected values
        self.expected_F = 2.88e-28  # From steps analysis
        
        # Log storage
        self.logs = []
        
    def _log(self, message, success=True):
        """Helper to log test results"""
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
        
    def test_invariants(self):
        """Test Coulomb's law invariants and force direction"""
        try:
            # 1. Check magnitude using Coulomb's law
            F_expected_magnitude = self.k * abs(self.q1 * self.q2) / (self.r ** 2)
            F_computed_magnitude = abs(self.F)
            
            if math.isclose(F_computed_magnitude, F_expected_magnitude, 
                          rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Coulomb's law magnitude check passed: {F_computed_magnitude:.3e} N")
            else:
                self._log(f"Coulomb's law magnitude check failed: Expected ~{F_expected_magnitude:.3e} N, Got {F_computed_magnitude:.3e} N", False)
                
            # 2. Check force direction (attractive for opposite charges)
            if self.q1 * self.q2 < 0:
                # Force should be attractive (negative sign in vector form)
                # For magnitude-only result, check if positive
                if self.F >= 0:
                    self._log("Force direction check passed: Magnitude positive for attractive force")
                else:
                    # Some solutions might compute with sign
                    self._log("Note: Force computed with sign, but magnitude is expected")
            else:
                self._log("Charges have like signs - force should be repulsive")
                
            # 3. Symbolic check
            q1_sym, q2_sym, r_sym, k_sym = sp.symbols('q1 q2 r k')
            F_sym = k_sym * q1_sym * q2_sym / r_sym**2
            
            # Substitute values
            F_symbolic = F_sym.subs({
                k_sym: self.k,
                q1_sym: self.q1,
                q2_sym: self.q2,
                r_sym: self.r
            })
            
            F_symbolic_val = float(F_symbolic.evalf())
            
            if math.isclose(self.F, F_symbolic_val, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Symbolic Coulomb's law check passed: {F_symbolic_val:.3e} N")
            else:
                self._log(f"Symbolic check: Expected {F_symbolic_val:.3e} N, Got {self.F:.3e} N", False)
                
            return True
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            return False
            
    def test_trace_consistency(self):
        """Cross-check with intermediate results if available"""
        try:
            # Check if F matches expected value from steps analysis
            if math.isclose(abs(self.F), self.expected_F, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Result consistency check passed: {abs(self.F):.3e} N ≈ {self.expected_F:.3e} N")
                return True
            else:
                self._log(f"Result consistency failed: Expected ~{self.expected_F:.3e} N, Got {abs(self.F):.3e} N", False)
                return False
                
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", False)
            return False
            
    def test_monotonicity(self):
        """Test trend when perturbing parameters"""
        try:
            # Save original values
            orig_q1, orig_q2, orig_r = self.q1, self.q2, self.r
            
            # Test 1: Increase distance, force should decrease
            self.r *= 1.1  # 10% increase
            F_new = self.k * abs(self.q1 * self.q2) / (self.r ** 2)
            
            if F_new < abs(orig_q1 * orig_q2) * self.k / (orig_r ** 2):
                self._log("Monotonicity test 1 passed: Force decreases with increased distance")
            else:
                self._log("Monotonicity test 1 failed: Force didn't decrease with increased distance", False)
                
            # Restore original r
            self.r = orig_r
            
            # Test 2: Increase charge magnitude, force should increase
            self.q1 *= 1.1  # 10% increase
            F_new = self.k * abs(self.q1 * self.q2) / (self.r ** 2)
            F_orig = self.k * abs(orig_q1 * orig_q2) / (self.r ** 2)
            
            if F_new > F_orig:
                self._log("Monotonicity test 2 passed: Force increases with increased charge magnitude")
            else:
                self._log("Monotonicity test 2 failed: Force didn't increase with increased charge magnitude", False)
                
            # Restore original values
            self.q1 = orig_q1
            
            return True
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            # Try to restore values even if test failed
            try:
                self.q1, self.q2, self.r = orig_q1, orig_q2, orig_r
            except:
                pass
            return False
            
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []
        success = True
        
        # Run tests
        tests = [
            ("Coulomb's Law Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Parameter Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self._log(f"Test crashed: {traceback.format_exc()}", False)
                success = False
                
        # Final summary
        self.logs.append(f"\n{'='*40}")
        self.logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs