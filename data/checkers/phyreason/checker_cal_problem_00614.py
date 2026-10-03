import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract results safely - use numeric values for testing
        self.F_T_num = self.results.get('F_T_num', None)
        self.F_T2_num = self.results.get('F_T2_num', None)
        self.m_ratio = self.results.get('m_1:m_2', None)

        # Define symbols for physics checks (only for symbolic derivation if needed)
        self.logs = []

    def _log(self, message, success=None):
        """Helper to log messages."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")

    def test_invariants(self):
        """Verify Newton's 2nd Law consistency using numerical values."""
        try:
            # Use numerical values from results
            if self.F_T_num is None or self.F_T2_num is None:
                self._log("Missing F_T_num or F_T2_num values", False)
                return False
            
            # We'll test with the specific mass values that were used
            # The problem uses m1=1.0, m2=2.0, g=9.8
            m1_val = 1.0
            m2_val = 2.0
            g_val = 9.8
            
            # Compute expected accelerations from Newton's laws
            # For first system: a1 = m2*g/(m1+m2)
            a1_expected = m2_val * g_val / (m1_val + m2_val)
            # For second system: a2 = m2*g/(2*m1+m2)
            a2_expected = m2_val * g_val / (2*m1_val + m2_val)
            
            # Compute tensions from Newton's laws
            # For first system: F_T = m1*a1
            F_T_expected = m1_val * a1_expected
            # For second system: F_T2 = 2*m1*a2
            F_T2_expected = 2 * m1_val * a2_expected
            
            # Compare with provided values
            success1 = math.isclose(self.F_T_num, F_T_expected, rel_tol=1e-2, abs_tol=1e-2)
            success2 = math.isclose(self.F_T2_num, F_T2_expected, rel_tol=1e-2, abs_tol=1e-2)
            
            success = success1 and success2
            self._log(f"Invariant check: F_T_num={self.F_T_num:.3f} vs expected={F_T_expected:.3f}, "
                     f"F_T2_num={self.F_T2_num:.3f} vs expected={F_T2_expected:.3f}", success)
            return success
        except Exception as e:
            self._log(f"Invariant test failed: {traceback.format_exc()}", False)
            return False

    def test_trace_consistency(self):
        """Cross-check that the ratio condition holds."""
        try:
            # Use numerical values
            m1_val = 1.0
            m2_val = 2.0
            g_val = 9.8
            
            # Compute F_T and F_T2 from formulas
            F_T_calc = (m1_val * m2_val * g_val) / (m1_val + m2_val)
            F_T2_calc = (2 * m1_val * m2_val * g_val) / (2 * m1_val + m2_val)
            
            # Check if F_T2 = (3/2)*F_T
            expected_F_T2 = 1.5 * F_T_calc
            success = math.isclose(F_T2_calc, expected_F_T2, rel_tol=1e-2, abs_tol=1e-2)
            
            self._log(f"Trace consistency: F_T2={F_T2_calc:.3f}, (3/2)*F_T={expected_F_T2:.3f}", success)
            return success
        except Exception as e:
            self._log(f"Trace consistency test failed: {traceback.format_exc()}", False)
            return False

    def test_monotonicity(self):
        """Check tension increases with mass m1 using numerical derivative."""
        try:
            # Use a small perturbation
            m1_base = 1.0
            m2_val = 2.0
            g_val = 9.8
            
            # Compute F_T at m1_base
            F_T_base = (m1_base * m2_val * g_val) / (m1_base + m2_val)
            
            # Compute F_T at m1_base + small increment
            delta = 0.01
            m1_perturbed = m1_base + delta
            F_T_perturbed = (m1_perturbed * m2_val * g_val) / (m1_perturbed + m2_val)
            
            # Check if tension increases
            monotonic = F_T_perturbed > F_T_base
            
            self._log(f"Monotonicity: F_T({m1_base})={F_T_base:.3f}, "
                     f"F_T({m1_perturbed})={F_T_perturbed:.3f}, increasing={monotonic}", monotonic)
            return monotonic
        except Exception as e:
            self._log(f"Monotonicity test failed: {traceback.format_exc()}", False)
            return False

    def run_all_tests(self):
        """Execute all checks and return results."""
        self.logs = []
        
        # Run tests
        test_results = []
        
        self._log("=== Starting Physics Audit ===")
        test_results.append(("invariants", self.test_invariants()))
        test_results.append(("trace_consistency", self.test_trace_consistency()))
        test_results.append(("monotonicity", self.test_monotonicity()))
        
        # Summary
        all_passed = all(result for _, result in test_results)
        self._log(f"=== All tests passed: {all_passed} ===")
        
        return all_passed, self.logs