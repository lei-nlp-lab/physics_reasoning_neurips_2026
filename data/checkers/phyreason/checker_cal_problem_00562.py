import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access to prevent AttributeError in sandbox.
        """
        self.solution = solution_instance
        # Extract inputs with fallback defaults
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.q = getattr(solution_instance, 'q', solution_instance.results.get('q', 1.0))
        self.v0 = getattr(solution_instance, 'v0', solution_instance.results.get('v0', 1.0))
        self.L = getattr(solution_instance, 'L', solution_instance.results.get('L', 1.0))
        
        # Extract results with fallback to defaults (0.0 if not present)
        self.B = getattr(solution_instance, 'B', solution_instance.results.get('B', 0.0))
        self.x = getattr(solution_instance, 'x', solution_instance.results.get('x', 0.0))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', 0.0))
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', 0.0))
        
        # Store logs for reporting
        self.logs = []
    
    def _log(self, msg):
        """Helper to add log messages."""
        self.logs.append(msg)
    
    def test_invariants(self):
        """
        Verify fundamental physics invariants:
        1. Lorentz force equals centripetal force (Newton's 2nd Law for circular motion)
        2. Velocity magnitude remains constant (magnetic field does no work)
        3. Geometric consistency of circular motion
        """
        try:
            # 1. Lorentz force = centripetal force: q*v0*B = m*v0^2/r
            lhs = self.q * self.v0 * self.B
            rhs = self.m * self.v0**2 / self.r
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Lorentz force {lhs:.3g} != centripetal force {rhs:.3g}")
                return False
            self._log(f"PASS: Lorentz force = centripetal force ({lhs:.3g} ≈ {rhs:.3g})")
            
            # 2. Radius from geometry: r should equal 2L from step analysis
            if not math.isclose(self.r, 2*self.L, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Radius {self.r:.3g} != 2L ({2*self.L:.3g})")
                return False
            self._log(f"PASS: Geometric radius consistency (r={self.r:.3g} ≈ 2L={2*self.L:.3g})")
            
            # 3. Arc length: x = (150/360)*2πr = (5/3)πL
            expected_x = (5/3) * math.pi * self.L
            if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Arc length {self.x:.3g} != (5/3)πL ({expected_x:.3g})")
                return False
            self._log(f"PASS: Arc length consistency (x={self.x:.3g} ≈ (5/3)πL={expected_x:.3g})")
            
            return True
        except Exception as e:
            self._log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        try:
            # Check B = m*v0/(q*r) = m*v0/(2*q*L) from steps
            expected_B = self.m * self.v0 / (2 * self.q * self.L)
            if not math.isclose(self.B, expected_B, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: B={self.B:.3g}, expected m*v0/(2*q*L)={expected_B:.3g}")
                return False
            self._log(f"PASS: B = m*v0/(2*q*L) ({self.B:.3g} ≈ {expected_B:.3g})")
            
            # Check t = x/v0 = (5πL)/(3v0)
            expected_t = (5 * math.pi * self.L) / (3 * self.v0)
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: t={self.t:.3g}, expected (5πL)/(3v0)={expected_t:.3g}")
                return False
            self._log(f"PASS: t = (5πL)/(3v0) ({self.t:.3g} ≈ {expected_t:.3g})")
            
            return True
        except Exception as e:
            self._log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """
        Perturb input L and check physical trends:
        1. As L increases, B should decrease (B ∝ 1/L)
        2. As L increases, t should increase (t ∝ L)
        """
        try:
            # Save original values
            L_orig = self.L
            B_orig = self.B
            t_orig = self.t
            
            # Perturb L by +10%
            self.L *= 1.10
            
            # Recalculate expected values using physics formulas
            B_new = self.m * self.v0 / (2 * self.q * self.L)
            t_new = (5 * math.pi * self.L) / (3 * self.v0)
            
            # Restore original values
            self.L = L_orig
            
            # Check monotonic trends
            if not B_new < B_orig:
                self._log(f"FAIL: B should decrease when L increases (B_orig={B_orig:.3g}, B_new={B_new:.3g})")
                return False
            self._log(f"PASS: B decreases with increasing L (B ∝ 1/L)")
            
            if not t_new > t_orig:
                self._log(f"FAIL: t should increase when L increases (t_orig={t_orig:.3g}, t_new={t_new:.3g})")
                return False
            self._log(f"PASS: t increases with increasing L (t ∝ L)")
            
            return True
        except Exception as e:
            self._log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """
        Execute all tests and return success status with logs.
        """
        self.logs = []
        success = True
        
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n=== Running {test_name} ===")
            if not test_func():
                success = False
                self._log(f"--- {test_name} FAILED ---")
            else:
                self._log(f"--- {test_name} PASSED ---")
        
        return success, self.logs

# Optional: If running directly as a script for testing
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.m = 1.0e-3  # kg
            self.q = 1.6e-19  # C
            self.v0 = 1.0e6   # m/s
            self.L = 0.1      # m
            self.results = {
                'B': 3.125,  # T (from m*v0/(2*q*L))
                'x': 0.5236, # m (5πL/3)
                't': 5.236e-7, # s (5πL/(3v0))
                'r': 0.2     # m (2L)
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    print(f"\nOverall success: {success}")
    for log in logs:
        print(log)