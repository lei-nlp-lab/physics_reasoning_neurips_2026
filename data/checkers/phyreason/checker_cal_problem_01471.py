import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for results
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract computed results with fallbacks
        self.v_3 = getattr(solution_instance, 'v_3', self.results.get('v_3', None))
        self.t = getattr(solution_instance, 't', self.results.get('t', None))
        
        # Problem constants (from problem statement)
        self.v0 = 1.0  # m/s
        self.a = 2.0   # m/s²
        self.t3 = 3.0  # s
        self.v_target = 21.0  # m/s
        
        # Expected answers (for reference)
        self.expected_v3 = 7.0
        self.expected_t = 10.0
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs storage
        self.logs = []

    def _log(self, message, is_error=False):
        """Helper to log messages."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
    
    def _is_close(self, a, b):
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self):
        """Verify physical laws are satisfied."""
        success = True
        
        # Check step 1: v_3 = v0 + a*t3
        if self.v_3 is not None:
            v3_calc = self.v0 + self.a * self.t3
            if not self._is_close(self.v_3, v3_calc):
                self._log(f"Velocity equation violated: v_3={self.v_3}, expected {v3_calc}", is_error=True)
                success = False
            else:
                self._log(f"Step 1 invariant passed: v_3 = v0 + a*t3 = {v3_calc}")
        else:
            self._log("Missing v_3 result", is_error=True)
            success = False
        
        # Check step 2: t = (v_target - v0)/a
        if self.t is not None:
            t_calc = (self.v_target - self.v0) / self.a
            if not self._is_close(self.t, t_calc):
                self._log(f"Time equation violated: t={self.t}, expected {t_calc}", is_error=True)
                success = False
            else:
                self._log(f"Step 2 invariant passed: t = (v_target - v0)/a = {t_calc}")
        else:
            self._log("Missing t result", is_error=True)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        if self.v_3 is not None and self.t is not None:
            # Both steps use the same physical parameters v0 and a
            # We can check consistency by verifying the same acceleration is used in both
            # From step 1: a_calc1 = (v_3 - v0)/t3
            # From step 2: a_calc2 = (v_target - v0)/t
            
            a_calc1 = (self.v_3 - self.v0) / self.t3
            a_calc2 = (self.v_target - self.v0) / self.t
            
            if not self._is_close(a_calc1, self.a):
                self._log(f"Inconsistent acceleration from step 1: {a_calc1} ≠ {self.a}", is_error=True)
                success = False
            if not self._is_close(a_calc2, self.a):
                self._log(f"Inconsistent acceleration from step 2: {a_calc2} ≠ {self.a}", is_error=True)
                success = False
            
            if success:
                self._log("Trace consistency passed: both steps use the same acceleration")
        else:
            self._log("Missing results for trace consistency check", is_error=True)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb input and check logical trend."""
        success = True
        
        # Perturb acceleration (increase by 1%)
        perturbed_a = self.a * 1.01
        
        # Recalculate with perturbed acceleration
        v3_perturbed = self.v0 + perturbed_a * self.t3
        t_perturbed = (self.v_target - self.v0) / perturbed_a
        
        # Check trends (since a > 0 and v_target > v0):
        # 1. v_3 should increase with increased acceleration
        # 2. t should decrease with increased acceleration
        
        if self.v_3 is not None:
            if not (v3_perturbed > self.v_3):
                self._log(f"Monotonicity violated for v_3: increased a should increase v_3", is_error=True)
                success = False
        
        if self.t is not None:
            if not (t_perturbed < self.t):
                self._log(f"Monotonicity violated for t: increased a should decrease t", is_error=True)
                success = False
        
        if success:
            self._log("Monotonicity passed: trends match physical intuition")
        
        return success
    
    def run_all_tests(self):
        """Execute all checks and return results."""
        self.logs = []
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Also perform direct comparison with expected answers (optional)
        if self.v_3 is not None and not self._is_close(self.v_3, self.expected_v3):
            self._log(f"v_3 differs from expected: {self.v_3} vs {self.expected_v3}", is_error=True)
            test1 = False
        
        if self.t is not None and not self._is_close(self.t, self.expected_t):
            self._log(f"t differs from expected: {self.t} vs {self.expected_t}", is_error=True)
            test1 = False
        
        # Overall success
        overall_success = test1 and test2 and test3
        
        if overall_success:
            self._log("All tests passed!")
        
        return overall_success, self.logs