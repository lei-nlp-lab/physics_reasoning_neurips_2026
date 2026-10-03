import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr and .get()
        self.c = 330.0  # speed of sound in air (m/s)
        
        # Extract results with fallback to known values
        # The observed frequencies are given constants, not computed variables
        self.f_approach = 1200.0  # Hz (given in problem)
        self.f_recede = 1000.0    # Hz (given in problem)
        
        # Extract car speed from results
        self.v = float(getattr(solution_instance, 'v', 
                              getattr(solution_instance, 'results', {}).get('v', 0.0)))
        
        # Extract source frequency if available
        self.f0 = float(getattr(solution_instance, 'f0',
                               getattr(solution_instance, 'results', {}).get('f0', 0.0)))
        
        # Derived quantities for checking
        self.f0_from_approach = None
        self.f0_from_recede = None
        self._compute_derived()
        
        self.logs = []
    
    def _compute_derived(self):
        """Compute source frequency from both equations for consistency checks."""
        try:
            # From f_approach = f0 * c / (c - v) => f0 = f_approach * (c - v) / c
            self.f0_from_approach = self.f_approach * (self.c - self.v) / self.c
            # From f_recede = f0 * c / (c + v) => f0 = f_recede * (c + v) / c
            self.f0_from_recede = self.f_recede * (self.c + self.v) / self.c
        except ZeroDivisionError:
            self.logs.append("ERROR: Division by zero in _compute_derived")
    
    def _log(self, msg, is_error=False):
        """Helper for logging."""
        self.logs.append(("ERROR" if is_error else "INFO") + ": " + msg)
    
    def test_invariants(self):
        """Check physical invariants: Doppler formula consistency."""
        success = True
        
        # 1. Check that both give the same source frequency
        if self.f0_from_approach is None or self.f0_from_recede is None:
            self._log("Cannot compute source frequency", is_error=True)
            return False
        
        if not math.isclose(self.f0_from_approach, self.f0_from_recede, 
                           rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Source frequency mismatch: f0_approach={self.f0_from_approach:.2f}, "
                     f"f0_recede={self.f0_from_recede:.2f}")
            success = False
        else:
            self._log(f"Source frequency consistent: {self.f0_from_approach:.2f} Hz")
        
        # 2. Check velocity from frequency ratio matches given v
        if self.f_recede == 0:
            self._log("f_recede is zero, cannot compute ratio", is_error=True)
            success = False
        else:
            r = self.f_approach / self.f_recede
            v_calc = self.c * (r - 1) / (r + 1)
            if not math.isclose(self.v, v_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Velocity mismatch: given v={self.v:.2f}, "
                         f"calculated v={v_calc:.2f}")
                success = False
            else:
                self._log(f"Velocity consistent: {self.v:.2f} m/s")
        
        # 3. Check against stored f0 if available
        if self.f0 > 0:
            if not math.isclose(self.f0, self.f0_from_approach, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Stored f0 mismatch: stored={self.f0:.2f}, "
                         f"calculated={self.f0_from_approach:.2f}")
                success = False
            else:
                self._log(f"Stored f0 matches calculated: {self.f0:.2f} Hz")
        
        return success
    
    def test_trace_consistency(self):
        """Check that intermediate steps (if present) are consistent with final answer."""
        success = True
        
        # Since we don't have explicit intermediate results, we check that
        # the computed source frequency is physically reasonable.
        if self.f0_from_approach is not None:
            # Source frequency should be between the two observed frequencies
            # Actually, for approaching: f_approach > f0, for receding: f_recede < f0
            # So f0 should be between f_recede and f_approach.
            if not (self.f_recede <= self.f0_from_approach <= self.f_approach):
                self._log(f"Source frequency {self.f0_from_approach:.2f} Hz not between "
                         f"{self.f_recede:.2f} Hz and {self.f_approach:.2f} Hz")
                success = False
            else:
                self._log(f"Source frequency {self.f0_from_approach:.2f} Hz is within bounds")
        
        # Check that the velocity is positive and less than speed of sound
        if self.v <= 0 or self.v >= self.c:
            self._log(f"Velocity {self.v:.2f} m/s is not in (0, {self.c})", is_error=True)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and verify output trends."""
        success = True
        
        # Use the correct formula for velocity from frequency ratio
        r = self.f_approach / self.f_recede
        v_current = self.c * (r - 1) / (r + 1)
        
        # Perturb f_approach upward by 1%
        f_approach_pert = self.f_approach * 1.01
        r_pert = f_approach_pert / self.f_recede
        v_pert = self.c * (r_pert - 1) / (r_pert + 1)
        
        # Expect v to increase when f_approach increases (with f_recede fixed)
        if not v_pert > v_current:
            self._log(f"Monotonicity violated: increasing f_approach from {self.f_approach:.2f} to "
                     f"{f_approach_pert:.2f} did not increase v (got {v_pert:.2f} vs {v_current:.2f})")
            success = False
        else:
            self._log(f"Monotonicity check 1 passed: v increases with f_approach")
        
        # Perturb f_recede downward by 1%
        f_recede_pert = self.f_recede * 0.99
        r_pert2 = self.f_approach / f_recede_pert
        v_pert2 = self.c * (r_pert2 - 1) / (r_pert2 + 1)
        
        # Expect v to increase when f_recede decreases (with f_approach fixed)
        if not v_pert2 > v_current:
            self._log(f"Monotonicity violated: decreasing f_recede from {self.f_recede:.2f} to "
                     f"{f_recede_pert:.2f} did not increase v (got {v_pert2:.2f} vs {v_current:.2f})")
            success = False
        else:
            self._log(f"Monotonicity check 2 passed: v increases as f_recede decreases")
        
        # Perturb speed of sound (optional)
        c_pert = self.c * 1.01  # increase by 1%
        v_pert3 = c_pert * (r - 1) / (r + 1)
        # v is proportional to c, so v should increase when c increases
        if not v_pert3 > v_current:
            self._log(f"Monotonicity violated: increasing c from {self.c:.2f} to "
                     f"{c_pert:.2f} did not increase v (got {v_pert3:.2f} vs {v_current:.2f})")
            success = False
        else:
            self._log(f"Monotonicity check 3 passed: v increases with c")
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success and logs."""
        self.logs = []  # Reset logs
        results = []
        
        try:
            results.append(("Invariants", self.test_invariants()))
            results.append(("Trace Consistency", self.test_trace_consistency()))
            results.append(("Monotonicity", self.test_monotonicity()))
        except Exception as e:
            self._log(f"Unexpected error: {traceback.format_exc()}", is_error=True)
            results.append(("Execution", False))
        
        success = all(r[1] for r in results)
        
        # Summary
        self._log(f"All tests passed: {success}")
        for name, passed in results:
            self._log(f"  {name}: {'PASS' if passed else 'FAIL'}")
        
        return success, self.logs