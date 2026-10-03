import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts inputs and results from solution instance.
        Uses getattr with fallback to results dictionary to prevent AttributeError.
        """
        self.solution = solution_instance
        
        # Extract inputs with safe access
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 0.5))
        self.H = getattr(solution_instance, 'H', solution_instance.results.get('H', 0.8))
        self.mu = getattr(solution_instance, 'mu', solution_instance.results.get('mu', 0.2))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 10.0))
        
        # Extract results with safe access
        self.omega = getattr(solution_instance, 'omega', 
                            solution_instance.results.get('omega', 
                                                         solution_instance.results.get('ω', 0.0)))
        self.s = getattr(solution_instance, 's', 
                        solution_instance.results.get('s', 0.0))
        self.t = getattr(solution_instance, 't', 
                        solution_instance.results.get('t', 0.0))
        
        # Derived expected values for checking
        self.expected_omega = math.sqrt(self.mu * self.g / self.R)
        self.expected_t = math.sqrt(2 * self.H / self.g)
        self.expected_s = self.expected_omega * self.R * self.expected_t
        
        self.logs = []
        
    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
        
    def _fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physics laws are satisfied."""
        success = True
        
        # 1. Check centripetal force equals friction force (Newton's 2nd Law for rotation)
        centripetal_force = self.omega**2 * self.R  # ω²R (mass cancels)
        friction_force = self.mu * self.g  # μg (mass cancels)
        if not self._fuzzy_equal(centripetal_force, friction_force):
            success = self._log(f"Centripetal force ({centripetal_force:.3f}) != friction force ({friction_force:.3f})", False)
        else:
            self._log("Centripetal force = friction force ✓", True)
        
        # 2. Check free fall time from vertical motion
        if not self._fuzzy_equal(self.t, self.expected_t):
            success = self._log(f"Time t={self.t:.3f}s differs from expected sqrt(2H/g)={self.expected_t:.3f}s", False)
        else:
            self._log(f"Free fall time t={self.t:.3f}s ✓", True)
        
        # 3. Check horizontal range from projectile motion
        # s = ωR * t = ωR * sqrt(2H/g)
        computed_s = self.omega * self.R * self.expected_t
        if not self._fuzzy_equal(self.s, computed_s):
            success = self._log(f"Range s={self.s:.3f}m differs from ωRt={computed_s:.3f}m", False)
        else:
            self._log(f"Horizontal range s={self.s:.3f}m ✓", True)
            
        # 4. Check energy conservation (rotational KE to translational KE)
        # At sliding point: KE_rot = ½m(ωR)², friction work = -μmg*(arc length?)
        # Since friction is tangential, it does work reducing rotational energy
        # But this check is complex without arc length; we'll skip explicit energy check
        # as friction work path isn't provided
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        success = True
        
        # 1. Verify ω calculation from step 1
        if not self._fuzzy_equal(self.omega, self.expected_omega):
            success = self._log(f"ω={self.omega:.3f} rad/s differs from sqrt(μg/R)={self.expected_omega:.3f} rad/s", False)
        else:
            self._log(f"Angular velocity ω={self.omega:.3f} rad/s consistent ✓", True)
        
        # 2. Verify t calculation from step 2
        if not self._fuzzy_equal(self.t, self.expected_t):
            success = self._log(f"t={self.t:.3f}s differs from sqrt(2H/g)={self.expected_t:.3f}s", False)
        else:
            self._log(f"Time of flight t={self.t:.3f}s consistent ✓", True)
        
        # 3. Verify s calculation from step 4: s = ωR * t
        computed_s_from_t = self.omega * self.R * self.t
        if not self._fuzzy_equal(self.s, computed_s_from_t):
            success = self._log(f"s={self.s:.3f}m differs from ωRt={computed_s_from_t:.3f}m", False)
        else:
            self._log(f"Range s={self.s:.3f}m = ωRt ✓", True)
        
        # 4. Alternative formula check: s = sqrt(2H/g) * ωR
        computed_s_alt = self.expected_t * self.omega * self.R
        if not self._fuzzy_equal(self.s, computed_s_alt):
            success = self._log(f"s={self.s:.3f}m differs from sqrt(2H/g)*ωR={computed_s_alt:.3f}m", False)
        else:
            self._log(f"Range s matches alternative formula ✓", True)
        
        return success
    
    def test_monotonicity(self):
        """Perturb input parameters and verify expected trends."""
        success = True
        
        # Save original values
        original_mu = self.mu
        original_omega = self.omega
        original_s = self.s
        
        # Test 1: Increase μ by 10%, ω should increase (ω ∝ √μ)
        self.mu *= 1.10
        new_omega = math.sqrt(self.mu * self.g / self.R)
        new_s = new_omega * self.R * math.sqrt(2 * self.H / self.g)
        
        if new_omega <= original_omega:
            success = self._log(f"ω doesn't increase when μ increases: {new_omega:.3f} ≤ {original_omega:.3f}", False)
        else:
            self._log(f"ω increases with μ ({new_omega:.3f} > {original_omega:.3f}) ✓", True)
        
        if new_s <= original_s:
            success = self._log(f"s doesn't increase when μ increases: {new_s:.3f} ≤ {original_s:.3f}", False)
        else:
            self._log(f"s increases with μ ({new_s:.3f} > {original_s:.3f}) ✓", True)
        
        # Restore μ and test decrease
        self.mu = original_mu * 0.90
        new_omega = math.sqrt(self.mu * self.g / self.R)
        new_s = new_omega * self.R * math.sqrt(2 * self.H / self.g)
        
        if new_omega >= original_omega:
            success = self._log(f"ω doesn't decrease when μ decreases: {new_omega:.3f} ≥ {original_omega:.3f}", False)
        else:
            self._log(f"ω decreases with μ ({new_omega:.3f} < {original_omega:.3f}) ✓", True)
        
        if new_s >= original_s:
            success = self._log(f"s doesn't decrease when μ decreases: {new_s:.3f} ≥ {original_s:.3f}", False)
        else:
            self._log(f"s decreases with μ ({new_s:.3f} < {original_s:.3f}) ✓", True)
        
        # Test 2: Increase R by 10%, ω should decrease (ω ∝ 1/√R)
        original_R = self.R
        self.R *= 1.10
        new_omega = math.sqrt(original_mu * self.g / self.R)
        new_s = new_omega * self.R * math.sqrt(2 * self.H / self.g)
        
        if new_omega >= original_omega:
            success = self._log(f"ω doesn't decrease when R increases: {new_omega:.3f} ≥ {original_omega:.3f}", False)
        else:
            self._log(f"ω decreases with R ({new_omega:.3f} < {original_omega:.3f}) ✓", True)
        
        # s = ωR, and ω ∝ 1/√R, so s ∝ √R, should increase
        if new_s <= original_s:
            success = self._log(f"s doesn't increase when R increases: {new_s:.3f} ≤ {original_s:.3f}", False)
        else:
            self._log(f"s increases with R ({new_s:.3f} > {original_s:.3f}) ✓", True)
        
        # Restore all original values
        self.mu = original_mu
        self.R = original_R
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs = []
        self.logs.append("=== Physics Checker Started ===")
        
        overall_success = True
        
        try:
            # Run test suite
            self.logs.append("\n--- Test 1: Physics Invariants ---")
            if not self.test_invariants():
                overall_success = False
                
            self.logs.append("\n--- Test 2: Trace Consistency ---")
            if not self.test_trace_consistency():
                overall_success = False
                
            self.logs.append("\n--- Test 3: Monotonicity Checks ---")
            if not self.test_monotonicity():
                overall_success = False
                
        except Exception as e:
            self.logs.append(f"ERROR: Exception during testing: {str(e)}")
            self.logs.append(traceback.format_exc())
            overall_success = False
        
        self.logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, self.logs

# Example usage guard
if __name__ == "__main__":
    # Example mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.R = 0.5
            self.H = 0.8
            self.mu = 0.2
            self.g = 10.0
            self.results = {
                'omega': 2.0,
                's': 0.4,
                't': 0.4
            }
    
    mock = MockSolution()
    checker = PhysicsChecker(mock)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)