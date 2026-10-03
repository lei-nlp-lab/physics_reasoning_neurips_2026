import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.inputs = getattr(solution_instance, 'inputs', {})
        
        # Extract inputs with defaults
        self.m = float(self.inputs.get('m', 0.5))
        self.v0 = float(self.inputs.get('v0', 8.0))
        self.theta_deg = float(self.inputs.get('theta_deg', 37.0))
        self.mu = float(self.inputs.get('mu', 0.25))
        self.g = float(self.inputs.get('g', 10.0))
        self.sin37 = float(self.inputs.get('sin37', 0.6))
        self.cos37 = float(self.inputs.get('cos37', 0.8))
        
        # Convert theta to radians
        self.theta = math.radians(self.theta_deg)
        
        # Extract results with safe access
        self.a1 = float(getattr(solution_instance, 'a_1', self.results.get('a_1', 0.0)))
        self.t1 = float(getattr(solution_instance, 't_1', self.results.get('t_1', 0.0)))
        self.x = float(getattr(solution_instance, 'x', self.results.get('x', 0.0)))
        self.a2 = float(getattr(solution_instance, 'a_2', self.results.get('a_2', 0.0)))
        self.t2 = float(getattr(solution_instance, 't_2', self.results.get('t_2', 0.0)))
        self.t_total = float(getattr(solution_instance, 't', self.results.get('t', 0.0)))
        
        # Expected values from steps analysis
        self.expected_a1 = 8.0  # m/s²
        self.expected_t1 = 1.0  # s
        self.expected_x = 4.0   # m
        self.expected_a2 = 4.0  # m/s²
        self.expected_t2 = math.sqrt(2)  # ≈1.414 s
        self.expected_t_total = 1.0 + math.sqrt(2)  # ≈2.414 s
        
        self.logs = []
    
    def _log(self, msg, success=True):
        self.logs.append((success, msg))
        return success
    
    def test_invariants(self):
        """Verify Newton's Laws and kinematic consistency."""
        success = True
        
        # 1. Check Newton's 2nd Law for upward motion
        # a1 = g*sinθ + μ*g*cosθ
        expected_a1_formula = self.g * self.sin37 + self.mu * self.g * self.cos37
        if not math.isclose(self.a1, expected_a1_formula, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"Newton's 2nd Law (upward) failed: a1={self.a1}, expected {expected_a1_formula}"
            success = success and self._log(msg, False)
        else:
            self._log(f"Newton's 2nd Law (upward) check passed: a1={self.a1}")
        
        # 2. Check Newton's 2nd Law for downward motion
        # a2 = g*sinθ - μ*g*cosθ
        expected_a2_formula = self.g * self.sin37 - self.mu * self.g * self.cos37
        if not math.isclose(self.a2, expected_a2_formula, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"Newton's 2nd Law (downward) failed: a2={self.a2}, expected {expected_a2_formula}"
            success = success and self._log(msg, False)
        else:
            self._log(f"Newton's 2nd Law (downward) check passed: a2={self.a2}")
        
        # 3. Check kinematic consistency for upward motion
        # t1 = v0 / a1
        expected_t1 = self.v0 / self.a1 if abs(self.a1) > 1e-10 else float('inf')
        if not math.isclose(self.t1, expected_t1, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"Kinematic check (upward) failed: t1={self.t1}, expected v0/a1={expected_t1}"
            success = success and self._log(msg, False)
        else:
            self._log(f"Kinematic check (upward) passed: t1={self.t1}")
        
        # 4. Check kinematic consistency for downward motion
        # x = 0.5*a2*t2²
        if abs(self.a2) > 1e-10:
            expected_t2 = math.sqrt(2 * self.x / self.a2)
            if not math.isclose(self.t2, expected_t2, rel_tol=1e-2, abs_tol=1e-2):
                msg = f"Kinematic check (downward) failed: t2={self.t2}, expected √(2x/a2)={expected_t2}"
                success = success and self._log(msg, False)
            else:
                self._log(f"Kinematic check (downward) passed: t2={self.t2}")
        
        # 5. Check total time consistency
        expected_total = self.t1 + self.t2
        if not math.isclose(self.t_total, expected_total, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"Total time consistency failed: t={self.t_total}, expected t1+t2={expected_total}"
            success = success and self._log(msg, False)
        else:
            self._log(f"Total time consistency passed: t={self.t_total}")
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        success = True
        
        # Check if upward displacement matches downward displacement
        # Both should use the same x
        upward_x = 0.5 * self.a1 * (self.t1 ** 2)
        if not math.isclose(self.x, upward_x, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"Displacement mismatch: x={self.x}, upward calculation gives {upward_x}"
            success = success and self._log(msg, False)
        else:
            self._log(f"Displacement consistency passed: x={self.x}")
        
        # Check if downward time matches using x and a2
        if abs(self.a2) > 1e-10:
            downward_t2 = math.sqrt(2 * self.x / self.a2)
            if not math.isclose(self.t2, downward_t2, rel_tol=1e-2, abs_tol=1e-2):
                msg = f"Downward time mismatch: t2={self.t2}, calculated from x and a2 gives {downward_t2}"
                success = success and self._log(msg, False)
            else:
                self._log(f"Downward time consistency passed: t2={self.t2}")
        
        # Check final answer format against target answers
        # Target answers are: "8 m/s²" and "(1+√2)s"
        # We'll check if our computed values are close to these
        if not math.isclose(self.a1, self.expected_a1, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"Final answer a1 mismatch: {self.a1} vs expected {self.expected_a1}"
            success = success and self._log(msg, False)
        else:
            self._log(f"Final answer a1 check passed: {self.a1}")
        
        if not math.isclose(self.t_total, self.expected_t_total, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"Final answer total time mismatch: {self.t_total} vs expected {self.expected_t_total}"
            success = success and self._log(msg, False)
        else:
            self._log(f"Final answer total time check passed: {self.t_total}")
        
        return success
    
    def test_monotonicity(self):
        """Perturb input (μ) and verify expected trends."""
        success = True
        
        # Save original values
        original_mu = self.mu
        
        # Perturb μ by +10%
        self.mu *= 1.1
        
        # Recalculate using formulas
        new_a1 = self.g * self.sin37 + self.mu * self.g * self.cos37
        new_a2 = self.g * self.sin37 - self.mu * self.g * self.cos37
        new_t1 = self.v0 / new_a1
        new_x = 0.5 * new_a1 * (new_t1 ** 2)
        new_t2 = math.sqrt(2 * new_x / new_a2) if new_a2 > 1e-10 else float('inf')
        new_total = new_t1 + new_t2
        
        # Restore original μ
        self.mu = original_mu
        
        # Expected trends when μ increases:
        # 1. a1 should increase (more friction opposing upward motion)
        # 2. a2 should decrease (more friction opposing downward motion)
        # 3. t1 should decrease (larger deceleration)
        # 4. x should decrease (shorter distance before stopping)
        # 5. t2 might increase or decrease depending on x and a2
        
        if not (new_a1 > self.a1):
            msg = f"Monotonicity failed: a1 should increase with μ. Original: {self.a1}, New: {new_a1}"
            success = success and self._log(msg, False)
        else:
            self._log(f"Monotonicity a1 passed: {self.a1} -> {new_a1} (increased)")
        
        if not (new_a2 < self.a2):
            msg = f"Monotonicity failed: a2 should decrease with μ. Original: {self.a2}, New: {new_a2}"
            success = success and self._log(msg, False)
        else:
            self._log(f"Monotonicity a2 passed: {self.a2} -> {new_a2} (decreased)")
        
        if not (new_t1 < self.t1):
            msg = f"Monotonicity failed: t1 should decrease with μ. Original: {self.t1}, New: {new_t1}"
            success = success and self._log(msg, False)
        else:
            self._log(f"Monotonicity t1 passed: {self.t1} -> {new_t1} (decreased)")
        
        if not (new_x < self.x):
            msg = f"Monotonicity failed: x should decrease with μ. Original: {self.x}, New: {new_x}"
            success = success and self._log(msg, False)
        else:
            self._log(f"Monotonicity x passed: {self.x} -> {new_x} (decreased)")
        
        # Check that new values are physically plausible
        if new_a1 <= 0 or new_a2 <= 0:
            msg = f"Monotonicity check: invalid accelerations after perturbation: a1={new_a1}, a2={new_a2}"
            success = success and self._log(msg, False)
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        self.logs = []  # Reset logs
        
        try:
            success = True
            
            # Run tests
            success = self.test_invariants() and success
            success = self.test_trace_consistency() and success
            success = self.test_monotonicity() and success
            
            # Summary
            if success:
                self._log("All physics checks passed!", True)
            else:
                self._log("Some physics checks failed.", False)
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Checker crashed: {str(e)}\n{traceback.format_exc()}"
            return False, [(False, error_msg)]