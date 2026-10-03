import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Robust initialization with safe attribute access.
        """
        # Extract inputs with fallbacks
        self.m = getattr(solution_instance, 'm', 5.0)
        self.mu = getattr(solution_instance, 'mu', 0.2)
        self.F = getattr(solution_instance, 'F', 15.0)
        self.t = getattr(solution_instance, 't', 5.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results safely from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        self.a = results.get('a', 0.0)
        self.a_2 = results.get('a_2', 0.0)
        self.x_2 = results.get('x_2', 0.0)
        self.v = results.get('v', 0.0)
        self.x_1 = results.get('x_1', 0.0)
        
        self.logs = []
        
    def log(self, message):
        """Helper for logging."""
        self.logs.append(message)
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        try:
            # Step 1: Newton's Second Law (during pulling)
            # Net force = F - friction
            friction = self.mu * self.m * self.g
            net_force = self.F - friction
            expected_a = net_force / self.m if self.m != 0 else 0
            if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Invariant violation: a={self.a} but expected {expected_a} from F_net/m")
                return False
            
            # Step 2 & 3: Kinematic consistency with acceleration a
            # v = a*t
            expected_v = self.a * self.t
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Invariant violation: v={self.v} but expected a*t={expected_v}")
                return False
            
            # x_1 = 0.5*a*t^2
            expected_x1 = 0.5 * self.a * (self.t ** 2)
            if not math.isclose(self.x_1, expected_x1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Invariant violation: x_1={self.x_1} but expected 0.5*a*t^2={expected_x1}")
                return False
            
            # Step 4: Newton's Second Law (after force removal)
            # Only friction acts: a_2 = friction/m = mu*g
            expected_a2 = self.mu * self.g
            if not math.isclose(self.a_2, expected_a2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Invariant violation: a_2={self.a_2} but expected mu*g={expected_a2}")
                return False
            
            # Step 5: Kinematic deceleration (v^2 = 2*a_2*x_2)
            # x_2 = v^2/(2*a_2) if a_2 > 0
            if self.a_2 != 0:
                expected_x2 = (self.v ** 2) / (2 * abs(self.a_2))
                if not math.isclose(self.x_2, expected_x2, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Invariant violation: x_2={self.x_2} but expected v^2/(2*a_2)={expected_x2}")
                    return False
            
            self.log("All invariants passed.")
            return True
        except Exception as e:
            self.log(f"Error in test_invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Total displacement should be x_1 + x_2
            total_displacement = self.x_1 + self.x_2
            # Check if x_2 is consistent with x_1 and v via step-by-step
            # From step 5, x_2 depends on v and a_2, which themselves depend on earlier steps
            # Already checked in invariants, but double-check here
            if self.a_2 != 0:
                recalc_x2 = (self.v ** 2) / (2 * abs(self.a_2))
                if not math.isclose(self.x_2, recalc_x2, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Trace inconsistency: x_2={self.x_2} vs recalculated {recalc_x2}")
                    return False
            
            # Check that a and a_2 have correct signs
            if self.a <= 0:
                self.log(f"Trace inconsistency: a should be positive but got {self.a}")
                return False
            if self.a_2 <= 0:
                self.log(f"Trace inconsistency: a_2 should be positive (deceleration) but got {self.a_2}")
                return False
            
            # Check that v is consistent with a and t
            recalc_v = self.a * self.t
            if not math.isclose(self.v, recalc_v, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Trace inconsistency: v={self.v} vs a*t={recalc_v}")
                return False
            
            self.log("Trace consistency passed.")
            return True
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb mass and check expected trends."""
        original_m = self.m
        try:
            # Increase mass by 10%
            self.m *= 1.1
            # Recalculate using the same formulas (but without modifying other results)
            # Step 1: a_new = (F - mu*m*g)/m
            a_new = (self.F - self.mu * self.m * self.g) / self.m
            # Since m increased, denominator increased and friction increased, so a should decrease
            if a_new >= self.a:  # Note: using original a stored in self.a
                self.log(f"Monotonicity violation: a should decrease when mass increases (old={self.a}, new={a_new})")
                self.m = original_m
                return False
            
            # Step 4: a_2_new = mu*g (independent of mass)
            a_2_new = self.mu * self.g
            if not math.isclose(a_2_new, self.a_2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Monotonicity anomaly: a_2 changed with mass (old={self.a_2}, new={a_2_new})")
                self.m = original_m
                return False
            
            # Step 3: v_new = a_new * t
            v_new = a_new * self.t
            if v_new >= self.v:
                self.log(f"Monotonicity violation: v should decrease when mass increases (old={self.v}, new={v_new})")
                self.m = original_m
                return False
            
            # Step 5: x_2_new = v_new^2/(2*a_2_new)
            x_2_new = (v_new ** 2) / (2 * a_2_new)
            if x_2_new >= self.x_2:
                self.log(f"Monotonicity violation: x_2 should decrease when mass increases (old={self.x_2}, new={x_2_new})")
                self.m = original_m
                return False
            
            # Restore original mass
            self.m = original_m
            self.log("Monotonicity test passed.")
            return True
        except Exception as e:
            self.m = original_m
            self.log(f"Error in test_monotonicity: {traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        success = True
        
        # Run tests in order
        if not self.test_invariants():
            success = False
        if not self.test_trace_consistency():
            success = False
        if not self.test_monotonicity():
            success = False
        
        return success, self.logs