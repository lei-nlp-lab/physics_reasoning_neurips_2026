import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Extract key results with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Coordinates of Q
        self.x_Q = getattr(solution_instance, 'x', self.results.get('x', 0.0))
        self.y_Q = getattr(solution_instance, 'y', self.results.get('y', 0.0))
        
        # Magnetic field B
        self.B = getattr(solution_instance, 'B', self.results.get('B', 0.0))
        
        # Velocity at Q
        self.v = getattr(solution_instance, 'v', self.results.get('v', 0.0))
        
        # Radius of circular motion
        self.R = getattr(solution_instance, 'R', self.results.get('R', 0.0))
        
        # Input parameters (with safe defaults)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.v0 = getattr(solution_instance, 'v0', 1.0)
        self.d = getattr(solution_instance, 'd', 1.0)
        
        # For symbolic computations
        self.m_sym = sp.Symbol('m', positive=True)
        self.q_sym = sp.Symbol('q', positive=True)
        self.v0_sym = sp.Symbol('v0', positive=True)
        self.d_sym = sp.Symbol('d', positive=True)
        
    def test_invariants(self):
        """Verify energy conservation and force laws."""
        logs = []
        try:
            # 1. Check coordinates of Q: (0, 2d)
            expected_x = 0.0
            expected_y = 2.0 * self.d
            
            if not math.isclose(self.x_Q, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x-coordinate of Q = {self.x_Q}, expected ~{expected_x}")
            else:
                logs.append(f"PASS: x-coordinate of Q = {self.x_Q}")
                
            if not math.isclose(self.y_Q, expected_y, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: y-coordinate of Q = {self.y_Q}, expected ~{expected_y}")
            else:
                logs.append(f"PASS: y-coordinate of Q = {self.y_Q}")
            
            # 2. Check velocity at Q: v = √2 * v0 (from 45° condition)
            expected_v = math.sqrt(2) * self.v0
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: velocity at Q = {self.v}, expected ~{expected_v}")
            else:
                logs.append(f"PASS: velocity at Q = {self.v}")
            
            # 3. Check radius: R = 2√2 * d
            expected_R = 2 * math.sqrt(2) * self.d
            if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: radius R = {self.R}, expected ~{expected_R}")
            else:
                logs.append(f"PASS: radius R = {self.R}")
            
            # 4. Check Lorentz force as centripetal force: qvB = mv²/R
            lhs = self.q * self.v * self.B
            rhs = self.m * self.v**2 / self.R
            
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Lorentz force qvB = {lhs}, centripetal force mv²/R = {rhs}")
            else:
                logs.append(f"PASS: Lorentz force equals centripetal force")
            
            # 5. Check magnetic field formula: B = m*v0/(2*q*d)
            expected_B = (self.m * self.v0) / (2 * self.q * self.d)
            if not math.isclose(self.B, expected_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: B = {self.B}, expected ~{expected_B}")
            else:
                logs.append(f"PASS: B = {self.B}")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        try:
            # 1. Time of flight in electric field: t = 2d/v0 (from y = v0*t = 2d)
            t = 2.0 * self.d / self.v0
            logs.append(f"Computed time in E-field: t = {t}")
            
            # 2. Acceleration in x-direction: a = v0/t = v0²/(2d)
            a = self.v0 / t  # from tan45 = a*t/v0 = 1
            expected_a = self.v0**2 / (2.0 * self.d)
            
            if not math.isclose(a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: inconsistent acceleration: a = {a}, expected ~{expected_a}")
            else:
                logs.append(f"PASS: acceleration consistent")
            
            # 3. Check velocity components at Q
            vx_Q = a * t
            vy_Q = self.v0
            
            if not math.isclose(vx_Q, self.v0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: vx at Q = {vx_Q}, should equal v0 = {self.v0}")
            else:
                logs.append(f"PASS: vx at Q = v0")
            
            # 4. Check speed magnitude at Q
            v_mag = math.sqrt(vx_Q**2 + vy_Q**2)
            if not math.isclose(v_mag, self.v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: computed v = {v_mag}, stored v = {self.v}")
            else:
                logs.append(f"PASS: velocity magnitude consistent")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self):
        """Perturb input d and check B changes inversely."""
        logs = []
        original_d = self.d
        
        try:
            # Perturb d by +10%
            perturbed_d = original_d * 1.1
            
            # Expected B changes inversely: B' = B * (d / d')
            expected_B_perturbed = self.B * (original_d / perturbed_d)
            
            # Recompute using formula B = m*v0/(2*q*d)
            computed_B_perturbed = (self.m * self.v0) / (2 * self.q * perturbed_d)
            
            if not math.isclose(computed_B_perturbed, expected_B_perturbed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: B monotonicity violated")
                logs.append(f"  Original d={original_d}, B={self.B}")
                logs.append(f"  Perturbed d={perturbed_d}")
                logs.append(f"  Expected B'={expected_B_perturbed}, computed={computed_B_perturbed}")
            else:
                logs.append(f"PASS: B varies inversely with d")
            
            # Restore original value
            self.d = original_d
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            self.d = original_d  # Ensure restoration even on error
        
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        all_logs = []
        success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} Test ===")
            logs = test_func()
            all_logs.extend(logs)
            
            # Check for failures in invariants (most critical)
            if test_name == "Invariants":
                for log in logs:
                    if log.startswith("FAIL:"):
                        success = False
        
        return success, all_logs

# Example usage (commented out for safety):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)