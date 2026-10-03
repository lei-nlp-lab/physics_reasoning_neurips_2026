import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Use getattr to avoid AttributeError in sandbox.
        """
        # Inputs from the problem statement
        self.g = 10.0  # gravitational acceleration
        self.v_ball_initial = 14.0  # m/s
        self.resistance_coef = 0.2
        self.forward_accel = 2.5  # m/s^2
        self.forward_vmax = 10.0  # m/s
        self.delta_x0 = 1.0  # m, forward starts 1m behind ball
        
        # Extract results safely, fallback to 0.0 if missing
        self.results = getattr(solution_instance, 'results', {})
        self.a1 = self.results.get('a_1', 0.0)
        self.v2 = self.results.get('v_2', 0.0)
        self.x3 = self.results.get('x_3', 0.0)
        self.t2 = self.results.get('t_2', 0.0)
        self.x1 = self.results.get('x_1', 0.0)
        self.t3 = self.results.get('t_3', 0.0)
        self.a2 = self.results.get('a_2', 0.0)
        self.x2 = self.results.get('x_2', 0.0)
        self.v1 = self.results.get('v_1', 0.0)
        self.vm = self.results.get('v_m', 0.0)
        
        # Symbols for symbolic checks
        self.m = sp.symbols('m', positive=True)  # ball mass
        self.t = sp.symbols('t', positive=True)
        
    def test_invariants(self):
        """Verify physics invariants: Newton's 2nd Law, kinematic formulas."""
        logs = []
        
        # 1. Newton's 2nd Law for the ball: F_resistance = m * a1, with F_resistance = 0.2*m*g
        try:
            F_res = 0.2 * self.m * self.g
            a1_expected = sp.simplify(F_res / self.m)
            a1_expected_val = float(a1_expected.subs({self.g: 10.0}))
            if not math.isclose(self.a1, a1_expected_val, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Newton's 2nd Law: a1={self.a1}, expected {a1_expected_val}")
            else:
                logs.append(f"PASS: Newton's 2nd Law: a1={self.a1} matches resistance force.")
        except Exception as e:
            logs.append(f"ERROR in Newton's 2nd Law check: {e}")
        
        # 2. Uniformly decelerated motion for the ball: v2 = v1 - a1*t1 (with t1=2s from KB)
        try:
            t1 = 2.0  # time for step 2
            v2_expected = self.v1 - self.a1 * t1
            if not math.isclose(self.v2, v2_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Kinematic consistency v2: {self.v2} vs expected {v2_expected}")
            else:
                logs.append(f"PASS: v2 = v1 - a1*t1 holds.")
        except Exception as e:
            logs.append(f"ERROR in v2 check: {e}")
        
        # 3. Time to stop for ball: t2 = v1 / a1
        try:
            t2_expected = self.v1 / self.a1 if abs(self.a1) > 1e-12 else float('inf')
            if not math.isclose(self.t2, t2_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: t2 = v1/a1: {self.t2} vs expected {t2_expected}")
            else:
                logs.append(f"PASS: t2 = v1/a1 holds.")
        except Exception as e:
            logs.append(f"ERROR in t2 check: {e}")
        
        # 4. Displacement for ball until stop: x1 = (v1/2)*t2
        try:
            x1_expected = (self.v1 / 2.0) * self.t2
            if not math.isclose(self.x1, x1_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: x1 = (v1/2)*t2: {self.x1} vs expected {x1_expected}")
            else:
                logs.append(f"PASS: x1 = (v1/2)*t2 holds.")
        except Exception as e:
            logs.append(f"ERROR in x1 check: {e}")
        
        # 5. Forward's acceleration to max speed: t3 = vm / a2
        try:
            t3_expected = self.vm / self.a2 if abs(self.a2) > 1e-12 else float('inf')
            if not math.isclose(self.t3, t3_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: t3 = vm/a2: {self.t3} vs expected {t3_expected}")
            else:
                logs.append(f"PASS: t3 = vm/a2 holds.")
        except Exception as e:
            logs.append(f"ERROR in t3 check: {e}")
        
        # 6. Forward's displacement during acceleration: x2 = (vm/2)*t3
        try:
            x2_expected = (self.vm / 2.0) * self.t3
            if not math.isclose(self.x2, x2_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: x2 = (vm/2)*t3: {self.x2} vs expected {x2_expected}")
            else:
                logs.append(f"PASS: x2 = (vm/2)*t3 holds.")
        except Exception as e:
            logs.append(f"ERROR in x2 check: {e}")
        
        # 7. Forward's displacement at constant speed: x3 = vm*(t2 - t3)
        try:
            x3_expected = self.vm * (self.t2 - self.t3)
            if not math.isclose(self.x3, x3_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: x3 = vm*(t2 - t3): {self.x3} vs expected {x3_expected}")
            else:
                logs.append(f"PASS: x3 = vm*(t2 - t3) holds.")
        except Exception as e:
            logs.append(f"ERROR in x3 check: {e}")
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        
        # Final answers: v2=10 m/s, and catch-up time=7 s (t2)
        expected_v2 = 10.0
        expected_catch_time = 7.0
        
        if not math.isclose(self.v2, expected_v2, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: v2={self.v2} does not match expected answer {expected_v2} m/s")
        else:
            logs.append(f"PASS: v2 matches target answer {expected_v2} m/s")
        
        if not math.isclose(self.t2, expected_catch_time, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: catch-up time t2={self.t2} does not match expected {expected_catch_time} s")
        else:
            logs.append(f"PASS: catch-up time t2 matches target answer {expected_catch_time} s")
        
        # Check if forward catches the ball: total forward displacement must be >= ball displacement + initial gap
        total_forward_displacement = self.x2 + self.x3  # from solution
        required_displacement = self.x1 + self.delta_x0  # ball displacement + 1m gap
        
        if total_forward_displacement < required_displacement - 0.1:  # tolerance
            logs.append(f"FAIL: forward displacement {total_forward_displacement} < required {required_displacement}")
        elif total_forward_displacement > required_displacement + 0.1:
            logs.append(f"WARN: forward displacement {total_forward_displacement} > required {required_displacement} (overshoot)")
        else:
            logs.append(f"PASS: forward catches ball exactly at t={self.t2} s")
        
        return logs
    
    def test_monotonicity(self):
        """
        Perturb an input parameter and check the trend.
        Perturb resistance coefficient (increase) and verify that ball's stopping distance decreases.
        Restore original values after.
        """
        logs = []
        
        # Save original values
        original_a1 = self.a1
        original_t2 = self.t2
        original_x1 = self.x1
        
        # Perturb resistance coefficient: increase by 10%
        perturbed_coef = self.resistance_coef * 1.10
        perturbed_a1 = perturbed_coef * self.g  # a1 = 0.2g -> now 0.22g
        
        # Recalculate using formulas
        perturbed_t2 = self.v1 / perturbed_a1
        perturbed_x1 = (self.v1 / 2.0) * perturbed_t2
        
        # Check monotonicity: if resistance increases, stopping distance should decrease
        if perturbed_x1 < original_x1:
            logs.append(f"PASS: monotonicity - increased resistance reduces stopping distance (from {original_x1:.2f} to {perturbed_x1:.2f} m)")
        else:
            logs.append(f"FAIL: monotonicity - increased resistance did not reduce stopping distance")
        
        # Restore original values
        self.a1 = original_a1
        self.t2 = original_t2
        self.x1 = original_x1
        
        return logs
    
    def run_all_tests(self):
        """Run all tests and return success boolean and logs."""
        logs = []
        success = True
        
        try:
            logs.append("=== PhysicsChecker starting ===")
            
            # Test invariants
            inv_logs = self.test_invariants()
            logs.extend(inv_logs)
            if any("FAIL" in log for log in inv_logs):
                success = False
            
            # Test trace consistency
            trace_logs = self.test_trace_consistency()
            logs.extend(trace_logs)
            if any("FAIL" in log for log in trace_logs):
                success = False
            
            # Test monotonicity
            mono_logs = self.test_monotonicity()
            logs.extend(mono_logs)
            if any("FAIL" in log for log in mono_logs):
                success = False
            
            logs.append(f"=== PhysicsChecker finished: {'SUCCESS' if success else 'FAILURE'} ===")
        except Exception as e:
            logs.append(f"CRITICAL ERROR: {e}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs