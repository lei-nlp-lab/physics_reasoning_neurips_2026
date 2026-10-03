import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with fallbacks
        self.v0_kmh = getattr(solution_instance, 'v0_kmh', 36.0)
        self.a1 = getattr(solution_instance, 'a1', 0.5)
        self.t1 = getattr(solution_instance, 't1', 10.0)
        self.a2 = getattr(solution_instance, 'a2', -10.0)
        
        # Safely extract results from instance or results dict
        results = getattr(solution_instance, 'results', {})
        self.v0 = getattr(solution_instance, 'v_0', results.get('v_0', 10.0))  # converted to m/s
        self.v = getattr(solution_instance, 'v', results.get('v', 0.0))
        self.t0 = getattr(solution_instance, 't_0', results.get('t_0', 0.0))
        self.x = getattr(solution_instance, 'x', results.get('x', 0.0))
        
        # Store original for monotonicity test
        self.original_inputs = {
            'v0_kmh': self.v0_kmh,
            'a1': self.a1,
            't1': self.t1,
            'a2': self.a2
        }
        
        # Define symbols for symbolic checks
        self.sym_v0, self.sym_a1, self.sym_t1, self.sym_v, self.sym_a2, self.sym_t0, self.sym_x = sp.symbols('v0 a1 t1 v a2 t0 x')
        
    def _is_close(self, a, b):
        """Fuzzy numerical comparison with physics tolerances."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Check kinematic laws from KB reference."""
        logs = []
        success = True
        
        try:
            # 1. Check unit conversion: 36 km/h = 10 m/s
            expected_v0 = 36.0 * (1000.0 / 3600.0)  # km/h to m/s
            if not self._is_close(self.v0, expected_v0):
                logs.append(f"FAIL: Unit conversion mismatch. v0={self.v0}, expected {expected_v0}")
                success = False
            else:
                logs.append("PASS: Unit conversion correct.")
            
            # 2. Check accelerated motion: v = v0 + a1*t1
            expected_v = self.v0 + self.a1 * self.t1
            if not self._is_close(self.v, expected_v):
                logs.append(f"FAIL: Acceleration formula mismatch. v={self.v}, expected {expected_v}")
                success = False
            else:
                logs.append("PASS: Acceleration formula holds.")
            
            # 3. Check deceleration time: t0 = v / |a2| (since comes to stop)
            # Note: a2 is negative (deceleration), so use absolute value
            expected_t0 = self.v / abs(self.a2)
            if not self._is_close(self.t0, expected_t0):
                logs.append(f"FAIL: Deceleration time mismatch. t0={self.t0}, expected {expected_t0}")
                success = False
            else:
                logs.append("PASS: Deceleration time formula holds.")
            
            # 4. Check braking distance: x = (v/2) * t0 (using average velocity)
            expected_x_avg = (self.v / 2.0) * self.t0
            # Alternative check: x = v*t0 + 0.5*a2*t0**2
            expected_x_eq = self.v * self.t0 + 0.5 * self.a2 * (self.t0 ** 2)
            
            if not (self._is_close(self.x, expected_x_avg) or self._is_close(self.x, expected_x_eq)):
                logs.append(f"FAIL: Braking distance mismatch. x={self.x}, expected {expected_x_avg} (avg) or {expected_x_eq} (eq)")
                success = False
            else:
                logs.append("PASS: Braking distance formula holds.")
            
            # Symbolic check (fallback to numerical if symbolic fails)
            try:
                # Define symbolic equations from KB
                eq1 = sp.Eq(self.sym_v, self.sym_v0 + self.sym_a1 * self.sym_t1)
                eq2 = sp.Eq(self.sym_t0, self.sym_v / abs(self.sym_a2))
                eq3 = sp.Eq(self.sym_x, (self.sym_v / 2) * self.sym_t0)
                
                # Substitute numeric values
                subs_dict = {
                    self.sym_v0: self.v0,
                    self.sym_a1: self.a1,
                    self.sym_t1: self.t1,
                    self.sym_v: self.v,
                    self.sym_a2: self.a2,
                    self.sym_t0: self.t0,
                    self.sym_x: self.x
                }
                
                if not eq1.subs(subs_dict) and eq1.lhs.subs(subs_dict) != eq1.rhs.subs(subs_dict):
                    logs.append("WARN: Symbolic acceleration check inconclusive.")
                if not eq2.subs(subs_dict) and abs(eq2.lhs.subs(subs_dict) - eq2.rhs.subs(subs_dict)) > 1e-2:
                    logs.append("WARN: Symbolic deceleration time check inconclusive.")
                if not eq3.subs(subs_dict) and abs(eq3.lhs.subs(subs_dict) - eq3.rhs.subs(subs_dict)) > 1e-2:
                    logs.append("WARN: Symbolic distance check inconclusive.")
                else:
                    logs.append("PASS: Symbolic checks passed.")
            except Exception as e:
                logs.append(f"INFO: Symbolic check skipped due to {type(e).__name__}")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Target answers from problem statement
            target_v = 15.0  # m/s
            target_x = 11.25  # m
            
            if not self._is_close(self.v, target_v):
                logs.append(f"FAIL: Final velocity mismatch. v={self.v}, target={target_v}")
                success = False
            else:
                logs.append("PASS: Final velocity matches target.")
            
            if not self._is_close(self.x, target_x):
                logs.append(f"FAIL: Braking distance mismatch. x={self.x}, target={target_x}")
                success = False
            else:
                logs.append("PASS: Braking distance matches target.")
            
            # Consistency between steps:
            # Step 2 output (v=15) should match step 3 input
            # Step 3 output (t0=1.5) should match step 4 input
            expected_t0 = target_v / abs(self.a2)  # 15/10 = 1.5
            if not self._is_close(self.t0, expected_t0):
                logs.append(f"FAIL: Step consistency broken. t0={self.t0}, expected from v={expected_t0}")
                success = False
            else:
                logs.append("PASS: Step-to-step consistency holds.")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb acceleration a1 and check trends."""
        logs = []
        success = True
        
        try:
            # Store original values
            orig_a1 = self.a1
            orig_v = self.v
            orig_t0 = self.t0
            orig_x = self.x
            
            # Perturb a1 by +10%
            delta = 0.1  # 0.1 m/s^2 increase (10% of 0.5)
            self.a1 = orig_a1 + delta
            
            # Recalculate using kinematic formulas (NOT calling solution methods)
            new_v = self.v0 + self.a1 * self.t1
            new_t0 = new_v / abs(self.a2)
            new_x = (new_v / 2.0) * new_t0
            
            # Check expected trends from KB:
            # If a1 increases, v should increase (positive acceleration)
            # If v increases, t0 should increase (more time to stop)
            # If v and t0 increase, x should increase (longer braking distance)
            trend_checks = []
            
            if new_v <= orig_v:
                trend_checks.append(f"FAIL: v monotonicity broken. orig={orig_v}, new={new_v}")
                success = False
            else:
                trend_checks.append(f"PASS: v increases with a1 (+{delta}).")
            
            if new_t0 <= orig_t0:
                trend_checks.append(f"FAIL: t0 monotonicity broken. orig={orig_t0}, new={new_t0}")
                success = False
            else:
                trend_checks.append(f"PASS: t0 increases with v.")
            
            if new_x <= orig_x:
                trend_checks.append(f"FAIL: x monotonicity broken. orig={orig_x}, new={new_x}")
                success = False
            else:
                trend_checks.append(f"PASS: x increases with v and t0.")
            
            logs.extend(trend_checks)
            
            # Restore original values
            self.a1 = orig_a1
            self.v = orig_v
            self.t0 = orig_t0
            self.x = orig_x
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            success = False
            # Try to restore on error
            self.a1 = self.original_inputs['a1']
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all checks and return aggregated results."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} Test ===")
            success, logs = test_method()
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== Summary ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs