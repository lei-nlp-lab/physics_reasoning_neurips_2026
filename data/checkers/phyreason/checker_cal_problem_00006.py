import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.S = getattr(solution_instance, 'S', solution_instance.results.get('S', 0.0))
        self.h = getattr(solution_instance, 'h', solution_instance.results.get('h', 0.0))
        self.d = getattr(solution_instance, 'd', solution_instance.results.get('d', 0.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.81))
        
        # Safe attribute access for results
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', 0.0))
        self.v_0 = getattr(solution_instance, 'v_0', solution_instance.results.get('v_0', 0.0))
        self.Q = getattr(solution_instance, 'Q', solution_instance.results.get('Q', 0.0))
        
        self.logs = []
        
    def test_invariants(self):
        """Verify energy conservation and Newton's laws based on KB."""
        try:
            # Invariant 1: Projectile motion equations consistency
            # From step_1: t = sqrt(2h/g)
            t_expected = math.sqrt(2 * self.h / self.g)
            if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Time of flight invariant. Expected t={t_expected:.3f}, got {self.t:.3f}")
                return False
            else:
                self.logs.append(f"PASS: Time of flight invariant (t = sqrt(2h/g))")
            
            # Invariant 2: Horizontal motion v0 = d/t
            v0_from_horizontal = self.d / self.t
            if not math.isclose(self.v_0, v0_from_horizontal, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Horizontal motion invariant. Expected v0={v0_from_horizontal:.3f}, got {self.v_0:.3f}")
                return False
            else:
                self.logs.append(f"PASS: Horizontal motion invariant (v0 = d/t)")
            
            # Invariant 3: Volume flow rate Q = S * v0
            Q_expected = self.S * self.v_0
            if not math.isclose(self.Q, Q_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Flow rate invariant. Expected Q={Q_expected:.3f}, got {self.Q:.3f}")
                return False
            else:
                self.logs.append(f"PASS: Flow rate invariant (Q = S * v0)")
            
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # Step 1 -> Step 2: v0 = d * sqrt(g/(2h))
            v0_expected = self.d * math.sqrt(self.g / (2 * self.h))
            if not math.isclose(self.v_0, v0_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Trace consistency v0. Expected {v0_expected:.3f}, got {self.v_0:.3f}")
                return False
            else:
                self.logs.append(f"PASS: Trace consistency for v0")
            
            # Step 2 -> Step 3: Q = S * v0 = S * d * sqrt(g/(2h))
            Q_expected = self.S * self.d * math.sqrt(self.g / (2 * self.h))
            if not math.isclose(self.Q, Q_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Trace consistency Q. Expected {Q_expected:.3f}, got {self.Q:.3f}")
                return False
            else:
                self.logs.append(f"PASS: Trace consistency for Q")
            
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and check trends, then restore."""
        try:
            original_h = self.h
            original_d = self.d
            
            # Test 1: Increase h -> t should increase
            self.h *= 1.1  # 10% increase
            t_new = math.sqrt(2 * self.h / self.g)
            if not (t_new > self.t * 1.001):  # Allow small tolerance
                self.logs.append(f"FAIL: Monotonicity t(h). t didn't increase when h increased")
                self.h = original_h
                return False
            else:
                self.logs.append(f"PASS: t increases with h")
            
            # Restore h
            self.h = original_h
            
            # Test 2: Increase h -> v0 should decrease
            self.h *= 1.1
            v0_new = self.d * math.sqrt(self.g / (2 * self.h))
            if not (v0_new < self.v_0 * 0.999):  # Allow small tolerance
                self.logs.append(f"FAIL: Monotonicity v0(h). v0 didn't decrease when h increased")
                self.h = original_h
                return False
            else:
                self.logs.append(f"PASS: v0 decreases with h")
            
            self.h = original_h
            
            # Test 3: Increase d -> v0 should increase
            self.d *= 1.1
            v0_new = self.d * math.sqrt(self.g / (2 * self.h))
            if not (v0_new > self.v_0 * 1.001):
                self.logs.append(f"FAIL: Monotonicity v0(d). v0 didn't increase when d increased")
                self.d = original_d
                return False
            else:
                self.logs.append(f"PASS: v0 increases with d")
            
            self.d = original_d
            
            # Test 4: Increase d -> Q should increase
            self.d *= 1.1
            Q_new = self.S * self.d * math.sqrt(self.g / (2 * self.h))
            if not (Q_new > self.Q * 1.001):
                self.logs.append(f"FAIL: Monotonicity Q(d). Q didn't increase when d increased")
                self.d = original_d
                return False
            else:
                self.logs.append(f"PASS: Q increases with d")
            
            self.d = original_d
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {str(e)}")
            # Restore values on error
            self.h = original_h
            self.d = original_d
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs.clear()
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n=== Running {test_name} Test ===")
            if not test_func():
                success = False
                self.logs.append(f"FAILED: {test_name}")
            else:
                self.logs.append(f"PASSED: {test_name}")
        
        self.logs.append(f"\n=== Overall Result ===")
        self.logs.append(f"SUCCESS: {success}")
        
        return success, self.logs