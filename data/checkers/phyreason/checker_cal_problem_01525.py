import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs (with defaults if missing)
        self.m = getattr(solution_instance, 'm', 1.0)  # mass not given, default to 1 for numerical checks
        self.g = getattr(solution_instance, 'g', 10.0)
        self.h = getattr(solution_instance, 'h', 1.2)
        self.d = getattr(solution_instance, 'd', 0.8)
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.v_C = results.get('v_C', 0.0)
        self.v_all = results.get('v_all', 0.0)
        self.a = results.get('a', 0.0)
        self.t = results.get('t', 0.0)
        
        # Store logs
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
        
    def test_invariants(self):
        """Verify physical invariants from the KB and steps."""
        try:
            # Step 1: Work-energy theorem for ring C falling distance d
            # 2mgd = 0.5 * (2m) * v_C^2  => v_C^2 = 2gd
            expected_v_C_sq = 2 * self.g * self.d
            computed_v_C_sq = self.v_C ** 2
            if not math.isclose(computed_v_C_sq, expected_v_C_sq, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Step 1 (Work-energy): v_C^2 = {computed_v_C_sq}, expected {expected_v_C_sq}")
                return False
            self.log("PASS Step 1: Work-energy theorem holds for ring C.")
            
            # Step 2: Conservation of momentum during collision
            # Before: momentum = 2m * v_C (ring C) + 0 (block B at rest) = 2m * v_C
            # After: momentum = (2m + m + m) * v_all = 4m * v_all
            # So: 2m * v_C = 4m * v_all => v_all = v_C / 2
            expected_v_all = self.v_C / 2
            if not math.isclose(self.v_all, expected_v_all, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Step 2 (Momentum): v_all = {self.v_all}, expected {expected_v_all}")
                return False
            self.log("PASS Step 2: Momentum conservation holds during collision.")
            
            # Step 3: Newton's second law after collision
            # Net force on system: (3m)g (downward on B+C) - (1m)g (downward on A) = 2mg
            # Total mass: 4m => a = (2mg)/(4m) = g/2
            expected_a = self.g / 2
            if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Step 3 (Newton's second law): a = {self.a}, expected {expected_a}")
                return False
            self.log("PASS Step 3: Newton's second law holds after collision.")
            
            # Step 4 & 5: Kinematic consistency
            # h = v_all * t + 0.5 * a * t^2
            computed_h = self.v_all * self.t + 0.5 * self.a * (self.t ** 2)
            if not math.isclose(computed_h, self.h, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Step 4/5 (Kinematic): computed h = {computed_h}, given h = {self.h}")
                return False
            self.log("PASS Step 4/5: Kinematic equation is satisfied.")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check that intermediate results lead to the final answer."""
        try:
            # From v_all and a, compute t using kinematic equation (positive root)
            # Solve h = v_all * t + 0.5 * a * t^2 for t
            # t = [-v_all + sqrt(v_all^2 + 2*a*h)] / a   (since t>0)
            discriminant = self.v_all ** 2 + 2 * self.a * self.h
            if discriminant < 0:
                self.log("FAIL Trace: discriminant negative, no real t")
                return False
            expected_t = (-self.v_all + math.sqrt(discriminant)) / self.a
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Trace: t = {self.t}, expected from intermediates = {expected_t}")
                return False
            self.log("PASS Trace: Intermediate results consistently yield the final t.")
            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb input h and check that t changes in the expected direction."""
        original_h = self.h
        original_t = self.t
        
        try:
            # Increase h by 10%
            perturbed_h = self.h * 1.1
            # Recompute t using the same kinematic formula and current v_all, a
            discriminant = self.v_all ** 2 + 2 * self.a * perturbed_h
            if discriminant < 0:
                self.log("FAIL Monotonicity: discriminant negative for perturbed h")
                return False
            perturbed_t = (-self.v_all + math.sqrt(discriminant)) / self.a
            
            # Expect t to increase when h increases
            if not (perturbed_t > original_t):
                self.log(f"FAIL Monotonicity: increasing h did not increase t (t_orig={original_t}, t_pert={perturbed_t})")
                return False
            
            # Restore original h
            self.h = original_h
            self.log("PASS Monotonicity: t increases with h as expected.")
            return True
        except Exception as e:
            self.h = original_h  # restore on error
            self.log(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs)."""
        self.logs.clear()
        
        test_results = []
        try:
            test_results.append(self.test_invariants())
            test_results.append(self.test_trace_consistency())
            test_results.append(self.test_monotonicity())
        except Exception as e:
            self.log(f"CRITICAL ERROR in run_all_tests: {e}\n{traceback.format_exc()}")
            return False, self.logs
        
        success = all(test_results)
        return success, self.logs