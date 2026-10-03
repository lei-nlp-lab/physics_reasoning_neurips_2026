import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        self.logs = []
        
        # Extract results with safe fallbacks
        results_dict = getattr(solution_instance, 'results', {})
        self.v = results_dict.get('v', None)
        self.s = results_dict.get('s', None)
        self.t = results_dict.get('t', None)
        
        # Extract input parameters with safe fallbacks
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.h = getattr(solution_instance, 'h', 20.0)
        self.F = getattr(solution_instance, 'F', 200.0)
        
        # Air resistance force magnitude
        self.f = 0.2 * self.m * self.g
        
        # Store original for monotonicity tests
        self._original_m = self.m
        self._original_results = {'v': self.v, 's': self.s, 't': self.t}
        
    def _log(self, msg, level="INFO"):
        """Helper for logging."""
        self.logs.append(f"[{level}] {msg}")
    
    def test_invariants(self):
        """
        Test physical invariants using EXACT ground truth equations.
        """
        try:
            # Test 1: Step 1 equation WITHOUT air resistance during pulling (ground truth)
            # (2F cos37° - mg)h = 1/2 m v^2
            cos37 = 0.8  # cos(37°) = 4/5
            lhs_step1 = (2 * self.F * cos37 - self.m * self.g) * self.h
            rhs_step1 = 0.5 * self.m * self.v**2
                
            if not math.isclose(lhs_step1, rhs_step1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 1 invariant failed: LHS={lhs_step1:.2f}, RHS={rhs_step1:.2f}")
                return False
            self._log("Step 1 invariant passed")
            
            # Test 2: Step 3 energy conservation (entire motion)
            # (2F cos37° - mg)h - f*s = 0
            work_initial = (2 * self.F * cos37 - self.m * self.g) * self.h
            work_friction = self.f * self.s
                
            if not math.isclose(work_initial, work_friction, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Energy conservation failed: W_initial={work_initial:.2f}, W_friction={work_friction:.2f}")
                return False
            self._log("Energy conservation passed")
            
            # Test 3: Time t should be positive and reasonable
            if self.t is not None:
                if self.t <= 0:
                    self._log(f"Time must be positive, got {self.t}")
                    return False
                # With air resistance during free flight, time should be > than without air resistance
                # But ground truth step 2 uses -h = v*t - ½g*t² (NO air resistance)
                # So we should check that t matches this equation
                # Solve -h = v*t - ½g*t² for t
                # Discriminant: v² - 2g(-h) = v² + 2gh
                t_calc = (self.v + math.sqrt(self.v**2 + 2*self.g*self.h)) / self.g
                if not math.isclose(self.t, t_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Time mismatch: calculated from kinematics={t_calc:.2f}, reported={self.t:.2f}")
                    return False
                self._log("Kinematics check passed")
            
            return True
            
        except Exception as e:
            self._log(f"Invariant test error: {str(e)}", "ERROR")
            return False
    
    def test_trace_consistency(self):
        """
        Check consistency between intermediate and final results.
        """
        try:
            # Check 1: s from step 3 should be consistent with energy
            cos37 = 0.8
            work_initial = (2 * self.F * cos37 - self.m * self.g) * self.h
            s_calc = work_initial / self.f
                
            if not math.isclose(self.s, s_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"s trace mismatch: calculated={s_calc:.2f}, reported={self.s:.2f}")
                return False
            self._log("s trace consistency passed")
            
            # Check 2: v from step 1 should be 30.0 (given in ground truth)
            if not math.isclose(self.v, 30.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v should be 30.0, got {self.v}")
                return False
            self._log("v trace consistency passed")
            
            return True
            
        except Exception as e:
            self._log(f"Trace consistency error: {str(e)}", "ERROR")
            return False
    
    def test_monotonicity(self):
        """
        Perturb mass and check physical trends.
        """
        try:
            # Store original
            original_m = self.m
            original_v = self.v
            
            # Perturb mass by +10%
            self.m *= 1.1
            self.f = 0.2 * self.m * self.g
            
            # Recalculate v using step 1 equation WITHOUT air resistance (ground truth)
            cos37 = 0.8
            v_new = math.sqrt(2 * ((2 * self.F * cos37 - self.m * self.g) * self.h) / self.m)
                
            # v should DECREASE with increased mass (since mg term increases)
            if v_new > original_v:
                self._log(f"Monotonicity failed: v increased from {original_v:.2f} to {v_new:.2f} with mass increase")
                return False
            self._log("Mass monotonicity passed")
            
            # Restore original values
            self.m = original_m
            self.f = 0.2 * self.m * self.g
            self.v = original_v
            
            return True
            
        except Exception as e:
            self._log(f"Monotonicity test error: {str(e)}", "ERROR")
            # Restore on error
            self.m = self._original_m
            self.f = 0.2 * self.m * self.g
            self.v = self._original_results['v']
            return False
    
    def run_all_tests(self):
        """
        Execute all tests and return success status with logs.
        """
        self.logs = []  # Reset logs
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    self._log(f"{test_name} failed", "FAIL")
                    success = False
                else:
                    self._log(f"{test_name} passed", "PASS")
            except Exception as e:
                self._log(f"{test_name} crashed: {str(e)}", "ERROR")
                success = False
        
        return success, self.logs