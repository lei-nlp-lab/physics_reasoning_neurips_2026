import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs and results safely
        self.g = 10.0  # Given constant
        self.U = getattr(solution_instance, 'U', self.results.get('U', 0.0))
        self.h_minus_9_25h = getattr(solution_instance, 'h_minus_9_25h', 
                                     self.results.get('h-9/25h', self.results.get('h_minus_9_25h', 0.0)))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.h = getattr(solution_instance, 'h', self.results.get('h', 0.0))
        self.h1 = getattr(solution_instance, 'h1', self.results.get('h_1', 0.0))
        self.h2 = getattr(solution_instance, 'h2', self.results.get('h_2', 0.0))
        
        self.logs = []
        
    def _log(self, msg, success=True):
        """Helper to log test results"""
        self.logs.append((success, msg))
        
    def _isclose(self, a, b):
        """Robust fuzzy numerical comparison"""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        
    def test_invariants(self):
        """Verify kinematic equations and logical consistency"""
        try:
            # 1. Check basic kinematic relation: h = 0.5*g*t^2
            expected_h = 0.5 * self.g * self.t**2
            if not self._isclose(self.h, expected_h):
                self._log(f"Kinematic invariant failed: h={self.h}, 0.5*g*t^2={expected_h}", False)
                return False
            else:
                self._log(f"✓ Kinematic invariant satisfied: h = 0.5*g*t^2")
            
            # 2. Check the last-second displacement condition
            # Displacement in first (t-1) seconds
            h_before_last = 0.5 * self.g * (self.t - 1)**2
            # This should equal h - (9/25)h = (16/25)h
            expected_h_before_last = (16/25) * self.h
            
            if not self._isclose(h_before_last, expected_h_before_last):
                self._log(f"Last-second condition failed: h_before_last={h_before_last}, (16/25)h={expected_h_before_last}", False)
                return False
            else:
                self._log(f"✓ Last-second condition satisfied")
            
            # 3. Check that h_minus_9_25h equals (16/25)h
            if not self._isclose(self.h_minus_9_25h, (16/25) * self.h):
                self._log(f"h-9/25h inconsistency: {self.h_minus_9_25h} vs {(16/25)*self.h}", False)
                return False
            else:
                self._log(f"✓ h-9/25h consistent with total h")
            
            # 4. Check h1 calculation
            expected_h1 = 0.5 * self.g * (3)**2  # First 3 seconds
            if not self._isclose(self.h1, expected_h1):
                self._log(f"h1 calculation failed: {self.h1} vs {expected_h1}", False)
                return False
            else:
                self._log(f"✓ h1 calculation correct")
            
            # 5. Check h2 = h - h1
            if not self._isclose(self.h2, self.h - self.h1):
                self._log(f"h2 = h - h1 failed: {self.h2} vs {self.h - self.h1}", False)
                return False
            else:
                self._log(f"✓ h2 = h - h1 satisfied")
            
            return True
        except Exception as e:
            self._log(f"Exception in test_invariants: {str(e)}", False)
            return False
            
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # Check that t=5 leads to h=125
            if not self._isclose(self.t, 5.0):
                self._log(f"Total time t={self.t} not consistent with expected 5s", False)
                return False
            
            if not self._isclose(self.h, 125.0):
                self._log(f"Total height h={self.h} not consistent with expected 125m", False)
                return False
            
            # Check that last-second displacement is 9/25 of 125 = 45m
            last_second_disp = self.h - 0.5 * self.g * (self.t - 1)**2
            expected_last_sec = (9/25) * self.h
            
            if not self._isclose(last_second_disp, expected_last_sec):
                self._log(f"Last second displacement {last_second_disp} ≠ (9/25)h={expected_last_sec}", False)
                return False
            
            # Check target answers
            target_time = "5s"
            target_displacement = "80m"
            
            time_ok = self._isclose(self.t, 5.0)
            displacement_ok = self._isclose(self.h2, 80.0)
            
            if not (time_ok and displacement_ok):
                self._log(f"Target answers mismatch: t={self.t}s (expected 5s), h2={self.h2}m (expected 80m)", False)
                return False
            
            self._log(f"✓ All intermediate results consistent with target answers: {target_time}, {target_displacement}")
            return True
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {str(e)}", False)
            return False
            
    def test_monotonicity(self):
        """Perturb input (g) and verify output trends"""
        try:
            # Original calculation with g=10
            t_original = self.t
            h_original = self.h
            h2_original = self.h2
            
            # Perturb g by +10%
            g_perturbed = self.g * 1.1
            
            # Recalculate t from equations
            # From: h = 0.5*g*t^2 and h_before_last = 0.5*g*(t-1)^2 = (16/25)h
            # Substitute: 0.5*g*(t-1)^2 = (16/25)*(0.5*g*t^2)
            # Cancel 0.5*g: (t-1)^2 = (16/25)*t^2
            # This gives: 25(t-1)^2 = 16t^2
            # 25(t^2 - 2t + 1) = 16t^2
            # 25t^2 - 50t + 25 = 16t^2
            # 9t^2 - 50t + 25 = 0
            # Solutions: t = (50 ± sqrt(2500 - 900))/18 = (50 ± 40)/18
            # Positive solutions: t=5 or t=5/9 ≈ 0.555...
            # For free fall from rest, t>1 (since we measure last second), so t=5 is the valid solution
            
            # Interestingly, t doesn't depend on g! Both equations have g factor that cancels.
            # So t should remain 5 even with perturbed g.
            t_perturbed = 5.0  # From equation derivation
            
            # Calculate h with perturbed g
            h_perturbed = 0.5 * g_perturbed * t_perturbed**2
            
            # Calculate h2 with perturbed g
            h1_perturbed = 0.5 * g_perturbed * (3)**2
            h2_perturbed = h_perturbed - h1_perturbed
            
            # Check monotonicity: if g increases, h should increase proportionally
            if h_perturbed <= h_original:
                self._log(f"Monotonicity failed: h didn't increase with increased g", False)
                return False
            
            # h2 should also increase with g
            if h2_perturbed <= h2_original:
                self._log(f"Monotonicity failed: h2 didn't increase with increased g", False)
                return False
            
            # Verify the ratio h2/h remains constant (should be 80/125 = 0.64)
            ratio_original = h2_original / h_original if h_original != 0 else 0
            ratio_perturbed = h2_perturbed / h_perturbed if h_perturbed != 0 else 0
            
            if not self._isclose(ratio_original, ratio_perturbed):
                self._log(f"Ratio h2/h not invariant under g perturbation", False)
                return False
            
            self._log(f"✓ Monotonicity test passed: outputs change correctly with g perturbation")
            return True
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {str(e)}", False)
            return False
            
    def run_all_tests(self):
        """Execute all tests and return summary"""
        self.logs = []
        
        tests = [
            ("Kinematic Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_pass = True
        for test_name, test_func in tests:
            try:
                if not test_func():
                    all_pass = False
            except Exception as e:
                self._log(f"Test '{test_name}' crashed: {traceback.format_exc()}", False)
                all_pass = False
        
        return all_pass, self.logs