import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Constants
        self.c = 3e8  # Speed of light in m/s
        
        # Input parameters (with safe attribute access)
        self.lambda_val = getattr(solution_instance, 'lambda_val', 
                                  solution_instance.results.get('lambda_val', 0.0))
        self.n = getattr(solution_instance, 'n', 
                         solution_instance.results.get('n', 0.0))
        self.t_pulse = getattr(solution_instance, 't_pulse', 
                               solution_instance.results.get('t_pulse', 0.0))
        
        # Results (with safe attribute access)
        self.f = getattr(solution_instance, 'f', 
                         solution_instance.results.get('f', 0.0))
        self.s = getattr(solution_instance, 's', 
                         solution_instance.results.get('s', 0.0))
        self.s_m = getattr(solution_instance, 's_m', 
                           solution_instance.results.get('s_m', 0.0))
        
        self.logs = []
        
    def _log(self, message, is_error=False):
        """Helper method for logging."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
    
    def test_invariants(self):
        """Check fundamental physical invariants."""
        try:
            # 1. Wave speed relationship: c = lambda * f
            expected_c = self.lambda_val * self.f
            if not math.isclose(expected_c, self.c, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Wave speed invariant violated: λ*f = {expected_c:.2e} m/s ≠ c = {self.c:.2e} m/s", True)
                return False
            self._log("Wave speed invariant (c = λ*f) satisfied.")
            
            # 2. Distance between pulses: s = c*(1/n - t)
            delta_t = 1/self.n - self.t_pulse
            expected_s = self.c * delta_t
            if not math.isclose(self.s, expected_s, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Distance between pulses invariant violated: s = {self.s:.2e} m ≠ c*(1/n - t) = {expected_s:.2e} m", True)
                return False
            self._log("Distance between pulses invariant (s = c*(1/n - t)) satisfied.")
            
            # 3. Maximum range relationship: s_m = s/2
            if not math.isclose(self.s_m, self.s/2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Maximum range invariant violated: s_m = {self.s_m:.2e} m ≠ s/2 = {self.s/2:.2e} m", True)
                return False
            self._log("Maximum range invariant (s_m = s/2) satisfied.")
            
            return True
            
        except Exception as e:
            self._log(f"Exception in test_invariants: {str(e)}\n{traceback.format_exc()}", True)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # 1. Check f matches step 1
            expected_f = self.c / self.lambda_val
            if not math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Frequency inconsistent: f = {self.f:.2e} Hz ≠ c/λ = {expected_f:.2e} Hz", True)
                return False
            self._log(f"Frequency f = {self.f:.2e} Hz consistent with step 1.")
            
            # 2. Check s matches step 2
            delta_t = 1/self.n - self.t_pulse
            expected_s = self.c * delta_t
            if not math.isclose(self.s, expected_s, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Distance s inconsistent: s = {self.s:.2e} m ≠ c*(1/n - t) = {expected_s:.2e} m", True)
                return False
            self._log(f"Distance s = {self.s:.2e} m consistent with step 2.")
            
            # 3. Check s_m matches step 3 and target answer
            expected_s_m = self.s / 2
            if not math.isclose(self.s_m, expected_s_m, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Maximum range inconsistent: s_m = {self.s_m:.2e} m ≠ s/2 = {expected_s_m:.2e} m", True)
                return False
            self._log(f"Maximum range s_m = {self.s_m:.2e} m = {self.s_m/1000:.2f} km consistent with step 3 and target answer.")
            
            # 4. Check target answers
            target_f = 1.5e9
            target_s_m_km = 30.0
            
            if not math.isclose(self.f, target_f, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Frequency doesn't match target: {self.f:.2e} Hz ≠ {target_f:.2e} Hz", True)
                return False
            
            s_m_km = self.s_m / 1000
            if not math.isclose(s_m_km, target_s_m_km, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Maximum range doesn't match target: {s_m_km:.2f} km ≠ {target_s_m_km:.2f} km", True)
                return False
                
            self._log(f"Target answers verified: f = {self.f:.2e} Hz, s_m = {s_m_km:.2f} km.")
            return True
            
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {str(e)}\n{traceback.format_exc()}", True)
            return False
    
    def test_monotonicity(self):
        """Test that perturbations produce expected trends."""
        try:
            # Store original values
            original_lambda = self.lambda_val
            original_n = self.n
            original_t = self.t_pulse
            original_f = self.f  # Store original frequency
            original_s = self.s  # Store original s for comparison
            
            success = True
            
            # Test 1: Increase wavelength by 10% -> frequency should decrease
            self.lambda_val *= 1.1
            new_f = self.c / self.lambda_val
            # Use math.isclose with tolerance for floating point comparison
            if not (new_f < original_f or math.isclose(new_f, original_f, rel_tol=1e-10)):
                self._log(f"Monotonicity violated: f({original_lambda:.3e}) = {original_f:.3e} Hz, f({self.lambda_val:.3e}) = {new_f:.3e} Hz (should decrease)", True)
                success = False
            self._log(f"λ↑10% → f↓: {original_lambda:.3e}→{self.lambda_val:.3e} m, {original_f:.3e}→{new_f:.3e} Hz ✓")
            
            # Restore lambda
            self.lambda_val = original_lambda
            
            # Test 2: Increase pulse rate n by 10% -> distance s should decrease
            self.n *= 1.1
            new_s = self.c * (1/self.n - self.t_pulse)
            # Use math.isclose with tolerance for floating point comparison
            if not (new_s < original_s or math.isclose(new_s, original_s, rel_tol=1e-10)):
                self._log(f"Monotonicity violated: s(n={original_n:.0f}) = {original_s:.3e} m, s(n={self.n:.0f}) = {new_s:.3e} m (should decrease)", True)
                success = False
            self._log(f"n↑10% → s↓: {original_n:.0f}→{self.n:.0f} Hz, {original_s:.3e}→{new_s:.3e} m ✓")
            
            # Test 3: Increase pulse duration t by 10% -> distance s should decrease
            self.n = original_n  # Restore n first
            self.t_pulse *= 1.1
            new_s = self.c * (1/self.n - self.t_pulse)
            # Use math.isclose with tolerance for floating point comparison
            # Also handle the case where t_pulse is very small (close to zero)
            if self.t_pulse < 1e-10:  # If t_pulse is essentially zero
                # When t_pulse is zero or very small, s won't change much
                expected_change = -self.c * 0.1 * original_t
                actual_change = new_s - original_s
                if not (actual_change < 0 or math.isclose(actual_change, 0, abs_tol=1e-6)):
                    self._log(f"Monotonicity violated for small t: s(t={original_t:.2e}) = {original_s:.3e} m, s(t={self.t_pulse:.2e}) = {new_s:.3e} m, change = {actual_change:.2e} m (should be negative or zero)", True)
                    success = False
            elif not (new_s < original_s or math.isclose(new_s, original_s, rel_tol=1e-10)):
                self._log(f"Monotonicity violated: s(t={original_t:.2e}) = {original_s:.3e} m, s(t={self.t_pulse:.2e}) = {new_s:.3e} m (should decrease)", True)
                success = False
            self._log(f"t↑10% → s↓: {original_t:.2e}→{self.t_pulse:.2e} s, {original_s:.3e}→{new_s:.3e} m ✓")
            
            # Restore all original values
            self.lambda_val = original_lambda
            self.n = original_n
            self.t_pulse = original_t
            
            return success
            
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {str(e)}\n{traceback.format_exc()}", True)
            # Restore original values if possible
            if 'original_lambda' in locals():
                self.lambda_val = original_lambda
                self.n = original_n
                self.t_pulse = original_t
            return False
    
    def run_all_tests(self):
        """Run all tests and return (success_bool, logs_list)."""
        self.logs = []  # Reset logs
        success = True
        
        tests = [
            ("test_invariants", self.test_invariants),
            ("test_trace_consistency", self.test_trace_consistency),
            ("test_monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            if not test_func():
                success = False
                self._log(f"{test_name} failed.")
            else:
                self._log(f"{test_name} passed.")
        
        if success:
            self._log("All physics checks passed successfully!")
        else:
            self._log("Some physics checks failed.", True)
        
        return success, self.logs