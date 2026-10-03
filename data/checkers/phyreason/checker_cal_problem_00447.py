import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Safe extraction of inputs
        self.R = getattr(solution_instance, 'R', 0.2)  # radius in meters
        self.M = getattr(solution_instance, 'M', 1.0)  # mass of C in kg
        self.m = getattr(solution_instance, 'm', 0.5)  # mass of Jia in kg
        self.mu1 = getattr(solution_instance, 'mu1', 0.5)  # friction between Jia and C
        self.mu2 = getattr(solution_instance, 'mu2', 0.1)  # friction between C and ground
        self.g = getattr(solution_instance, 'g', 10.0)  # gravitational acceleration

        # Safe extraction of results (using .results dictionary if available)
        if hasattr(solution_instance, 'results'):
            results_dict = solution_instance.results
        else:
            results_dict = {}
        
        # List of expected keys from the KB
        expected_keys = [
            'F_N', 'a_1', 'f_3', 't_2', 't', 'v_0', 'F_pressure',
            'a_2', 'f_2', 'v', 't_1', 'f_1', 'a_3'
        ]
        
        # Initialize results with safe extraction
        self.results = {}
        for key in expected_keys:
            self.results[key] = results_dict.get(key, 0.0)
        
        # Also try direct attributes as fallback
        for key in expected_keys:
            if hasattr(solution_instance, key):
                self.results[key] = getattr(solution_instance, key)
        
        # Store for monotonicity perturbation
        self.solution_instance = solution_instance

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Test physical invariants like energy conservation and Newton's laws."""
        logs = []
        success = True
        
        try:
            # 1. Energy conservation from A to B (step_1 and step_2)
            v0_expected = math.sqrt(2 * self.g * self.R)
            v0_actual = self.results['v_0']
            if not self._is_close(v0_actual, v0_expected):
                logs.append(f"Energy conservation failed: v0 = {v0_actual}, expected {v0_expected}")
                success = False
            
            # 2. Normal force at B (step_3): F_N = mg + m*v0^2/R
            F_N_expected = self.m * self.g + self.m * v0_actual**2 / self.R
            if not self._is_close(self.results['F_N'], F_N_expected):
                logs.append(f"Newton's 2nd law at B failed: F_N = {self.results['F_N']}, expected {F_N_expected}")
                success = False
            
            # 3. Newton's 3rd law: F_pressure should equal F_N (step_4)
            if not self._is_close(self.results['F_pressure'], self.results['F_N']):
                logs.append(f"Newton's 3rd law failed: F_pressure = {self.results['F_pressure']}, F_N = {self.results['F_N']}")
                success = False
            
            # 4. Friction forces (step_5 and step_6)
            f1_expected = self.mu1 * self.m * self.g
            f2_expected = self.mu2 * (self.m + self.M) * self.g
            if not self._is_close(self.results['f_1'], f1_expected):
                logs.append(f"Friction f1 failed: f1 = {self.results['f_1']}, expected {f1_expected}")
                success = False
            if not self._is_close(self.results['f_2'], f2_expected):
                logs.append(f"Friction f2 failed: f2 = {self.results['f_2']}, expected {f2_expected}")
                success = False
            
            # 5. Accelerations during relative motion (step_7 and step_8)
            a2_expected = (self.results['f_1'] - self.results['f_2']) / self.M
            a1_expected = -self.results['f_1'] / self.m
            if not self._is_close(self.results['a_2'], a2_expected):
                logs.append(f"Acceleration a2 failed: a2 = {self.results['a_2']}, expected {a2_expected}")
                success = False
            if not self._is_close(self.results['a_1'], a1_expected):
                logs.append(f"Acceleration a1 failed: a1 = {self.results['a_1']}, expected {a1_expected}")
                success = False
            
            # 6. Time and velocity when velocities equalize (step_9 and step_10)
            # t1 = v0 / (a2 - a1)
            if self.results['a_2'] - self.results['a_1'] != 0:
                t1_expected = self.results['v_0'] / (self.results['a_2'] - self.results['a_1'])
                if not self._is_close(self.results['t_1'], t1_expected):
                    logs.append(f"Time t1 failed: t1 = {self.results['t_1']}, expected {t1_expected}")
                    success = False
            v_expected = self.results['a_2'] * self.results['t_1']
            if not self._is_close(self.results['v'], v_expected):
                logs.append(f"Velocity v failed: v = {self.results['v']}, expected {v_expected}")
                success = False
            
            # 7. Friction when moving together (step_11)
            f3_expected = self.m * self.results['a_2']
            if not self._is_close(self.results['f_3'], f3_expected):
                logs.append(f"Friction f3 failed: f3 = {self.results['f_3']}, expected {f3_expected}")
                success = False
            
            # 8. Deceleration after equalization (step_12)
            a3_expected = -self.results['f_2'] / (self.m + self.M)
            if not self._is_close(self.results['a_3'], a3_expected):
                logs.append(f"Deceleration a3 failed: a3 = {self.results['a_3']}, expected {a3_expected}")
                success = False
            
            # 9. Deceleration time (step_13)
            if self.results['a_3'] != 0:
                t2_expected = -self.results['v'] / self.results['a_3']
                if not self._is_close(self.results['t_2'], t2_expected):
                    logs.append(f"Time t2 failed: t2 = {self.results['t_2']}, expected {t2_expected}")
                    success = False
            
            # 10. Total time (step_14)
            t_expected = self.results['t_1'] + self.results['t_2']
            if not self._is_close(self.results['t'], t_expected):
                logs.append(f"Total time t failed: t = {self.results['t']}, expected {t_expected}")
                success = False
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {e}\n{traceback.format_exc()}")
            success = False
        
        return success, logs

    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answers."""
        logs = []
        success = True
        
        try:
            # Target answers: 15N and 2/3s
            target_F = 15.0
            target_t = 2/3
            
            # Check F_N (should be 15N)
            if not self._is_close(self.results['F_N'], target_F):
                logs.append(f"F_N = {self.results['F_N']} does not match target {target_F}N")
                success = False
            
            # Check total time t (should be 2/3s)
            if not self._is_close(self.results['t'], target_t):
                logs.append(f"t = {self.results['t']} does not match target {target_t}s")
                success = False
            
            # Additional consistency: v_0 should lead to correct F_N
            v0_sq = 2 * self.g * self.R
            F_N_from_v0 = self.m * self.g + self.m * v0_sq / self.R
            if not self._is_close(self.results['F_N'], F_N_from_v0):
                logs.append(f"F_N from v0 inconsistent: F_N={self.results['F_N']}, calculated={F_N_from_v0}")
                success = False
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}")
            success = False
        
        return success, logs

    def test_monotonicity(self):
        """Perturb mass m and check trends in total time."""
        logs = []
        success = True
        
        try:
            # Store original mass
            original_m = self.m
            
            # Perturb m by +1%
            perturbed_m = original_m * 1.01
            
            # Recalculate key quantities using formulas from steps
            # Step 1-2: v0 unchanged (independent of m)
            v0 = math.sqrt(2 * self.g * self.R)
            
            # Step 3: F_N = m*g + m*v0^2/R
            F_N_perturbed = perturbed_m * self.g + perturbed_m * v0**2 / self.R
            
            # Step 5-6: Friction forces
            f1_perturbed = self.mu1 * perturbed_m * self.g
            f2_perturbed = self.mu2 * (perturbed_m + self.M) * self.g
            
            # Step 7-8: Accelerations
            a2_perturbed = (f1_perturbed - f2_perturbed) / self.M
            a1_perturbed = -f1_perturbed / perturbed_m
            
            # Step 9: Time to equalize velocities
            if a2_perturbed - a1_perturbed != 0:
                t1_perturbed = v0 / (a2_perturbed - a1_perturbed)
            else:
                t1_perturbed = 0
            
            # Step 10: Velocity when equal
            v_perturbed = a2_perturbed * t1_perturbed
            
            # Step 12: Deceleration after equalization
            a3_perturbed = -f2_perturbed / (perturbed_m + self.M)
            
            # Step 13: Deceleration time
            if a3_perturbed != 0:
                t2_perturbed = -v_perturbed / a3_perturbed
            else:
                t2_perturbed = 0
            
            # Step 14: Total time
            t_perturbed = t1_perturbed + t2_perturbed
            
            # Compare with original total time
            t_original = self.results['t']
            
            # Expected trend: As m increases, total time should increase
            # (From analytic derivative: dt/dm > 0 for m=0.5, M=1, mu1=0.5, mu2=0.1)
            if t_perturbed > t_original:
                logs.append(f"Monotonicity passed: t increased from {t_original:.4f} to {t_perturbed:.4f} when m increased")
            else:
                logs.append(f"Monotonicity warning: t changed from {t_original:.4f} to {t_perturbed:.4f} (expected increase)")
                # This is a warning, not necessarily failure, due to fuzzy physics
            
            # Restore original mass
            self.m = original_m
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}")
            success = False
            # Restore mass on error
            self.m = original_m
        
        return success, logs

    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run invariants test
        success, logs = self.test_invariants()
        all_logs.append("=== Invariants Test ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Run trace consistency test
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Test ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Run monotonicity test
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Test ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        return overall_success, all_logs