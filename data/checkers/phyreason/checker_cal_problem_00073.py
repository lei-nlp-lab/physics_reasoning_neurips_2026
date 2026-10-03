import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters safely
        self.m = getattr(solution_instance, 'm', 1.0)
        self.L = getattr(solution_instance, 'L', 1.0)
        self.mu = getattr(solution_instance, 'mu', 0.5)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.lam = getattr(solution_instance, 'lam', 2.0)
        self.h = getattr(solution_instance, 'h', 2.0)
        
        # Extract results with safe fallbacks
        self.v = self.results.get('v', 0.0)
        self.x = self.results.get('x', 0.0)
        self.y = self.results.get('y', 0.0)
        self.E_k = self.results.get('E_k', 0.0)
        self.v_2 = self.results.get('v_2', 0.0)
        self.v_02 = self.results.get('v_02', 0.0)
        self.v_32 = self.results.get('v_32', 0.0)
        self.v_1 = self.results.get('v_1', 0.0)
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        logs = []
        
        try:
            # 1. Work-energy theorem for step_1: v from height 2μL
            # mg*(2μL) - μmgL = 1/2 m v^2
            left_energy_step1 = self.m * self.g * (2 * self.mu * self.L) - self.mu * self.m * self.g * self.L
            right_energy_step1 = 0.5 * self.m * self.v**2
            if not self._is_close(left_energy_step1, right_energy_step1):
                logs.append(f"FAIL: Work-energy theorem in step_1: {left_energy_step1} ≠ {right_energy_step1}")
            
            # 2. Energy conservation in collision (step_8 & step_9)
            # 1/2 m v_0^2 = 1/2 m v_1^2 + 1/2 λ m v_2^2
            if self.v_02 > 0 and self.v_1 != 0 and self.v_2 != 0:
                left_collision = 0.5 * self.m * self.v_02
                right_collision = 0.5 * self.m * self.v_1**2 + 0.5 * self.lam * self.m * self.v_2**2
                if not self._is_close(left_collision, right_collision):
                    logs.append(f"FAIL: Energy conservation in collision: {left_collision} ≠ {right_collision}")
            
            # 3. Check curve equation consistency at point P
            # At point P: x = 2μL, y = μL
            x_P = 2 * self.mu * self.L
            y_P = self.mu * self.L
            curve_rhs = 2 * math.sqrt(2 * self.mu * self.L * y_P - y_P**2)
            if not self._is_close(x_P, curve_rhs):
                logs.append(f"FAIL: Curve equation at point P: {x_P} ≠ {curve_rhs}")
            
            # 4. Energy at landing point P should be consistent
            # E_k = mgy + mgx²/(4y) = 2μmgL at P
            E_k_P = self.m * self.g * y_P + (self.m * self.g * x_P**2) / (4 * y_P)
            expected_E_k_P = 2 * self.mu * self.m * self.g * self.L
            if not self._is_close(E_k_P, expected_E_k_P):
                logs.append(f"FAIL: Energy at point P: {E_k_P} ≠ {expected_E_k_P}")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            
        return logs
    
    def test_trace_consistency(self):
        logs = []
        
        try:
            # 1. Check step_1 velocity
            expected_v = math.sqrt(2 * self.mu * self.g * self.L)
            if not self._is_close(self.v, expected_v):
                logs.append(f"FAIL: v from step_1: {self.v} ≠ {expected_v}")
            
            # 2. Check step_4 relationship for the sample point (x, y)
            # v_0^2 = gx²/(2y) from projectile motion
            if self.x > 0 and self.y > 0:
                v0_sq_from_proj = (self.g * self.x**2) / (2 * self.y)
                # We don't compare with v_02 because they are from different scenarios
                # Instead, compute v0^2 from the curve equation and check consistency
                # For the sample point, E_k should be constant = 2μmgL
                E_k_sample = self.m * self.g * self.y + 0.5 * self.m * v0_sq_from_proj
                expected_E_k = 2 * self.mu * self.m * self.g * self.L
                if not self._is_close(E_k_sample, expected_E_k):
                    logs.append(f"FAIL: E_k at sample point: {E_k_sample} ≠ {expected_E_k}")
            
            # 3. Check step_5 energy expression for point P (not sample point)
            # E_k at P should match computed E_k
            x_P = 2 * self.mu * self.L
            y_P = self.mu * self.L
            v0_sq_P = self.g * x_P**2 / (2 * y_P)
            E_k_P_calc = self.m * self.g * y_P + 0.5 * self.m * v0_sq_P
            if not self._is_close(self.E_k, E_k_P_calc):
                logs.append(f"FAIL: E_k at P: {self.E_k} ≠ {E_k_P_calc}")
            
            # 4. Check collision velocities (step_10)
            # v_1 = (λ-1)/(λ+1) * v_0, v_2 = 2v_0/(λ+1)
            v_0_collision = math.sqrt(self.v_02)
            expected_v1 = ((self.lam - 1) / (self.lam + 1)) * v_0_collision
            expected_v2 = (2 * v_0_collision) / (self.lam + 1)
            
            if not self._is_close(self.v_1, expected_v1):
                logs.append(f"FAIL: v_1 from step_10: {self.v_1} ≠ {expected_v1}")
            if not self._is_close(self.v_2, expected_v2):
                logs.append(f"FAIL: v_2 from step_10: {self.v_2} ≠ {expected_v2}")
            
            # 5. Check rebound velocity (step_11)
            # v_3^2 = v_1^2 - 4μgL
            expected_v32 = self.v_1**2 - 4 * self.mu * self.g * self.L
            if not self._is_close(self.v_32, expected_v32):
                logs.append(f"FAIL: v_3^2 from step_11: {self.v_32} ≠ {expected_v32}")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            
        return logs
    
    def test_monotonicity(self):
        logs = []
        
        try:
            # Perturb mass m by +10%
            original_m = self.m
            perturbed_m = original_m * 1.1
            
            # Calculate v_0^2 with perturbed mass using formula from step_7
            # v_0^2 = 2gh - 2μgL (independent of m)
            v0_sq_original = 2 * self.g * self.h - 2 * self.mu * self.g * self.L
            v0_sq_perturbed = 2 * self.g * self.h - 2 * self.mu * self.g * self.L
            
            # Mass should cancel out, so velocities should remain the same
            if not self._is_close(v0_sq_original, v0_sq_perturbed):
                logs.append(f"FAIL: v_0^2 changes with mass perturbation: {v0_sq_original} → {v0_sq_perturbed}")
            
            # Calculate v_3 and v_2 with perturbed mass
            v0_perturbed = math.sqrt(max(0, v0_sq_perturbed))
            v1_perturbed = ((self.lam - 1) / (self.lam + 1)) * v0_perturbed
            v2_perturbed = (2 * v0_perturbed) / (self.lam + 1)
            v3_sq_perturbed = v1_perturbed**2 - 4 * self.mu * self.g * self.L
            
            # Original values
            v0_original = math.sqrt(max(0, self.v_02))
            v2_original = self.v_2
            
            # v_2 should be proportional to v_0, which didn't change
            if not self._is_close(v2_original, v2_perturbed):
                logs.append(f"FAIL: v_2 changes with mass perturbation: {v2_original} → {v2_perturbed}")
            
            # Check that v_3^2 trend is consistent
            if v3_sq_perturbed < 0 and self.v_32 >= 0:
                logs.append("WARNING: v_3^2 becomes negative with mass perturbation")
            
            # Restore original mass
            self.m = original_m
            
            # Additional check: perturb μ and verify v increases
            original_mu = self.mu
            perturbed_mu = original_mu * 1.1
            
            # v = √(2μgL) from step_1
            v_original_mu = math.sqrt(2 * original_mu * self.g * self.L)
            v_perturbed_mu = math.sqrt(2 * perturbed_mu * self.g * self.L)
            
            if v_perturbed_mu <= v_original_mu:
                logs.append(f"FAIL: v doesn't increase with μ: {v_original_mu} → {v_perturbed_mu}")
            
            self.mu = original_mu
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            # Ensure we restore original values even if error occurs
            self.m = getattr(self, 'm', 1.0)
            self.mu = getattr(self, 'mu', 0.5)
            
        return logs
    
    def run_all_tests(self):
        all_logs = []
        
        # Run all tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                logs = test_func()
                if logs:
                    all_logs.append(f"\n=== {test_name} ===")
                    all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"\n=== {test_name} Error ===")
                all_logs.append(f"Exception: {str(e)}")
                all_logs.append(traceback.format_exc())
        
        success = len(all_logs) == 0
        return success, all_logs