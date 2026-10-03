import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        """
        # Extract results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Inputs: try to get from results, otherwise use defaults
        self.R = results.get('R', 1.0)  # Use 1.0 as default for testing
        self.B = results.get('B', 1.0)
        self.m = results.get('m', 1.0)
        self.q = results.get('q', 1.0)
        
        # Compute θ from tan(θ/2)=0.5
        self.theta = 2 * math.atan(0.5)  # ~0.9273 rad
        
        # Results
        self.alpha = results.get('alpha', 0.0)
        self.S_1 = results.get('S_1', 0.0)
        self.S = results.get('S', 0.0)
        self.S_2 = results.get('S_2', 0.0)
        self.S_3 = results.get('S_3', 0.0)
        self.r_1 = results.get('r_1', 0.0)
        self.r_2 = results.get('r_2', 0.0)
        self.t_max = results.get('t_max', 0.0)
        self.v_1 = results.get('v_1', 0.0)
        
        # Compute v_2 from formula
        self.v_2 = (self.q * self.B * self.r_2) / self.m if self.m != 0 else 0.0
        
        # Constants
        self.pi = math.pi
        self.sqrt3 = math.sqrt(3)
        
        # Logs
        self.logs = []
        
    def _log(self, msg, success):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
        
    def test_invariants(self):
        """
        Verify fundamental physics laws: Lorentz force as centripetal, geometric constraints.
        """
        success = True
        
        # 1. Check step_1: r_1 = R * tan(θ/2) = R/2
        expected_r1 = self.R * math.tan(self.theta / 2)
        if not math.isclose(self.r_1, expected_r1, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"r_1 mismatch: expected {expected_r1}, got {self.r_1}", False) and success
            
        # 2. Check step_2: q v_1 B = m v_1^2 / r_1  => v_1 = (q B r_1)/m
        expected_v1 = (self.q * self.B * self.r_1) / self.m if self.m != 0 else 0.0
        if not math.isclose(self.v_1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"v_1 mismatch: expected {expected_v1}, got {self.v_1}", False) and success
            
        # 3. Check step_3: r_2 = 2R (given, but also from Lorentz: r_2 = m v_2 / (q B))
        expected_r2 = 2 * self.R
        if not math.isclose(self.r_2, expected_r2, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"r_2 mismatch: expected {expected_r2}, got {self.r_2}", False) and success
            
        # 4. Check step_4: α = arcsin(R / r_2) = π/6
        expected_alpha = math.asin(self.R / self.r_2) if self.r_2 != 0 else 0.0
        if not math.isclose(self.alpha, expected_alpha, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"α mismatch: expected {expected_alpha}, got {self.alpha}", False) and success
            
        # 5. Check step_5: t_max = (2 * α * r_2) / v_2
        expected_tmax = (2 * self.alpha * self.r_2) / self.v_2 if self.v_2 != 0 else 0.0
        if not math.isclose(self.t_max, expected_tmax, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"t_max mismatch: expected {expected_tmax}, got {self.t_max}", False) and success
            
        # 6. Check step_7: S_1 = (π * r_1^2) / 2
        expected_S1 = (self.pi * self.r_1**2) / 2
        if not math.isclose(self.S_1, expected_S1, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"S_1 mismatch: expected {expected_S1}, got {self.S_1}", False) and success
            
        # 7. Check step_8: S_2 = (1/2) * R^2 * (π/3)  (sector of radius R, angle π/3)
        expected_S2 = 0.5 * self.R**2 * (self.pi / 3)
        if not math.isclose(self.S_2, expected_S2, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"S_2 mismatch: expected {expected_S2}, got {self.S_2}", False) and success
            
        # 8. Check step_9: S_3 = (π/6 - √3/4) * R^2
        expected_S3 = (self.pi/6 - self.sqrt3/4) * self.R**2
        if not math.isclose(self.S_3, expected_S3, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"S_3 mismatch: expected {expected_S3}, got {self.S_3}", False) and success
            
        # 9. Check step_10: S = S_1 + S_2 + S_3
        expected_S = self.S_1 + self.S_2 + self.S_3
        if not math.isclose(self.S, expected_S, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"S mismatch: expected {expected_S}, got {self.S}", False) and success
            
        return success
        
    def test_trace_consistency(self):
        """
        Cross-check that intermediate results match the final answers.
        """
        success = True
        
        # Final answers from target:
        # v1 = q B R / (2 m)
        expected_v1_final = (self.q * self.B * self.R) / (2 * self.m) if self.m != 0 else 0.0
        if not math.isclose(self.v_1, expected_v1_final, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Final v1 mismatch: expected {expected_v1_final}, got {self.v_1}", False) and success
            
        # t_max = π m / (3 q B)
        expected_tmax_final = (self.pi * self.m) / (3 * self.q * self.B) if self.q*self.B != 0 else 0.0
        if not math.isclose(self.t_max, expected_tmax_final, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Final t_max mismatch: expected {expected_tmax_final}, got {self.t_max}", False) and success
            
        # S = (11π/24 - √3/4) R^2
        expected_S_final = (11*self.pi/24 - self.sqrt3/4) * self.R**2
        if not math.isclose(self.S, expected_S_final, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Final S mismatch: expected {expected_S_final}, got {self.S}", False) and success
            
        return success
        
    def test_monotonicity(self):
        """
        Perturb each input (m, q, B, R) and verify expected trends using formulas.
        """
        success = True
        perturb_factor = 1.01  # 1% increase
        
        # Helper to compute outputs from formulas
        def compute_outputs(R, B, m, q):
            r1 = R * math.tan(self.theta/2)
            v1 = (q * B * r1) / m if m != 0 else 0.0
            r2 = 2 * R
            alpha = math.asin(R / r2) if r2 != 0 else 0.0
            v2 = (q * B * r2) / m if m != 0 else 0.0
            tmax = (2 * alpha * r2) / v2 if v2 != 0 else 0.0
            S = (11*math.pi/24 - math.sqrt(3)/4) * R**2
            return v1, tmax, S
            
        # Original outputs
        v1_orig, tmax_orig, S_orig = compute_outputs(self.R, self.B, self.m, self.q)
        
        # Perturb m
        m_pert = self.m * perturb_factor
        v1_m, tmax_m, S_m = compute_outputs(self.R, self.B, m_pert, self.q)
        if not (v1_m < v1_orig or math.isclose(v1_m, v1_orig, rel_tol=1e-2)):
            success = self._log(f"Monotonicity: v1 should decrease when m increases (orig {v1_orig}, pert {v1_m})", False) and success
        if not (tmax_m > tmax_orig or math.isclose(tmax_m, tmax_orig, rel_tol=1e-2)):
            success = self._log(f"Monotonicity: t_max should increase when m increases (orig {tmax_orig}, pert {tmax_m})", False) and success
        if not math.isclose(S_m, S_orig, rel_tol=1e-2):
            success = self._log(f"Monotonicity: S should be invariant under m change (orig {S_orig}, pert {S_m})", False) and success
            
        # Perturb q
        q_pert = self.q * perturb_factor
        v1_q, tmax_q, S_q = compute_outputs(self.R, self.B, self.m, q_pert)
        if not (v1_q > v1_orig or math.isclose(v1_q, v1_orig, rel_tol=1e-2)):
            success = self._log(f"Monotonicity: v1 should increase when q increases (orig {v1_orig}, pert {v1_q})", False) and success
        if not (tmax_q < tmax_orig or math.isclose(tmax_q, tmax_orig, rel_tol=1e-2)):
            success = self._log(f"Monotonicity: t_max should decrease when q increases (orig {tmax_orig}, pert {tmax_q})", False) and success
        if not math.isclose(S_q, S_orig, rel_tol=1e-2):
            success = self._log(f"Monotonicity: S should be invariant under q change (orig {S_orig}, pert {S_q})", False) and success
            
        # Perturb B
        B_pert = self.B * perturb_factor
        v1_B, tmax_B, S_B = compute_outputs(self.R, B_pert, self.m, self.q)
        if not (v1_B > v1_orig or math.isclose(v1_B, v1_orig, rel_tol=1e-2)):
            success = self._log(f"Monotonicity: v1 should increase when B increases (orig {v1_orig}, pert {v1_B})", False) and success
        if not (tmax_B < tmax_orig or math.isclose(tmax_B, tmax_orig, rel_tol=1e-2)):
            success = self._log(f"Monotonicity: t_max should decrease when B increases (orig {tmax_orig}, pert {tmax_B})", False) and success
        if not math.isclose(S_B, S_orig, rel_tol=1e-2):
            success = self._log(f"Monotonicity: S should be invariant under B change (orig {S_orig}, pert {S_B})", False) and success
            
        # Perturb R
        R_pert = self.R * perturb_factor
        v1_R, tmax_R, S_R = compute_outputs(R_pert, self.B, self.m, self.q)
        if not (v1_R > v1_orig or math.isclose(v1_R, v1_orig, rel_tol=1e-2)):
            success = self._log(f"Monotonicity: v1 should increase when R increases (orig {v1_orig}, pert {v1_R})", False) and success
        if not math.isclose(tmax_R, tmax_orig, rel_tol=1e-2):
            success = self._log(f"Monotonicity: t_max should be invariant under R change (orig {tmax_orig}, pert {tmax_R})", False) and success
        if not (S_R > S_orig or math.isclose(S_R, S_orig, rel_tol=1e-2)):
            success = self._log(f"Monotonicity: S should increase when R increases (orig {S_orig}, pert {S_R})", False) and success
            
        return success
        
    def run_all_tests(self):
        """
        Execute all tests and return (success_bool, logs_list).
        """
        self.logs = []
        overall_success = True
        
        try:
            if not self.test_invariants():
                overall_success = False
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {e}")
            overall_success = False
            
        try:
            if not self.test_trace_consistency():
                overall_success = False
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {e}")
            overall_success = False
            
        try:
            if not self.test_monotonicity():
                overall_success = False
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {e}")
            overall_success = False
            
        return overall_success, self.logs