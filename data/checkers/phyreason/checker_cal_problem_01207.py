import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Do NOT directly access solution_instance.attr - use getattr with fallbacks.
        """
        self.solution = solution_instance
        self.logs = []
        
        # Constants from problem statement
        self.m = 1.0  # kg
        self.v_belt = 3.0  # m/s
        self.L = 6.0  # m
        self.mu_k = 0.15
        self.g = 10.0  # m/s²
        
        # Expected results (use safe extraction)
        self.results = {}
        expected_keys = ['t_2', 't', 'a', 't_1', 'x_1', 'v_m']
        for key in expected_keys:
            self.results[key] = getattr(solution_instance, key, 
                                        getattr(solution_instance, 'results', {}).get(key, 0.0))
        
        # Derived quantities for checking
        self.v_belt = 3.0  # Given constant
        self.expected_a = self.mu_k * self.g  # μmg = ma ⇒ a = μg
        
    def _log(self, msg, success):
        """Helper to log test results."""
        self.logs.append((success, msg))
        return success
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        try:
            # 1. Newton's Second Law check: a = μkg
            a_calculated = self.mu_k * self.g
            if not math.isclose(self.results['a'], a_calculated, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Newton's Law violation: a={self.results['a']} but μkg={a_calculated}", False)
            
            # 2. Energy conservation: Work done by friction = ΔKE + ΔPE (PE doesn't change horizontally)
            # During acceleration phase: friction does positive work, increasing KE from 0 to (1/2)m*v_belt^2
            # Work = friction_force * displacement = μmg * x_1
            # ΔKE = (1/2)*m*v_belt^2 - 0
            x1 = self.results['x_1']
            work_friction = self.mu_k * self.m * self.g * x1
            delta_ke = 0.5 * self.m * self.v_belt**2
            
            if not math.isclose(work_friction, delta_ke, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Energy violation: Work={work_friction} ≠ ΔKE={delta_ke}", False)
            
            # 3. Kinematic consistency for acceleration phase
            # v_belt = a*t_1
            t1 = self.results['t_1']
            v_from_a = self.results['a'] * t1
            if not math.isclose(v_from_a, self.v_belt, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Kinematic violation: a*t1={v_from_a} ≠ v_belt={self.v_belt}", False)
            
            # 4. Displacement during acceleration: x_1 = (1/2)*a*t1^2
            x1_calc = 0.5 * self.results['a'] * t1**2
            if not math.isclose(self.results['x_1'], x1_calc, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Displacement mismatch: x1={self.results['x_1']} ≠ 0.5*a*t1²={x1_calc}", False)
            
            # 5. Total distance: x_1 + v_belt*t_2 = L
            t2 = self.results['t_2']
            total_dist = self.results['x_1'] + self.v_belt * t2
            if not math.isclose(total_dist, self.L, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Total distance violation: {total_dist} ≠ L={self.L}", False)
            
            return self._log("All invariants satisfied", True)
        except Exception as e:
            return self._log(f"Invariant test failed with error: {str(e)}", False)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check step-by-step consistency
            t1 = self.results['t_1']
            t2 = self.results['t_2']
            t_total = self.results['t']
            
            # 1. t = t1 + t2
            if not math.isclose(t_total, t1 + t2, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Time addition violation: t={t_total} ≠ t1+t2={t1+t2}", False)
            
            # 2. t2 = (L - x1)/v_belt
            t2_calc = (self.L - self.results['x_1']) / self.v_belt
            if not math.isclose(t2, t2_calc, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"t2 calculation: {t2} ≠ (L-x1)/v={t2_calc}", False)
            
            # 3. v_m^2 = 2*a*L (for minimum speed calculation)
            v_m_sq = self.results['v_m']**2
            expected_vm_sq = 2 * self.results['a'] * self.L
            if not math.isclose(v_m_sq, expected_vm_sq, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"v_m violation: v_m²={v_m_sq} ≠ 2aL={expected_vm_sq}", False)
            
            # 4. t1 = v_belt/a
            t1_calc = self.v_belt / self.results['a']
            if not math.isclose(t1, t1_calc, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"t1 calculation: {t1} ≠ v/a={t1_calc}", False)
            
            return self._log("Trace consistency check passed", True)
        except Exception as e:
            return self._log(f"Trace consistency test failed: {str(e)}", False)
    
    def test_monotonicity(self):
        """Perturb inputs and check if results change in expected direction."""
        try:
            original_mu = self.mu_k
            
            # Test 1: Increase μ (friction coefficient)
            self.mu_k = original_mu * 1.1  # 10% increase
            
            # Recalculate expected changes
            new_a = self.mu_k * self.g  # a increases
            new_t1 = self.v_belt / new_a  # t1 decreases
            new_x1 = 0.5 * new_a * new_t1**2  # x1 = v_belt^2/(2a) - decreases
            new_t2 = (self.L - new_x1) / self.v_belt  # t2 increases
            new_t = new_t1 + new_t2
            
            # Restore original
            self.mu_k = original_mu
            
            # Check monotonic trends
            # Higher μ → higher a
            if not (new_a > self.results['a']):
                return self._log("Monotonicity fail: a should increase with μ", False)
            
            # Higher a → shorter t1
            if not (new_t1 < self.results['t_1']):
                return self._log("Monotonicity fail: t1 should decrease with a", False)
            
            # Higher a → shorter x1 (since v_belt fixed)
            if not (new_x1 < self.results['x_1']):
                return self._log("Monotonicity fail: x1 should decrease with a", False)
            
            # For total time: ambiguous, but let's check the derivative
            # t = v_belt/a + (L - v_belt^2/(2a))/v_belt
            # dt/da = -v_belt/a^2 + (v_belt^2/(2a^2))/v_belt = -v_belt/a^2 + v_belt/(2a^2) = -v_belt/(2a^2) < 0
            # So total time should decrease with increased a
            if not (new_t < self.results['t']):
                return self._log("Monotonicity fail: total time should decrease with increased μ", False)
            
            # Test 2: Increase belt speed (perturbation)
            original_v = self.v_belt
            self.v_belt = original_v * 1.1
            
            # Recalculate
            new_t1_v = self.v_belt / self.results['a']  # increases
            new_x1_v = 0.5 * self.results['a'] * new_t1_v**2  # increases
            new_t2_v = (self.L - new_x1_v) / self.v_belt  # ambiguous
            new_t_v = new_t1_v + new_t2_v
            
            # Restore
            self.v_belt = original_v
            
            # Higher v_belt → longer t1
            if not (new_t1_v > self.results['t_1']):
                return self._log("Monotonicity fail: t1 should increase with v_belt", False)
            
            return self._log("Monotonicity tests passed", True)
        except Exception as e:
            return self._log(f"Monotonicity test failed: {str(e)}", False)
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        
        tests = [
            self.test_invariants,
            self.test_trace_consistency,
            self.test_monotonicity,
        ]
        
        all_passed = True
        for test in tests:
            try:
                passed = test()
                if not passed:
                    all_passed = False
            except Exception as e:
                self._log(f"Test {test.__name__} crashed: {traceback.format_exc()}", False)
                all_passed = False
        
        return all_passed, self.logs