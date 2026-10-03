import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of input parameters
        self.m = getattr(solution_instance, 'm', 0.12)
        self.R = getattr(solution_instance, 'R', 0.4)
        self.k = getattr(solution_instance, 'k', 100)
        self.v0 = getattr(solution_instance, 'v0', 2 * math.sqrt(21))
        self.L = getattr(solution_instance, 'L', 0.8)
        self.v_conveyor = getattr(solution_instance, 'v', 2)  # belt speed
        self.mu = getattr(solution_instance, 'mu', 0.5)
        self.g = getattr(solution_instance, 'g', 10.0)  # inferred from steps
        
        # Safe extraction of results with fallbacks
        self.results = {}
        result_keys = [
            'F_N', 'v\'', 'DeltaE', 'v_2', 'Deltax', 'a', 'x_1', 'v_C',
            'v', 'x_2', 'v_F', 'v_1'
        ]
        for key in result_keys:
            # First try direct attribute, then results dictionary
            value = getattr(solution_instance, key, None)
            if value is None:
                value = getattr(solution_instance, 'results', {}).get(key, None)
            self.results[key] = value
        
        # Alias for convenience
        self.F_N = self.results.get('F_N')
        self.v_prime = self.results.get("v'")
        self.DeltaE = self.results.get('DeltaE')
        self.v_2 = self.results.get('v_2')
        self.Deltax = self.results.get('Deltax')
        self.a = self.results.get('a')
        self.x_1 = self.results.get('x_1')
        self.v_C = self.results.get('v_C')
        self.v_common = self.results.get('v')  # from step 9
        self.x_2 = self.results.get('x_2')
        self.v_F = self.results.get('v_F')
        self.v_1 = self.results.get('v_1')
        
        self.logs = []
        
    def _log(self, msg, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {msg}")
        else:
            self.logs.append(f"[INFO] {msg}")
            
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        
    def test_invariants(self):
        """Verify energy conservation, Newton's laws based on KB."""
        try:
            # 1. Conservation of Mechanical Energy (D to F)
            if self.v_F is not None:
                K0 = 0.5 * self.m * self.v0**2
                K_F = 0.5 * self.m * self.v_F**2
                delta_U = self.m * self.g * 2 * self.R
                energy_balance = K_F - K0
                if not self._is_close(energy_balance, delta_U):
                    self._log(f"Energy D→F mismatch: ΔK={energy_balance}, ΔU={delta_U}", False)
                    return False
                self._log(f"Energy D→F conserved: ΔK={energy_balance:.2f}, ΔU={delta_U:.2f}", True)
            
            # 2. Newton's 2nd Law at point F (centripetal)
            if self.F_N is not None and self.v_F is not None:
                centripetal = self.m * self.v_F**2 / self.R
                normal_force_eq = self.F_N - self.m * self.g
                if not self._is_close(normal_force_eq, centripetal):
                    self._log(f"Normal force mismatch: F_N-mg={normal_force_eq}, mv²/R={centripetal}", False)
                    return False
                self._log(f"Normal force at F satisfies Newton's 2nd law", True)
            
            # 3. Acceleration on conveyor belt
            if self.a is not None:
                expected_a = self.mu * self.g
                if not self._is_close(self.a, expected_a):
                    self._log(f"Acceleration mismatch: a={self.a}, μ·g={expected_a}", False)
                    return False
                self._log(f"Acceleration a = μ·g = {expected_a} m/s²", True)
            
            # 4. Conservation of Momentum (collision a-b)
            if all(v is not None for v in [self.v_F, self.v_1, self.v_2]):
                p_initial = self.m * self.v_F
                p_final = -self.m * self.v_1 + 3 * self.m * self.v_2
                if not self._is_close(p_initial, p_final):
                    self._log(f"Momentum mismatch: initial={p_initial}, final={p_final}", False)
                    return False
                self._log(f"Momentum conserved in collision a-b", True)
            
            # 5. Energy loss in collision
            if self.DeltaE is not None and self.v_F is not None and self.v_1 is not None and self.v_2 is not None:
                E_initial = 0.5 * self.m * self.v_F**2
                E_final = 0.5 * self.m * self.v_1**2 + 1.5 * self.m * self.v_2**2
                delta_E_calc = E_initial - E_final
                if not self._is_close(self.DeltaE, delta_E_calc):
                    self._log(f"ΔE mismatch: reported={self.DeltaE}, computed={delta_E_calc}", False)
                    return False
                self._log(f"Energy loss ΔE = {delta_E_calc:.2f} J", True)
            
            # 6. Spring energy conservation (step 11)
            if all(v is not None for v in [self.v_common, self.v_prime, self.x_1]):
                E_initial_spring = 0.5 * 4 * self.m * self.v_common**2
                E_final_spring = 0.5 * 6 * self.m * self.v_prime**2 + 0.5 * self.k * self.x_1**2
                if not self._is_close(E_initial_spring, E_final_spring):
                    self._log(f"Spring energy mismatch: initial={E_initial_spring}, final={E_final_spring}", False)
                    return False
                self._log(f"Spring energy conserved", True)
            
            # 7. Symmetry of spring oscillation (step 12)
            if self.x_1 is not None and self.x_2 is not None:
                if not self._is_close(self.x_1, self.x_2):
                    self._log(f"Spring symmetry broken: x1={self.x_1}, x2={self.x_2}", False)
                    return False
                self._log(f"Spring oscillation symmetric: x1 = x2 = {self.x_1} m", True)
            
            return True
        except Exception as e:
            self._log(f"test_invariants failed: {e}", False)
            return False
        
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Final answers from target list
            # 1. v_F and F_N
            if self.v_F is not None and not self._is_close(self.v_F, 10.0):
                self._log(f"v_F inconsistent: {self.v_F} ≠ 10", False)
                return False
            if self.F_N is not None and not self._is_close(self.F_N, 31.2):
                self._log(f"F_N inconsistent: {self.F_N} ≠ 31.2", False)
                return False
                
            # 2. ΔE
            if self.DeltaE is not None and not self._is_close(self.DeltaE, 0.0):
                self._log(f"ΔE inconsistent: {self.DeltaE} ≠ 0", False)
                return False
                
            # 3. Δx
            if self.Deltax is not None and not self._is_close(self.Deltax, 0.2):
                self._log(f"Δx inconsistent: {self.Deltax} ≠ 0.2", False)
                return False
                
            # Additional cross-checks
            if self.x_1 is not None and self.Deltax is not None:
                if not self._is_close(self.Deltax, 2 * self.x_1):
                    self._log(f"Δx ≠ 2*x1: {self.Deltax} ≠ 2*{self.x_1}", False)
                    return False
            
            self._log("All trace checks consistent with final answers", True)
            return True
        except Exception as e:
            self._log(f"test_trace_consistency failed: {e}", False)
            return False
        
    def test_monotonicity(self):
        """Perturb mass m and check trends, then restore."""
        original_m = self.m
        perturb_factor = 1.05  # 5% increase
        success = True
        
        try:
            # Perturb mass
            self.m = original_m * perturb_factor
            
            # Recalculate expected trends (symbolic fallback)
            # v_F from energy conservation: v_F = sqrt(v0² + 4gR) independent of m
            v_F_independent = math.sqrt(self.v0**2 + 4 * self.g * self.R)
            
            # F_N = m*g + m*v_F²/R ∝ m
            if self.F_N is not None:
                F_N_expected = self.F_N * perturb_factor  # linear scaling
                self._log(f"F_N should scale linearly with m: factor {perturb_factor}", None)
            
            # Spring compression x1 from energy: x1 = sqrt((4m*v² - 6m*v'²)/k) ∝ sqrt(m)
            if self.x_1 is not None:
                x1_expected = self.x_1 * math.sqrt(perturb_factor)
                self._log(f"x1 should scale with √m: factor {math.sqrt(perturb_factor):.3f}", None)
            
            # Δx = 2*x1 ∝ sqrt(m)
            if self.Deltax is not None:
                Deltax_expected = self.Deltax * math.sqrt(perturb_factor)
                self._log(f"Δx should scale with √m: factor {math.sqrt(perturb_factor):.3f}", None)
            
            # For actual verification, we'd need to re-run the solution with perturbed m.
            # Since we can't, we at least verify the symbolic relationships.
            if self.v_F is not None:
                v_F_new = math.sqrt(self.v0**2 + 4 * self.g * self.R)
                if not self._is_close(v_F_new, self.v_F):
                    self._log(f"v_F changed with m perturbation (should be independent)", False)
                    success = False
                else:
                    self._log(f"v_F correctly independent of m", True)
            
        except Exception as e:
            self._log(f"test_monotonicity failed: {e}", False)
            success = False
        finally:
            # Restore original mass
            self.m = original_m
            
        return success
        
    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs.clear()
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_passed = True
        for name, test_func in tests:
            try:
                passed = test_func()
                if not passed:
                    all_passed = False
                    self._log(f"Test '{name}' failed", False)
            except Exception as e:
                self._log(f"Test '{name}' crashed: {e}", False)
                all_passed = False
        
        return all_passed, self.logs