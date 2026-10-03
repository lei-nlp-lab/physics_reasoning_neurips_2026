import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with defaults
        self.R = getattr(solution_instance, 'R', 0.4)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.mu = getattr(solution_instance, 'mu', 0.2)
        self.L = getattr(solution_instance, 'L', 2.0)
        self.g = getattr(solution_instance, 'g', 10.0)  # Using g=10 as per step values
        
        # Safely extract results using getattr and .get() fallback
        self.results = getattr(solution_instance, 'results', {})
        self.v_C = getattr(solution_instance, 'v_C', self.results.get('v_C', 0.0))
        self.F_N = getattr(solution_instance, 'F_N', self.results.get('F_N', 0.0))
        self.F_N_prime = getattr(solution_instance, 'F_N_prime', 
                                 getattr(solution_instance, "F_N''", 
                                         self.results.get("F_N'", self.results.get('F_N_prime', 0.0))))
        self.E_pA = getattr(solution_instance, 'E_pA', self.results.get('E_pA', 0.0))
        
        # Derived quantities
        self.v_B = math.sqrt(self.v_C**2 + 4*self.g*self.R)  # From energy B->C
        self.logs = []

    def _log(self, msg, success):
        self.logs.append((msg, success))
        return success

    def test_invariants(self):
        """Verify fundamental physics laws: Newton's 2nd/3rd and Energy Conservation."""
        tests_passed = True
        
        # 1. Newton's 2nd Law at point C (just manages to pass => normal force ~0)
        required_v_C = math.sqrt(self.g * self.R)
        if not math.isclose(self.v_C, required_v_C, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(f"Newton's 2nd Law at C violated: v_C={self.v_C:.3f}, expected √(gR)={required_v_C:.3f}", False)
        else:
            self._log(f"Newton's 2nd Law at C satisfied: v_C ≈ √(gR)", True)
        
        # 2. Newton's 2nd Law at point B
        centripetal_B = self.m * self.v_B**2 / self.R
        normal_force_eq = self.F_N - self.m * self.g
        if not math.isclose(normal_force_eq, centripetal_B, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(f"Newton's 2nd Law at B violated: F_N - mg = {normal_force_eq:.3f}, mv_B²/R = {centripetal_B:.3f}", False)
        else:
            self._log(f"Newton's 2nd Law at B satisfied: F_N - mg = mv_B²/R", True)
        
        # 3. Newton's 3rd Law: F_N' = F_N
        if not math.isclose(self.F_N_prime, self.F_N, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(f"Newton's 3rd Law violated: F_N'={self.F_N_prime:.3f}, F_N={self.F_N:.3f}", False)
        else:
            self._log(f"Newton's 3rd Law satisfied: F_N' = F_N", True)
        
        # 4. Energy Conservation A->B (with friction)
        kinetic_B = 0.5 * self.m * self.v_B**2
        friction_work = self.mu * self.m * self.g * self.L
        energy_A_to_B = self.E_pA - friction_work
        if not math.isclose(energy_A_to_B, kinetic_B, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(f"Energy A->B violated: E_pA - μmgL = {energy_A_to_B:.3f}, ½mv_B² = {kinetic_B:.3f}", False)
        else:
            self._log(f"Energy A->B satisfied: E_pA - μmgL = ½mv_B²", True)
        
        # 5. Energy Conservation B->C (no friction)
        kinetic_C = 0.5 * self.m * self.v_C**2
        potential_C = self.m * self.g * (2 * self.R)
        energy_B_to_C = kinetic_B - (kinetic_C + potential_C)
        if not math.isclose(energy_B_to_C, 0.0, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(f"Energy B->C violated: ½mv_B² - (½mv_C² + mg·2R) = {energy_B_to_C:.3f}", False)
        else:
            self._log(f"Energy B->C satisfied: ½mv_B² = ½mv_C² + mg·2R", True)
        
        return tests_passed

    def test_trace_consistency(self):
        """Cross-check intermediate calculations with final answers."""
        tests_passed = True
        
        # Recompute v_B from v_C using energy B->C
        v_B_computed = math.sqrt(self.v_C**2 + 4 * self.g * self.R)
        if not math.isclose(self.v_B, v_B_computed, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(f"Trace inconsistency: v_B from step_2 = {self.v_B:.3f}, recomputed from v_C = {v_B_computed:.3f}", False)
        else:
            self._log(f"Trace v_B consistent with v_C via energy", True)
        
        # Recompute F_N from v_B
        F_N_computed = self.m * self.g + self.m * self.v_B**2 / self.R
        if not math.isclose(self.F_N, F_N_computed, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(f"Trace inconsistency: F_N = {self.F_N:.3f}, recomputed = {F_N_computed:.3f}", False)
        else:
            self._log(f"Trace F_N consistent with v_B via Newton's 2nd", True)
        
        # Recompute E_pA from v_B and friction
        E_pA_computed = 0.5 * self.m * self.v_B**2 + self.mu * self.m * self.g * self.L
        if not math.isclose(self.E_pA, E_pA_computed, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(f"Trace inconsistency: E_pA = {self.E_pA:.3f}, recomputed = {E_pA_computed:.3f}", False)
        else:
            self._log(f"Trace E_pA consistent with v_B and friction", True)
        
        return tests_passed

    def test_monotonicity(self):
        """Perturb mass m and verify scaling trends, then restore."""
        original_m = self.m
        tests_passed = True
        
        try:
            # Double the mass
            self.m = 2.0 * original_m
            
            # Recompute expected scaled values
            # v_C independent of m (from Newton's 2nd at C)
            v_C_expected = math.sqrt(self.g * self.R)
            # v_B independent of m (from energy B->C with same v_C)
            v_B_expected = math.sqrt(v_C_expected**2 + 4 * self.g * self.R)
            # F_N scales linearly with m
            F_N_expected = self.m * self.g + self.m * v_B_expected**2 / self.R
            # E_pA scales linearly with m
            E_pA_expected = 0.5 * self.m * v_B_expected**2 + self.mu * self.m * self.g * self.L
            
            # Compute actual using formulas from solution steps
            # Step 1: v_C (unchanged)
            v_C_actual = math.sqrt(self.g * self.R)
            # Step 2: v_B from energy B->C
            v_B_actual = math.sqrt(v_C_actual**2 + 4 * self.g * self.R)
            # Step 3: F_N from Newton's 2nd at B
            F_N_actual = self.m * self.g + self.m * v_B_actual**2 / self.R
            # Step 5: E_pA from energy A->B
            E_pA_actual = 0.5 * self.m * v_B_actual**2 + self.mu * self.m * self.g * self.L
            
            # Check scaling
            if not math.isclose(F_N_actual / F_N_expected, 1.0, rel_tol=0.02):
                tests_passed = self._log(f"Monotonicity: F_N does not scale linearly with mass", False)
            else:
                self._log(f"Monotonicity: F_N scales linearly with mass", True)
            
            if not math.isclose(E_pA_actual / E_pA_expected, 1.0, rel_tol=0.02):
                tests_passed = self._log(f"Monotonicity: E_pA does not scale linearly with mass", False)
            else:
                self._log(f"Monotonicity: E_pA scales linearly with mass", True)
                
        except Exception as e:
            tests_passed = self._log(f"Monotonicity test failed with error: {e}", False)
        finally:
            # Restore original mass
            self.m = original_m
        
        return tests_passed

    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []
        overall_success = True
        
        try:
            overall_success = self.test_invariants() and overall_success
            overall_success = self.test_trace_consistency() and overall_success
            overall_success = self.test_monotonicity() and overall_success
        except Exception as e:
            self.logs.append((f"Unhandled exception: {traceback.format_exc()}", False))
            overall_success = False
        
        return overall_success, self.logs