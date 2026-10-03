import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters with defaults
        self.L = getattr(solution_instance, 'L', 1.0)
        self.M = getattr(solution_instance, 'M', 2.0)
        self.v_belt = getattr(solution_instance, 'v', 2.0)  # conveyor belt speed
        self.m = getattr(solution_instance, 'm', 1.0)
        self.h = getattr(solution_instance, 'h', 1.0)
        self.mu = getattr(solution_instance, 'mu', 0.2)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Results with fallbacks
        self.v_0 = self.results.get('v_0', 0.0)  # initial speed after falling
        self.v_1 = self.results.get('v_1', 0.0)  # speed before first collision
        self.v_B1 = self.results.get('v_B1', 0.0)  # velocity after first collision
        self.x_B = self.results.get('x_B', 0.0)  # distance after collision
        self.t_0 = self.results.get('t_0', 0.0)  # initial deceleration time
        self.t = self.results.get('t', 0.0)  # total time
        
        # For recursive patterns
        self.v_Bn_value = self.results.get('v_Bn', 0.0)  # general term magnitude
        
        # Expected values from steps analysis
        self.v_0_expected = 2 * math.sqrt(5)  # ~4.472
        self.v_1_expected = 4.0
        self.v_B1_expected = -4/3  # ~-1.333
        self.x_B_expected = 4/9  # ~0.444
        self.t_0_expected = math.sqrt(5) - 2  # ~0.236
        
        # Logs storage
        self.logs = []

    def _log(self, message: str, success: bool = True):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success

    def test_invariants(self) -> bool:
        """Verify energy conservation, momentum conservation, and work-energy theorem."""
        success = True
        
        # 1. Conservation of mechanical energy for falling block B
        E_initial = self.m * self.g * self.h
        E_final = 0.5 * self.m * self.v_0 ** 2
        if not math.isclose(E_initial, E_final, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Energy conservation failed: mgh={E_initial} != 0.5*m*v0²={E_final}", False)
        else:
            self._log("Energy conservation (falling) passed.")
        
        # 2. Work-energy theorem on conveyor belt (first pass)
        # Friction does negative work: -μmgL = ΔKE
        friction_work = -self.mu * self.m * self.g * self.L
        delta_KE = 0.5 * self.m * (self.v_1 ** 2 - self.v_0 ** 2)
        if not math.isclose(friction_work, delta_KE, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Work-energy theorem failed: W_friction={friction_work} != ΔKE={delta_KE}", False)
        else:
            self._log("Work-energy theorem (first pass) passed.")
        
        # 3. Elastic collision: momentum conservation
        # Before collision: p = m*v_1, after: p = m*v_B1 + M*v_A1
        # v_A1 from energy conservation in collision
        # Using formulas from step 11: v_A1 = (2/3)*v_1 = 8/3 ≈ 2.667
        v_A1_expected = (2/3) * self.v_1_expected  # From step 11
        p_before = self.m * self.v_1
        p_after = self.m * self.v_B1 + self.M * v_A1_expected
        if not math.isclose(p_before, p_after, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Momentum conservation failed: p_before={p_before} != p_after={p_after}", False)
        else:
            self._log("Momentum conservation (first collision) passed.")
        
        # 4. Elastic collision: kinetic energy conservation
        KE_before = 0.5 * self.m * self.v_1 ** 2
        KE_after = 0.5 * self.m * self.v_B1 ** 2 + 0.5 * self.M * v_A1_expected ** 2
        if not math.isclose(KE_before, KE_after, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"KE conservation failed: KE_before={KE_before} != KE_after={KE_after}", False)
        else:
            self._log("KE conservation (first collision) passed.")
        
        # 5. Impulse theorem for initial deceleration
        # -μmg t_0 = m(v_1 - v_0)
        impulse_lhs = -self.mu * self.m * self.g * self.t_0
        impulse_rhs = self.m * (self.v_1 - self.v_0)
        if not math.isclose(impulse_lhs, impulse_rhs, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Impulse theorem failed: LHS={impulse_lhs} != RHS={impulse_rhs}", False)
        else:
            self._log("Impulse theorem (initial deceleration) passed.")
        
        return success

    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with formulas."""
        success = True
        
        # 1. Check v_0 = sqrt(2gh)
        v_0_calc = math.sqrt(2 * self.g * self.h)
        if not math.isclose(self.v_0, v_0_calc, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"v_0 mismatch: {self.v_0} != sqrt(2gh)={v_0_calc}", False)
        else:
            self._log("v_0 consistent with sqrt(2gh).")
        
        # 2. Check v_1 from work-energy: v_1² = v_0² - 2μgL
        v_1_sq_calc = max(0, self.v_0 ** 2 - 2 * self.mu * self.g * self.L)
        v_1_calc = math.sqrt(v_1_sq_calc)
        if not math.isclose(self.v_1, v_1_calc, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"v_1 mismatch: {self.v_1} != sqrt(v0²-2μgL)={v_1_calc}", False)
        else:
            self._log("v_1 consistent with work-energy theorem.")
        
        # 3. Check elastic collision formulas
        # For elastic collision with m1=m, m2=M, v2i=0:
        # v1f = (m1-m2)/(m1+m2) * v1i
        # v2f = (2m1)/(m1+m2) * v1i
        v_B1_calc = (self.m - self.M) / (self.m + self.M) * self.v_1
        if not math.isclose(self.v_B1, v_B1_calc, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"v_B1 mismatch: {self.v_B1} != formula={v_B1_calc}", False)
        else:
            self._log("v_B1 consistent with elastic collision formula.")
        
        # 4. Check x_B = v_B1²/(2μg)
        x_B_calc = self.v_B1 ** 2 / (2 * self.mu * self.g)
        if not math.isclose(self.x_B, x_B_calc, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"x_B mismatch: {self.x_B} != formula={x_B_calc}", False)
        else:
            self._log("x_B consistent with work-energy after collision.")
        
        # 5. Check t_0 = (v_0 - v_1)/(μg)
        t_0_calc = (self.v_0 - self.v_1) / (self.mu * self.g)
        if not math.isclose(self.t_0, t_0_calc, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"t_0 mismatch: {self.t_0} != formula={t_0_calc}", False)
        else:
            self._log("t_0 consistent with impulse theorem.")
        
        # 6. Check recursive pattern (for n=1)
        # v_B1 should be -1/3 * v_1
        v_B1_recursive = (-1/3) * self.v_1
        if not math.isclose(self.v_B1, v_B1_recursive, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Recursive pattern mismatch: v_B1={self.v_B1} != (-1/3)*v_1={v_B1_recursive}", False)
        else:
            self._log("Recursive pattern (n=1) consistent.")
        
        return success

    def test_monotonicity(self) -> bool:
        """Perturb inputs and check directional trends."""
        success = True
        
        # Save original values
        original_mu = self.mu
        original_results = {
            'v_1': self.v_1,
            'x_B': self.x_B,
            'v_B1': self.v_B1
        }
        
        # Test 1: Increase μ (friction)
        self.mu *= 1.1  # 10% increase
        
        # Recalculate using formulas (not running full solution)
        # v_1² = v_0² - 2μgL
        new_v_1_sq = max(0, self.v_0 ** 2 - 2 * self.mu * self.g * self.L)
        new_v_1 = math.sqrt(new_v_1_sq) if new_v_1_sq > 0 else 0
        
        # v_B1 from elastic collision formula
        new_v_B1 = (self.m - self.M) / (self.m + self.M) * new_v_1
        
        # x_B = v_B1²/(2μg)
        new_x_B = new_v_B1 ** 2 / (2 * self.mu * self.g)
        
        # Check trends
        # Higher μ → lower v_1
        if not (new_v_1 < original_results['v_1'] or math.isclose(new_v_1, original_results['v_1'], rel_tol=1e-2)):
            success = self._log(f"Monotonicity: v_1 didn't decrease with increased μ", False)
        
        # Higher μ → smaller |v_B1| (since v_1 decreased)
        if not (abs(new_v_B1) < abs(original_results['v_B1']) or 
                math.isclose(abs(new_v_B1), abs(original_results['v_B1']), rel_tol=1e-2)):
            success = self._log(f"Monotonicity: |v_B1| didn't decrease with increased μ", False)
        
        # Higher μ → smaller x_B
        if not (new_x_B < original_results['x_B'] or math.isclose(new_x_B, original_results['x_B'], rel_tol=1e-2)):
            success = self._log(f"Monotonicity: x_B didn't decrease with increased μ", False)
        
        if success:
            self._log("Monotonicity test (increasing μ) passed.")
        
        # Restore original μ
        self.mu = original_mu
        
        # Test 2: Increase mass m
        original_m = self.m
        self.m *= 1.1  # 10% increase
        
        # Recalculate v_0 (conservation of energy: independent of m)
        new_v_0 = math.sqrt(2 * self.g * self.h)
        
        # v_1² = v_0² - 2μgL (independent of m for v_0)
        new_v_1 = math.sqrt(max(0, new_v_0 ** 2 - 2 * self.mu * self.g * self.L))
        
        # v_B1 from elastic collision (now different mass ratio)
        new_v_B1 = (self.m - self.M) / (self.m + self.M) * new_v_1
        
        # Check: with m closer to M, v_B1 magnitude should decrease
        # Original m=1, M=2, so (m-M)/(m+M) = -1/3 ≈ -0.333
        # New m=1.1, so (1.1-2)/(1.1+2) ≈ -0.9/3.1 ≈ -0.290
        if not (abs(new_v_B1) < abs(original_results['v_B1']) or 
                math.isclose(abs(new_v_B1), abs(original_results['v_B1']), rel_tol=1e-2)):
            success = self._log(f"Monotonicity: |v_B1| didn't decrease with increased m", False)
        
        # Restore original m
        self.m = original_m
        
        if success:
            self._log("Monotonicity tests passed.")
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs.clear()
        self.logs.append("Starting PhysicsChecker tests...")
        
        # Run tests
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        # Summary
        self.logs.append(f"\n--- SUMMARY ---")
        self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs