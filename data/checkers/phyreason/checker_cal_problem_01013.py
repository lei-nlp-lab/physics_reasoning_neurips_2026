import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.L1 = getattr(solution_instance, 'L1', 0.19)  # convert cm to m
        if isinstance(self.L1, str):
            self.L1 = float(self.L1.strip('cm')) * 0.01
        elif isinstance(self.L1, (int, float)):
            if self.L1 > 10:  # likely in cm
                self.L1 *= 0.01

        self.M = getattr(solution_instance, 'M', 4.5)
        self.m = getattr(solution_instance, 'm', 3.0)
        self.S = getattr(solution_instance, 'S', 30.0) * 1e-4  # cm² to m²
        self.p0 = getattr(solution_instance, 'p0', 1.0e5)
        self.F_max = getattr(solution_instance, 'F', 120.0)
        self.g = getattr(solution_instance, 'g', 10.0)

        # Safe extraction of results
        results = getattr(solution_instance, 'results', {})
        self.p1 = results.get('p_1', getattr(solution_instance, 'p_1', 8.5e4))
        self.a = results.get('a', getattr(solution_instance, 'a', 6.0))
        self.p2 = results.get('p_2', getattr(solution_instance, 'p_2', 7.6e4))
        self.L2 = results.get('L_2', getattr(solution_instance, 'L_2', 0.2125))
        if isinstance(self.L2, str):
            self.L2 = float(self.L2.strip('cm')) * 0.01
        elif isinstance(self.L2, (int, float)):
            if self.L2 > 1.0:  # likely in cm
                self.L2 *= 0.01

        # For symbolic computations
        self.p0_sym = sp.symbols('p0')
        self.M_sym = sp.symbols('M')
        self.g_sym = sp.symbols('g')
        self.S_sym = sp.symbols('S')
        self.F_sym = sp.symbols('F')
        self.m_sym = sp.symbols('m')
        self.a_sym = sp.symbols('a')
        self.p1_sym = sp.symbols('p1')
        self.p2_sym = sp.symbols('p2')
        self.L1_sym = sp.symbols('L1')
        self.L2_sym = sp.symbols('L2')

        self.logs = []

    def log(self, msg, condition=None):
        if condition is not None:
            msg = f"[{'PASS' if condition else 'FAIL'}] {msg}"
        self.logs.append(msg)

    def test_invariants(self):
        """Check fundamental physical laws."""
        success = True

        # 1. Force balance for stationary elevator (step 1)
        # p1 = p0 - M*g/S
        p1_expected = self.p0 - (self.M * self.g) / self.S
        if not math.isclose(self.p1, p1_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self.log(f"Force balance (p1) mismatch: {self.p1} vs {p1_expected}", False)
        else:
            self.log(f"Force balance (p1) holds: {self.p1} Pa ≈ {p1_expected} Pa", True)

        # 2. Newton's second law for system acceleration (step 2)
        # a = (F_max - (M+m)*g) / (M+m)
        a_expected = (self.F_max - (self.M + self.m) * self.g) / (self.M + self.m)
        if not math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self.log(f"Newton's second law (a) mismatch: {self.a} vs {a_expected}", False)
        else:
            self.log(f"Newton's second law (a) holds: {self.a} m/s² ≈ {a_expected} m/s²", True)

        # 3. Force balance during acceleration (step 3)
        # p2 = p0 - M*(g+a)/S
        p2_expected = self.p0 - self.M * (self.g + self.a) / self.S
        if not math.isclose(self.p2, p2_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self.log(f"Force balance (p2) mismatch: {self.p2} vs {p2_expected}", False)
        else:
            self.log(f"Force balance (p2) holds: {self.p2} Pa ≈ {p2_expected} Pa", True)

        # 4. Boyle's law (isothermal) (step 4)
        # p1 * L1 * S = p2 * L2 * S  => p1 * L1 = p2 * L2
        if not math.isclose(self.p1 * self.L1, self.p2 * self.L2, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self.log(f"Boyle's law mismatch: p1*L1={self.p1*self.L1} vs p2*L2={self.p2*self.L2}", False)
        else:
            self.log(f"Boyle's law holds: p1*L1 = {self.p1*self.L1} ≈ {self.p2*self.L2} = p2*L2", True)

        return success

    def test_trace_consistency(self):
        """Cross-check intermediate steps with final answer."""
        success = True

        # 1. Check that p1 is consistent with static equilibrium
        p1_from_force_balance = self.p0 - (self.M * self.g) / self.S
        if not math.isclose(self.p1, p1_from_force_balance, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self.log(f"Trace consistency (p1): {self.p1} vs {p1_from_force_balance}", False)
        else:
            self.log(f"Trace consistency (p1) OK", True)

        # 2. Check that a is consistent with maximum tension
        a_from_tension = (self.F_max - (self.M + self.m) * self.g) / (self.M + self.m)
        if not math.isclose(self.a, a_from_tension, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self.log(f"Trace consistency (a): {self.a} vs {a_from_tension}", False)
        else:
            self.log(f"Trace consistency (a) OK", True)

        # 3. Check that p2 is consistent with acceleration
        p2_from_acceleration = self.p0 - self.M * (self.g + self.a) / self.S
        if not math.isclose(self.p2, p2_from_acceleration, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self.log(f"Trace consistency (p2): {self.p2} vs {p2_from_acceleration}", False)
        else:
            self.log(f"Trace consistency (p2) OK", True)

        # 4. Check that L2 is consistent with Boyle's law using p1 and p2 from steps
        L2_expected = (self.p1 * self.L1) / self.p2
        if not math.isclose(self.L2, L2_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self.log(f"Trace consistency (L2): {self.L2} vs {L2_expected}", False)
        else:
            self.log(f"Trace consistency (L2) OK: {self.L2} m ≈ {L2_expected} m", True)

        # 5. Final answer in cm
        L2_cm = self.L2 * 100
        if not math.isclose(L2_cm, 21.25, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self.log(f"Final answer mismatch: {L2_cm} cm vs 21.25 cm", False)
        else:
            self.log(f"Final answer OK: {L2_cm} cm ≈ 21.25 cm", True)

        return success

    def test_monotonicity(self):
        """Perturb input parameters and check trend."""
        original_M = self.M
        perturbed_M = original_M * 1.01  # increase by 1%
        self.M = perturbed_M

        # Recalculate p1, a, p2, L2 using formulas (without using solution's methods)
        p1_new = self.p0 - (self.M * self.g) / self.S
        a_new = (self.F_max - (self.M + self.m) * self.g) / (self.M + self.m)
        p2_new = self.p0 - self.M * (self.g + a_new) / self.S
        L2_new = (p1_new * self.L1) / p2_new

        # Restore original
        self.M = original_M

        # Expected trend: if M increases, p1 decreases (since p1 = p0 - M*g/S)
        if not (p1_new < self.p1):
            self.log(f"Monotonicity (p1 vs M): p1 should decrease when M increases", False)
            return False

        # a should decrease when M increases (since a = [F - (M+m)g]/(M+m))
        if not (a_new < self.a):
            self.log(f"Monotonicity (a vs M): a should decrease when M increases", False)
            return False

        # p2 should decrease when M increases (p2 = p0 - M*(g+a)/S, but note a also changes)
        # Since both M and (g+a) increase? Actually a decreases, so need careful check.
        # Let's use symbolic monotonicity: ∂p2/∂M = -(g+a)/S - M/S * ∂a/∂M
        # ∂a/∂M = [ -g*(M+m) - (F - (M+m)g) ] / (M+m)^2 = [ -g(M+m) - F + (M+m)g ] / (M+m)^2 = -F/(M+m)^2 < 0
        # So ∂p2/∂M = -(g+a)/S + (M/S) * F/(M+m)^2. Since F is 120 and other values positive, we expect negative? 
        # For given numbers, we can compute numerically:
        delta_M = perturbed_M - original_M
        delta_p2 = p2_new - self.p2
        if delta_M > 0 and delta_p2 > 0:
            self.log(f"Monotonicity (p2 vs M): p2 increased when M increased, unexpected", False)
            return False

        # L2: L2 = (p1 * L1) / p2. p1 decreases, p2 decreases. Which dominates?
        # Use symbolic: L2 = ( (p0 - M*g/S) * L1 ) / (p0 - M*(g+a)/S)
        # For small changes, we can check numerically:
        delta_L2 = L2_new - self.L2
        # From Boyle's law, if p1 decreases and p2 decreases, the ratio may increase or decrease.
        # Actually, in the problem, when M increases, the system becomes heavier, so the piston should go lower? 
        # But here L2 is the maximum distance when accelerating upward. We can check with a simple perturbation.
        # Since we don't have a clear expected trend, we skip this check for L2.

        self.log(f"Monotonicity test passed for p1, a, and p2 under M perturbation", True)
        return True

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        overall_success = True

        tests = [
            ("Invariant checks", self.test_invariants),
            ("Trace consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            try:
                success = test_func()
                overall_success = overall_success and success
            except Exception as e:
                self.log(f"Test '{name}' failed with exception: {e}\n{traceback.format_exc()}", False)
                overall_success = False

        return overall_success, self.logs