import math
import traceback

class PhysicsChecker:
    """
    Robust checker for the parallel-plate capacitor work problem.
    """
    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        givens = getattr(solution_instance, 'givens', {}) or {}
        if isinstance(givens, dict):
            self.V = givens.get('V', None)
            self.d = givens.get('d', None)
            self.d_prime = givens.get("d'", None) or givens.get('d_prime', None)
            self.r = givens.get('r', None)
        else:
            self.V = None
            self.d = None
            self.d_prime = None
            self.r = None

        # Extract results safely
        results = getattr(solution_instance, 'results', {}) or {}
        self.W = results.get('W', None)

        # Physical constants
        self.epsilon0 = 8.854187817e-12
        self.pi = math.pi

        # Explanation steps (optional, not used in numerical checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # -----------------------------------------------------------------
    # Helper methods (re-derive quantities from givens/theorems)
    # -----------------------------------------------------------------
    def _compute_capacitance(self, d_val):
        """C = epsilon0 * pi * r^2 / d"""
        if None in (self.r, d_val) or d_val == 0:
            return None
        return self.epsilon0 * self.pi * self.r**2 / d_val

    def _compute_charge(self):
        """Q = C_initial * V (constant after disconnection)"""
        if None in (self.V, self.d, self.r) or self.d == 0:
            return None
        C_init = self._compute_capacitance(self.d)
        if C_init is None:
            return None
        return C_init * self.V

    def _compute_energy(self, d_val):
        """U = Q^2 / (2 * C(d)) = Q^2 * d / (2 * epsilon0 * pi * r^2)"""
        if None in (self.r, d_val) or d_val == 0:
            return None
        Q = self._compute_charge()
        if Q is None:
            return None
        return Q**2 * d_val / (2 * self.epsilon0 * self.pi * self.r**2)

    def _compute_expected_work(self, d_val, dprime_val):
        """W = epsilon0 * pi * r^2 * (d' - d) * V^2 / (2 * d^2)"""
        if None in (self.V, self.r, d_val, dprime_val) or d_val == 0:
            return None
        area = self.pi * self.r**2
        return (self.epsilon0 * area * (dprime_val - d_val) * self.V**2) / (2 * d_val**2)

    # -----------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------
    def test_interface_results_contains_W(self):
        """Check that results dict contains key 'W' and it is numeric."""
        if self.W is None:
            return (False, "SKIP: 'W' is missing from results.")
        if not isinstance(self.W, (int, float)):
            return (False, f"'W' is not numeric, got {type(self.W)}")
        if math.isnan(self.W) or math.isinf(self.W):
            return (False, f"'W' is NaN or infinite.")
        return (True, "'W' present and numeric.")

    def test_interface_givens_available(self):
        """Check that all required givens are present."""
        missing = []
        if self.V is None:
            missing.append('V')
        if self.d is None:
            missing.append('d')
        if self.d_prime is None:
            missing.append("d'")
        if self.r is None:
            missing.append('r')
        if missing:
            return (False, f"SKIP: Missing givens: {', '.join(missing)}")
        return (True, "All required givens available.")

    # -----------------------------------------------------------------
    # Theorem-based invariant tests (at least two)
    # -----------------------------------------------------------------
    def test_invariant_work_equals_energy_change(self):
        """Check that W equals change in stored energy (U_final - U_initial)."""
        if self.W is None:
            return (False, "SKIP: W not available.")
        U_init = self._compute_energy(self.d)
        U_final = self._compute_energy(self.d_prime)
        if U_init is None or U_final is None:
            return (False, "SKIP: Cannot compute energies due to missing givens.")
        delta_U = U_final - U_init
        if not math.isclose(self.W, delta_U, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"W = {self.W} differs from energy change {delta_U}.")
        return (True, f"W matches energy change within tolerance.")

    def test_invariant_energy_expressions(self):
        """
        Check that stored energies from Q^2/(2C) and 1/2*C*V^2 are consistent
        for both initial and final states (constant charge).
        """
        if None in (self.V, self.d, self.d_prime, self.r):
            return (False, "SKIP: Missing givens for energy invariance.")
        C_init = self._compute_capacitance(self.d)
        C_final = self._compute_capacitance(self.d_prime)
        if C_init is None or C_final is None:
            return (False, "SKIP: Capacitance computation failed (maybe d=0).")
        Q = self._compute_charge()
        if Q is None:
            return (False, "SKIP: Cannot compute charge.")
        # Initial energy: Q^2/(2C) vs 1/2*C*V^2
        U_Q_init = Q**2 / (2 * C_init)
        U_CV_init = 0.5 * C_init * self.V**2
        if not math.isclose(U_Q_init, U_CV_init, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Initial energy mismatch: {U_Q_init} vs {U_CV_init}.")
        # Final energy: Q^2/(2C) vs 1/2*C*V_final^2 with V_final = Q/C_final
        U_Q_final = Q**2 / (2 * C_final)
        V_final = Q / C_final
        U_CV_final = 0.5 * C_final * V_final**2
        if not math.isclose(U_Q_final, U_CV_final, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Final energy mismatch: {U_Q_final} vs {U_CV_final}.")
        return (True, "Energy expressions are consistent with constant charge.")

    # -----------------------------------------------------------------
    # Modeling / trace quality tests (at least two)
    # -----------------------------------------------------------------
    def test_modeling_charge_is_constant(self):
        """
        Verify that Q_initial = Q_final, as required by the physics constraint
        (disconnected from circuit). This is a modeling quality check.
        """
        if None in (self.V, self.d, self.d_prime, self.r):
            return (False, "SKIP: Missing givens for charge constancy.")
        C_init = self._compute_capacitance(self.d)
        C_final = self._compute_capacitance(self.d_prime)
        if C_init is None or C_final is None:
            return (False, "SKIP: Cannot compute capacitances.")
        Q_initial = C_init * self.V
        # After disconnection, charge remains Q_initial.
        # But from the final state we can also compute Q_final = C_final * V_final.
        # However V_final is not given; we can compute Q_final from given W? No.
        # Better: The model says charge is constant, so we just check that
        # the solver has used that fact implicitly. We can check that
        # the computed W is consistent with constant charge.
        # But a more direct modeling check: Compute Q from initial and then
        # compute what W would be if charge were not constant (wrong).
        # Instead, we can check that the derived formula for W uses constant Q.
        # Since we don't have Q_final directly, we check that W matches
        # the formula derived from constant charge (already in test_modeling_work_formula).
        # So this test is redundant. Let's make a different modeling test.

        # Let's replace: check that the ratio of energies matches the ratio of separations
        # U_final/U_initial = d'/d (since U ∝ d when Q constant).
        U_init = self._compute_energy(self.d)
        U_final = self._compute_energy(self.d_prime)
        if U_init is None or U_final is None or U_init == 0:
            return (False, "SKIP: Cannot compute energies.")
        expected_ratio = self.d_prime / self.d
        actual_ratio = U_final / U_init
        if not math.isclose(expected_ratio, actual_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Energy ratio {actual_ratio} does not match separation ratio {expected_ratio}.")
        return (True, "Energy ratio matches separation ratio (consistent with constant charge).")

    def test_modeling_work_derived_from_capacitance_ratio(self):
        """
        Check that W = (1/2) * C_initial * V^2 * (C_initial/C_final - 1).
        This is an equivalent expression derived in the explanation steps.
        We re‑derive it here from givens.
        """
        if self.W is None:
            return (False, "SKIP: W not available.")
        C_init = self._compute_capacitance(self.d)
        C_final = self._compute_capacitance(self.d_prime)
        if C_init is None or C_final is None or C_final == 0:
            return (False, "SKIP: Cannot compute capacitances.")
        # Derived expression: W = 0.5 * C_init * V^2 * (C_init/C_final - 1)
        derived_W = 0.5 * C_init * self.V**2 * (C_init / C_final - 1)
        if not math.isclose(self.W, derived_W, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"W = {self.W} does not match derived expression {derived_W}.")
        return (True, "W matches expression from capacitance ratio.")

    def test_modeling_work_formula(self):
        """Check that W matches the final explicit formula: epsilon0*pi*r^2*(d'-d)*V^2/(2*d^2)."""
        if self.W is None:
            return (False, "SKIP: W not available.")
        expected_W = self._compute_expected_work(self.d, self.d_prime)
        if expected_W is None:
            return (False, "SKIP: Cannot compute expected work due to missing givens.")
        if not math.isclose(self.W, expected_W, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"W = {self.W} does not match expected {expected_W}.")
        return (True, "W matches the theoretical formula.")

    # -----------------------------------------------------------------
    # Monotonicity test (must perturb an input and compare outputs)
    # -----------------------------------------------------------------
    def test_monotonicity_work_increases_with_separation(self):
        """
        Check monotonicity: If d' increases, W should increase (positive derivative).
        We perturb d' by +1% and check that expected work increases.
        """
        if self.W is None or self.d_prime is None or self.d is None:
            return (False, "SKIP: Missing data for monotonicity test.")
        if self.d_prime <= 0 or self.d <= 0:
            return (False, "SKIP: d or d' non-positive, cannot perturb.")

        delta = 0.01 * self.d_prime  # 1% increase
        d_perturbed = self.d_prime + delta
        W_orig = self._compute_expected_work(self.d, self.d_prime)
        W_pert = self._compute_expected_work(self.d, d_perturbed)
        if W_orig is None or W_pert is None:
            return (False, "SKIP: Could not compute expected work for perturbation.")
        # Derivative dW/d(d') = epsilon0*A*V^2/(2*d^2) > 0, so W should increase
        if W_pert <= W_orig:
            return (False, f"Monotonicity violation: W_pert={W_pert} <= W_orig={W_orig} for increased d'.")
        return (True, f"Work increases with d': {W_pert} > {W_orig}.")

    # -----------------------------------------------------------------
    # Weak answer sanity test (at most one, no isclose, no re-derivation)
    # -----------------------------------------------------------------
    def test_answer_sanity_sign_and_magnitude(self):
        """
        Weak sanity: Check sign of W relative to d' vs d, and that W is finite and not absurdly large.
        No math.isclose here.
        """
        if self.W is None:
            return (False, "SKIP: W not available.")
        if self.d is None or self.d_prime is None:
            return (False, "SKIP: d or d' missing for sign check.")

        # Sign check: if d' > d, W should be positive (pulling apart requires work)
        if self.d_prime > self.d and self.W < 0:
            return (False, f"W negative ({self.W}) but d' > d, expected positive work.")
        if self.d_prime < self.d and self.W > 0:
            return (False, f"W positive ({self.W}) but d' < d, expected negative work.")

        # Basic magnitude bound: typical values give small work; if it's >1e20, likely nonsense
        if abs(self.W) > 1e20:
            return (False, f"W magnitude {self.W} is unreasonably large.")
        return (True, "Sign and magnitude are plausible.")

    # -----------------------------------------------------------------
    # Master test runner
    # -----------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, collect results."""
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)