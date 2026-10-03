import math
import traceback

class PhysicsChecker:
    """
    Audits solutions for a harmonic oscillator in an electric field.
    Checks for correct energy shift, expectation value of x, and dipole moment.
    """

    def __init__(self, solution_instance):
        # Safely extract results (the [emit] contract)
        results = solution_instance.results if hasattr(solution_instance, 'results') else {}
        self.x0 = results.get('x0', None)
        self.Delta_E = results.get('Delta_E', None)
        self.E_n = results.get('E_n', None)
        self.expect_x = results.get('expect_x', None)
        self.P_n = results.get('P_n', None)

        # Safely extract givens (from instance or givens dict)
        givens = (
            solution_instance.givens
            if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict)
            else {}
        )
        self.m = (
            getattr(solution_instance, 'm', None)
            or givens.get('m', None)
        )
        self.e = (
            getattr(solution_instance, 'e', None)
            or givens.get('e', None)
        )
        self.k = (
            getattr(solution_instance, 'k', None)
            or givens.get('k', None)
        )
        self.F = (
            getattr(solution_instance, 'F', None)
            or givens.get('F', None)
        )
        self.hbar = (
            getattr(solution_instance, 'hbar', None)
            or givens.get('hbar', None)
        )
        # n is optional (quantum number)
        self.n = getattr(solution_instance, 'n', givens.get('n', None))

        # Precompute derived quantities if possible (for invariants)
        self.omega = None
        if self.m is not None and self.k is not None and self.m > 0 and self.k > 0:
            self.omega = math.sqrt(self.k / self.m)

    # ------------------------------------------------------------------
    # Interface checks: required keys exist and are numeric
    # ------------------------------------------------------------------
    def test_interface_results_dict(self):
        """Check that results dict contains all expected keys."""
        keys = ['x0', 'Delta_E', 'E_n', 'expect_x', 'P_n']
        missing_actual = []
        for k in keys:
            val = getattr(self, k, None)
            if val is None:
                missing_actual.append(k)
        if missing_actual:
            return (False, f"Missing keys in results: {missing_actual}")
        return (True, "All required keys present.")

    def test_interface_results_types(self):
        """Check that emitted quantities are numeric (int or float)."""
        attrs = ['x0', 'Delta_E', 'E_n', 'expect_x', 'P_n']
        for attr in attrs:
            val = getattr(self, attr, None)
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                return (False, f"{attr} is not numeric (type: {type(val).__name__})")
        return (True, "All present results are numeric.")

    # ------------------------------------------------------------------
    # Invariant checks: physics relations that must hold exactly
    # ------------------------------------------------------------------
    def test_invariant_expect_x_from_x0(self):
        """Check that expect_x equals x0 (both derived from the coordinate shift)."""
        if self.expect_x is None or self.x0 is None:
            return (False, "SKIP: expect_x or x0 not available.")
        if not math.isclose(self.expect_x, self.x0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"expect_x ({self.expect_x}) != x0 ({self.x0})")
        return (True, f"expect_x = x0 = {self.x0}")

    def test_invariant_dipole_definition(self):
        """Check that P_n = -e * expect_x."""
        if self.P_n is None or self.expect_x is None or self.e is None:
            return (False, "SKIP: P_n, expect_x, or e missing.")
        expected_P = -self.e * self.expect_x
        if not math.isclose(self.P_n, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_n ({self.P_n}) != -e*expect_x ({expected_P})")
        return (True, f"P_n = -e*expect_x = {self.P_n}")

    def test_invariant_energy_shift_expression(self):
        """Check that Delta_E = - (eF)^2 / (2k)."""
        if self.Delta_E is None or self.e is None or self.F is None or self.k is None:
            return (False, "SKIP: required givens/Delta_E missing.")
        if self.k <= 0:
            return (False, "k must be positive.")
        expected = - (self.e * self.F)**2 / (2 * self.k)
        if not math.isclose(self.Delta_E, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Delta_E ({self.Delta_E}) != - (eF)^2/(2k) ({expected})")
        return (True, f"Delta_E = {self.Delta_E} matches formula.")

    def test_invariant_expect_x_field_relation(self):
        """Check that expect_x = eF/k (from shifted coordinate)."""
        if self.expect_x is None or self.e is None or self.F is None or self.k is None:
            return (False, "SKIP: required values missing.")
        if self.k <= 0:
            return (False, "k must be positive.")
        expected_x0 = self.e * self.F / self.k
        if not math.isclose(self.expect_x, expected_x0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"expect_x ({self.expect_x}) != eF/k ({expected_x0})")
        return (True, f"expect_x = eF/k = {self.expect_x}")

    def test_invariant_energy_level_structure(self):
        """If n is available, check that E_n = hbar*omega*(n+1/2) + Delta_E."""
        if self.E_n is None or self.Delta_E is None or self.hbar is None or self.omega is None or self.n is None:
            return (False, "SKIP: missing E_n, Delta_E, hbar, omega, or n.")
        expected_E = self.hbar * self.omega * (self.n + 0.5) + self.Delta_E
        if not math.isclose(self.E_n, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_n ({self.E_n}) != hbar*omega*(n+1/2)+Delta_E ({expected_E})")
        return (True, "E_n matches the known structure.")

    def test_invariant_delta_e_nonpositive(self):
        """Weak: Delta_E should be non-positive for nonzero field."""
        if self.Delta_E is None:
            return (False, "SKIP: missing.")
        if self.Delta_E > 0:
            return (False, f"Delta_E is positive ({self.Delta_E}), expected non-positive.")
        return (True, "Delta_E is non-positive.")

    # ------------------------------------------------------------------
    # Modeling / trace checks: verify that the solution derivation is consistent
    # ------------------------------------------------------------------
    def test_modeling_x0_definition(self):
        """Check that x0 is defined as eF/k (if appropriate givens exist)."""
        if self.x0 is None or self.e is None or self.F is None or self.k is None:
            return (False, "SKIP: missing data.")
        if self.k <= 0:
            return (False, "k must be positive.")
        expected_x0 = self.e * self.F / self.k
        if not math.isclose(self.x0, expected_x0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"x0 ({self.x0}) should be eF/k ({expected_x0})")
        return (True, "x0 is correctly defined.")

    def test_modeling_delta_e_negative(self):
        """Check that Delta_E is negative for nonzero field."""
        if self.Delta_E is None or self.F is None:
            return (False, "SKIP: missing.")
        if self.F == 0:
            return (True, "Field zero, shift zero - no check.")
        if self.Delta_E > 0:
            return (False, f"Delta_E should be negative, got {self.Delta_E}")
        return (True, "Delta_E is negative as expected.")

    def test_modeling_hamiltonian_shift_relation(self):
        """Check that Delta_E = -0.5*k*x0^2 (from completing square)."""
        if self.Delta_E is None or self.x0 is None or self.k is None:
            return (False, "SKIP: missing.")
        expected = -0.5 * self.k * (self.x0 ** 2)
        if not math.isclose(self.Delta_E, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Delta_E ({self.Delta_E}) != -0.5*k*x0^2 ({expected})")
        return (True, "Delta_E matches -0.5*k*x0^2.")

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_dipole_with_field(self):
        """
        If F increases, |P_n| increases (since P_n = -e^2 F/k).
        Use a 10% increase in F to check that magnitude grows.
        """
        if self.P_n is None or self.F is None or self.e is None or self.k is None:
            return (False, "SKIP: missing required data.")
        if self.F == 0:
            return (True, "Zero field: monotonicity not applicable.")
        baseline = self.P_n
        # perturb F by +10%
        F_prime = self.F * 1.10
        # expected new P_n from physics relation (re-derived from theorem)
        expected_new_P = - (self.e ** 2) * F_prime / self.k
        # For monotonicity, direction should match: if F increases in magnitude, |P| increases.
        # Test that the sign of change in |P| is positive when |F| increases.
        # Since linear, we check that (|expected_new_P| - |baseline|) > 0.
        if abs(expected_new_P) <= abs(baseline):
            return (False, f"Absolute dipole did not increase: |P|_new={abs(expected_new_P):.4f} <= |P|_old={abs(baseline):.4f} for 10% field increase.")
        return (True, f"Dipole magnitude increases with field: old |P|={abs(baseline):.4f}, new expected |P|={abs(expected_new_P):.4f}")

    def test_monotonicity_energy_shift_with_field(self):
        """
        If |F| increases, Delta_E becomes more negative (decreases).
        Perturb F by +10% and check that new Delta_E is more negative.
        """
        if self.Delta_E is None or self.F is None or self.e is None or self.k is None:
            return (False, "SKIP: missing required data.")
        if self.F == 0:
            return (True, "Zero field: monotonicity not applicable.")
        baseline = self.Delta_E
        # perturb F by +10%
        F_prime = self.F * 1.10
        # expected new Delta_E from physics relation
        expected_new_E = - (self.e * F_prime)**2 / (2 * self.k)
        # Since Delta_E is negative, more negative means lower value.
        if expected_new_E >= baseline:
            return (False, f"Energy shift did not decrease: new expected Delta_E={expected_new_E:.4f} >= old Delta_E={baseline:.4f}.")
        return (True, f"Energy shift decreases (becomes more negative): old={baseline:.4f}, new expected={expected_new_E:.4f}")

    def test_monotonicity_expect_x_with_field(self):
        """
        If F increases, expect_x (= eF/k) increases proportionally.
        Perturb F by +10% and check that new expect_x > old expect_x.
        """
        if self.expect_x is None or self.F is None or self.e is None or self.k is None:
            return (False, "SKIP: missing required data.")
        if self.F == 0:
            return (True, "Zero field: expect_x = 0, monotonicity not applicable.")
        baseline = self.expect_x
        F_prime = self.F * 1.10
        expected_new_x = self.e * F_prime / self.k
        # For positive e and F, expect_x positive; check increase.
        # For negative e or F, sign might flip. Use sign-independent comparison: compare magnitudes? Better: check that (new - old) has same sign as F.
        # Simpler: check that |new| > |old| if |F| increased.
        if abs(expected_new_x) <= abs(baseline):
            return (False, f"|expect_x| did not increase: new={abs(expected_new_x):.4f}, old={abs(baseline):.4f}")
        return (True, f"|expect_x| increases with field: old={abs(baseline):.4f}, new={abs(expected_new_x):.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_dipole_sign(self):
        """Check that sign(P_n) matches sign(F) when F != 0 (since P_n = -e^2 F/k, e>0)."""
        if self.P_n is None or self.F is None or self.e is None:
            return (False, "SKIP: missing.")
        if self.e <= 0:
            return (True, "e not positive; sign relation different.")
        if self.F == 0:
            return (True, "Field zero, dipole zero - no sign check.")
        # With e>0, P_n = -e^2 F/k, so P_n has opposite sign to F.
        if (self.P_n > 0 and self.F > 0) or (self.P_n < 0 and self.F < 0):
            return (False, f"P_n sign ({self.P_n}) should be opposite to F sign ({self.F}).")
        return (True, "Dipole sign correct (opposite to field).")

    # ------------------------------------------------------------------
    # Run all tests (discovery and execution)
    # ------------------------------------------------------------------
    def run_all_tests(self):
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)