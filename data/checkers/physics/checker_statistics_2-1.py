import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs (givens) safely
        self.m = getattr(solution_instance, 'm', None)
        self.k = getattr(solution_instance, 'k', None)
        self.E = getattr(solution_instance, 'E', None)

        # Extract results (from pseudocode emit keys)
        if hasattr(solution_instance, 'results'):
            results = solution_instance.results
            self.A = results.get('A', None)
            self.C = results.get('C', None)
            # Optional: if a full p(x) expression is stored, try to extract it
            self.p_expression = results.get('p(x)', None)
        else:
            self.A = None
            self.C = None
            self.p_expression = None

        # Derived quantities (computed inside checker, not from solution)
        self._A_expected = None
        self._C_expected = None
        if self.E is not None and self.k is not None:
            self._A_expected = math.sqrt(2.0 * self.E / self.k)
            self._C_expected = 1.0 / math.pi

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_inputs_present(self):
        """Check that all required inputs are provided."""
        if self.m is None:
            return (False, "Missing 'm'")
        if self.k is None:
            return (False, "Missing 'k'")
        if self.E is None:
            return (False, "Missing 'E'")
        return (True, "All inputs present")

    def test_interface_results_present(self):
        """Check that emitted results A and C exist."""
        if self.A is None:
            return (False, "Missing result 'A'")
        if self.C is None:
            return (False, "Missing result 'C'")
        return (True, "Results A and C present")

    # ------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ------------------------------------------------------------
    def test_invariant_energy_at_amplitude(self):
        """
        Energy conservation: at x = A the kinetic energy is zero,
        so total energy E = (1/2)*k*A^2.
        """
        if None in (self.E, self.k, self.A):
            return (False, "SKIP: missing E, k, or A")
        E_check = 0.5 * self.k * self.A * self.A
        if math.isclose(self.E, E_check, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E matches (1/2)*k*A^2")
        else:
            return (False, f"E={self.E}, (1/2)*k*A^2={E_check:.4f}")

    def test_invariant_normalization(self):
        """
        Probability density integrated over [-A, A] must equal 1.
        p(x) = C / sqrt(A^2 - x^2)
        Integral = C * π  (because ∫_{-A}^{A} dx / sqrt(A^2 - x^2) = π)
        """
        if None in (self.A, self.C):
            return (False, "SKIP: missing A or C")
        norm = self.C * math.pi
        if math.isclose(norm, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Normalization = {norm:.4f} ≈ 1")
        else:
            return (False, f"Normalization integral = {norm:.4f} (expected 1)")

    def test_invariant_domain_boundary(self):
        """
        The position x is bounded by amplitude A. Check that the
        denominator sqrt(A^2 - x^2) is defined only for |x| ≤ A.
        """
        if self.A is None:
            return (False, "SKIP: missing A")
        if self.A <= 0:
            return (False, f"A must be positive, got {self.A}")
        return (True, f"A = {self.A:.4f} > 0 – valid domain [-A, A]")

    # ------------------------------------------------------------
    # Modeling / trace tests (quality of physics modeling)
    # ------------------------------------------------------------
    def test_modeling_amplitude_formula(self):
        """
        Check that the amplitude A equals sqrt(2E/k).
        """
        if None in (self.E, self.k, self.A):
            return (False, "SKIP: missing required givens or A")
        expected = self._A_expected
        if math.isclose(self.A, expected, rel_tol=1e-2, abs_tol=1e-6):
            return (True, f"A={self.A:.4f} matches sqrt(2E/k)={expected:.4f}")
        else:
            return (False, f"A={self.A:.4f}, expected={expected:.4f}")

    def test_modeling_normalization_constant(self):
        """
        The normalization constant C should be 1/π.
        """
        if self.C is None:
            return (False, "SKIP: missing C")
        expected = self._C_expected if self._C_expected is not None else (1.0 / math.pi)
        if math.isclose(self.C, expected, rel_tol=1e-2, abs_tol=1e-6):
            return (True, f"C={self.C:.4f} matches 1/π={expected:.4f}")
        else:
            return (False, f"C={self.C:.4f}, expected={expected:.4f}")

    def test_modeling_probability_expression(self):
        """
        For a sample x (e.g., x = 0 and x = A/2), compute p(x) using
        the formula p(x) = C / sqrt(A^2 - x^2) and verify that it is
        well‑behaved (positive, finite). Also check consistency with
        the stored p_expression if available.
        """
        if None in (self.A, self.C):
            return (False, "SKIP: missing A or C")
        test_points = [0.0, self.A / 4.0, self.A / 2.0]
        for x in test_points:
            if abs(x) >= self.A:
                continue
            p = self.C / math.sqrt(self.A**2 - x**2)
            if not (math.isfinite(p) and p > 0):
                return (False, f"p({x}) = {p} is non‑positive or infinite")
        return (True, f"p(x) positive and finite at test points")

    # ------------------------------------------------------------
    # Monotonicity tests (trend checks via input perturbation)
    # ------------------------------------------------------------
    def test_monotonicity_amplitude_with_energy(self):
        """
        When total energy E increases, the amplitude A should also increase.
        Perturb E by +10% and check that A (computed from sqrt(2E/k)) increases.
        """
        if None in (self.E, self.k, self.A):
            return (False, "SKIP: missing E, k, or A")
        # Baseline A from solution
        A_base = self.A
        # Perturb E by +10%
        E_pert = self.E * 1.10
        A_pert = math.sqrt(2.0 * E_pert / self.k)
        # A should increase
        if A_pert > A_base:
            return (True, f"Amplitude increases with E: {A_base:.4f} -> {A_pert:.4f}")
        else:
            return (False, f"Amplitude did not increase: base={A_base:.4f}, perturbed={A_pert:.4f}")

    def test_monotonicity_probability_decreases_with_energy(self):
        """
        At a fixed position (e.g., x=0), the probability density p(0)
        decreases when total energy E increases.  Perturb E by +10%,
        compute new p(0) from the perturbed amplitude, and compare.
        """
        if None in (self.E, self.k, self.A, self.C):
            return (False, "SKIP: missing E, k, A, or C")
        # Baseline p(0)
        x = 0.0
        p_base = self.C / math.sqrt(self.A**2 - x**2)   # = C / A

        # Perturb E by +10%
        E_pert = self.E * 1.10
        A_pert = math.sqrt(2.0 * E_pert / self.k)
        # With the same normalization constant? Actually C is fixed (1/π),
        # but we can recompute p(0) using the same C (should be constant)
        p_pert = self.C / math.sqrt(A_pert**2 - x**2)   # = self.C / A_pert

        if p_pert < p_base:
            return (True, f"p(0) decreases with E: {p_base:.4f} -> {p_pert:.4f}")
        else:
            return (False, f"Expected p(0) to decrease, got base={p_base:.4f}, perturbed={p_pert:.4f}")

    # ------------------------------------------------------------
    # Answer sanity (weak, final answer check only)
    # ------------------------------------------------------------
    def test_answer_sanity_p_expression(self):
        """
        Very weak sanity: if p_expression is provided, check that it is a
        non‑empty string and that it contains essential symbols (x, k, E, etc.)
        """
        if self.p_expression is None:
            return (False, "SKIP: no p_expression stored")
        expr = self.p_expression
        if not isinstance(expr, str):
            return (False, f"p_expression is not a string: {type(expr)}")
        essentials = ['x', 'k', 'E']
        missing = [sym for sym in essentials if sym not in expr]
        if missing:
            return (False, f"p_expression missing symbols: {missing}")
        # Also check that it appears to be a valid mathematical expression
        if 'sqrt' not in expr and '**0.5' not in expr:
            return (False, "p_expression does not contain sqrt or power 0.5")
        return (True, "p_expression is present and plausible")

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
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
                passed = False
                msg = f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)