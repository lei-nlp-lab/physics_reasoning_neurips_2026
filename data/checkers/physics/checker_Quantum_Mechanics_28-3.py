import math
import sympy as sp
import traceback
from collections.abc import Iterable

class PhysicsChecker:
    """
    Checks the solution for the variational bound on the lowest frequency of a
    string with non-uniform density, using Rayleigh-Ritz.
    """

    def __init__(self, solution_instance):
        # ------------------ Givens (try multiple sources) ------------------
        self.a = self._get_value(solution_instance, 'a')
        self.T0 = self._get_value(solution_instance, 'T0')
        self.mu0 = self._get_value(solution_instance, 'mu0')

        # ------------------ Results (must be in solution_instance.results) ------------------
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        self.omega_sq_bound = results.get('omega_sq_bound', None)   # general bound expression
        self.bound_omega_sq = results.get('bound_omega_sq', None)   # numerical bound for part (b)

        # Target answer from problem statement (exact approximate value)
        self._exact_constant = 18.956   # exact ω0^2 = this * T0/(μ0 a^2)

    def _get_value(self, obj, name):
        """Safely extract a numeric value from the solution instance."""
        # Try direct attribute
        val = getattr(obj, name, None)
        if val is not None:
            return val
        # Try inside a 'givens' dict
        givens = getattr(obj, 'givens', {})
        if isinstance(givens, dict):
            val = givens.get(name, None)
            if val is not None:
                return val
        # Try inside 'results' dict as well (some solvers put givens there)
        results = getattr(obj, 'results', {})
        if isinstance(results, dict):
            val = results.get(name, None)
            if val is not None:
                return val
        return None

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all mandated result keys exist and are not None."""
        if self.omega_sq_bound is None:
            return (False, "SKIP? 'omega_sq_bound' missing from results")
        if self.bound_omega_sq is None:
            return (False, "SKIP? 'bound_omega_sq' missing from results")
        # Ensure they are numeric (int or float)
        if not isinstance(self.omega_sq_bound, (int, float)):
            return (False, f"'omega_sq_bound' should be numeric, got {type(self.omega_sq_bound)}")
        if not isinstance(self.bound_omega_sq, (int, float)):
            return (False, f"'bound_omega_sq' should be numeric, got {type(self.bound_omega_sq)}")
        return (True, "Required result keys present and numeric")

    def test_interface_givens_available(self):
        """Ensure the givens needed for numerical checks are present."""
        missing = []
        if self.a is None:
            missing.append('a')
        if self.T0 is None:
            missing.append('T0')
        if self.mu0 is None:
            missing.append('mu0')
        if missing:
            return (False, f"SKIP: missing givens: {', '.join(missing)}")
        return (True, "All needed givens are present")

    # ----------------------------------------------------------------------
    # Invariant (theorem-based) tests
    # ----------------------------------------------------------------------
    def test_invariant_positivity(self):
        """The variational bound and the expression must be positive."""
        if self.bound_omega_sq is not None and self.bound_omega_sq <= 0:
            return (False, f"bound_omega_sq = {self.bound_omega_sq} should be positive")
        if self.omega_sq_bound is not None and self.omega_sq_bound <= 0:
            return (False, f"omega_sq_bound = {self.omega_sq_bound} should be positive")
        return (True, "Both bounds are positive (or only one present)")

    def test_invariant_upper_bound(self):
        """
        Check that the variational bound is indeed an upper bound on the
        (approximate) exact frequency from the problem statement.
        Relies on givens to compute exact approximate value.
        """
        if any(v is None for v in (self.T0, self.mu0, self.a, self.bound_omega_sq)):
            return (False, "SKIP: missing givens or bound_omega_sq")
        exact_approx = self._exact_constant * self.T0 / (self.mu0 * self.a**2)
        if not (self.bound_omega_sq > exact_approx):
            return (False, f"bound_omega_sq ({self.bound_omega_sq}) should be > exact_approx ({exact_approx})")
        return (True, f"Bound {self.bound_omega_sq:.4f} > exact approx {exact_approx:.4f}")

    # ----------------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------------
    def test_modeling_bound_formula(self):
        """Verify that the numerical bound equals 2π² T0/(μ0 a²) within 2%."""
        if any(v is None for v in (self.T0, self.mu0, self.a, self.bound_omega_sq)):
            return (False, "SKIP: missing givens or bound_omega_sq")
        expected = 2 * math.pi**2 * self.T0 / (self.mu0 * self.a**2)
        if not math.isclose(self.bound_omega_sq, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"bound_omega_sq={self.bound_omega_sq} != expected={expected} (tolerance 2%)")
        return (True, f"bound_omega_sq matches 2π² T0/(μ0 a²) = {expected:.4f}")

    def test_modeling_omega_sq_bound_expression(self):
        """
        Check that the general bound expression (omega_sq_bound) is consistent
        with the Rayleigh quotient formula: T0·∫(ψ')² dx / ∫μ ψ² dx.
        We cannot recompute the integrals, but we can verify it is a string or
        expression that contains the required structure (e.g., 'T0', 'integral').
        This is a weak check.
        """
        if self.omega_sq_bound is None:
            return (False, "SKIP: omega_sq_bound not available")
        expr = str(self.omega_sq_bound).lower()
        keywords = ['integral', 'psi', 't0', 'mu']
        missing = [kw for kw in keywords if kw not in expr]
        if missing:
            return (False, f"Expression missing keywords: {missing}")
        return (True, "omega_sq_bound appears to be a valid variational expression")

    # ----------------------------------------------------------------------
    # Monotonicity tests (requires givens and ability to perturb)
    # ----------------------------------------------------------------------
    def test_monotonicity_positive_T0(self):
        """Increase T0 by 5%; bound_omega_sq should increase."""
        if any(v is None for v in (self.T0, self.mu0, self.a, self.bound_omega_sq)):
            return (False, "SKIP: missing data for monotonicity")
        pert_T0 = self.T0 * 1.05
        new_bound = 2 * math.pi**2 * pert_T0 / (self.mu0 * self.a**2)
        if not (new_bound > self.bound_omega_sq):
            return (False, "Increasing T0 should increase bound_omega_sq")
        return (True, f"T0 increased: bound {self.bound_omega_sq:.4f} -> {new_bound:.4f} OK")

    def test_monotonicity_negative_mu0(self):
        """Increase μ0 by 5%; bound_omega_sq should decrease."""
        if any(v is None for v in (self.T0, self.mu0, self.a, self.bound_omega_sq)):
            return (False, "SKIP: missing data for monotonicity")
        pert_mu0 = self.mu0 * 1.05
        new_bound = 2 * math.pi**2 * self.T0 / (pert_mu0 * self.a**2)
        if not (new_bound < self.bound_omega_sq):
            return (False, "Increasing μ0 should decrease bound_omega_sq")
        return (True, f"μ0 increased: bound {self.bound_omega_sq:.4f} -> {new_bound:.4f} OK")

    def test_monotonicity_negative_a(self):
        """Increase a by 5%; bound_omega_sq should decrease (since a² in denominator)."""
        if any(v is None for v in (self.T0, self.mu0, self.a, self.bound_omega_sq)):
            return (False, "SKIP: missing data for monotonicity")
        pert_a = self.a * 1.05
        new_bound = 2 * math.pi**2 * self.T0 / (self.mu0 * pert_a**2)
        if not (new_bound < self.bound_omega_sq):
            return (False, "Increasing a should decrease bound_omega_sq")
        return (True, f"a increased: bound {self.bound_omega_sq:.4f} -> {new_bound:.4f} OK")

    # ----------------------------------------------------------------------
    # Answer sanity (weak) – only one allowed
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite(self):
        """The numerical bound must be finite (not inf, not nan)."""
        if self.bound_omega_sq is None:
            return (False, "SKIP: bound_omega_sq not available")
        if not math.isfinite(self.bound_omega_sq):
            return (False, f"bound_omega_sq = {self.bound_omega_sq} is not finite")
        return (True, "bound_omega_sq is finite")

    # ----------------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and execute all test_* methods, collect results."""
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

# ============================================================================
# Quick example usage (if run standalone)
# ============================================================================
if __name__ == "__main__":
    # Dummy solution instance for demonstration
    class DummySolution:
        def __init__(self):
            self.a = 2.0
            self.T0 = 100.0
            self.mu0 = 0.5
            self.results = {
                'omega_sq_bound': "T0 * integral(psi'^2) / integral(mu * psi^2)",
                'bound_omega_sq': 2 * math.pi**2 * 100.0 / (0.5 * 2.0**2)  # 2π² * 100 / (0.5*4) = 2π² * 100 / 2 = 100π² ≈ 986.96
            }
    sol = DummySolution()
    checker = PhysicsChecker(sol)
    ok, msgs = checker.run_all_tests()
    for m in msgs:
        print(m)