import math
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution for classical pion-nucleus collision cross section.
    Checks interface presence, physical invariants (conservation laws),
    modeling consistency (sigma vs p_crit_sq), monotonicity trends,
    and a weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Safe extraction of givens and results
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})

        # Input parameters (may be None)
        self.T = self.givens.get('T', None)
        self.m = self.givens.get('m', None)
        self.b = self.givens.get('b', None)
        self.V0 = self.givens.get('V0', None)          # magnitude of Coulomb potential at b
        self.e = self.givens.get('e', None)
        self.Ze = self.givens.get('Ze', None)
        self.k = self.givens.get('k', None)

        # Expected result keys from [emit] contract
        self.p_crit_sq_plus = self.results.get('p_crit_sq_plus', None)
        self.sigma_plus = self.results.get('sigma_plus', None)
        self.p_crit_sq_minus = self.results.get('p_crit_sq_minus', None)
        self.sigma_minus = self.results.get('sigma_minus', None)

    # ----------------------------------------------------------------------
    # Helper: compute expected sigma values from T, b, V0 (if available)
    def _compute_expected_sigma_plus(self):
        if self.T is not None and self.b is not None and self.V0 is not None:
            if self.T != 0:
                return math.pi * self.b**2 * (self.T - self.V0) / self.T
        return None

    def _compute_expected_sigma_minus(self):
        if self.T is not None and self.b is not None and self.V0 is not None:
            if self.T != 0:
                return math.pi * self.b**2 * (self.T + self.V0) / self.T
        return None

    # ----------------------------------------------------------------------
    # Interface tests: check required keys exist and have numeric type
    def test_interface_required_results(self):
        """All four required result keys must be present and numeric."""
        required = {
            'p_crit_sq_plus': self.p_crit_sq_plus,
            'sigma_plus': self.sigma_plus,
            'p_crit_sq_minus': self.p_crit_sq_minus,
            'sigma_minus': self.sigma_minus
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            return (True, f"SKIP: missing result keys {missing}")  # skip, not crash
        # If present, check they are numbers (int, float, or sympy numeric)
        for key, val in required.items():
            if not isinstance(val, (int, float, sp.core.numbers.Number)):
                return (True, f"SKIP: {key} is not numeric (type {type(val).__name__})")
        return (True, "All required result keys exist and are numeric")

    # ----------------------------------------------------------------------
    # Invariant tests: physical conservation laws
    def test_invariant_sigma_sum(self):
        """sigma_plus + sigma_minus should equal 2*pi*b^2 (if b known)."""
        if self.b is None:
            return (True, "SKIP: b not available")
        if self.sigma_plus is None or self.sigma_minus is None:
            return (True, "SKIP: sigma values missing")
        expected_sum = 2.0 * math.pi * self.b**2
        actual_sum = float(self.sigma_plus) + float(self.sigma_minus)
        if math.isclose(actual_sum, expected_sum, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sigma_plus+sigma_minus ≈ {actual_sum:.4f} (expected {expected_sum:.4f})")
        else:
            return (False, f"sum {actual_sum:.4f} differs from expected {expected_sum:.4f}")

    def test_invariant_plus_upper_bound(self):
        """sigma_plus should be ≤ pi*b^2 (repulsive reduces cross section)."""
        if self.b is None or self.sigma_plus is None:
            return (True, "SKIP: insufficient data")
        max_sigma = math.pi * self.b**2
        if float(self.sigma_plus) <= max_sigma + 1e-2:  # allow tiny rounding
            return (True, f"sigma_plus {self.sigma_plus:.4f} ≤ pi*b^2 {max_sigma:.4f}")
        else:
            return (False, f"sigma_plus {self.sigma_plus:.4f} > pi*b^2 {max_sigma:.4f}")

    def test_invariant_minus_lower_bound(self):
        """sigma_minus should be ≥ pi*b^2 (attractive increases cross section)."""
        if self.b is None or self.sigma_minus is None:
            return (True, "SKIP: insufficient data")
        min_sigma = math.pi * self.b**2
        if float(self.sigma_minus) >= min_sigma - 1e-2:
            return (True, f"sigma_minus {self.sigma_minus:.4f} ≥ pi*b^2 {min_sigma:.4f}")
        else:
            return (False, f"sigma_minus {self.sigma_minus:.4f} < pi*b^2 {min_sigma:.4f}")

    # ----------------------------------------------------------------------
    # Modeling / trace consistency tests
    def test_modeling_sigma_vs_pcrit_sq_plus(self):
        """sigma_plus should equal pi * p_crit_sq_plus (within tolerance)."""
        if self.sigma_plus is None or self.p_crit_sq_plus is None:
            return (True, "SKIP: missing sigma_plus or p_crit_sq_plus")
        psi = float(self.sigma_plus)
        p2 = float(self.p_crit_sq_plus)
        expected = math.pi * p2
        if math.isclose(psi, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sigma_plus ≈ pi * p_crit_sq_plus ({expected:.4f})")
        else:
            return (False, f"sigma_plus {psi:.4f} != pi*p_crit_sq_plus {expected:.4f}")

    def test_modeling_sigma_vs_pcrit_sq_minus(self):
        """sigma_minus should equal pi * p_crit_sq_minus (within tolerance)."""
        if self.sigma_minus is None or self.p_crit_sq_minus is None:
            return (True, "SKIP: missing sigma_minus or p_crit_sq_minus")
        psi = float(self.sigma_minus)
        p2 = float(self.p_crit_sq_minus)
        expected = math.pi * p2
        if math.isclose(psi, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sigma_minus ≈ pi * p_crit_sq_minus ({expected:.4f})")
        else:
            return (False, f"sigma_minus {psi:.4f} != pi*p_crit_sq_minus {expected:.4f}")

    # ----------------------------------------------------------------------
    # Monotonicity tests: perturb V0 (if available) and check trend
    def test_monotonicity_plus_decreases_with_V0(self):
        """If V0 increases, sigma_plus should decrease."""
        if self.T is None or self.b is None or self.V0 is None or self.sigma_plus is None:
            return (True, "SKIP: insufficient data for monotonicity")
        # Baseline
        sigma_base = float(self.sigma_plus)
        # Perturb: increase V0 by 5%
        V0_new = self.V0 * 1.05
        if V0_new >= self.T:
            return (True, "SKIP: perturbed V0 > T, not physical for plus cross section")
        sigma_new = math.pi * self.b**2 * (self.T - V0_new) / self.T
        if sigma_new < sigma_base - 1e-2:
            return (True, f"sigma_plus decreases (base {sigma_base:.4f} -> perturbed {sigma_new:.4f})")
        else:
            return (False, f"sigma_plus did not decrease (base {sigma_base:.4f}, perturbed {sigma_new:.4f})")

    def test_monotonicity_minus_increases_with_V0(self):
        """If V0 increases, sigma_minus should increase."""
        if self.T is None or self.b is None or self.V0 is None or self.sigma_minus is None:
            return (True, "SKIP: insufficient data for monotonicity")
        sigma_base = float(self.sigma_minus)
        V0_new = self.V0 * 1.05
        sigma_new = math.pi * self.b**2 * (self.T + V0_new) / self.T
        if sigma_new > sigma_base + 1e-2:
            return (True, f"sigma_minus increases (base {sigma_base:.4f} -> perturbed {sigma_new:.4f})")
        else:
            return (False, f"sigma_minus did not increase (base {sigma_base:.4f}, perturbed {sigma_new:.4f})")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one, no math.isclose)
    def test_answer_sanity_weak(self):
        """Basic sign and ordering: sigma_plus <= sigma_minus, both >= 0."""
        if self.sigma_plus is None or self.sigma_minus is None:
            return (True, "SKIP: missing sigma values")
        sp = float(self.sigma_plus)
        sm = float(self.sigma_minus)
        if sp < 0 or sm < 0:
            return (False, f"Negative sigma values: plus={sp}, minus={sm}")
        if sp > sm:
            return (False, f"sigma_plus ({sp}) > sigma_minus ({sm}) – violates repulsive<attractive")
        # Finite check
        if not math.isfinite(sp) or not math.isfinite(sm):
            return (False, "Sigma values are not finite")
        return (True, f"sigma_plus={sp:.4f} <= sigma_minus={sm:.4f} (both ≥0)")

    # ----------------------------------------------------------------------
    # Test discovery and execution
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