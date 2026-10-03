import math
import traceback

class PhysicsChecker:
    """
    Robust checker for the bound state probability half-width problem.
    """
    def __init__(self, solution_instance):
        # Safely extract givens and results
        givens = getattr(solution_instance, 'givens', {}) or {}
        results = getattr(solution_instance, 'results', {}) or {}

        # Givens: particle mass m, delta strength a, reduced Planck constant hbar
        self.m = givens.get('m', None)
        self.a = givens.get('a', None)
        self.hbar = givens.get('hbar', None)

        # Result: half-width x0 for probability = 1/2
        self.x0 = results.get('x0', None)

        # Store explanation steps for reference (not used in tests here)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []

    def _has_all(self, *args):
        """Check that all given values are not None."""
        return all(v is not None for v in args)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required givens and the result key exist."""
        if not self._has_all(self.m, self.a, self.hbar):
            return (False, "Missing required givens: m, a, hbar")
        if self.x0 is None:
            return (False, "Missing required result: x0")
        return (True, "All required keys present")

    def test_interface_types(self):
        """Verify that all extracted values are numeric and positive."""
        for name, val in [('m', self.m), ('a', self.a), ('hbar', self.hbar), ('x0', self.x0)]:
            if not self._has_all(val):
                return (False, f"Missing value for {name}")
            if not isinstance(val, (int, float)):
                return (False, f"{name} not numeric, got {type(val)}")
            if val <= 0:
                return (False, f"{name} must be positive, got {val}")
        return (True, "All types and positivity OK")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ----------------------------------------------------------------------
    def test_invariant_kappa_positive(self):
        """Check that the decay constant kappa = m*a/hbar^2 is positive."""
        if not self._has_all(self.m, self.a, self.hbar):
            return (False, "SKIP: missing m, a, hbar")
        kappa = self.m * self.a / (self.hbar ** 2)
        if kappa > 0:
            return (True, f"kappa = {kappa} > 0 (consistent with bound state)")
        else:
            return (False, f"kappa = {kappa} should be positive")

    def test_invariant_self_consistency(self):
        """
        Verify that the decay constant from the jump condition (ma/hbar^2)
        matches the decay constant implied by the probability condition
        (ln2/(2*x0)). This checks the fundamental physics consistency.
        """
        if not self._has_all(self.m, self.a, self.hbar, self.x0):
            return (False, "SKIP: missing required values")
        try:
            kappa_jump = self.m * self.a / (self.hbar ** 2)
            if self.x0 <= 0:
                return (False, "x0 must be positive to infer kappa")
            kappa_prob = math.log(2) / (2.0 * self.x0)
            if math.isclose(kappa_jump, kappa_prob, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Jump kappa = {kappa_jump:.5f} ≈ prob kappa = {kappa_prob:.5f}")
            else:
                return (False, f"Discrepancy: jump kappa = {kappa_jump:.5f}, prob kappa = {kappa_prob:.5f}")
        except Exception:
            return (False, "Error computing self-consistency test")

    # ----------------------------------------------------------------------
    # Modeling and trace tests
    # ----------------------------------------------------------------------
    def test_modeling_probability_relation(self):
        """
        Verify the fundamental relation: kappa * x0 = ln(2)/2,
        derived from the probability condition 1 - e^{-2 kappa x0} = 1/2.
        """
        if not self._has_all(self.m, self.a, self.hbar, self.x0):
            return (False, "SKIP: missing required values")
        try:
            kappa = self.m * self.a / (self.hbar ** 2)
            lhs = kappa * self.x0
            rhs = math.log(2) / 2.0   # ≈ 0.3466
            if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"κ x0 = {lhs:.5f} ≈ {rhs:.5f} (ln2/2)")
            else:
                return (False, f"κ x0 = {lhs:.5f} is not close to {rhs:.5f}")
        except Exception:
            return (False, "Error computing probability relation")

    def test_modeling_normalization_consistency(self):
        """
        Check that the normalized wavefunction with A = sqrt(kappa) leads
        to the correct probability P(|x|<x0) = 1 - exp(-2κ x0) = 0.5.
        This traces the modeling step that combines normalization and
        the probability integral.
        """
        if not self._has_all(self.m, self.a, self.hbar, self.x0):
            return (False, "SKIP: missing required values")
        try:
            kappa = self.m * self.a / (self.hbar ** 2)
            # Normalization constant for the bound state wavefunction
            A = math.sqrt(kappa)   # |A| = sqrt(kappa)
            # Probability from -x0 to x0: 2 * A^2 * integral_0^x0 e^{-2κx} dx
            prob = 1.0 - math.exp(-2.0 * kappa * self.x0)
            if math.isclose(prob, 0.5, rel_tol=1e-2):
                return (True, f"Probability = {prob:.5f} ≈ 0.5")
            else:
                return (False, f"Probability = {prob:.5f} is not close to 0.5")
        except Exception:
            return (False, "Error computing normalization consistency")

    # ----------------------------------------------------------------------
    # Monotonicity test
    # ----------------------------------------------------------------------
    def test_monotonicity_x0_decreases_with_a(self):
        """
        Perturb the potential strength a by +10% and verify that the
        theoretical x0 decreases (since x0 = ħ^2/(2 m a) ln2).
        """
        if not self._has_all(self.m, self.a, self.hbar, self.x0):
            return (False, "SKIP: missing required values")
        try:
            a_pert = self.a * 1.1
            kappa_pert = self.m * a_pert / (self.hbar ** 2)
            x0_pert_expected = math.log(2) / (2 * kappa_pert)
            if x0_pert_expected < self.x0:
                return (True, f"Increased a -> smaller x0: {x0_pert_expected:.6f} < {self.x0:.6f}")
            else:
                return (False, f"Expected decrease, but {x0_pert_expected:.6f} >= {self.x0:.6f}")
        except Exception:
            return (False, "Error computing monotonicity test")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_x0_positive_finite(self):
        """Minimal sanity: x0 must be positive and finite."""
        if self.x0 is None:
            return (False, "SKIP: x0 missing")
        if not math.isfinite(self.x0):
            return (False, f"x0 is not finite: {self.x0}")
        if self.x0 <= 0:
            return (False, f"x0 should be positive, got {self.x0}")
        return (True, "x0 is positive and finite")

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)