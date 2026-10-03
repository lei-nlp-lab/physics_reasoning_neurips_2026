import math
import traceback
from typing import Any

class PhysicsChecker:
    """
    Robust physics checker for a charge near a grounded conducting plane.
    """

    def __init__(self, solution_instance: Any):
        # Extract givens and results safely (use None, never numeric defaults)
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.results = getattr(solution_instance, 'results', None) or {}

        # Physical parameters (None if missing)
        self.q = self.givens.get('q', None)
        self.a = self.givens.get('a', None)
        self.epsilon0 = self.givens.get('epsilon0', None)   # no numeric fallback

        # Extracted result values
        self.Q_ind = self.results.get('Q_ind', None)
        self.F_mag = self.results.get('F_mag', None)
        self.W = self.results.get('W', None)

    # ----------------------------------------------------------------------
    # Helper: compute epsilon0 only if needed (never stored as default)
    # ----------------------------------------------------------------------
    def _get_epsilon0(self):
        """Return epsilon0 from givens or use standard SI value."""
        if self.epsilon0 is not None:
            return self.epsilon0
        # Standard vacuum permittivity (SI) – only used for re-derivation
        return 8.854187817e-12

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> tuple:
        """Check that all required result keys are present."""
        required = ['Q_ind', 'F_mag', 'W']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_givens_present(self) -> tuple:
        """Check that required givens (q, a) are provided."""
        if self.q is None or self.a is None:
            return (False, "Missing givens: q or a")
        return (True, "Givens q and a available")

    def test_interface_Q_ind_numeric(self) -> tuple:
        if self.Q_ind is None:
            return (False, "SKIP: Q_ind not available")
        if not isinstance(self.Q_ind, (int, float)):
            return (False, f"Q_ind type {type(self.Q_ind)} not numeric")
        return (True, "Q_ind is numeric")

    def test_interface_F_mag_numeric(self) -> tuple:
        if self.F_mag is None:
            return (False, "SKIP: F_mag not available")
        if not isinstance(self.F_mag, (int, float)):
            return (False, f"F_mag type {type(self.F_mag)} not numeric")
        return (True, "F_mag is numeric")

    def test_interface_W_numeric(self) -> tuple:
        if self.W is None:
            return (False, "SKIP: W not available")
        if not isinstance(self.W, (int, float)):
            return (False, f"W type {type(self.W)} not numeric")
        return (True, "W is numeric")

    # ----------------------------------------------------------------------
    # Invariant checks: theorem-based physical laws
    # ----------------------------------------------------------------------
    def test_invariant_induced_charge_sign(self) -> tuple:
        """
        Induced charge on a grounded conductor must be opposite in sign to the
        real charge. Also checks magnitude.
        """
        if self.q is None or self.Q_ind is None:
            return (False, "SKIP: missing q or Q_ind")
        # Sign check (opposite)
        if (self.q > 0 and self.Q_ind > 0) or (self.q < 0 and self.Q_ind < 0):
            return (False, "Induced charge should have opposite sign to q")
        # Magnitude check: |Q_ind| should be |q|
        if not math.isclose(abs(self.Q_ind), abs(self.q), rel_tol=1e-2):
            return (False, f"Magnitude of induced charge {abs(self.Q_ind)} not close to |q|={abs(self.q)}")
        return (True, "Induced charge sign and magnitude consistent with image method")

    def test_invariant_force_equation(self) -> tuple:
        """F_mag should equal q^2/(16*pi*epsilon0*a^2)."""
        if None in (self.q, self.a, self.F_mag):
            return (False, "SKIP: missing required quantities for force")
        eps0 = self._get_epsilon0()
        expected = (self.q**2) / (16 * math.pi * eps0 * self.a**2)
        if not math.isclose(self.F_mag, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"F_mag {self.F_mag} not close to expected {expected}")
        return (True, "Force magnitude matches Coulomb law with image")

    def test_invariant_work_equation(self) -> tuple:
        """W should equal q^2/(16*pi*epsilon0*a)."""
        if None in (self.q, self.a, self.W):
            return (False, "SKIP: missing required quantities for work")
        eps0 = self._get_epsilon0()
        expected = (self.q**2) / (16 * math.pi * eps0 * self.a)
        if not math.isclose(self.W, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"W {self.W} not close to expected {expected}")
        return (True, "Work matches potential energy difference")

    def test_invariant_force_attractive(self) -> tuple:
        """Force magnitude is positive (attractive)."""
        if self.F_mag is None:
            return (False, "SKIP: F_mag missing")
        if self.F_mag < 0:
            return (False, "Force magnitude should be non-negative")
        return (True, "Force magnitude is non-negative (attractive)")

    def test_invariant_work_positive(self) -> tuple:
        """Work done by external agent is positive."""
        if self.W is None:
            return (False, "SKIP: W missing")
        if self.W <= 0:
            return (False, "Work should be positive (moving charge away from plane)")
        return (True, "Work is positive")

    def test_invariant_Q_ind_finite(self) -> tuple:
        if self.Q_ind is None:
            return (False, "SKIP: Q_ind missing")
        if math.isnan(self.Q_ind) or math.isinf(self.Q_ind):
            return (False, "Induced charge is NaN or Inf")
        return (True, "Induced charge is finite")

    # ----------------------------------------------------------------------
    # Modeling / trace quality checks (at least two)
    # ----------------------------------------------------------------------
    def test_modeling_induced_charge_magnitude(self) -> tuple:
        """Modeling check: induced charge magnitude should equal |q| (method of images)."""
        if self.q is None or self.Q_ind is None:
            return (False, "SKIP: missing q or Q_ind")
        if not math.isclose(abs(self.Q_ind), abs(self.q), rel_tol=1e-2):
            return (False, f"|Q_ind| = {abs(self.Q_ind)} not close to |q| = {abs(self.q)}")
        return (True, "Induced charge magnitude matches |q| (image method)")

    def test_modeling_force_work_consistency(self) -> tuple:
        """
        Modeling cross‑check: The work to move the charge from a to infinity
        should equal F_mag * a, because W = ∫ F dr = q²/(16πε₀a) and
        F(a) = q²/(16πε₀a²). If both are computed correctly, W ≈ F_mag * a.
        """
        if None in (self.F_mag, self.a, self.W):
            return (False, "SKIP: missing F_mag, a, or W")
        expected_W = self.F_mag * self.a
        if not math.isclose(self.W, expected_W, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"W = {self.W} is not close to F_mag*a = {expected_W}")
        return (True, "Work is consistent with force times distance (integral check)")

    # ----------------------------------------------------------------------
    # Monotonicity checks: verify derivative trends by perturbation
    # ----------------------------------------------------------------------
    def test_monotonicity_force_decreases_with_a(self) -> tuple:
        """If a increases, force magnitude should decrease."""
        if self.q is None or self.a is None or self.F_mag is None:
            return (False, "SKIP: missing parameters")
        eps0 = self._get_epsilon0()
        # Perturb a upward by 10%
        a_new = self.a * 1.1
        F_expected = (self.q**2) / (16 * math.pi * eps0 * a_new**2)
        # Expected perturbed force should be less than original (allow small tolerance)
        if F_expected >= self.F_mag * 1.02:
            return (False, f"Force should decrease when a increases: expected {F_expected} < original {self.F_mag}")
        return (True, "Force decreases when a increases")

    def test_monotonicity_work_decreases_with_a(self) -> tuple:
        """If a increases, work should decrease."""
        if self.q is None or self.a is None or self.W is None:
            return (False, "SKIP: missing parameters")
        eps0 = self._get_epsilon0()
        a_new = self.a * 1.1
        W_expected = (self.q**2) / (16 * math.pi * eps0 * a_new)
        if W_expected >= self.W * 1.02:
            return (False, f"Work should decrease when a increases: expected {W_expected} < original {self.W}")
        return (True, "Work decreases when a increases")

    def test_monotonicity_force_increases_with_q(self) -> tuple:
        """If |q| increases, force magnitude should increase."""
        if self.q is None or self.a is None or self.F_mag is None:
            return (False, "SKIP: missing parameters")
        eps0 = self._get_epsilon0()
        q_new = self.q * 1.1
        F_expected = (q_new**2) / (16 * math.pi * eps0 * self.a**2)
        # Expected perturbed force should be greater than original
        if F_expected <= self.F_mag * 0.98:
            return (False, f"Force should increase when |q| increases: expected {F_expected} > original {self.F_mag}")
        return (True, "Force increases when |q| increases")

    def test_monotonicity_work_increases_with_q(self) -> tuple:
        """If |q| increases, work should increase."""
        if self.q is None or self.a is None or self.W is None:
            return (False, "SKIP: missing parameters")
        eps0 = self._get_epsilon0()
        q_new = self.q * 1.1
        W_expected = (q_new**2) / (16 * math.pi * eps0 * self.a)
        if W_expected <= self.W * 0.98:
            return (False, f"Work should increase when |q| increases: expected {W_expected} > original {self.W}")
        return (True, "Work increases when |q| increases")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one method, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_reasonable_values(self) -> tuple:
        """
        Weak checks: results should be finite, non-negative, and within
        physically plausible orders of magnitude.
        """
        issues = []
        if self.Q_ind is not None:
            if math.isnan(self.Q_ind) or math.isinf(self.Q_ind):
                issues.append("Q_ind is NaN or Inf")
            elif abs(self.Q_ind) > 1e-3:
                issues.append("Q_ind magnitude too large (>1e-3 C)")
        if self.F_mag is not None:
            if math.isnan(self.F_mag) or math.isinf(self.F_mag):
                issues.append("F_mag is NaN or Inf")
            elif self.F_mag < 0:
                issues.append("F_mag negative")
            elif self.F_mag > 50:
                issues.append("F_mag unexpectedly large (>50 N)")
        if self.W is not None:
            if math.isnan(self.W) or math.isinf(self.W):
                issues.append("W is NaN or Inf")
            elif self.W < 0:
                issues.append("W negative")
            elif self.W > 10:
                issues.append("W unexpectedly large (>10 J)")
        if issues:
            return (False, "Sanity issues: " + "; ".join(issues))
        return (True, "All values appear physically plausible")

    # ----------------------------------------------------------------------
    # Main test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> tuple:
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