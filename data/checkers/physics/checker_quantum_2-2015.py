import math
import sympy as sp

class PhysicsChecker:
    """
    Audits the solution for heavy quarkonium bound states in a linear potential.
    Tests interface completeness, physical invariants, modeling consistency,
    monotonicity, and a weak sanity check on final answer magnitude.
    """

    def __init__(self, solution_instance):
        # Extract results dict safely
        results = getattr(solution_instance, 'results', {}) or {}

        self.delta_E_b = results.get('delta_E_b', None)
        self.E_Upsilon_prime = results.get('E_Upsilon_prime', None)
        self.R = results.get('R', None)
        self.delta_E_psi2 = results.get('delta_E_psi2', None)
        self.E_psi_double_prime = results.get('E_psi_double_prime', None)

        # Givens (from problem statement)
        self.m_c = 1.5       # GeV/c^2
        self.m_b = 4.5       # GeV/c^2
        self.E_psi = 3.1     # GeV
        self.E_psi_prime = 3.7
        self.E_Upsilon = 9.5
        self.delta_E_c = self.E_psi_prime - self.E_psi   # 0.6 GeV

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_delta_E_b(self):
        """Check that delta_E_b exists and is not None."""
        if self.delta_E_b is None:
            return (False, "Missing result key: delta_E_b")
        return (True, "delta_E_b is present")

    def test_interface_E_Upsilon_prime(self):
        """Check that E_Upsilon_prime exists."""
        if self.E_Upsilon_prime is None:
            return (False, "Missing result key: E_Upsilon_prime")
        return (True, "E_Upsilon_prime is present")

    def test_interface_R(self):
        """Check that R exists."""
        if self.R is None:
            return (False, "Missing result key: R")
        return (True, "R is present")

    def test_interface_delta_E_psi2(self):
        """Check that delta_E_psi2 exists."""
        if self.delta_E_psi2 is None:
            return (False, "Missing result key: delta_E_psi2")
        return (True, "delta_E_psi2 is present")

    def test_interface_E_psi_double_prime(self):
        """Check that E_psi_double_prime exists."""
        if self.E_psi_double_prime is None:
            return (False, "Missing result key: E_psi_double_prime")
        return (True, "E_psi_double_prime is present")

    # ------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ------------------------------------------------------------
    def test_invariant_delta_E_scaling(self):
        """
        Check that delta_E_b / delta_E_c = (m_c / m_b)^(1/3).
        This follows from dimensional analysis of the linear potential.
        """
        if None in (self.delta_E_b, self.delta_E_c, self.m_c, self.m_b):
            return (False, "SKIP: missing delta_E_b or givens")
        expected_ratio = (self.m_c / self.m_b) ** (1/3)
        actual_ratio = self.delta_E_b / self.delta_E_c
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Scaling holds: ratio {actual_ratio:.4f} ≈ {expected_ratio:.4f}")
        else:
            return (False, f"Scaling fails: ratio {actual_ratio:.4f} != expected {expected_ratio:.4f} (tolerance ±1%)")

    def test_invariant_WKB_splitting_ratio(self):
        """
        Check that R equals the WKB-predicted ratio:
        ( (2.5)^(2/3) - (1.5)^(2/3) ) / ( (1.5)^(2/3) - (0.5)^(2/3) )
        """
        if self.R is None:
            return (False, "SKIP: missing R")
        def epsilon(n):
            return (n + 0.5) ** (2/3)
        expected_R = (epsilon(2) - epsilon(1)) / (epsilon(1) - epsilon(0))
        if math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"WKB ratio R = {self.R:.4f} ≈ {expected_R:.4f}")
        else:
            return (False, f"WKB ratio R = {self.R:.4f} != expected {expected_R:.4f} (tolerance ±1%)")

    # ------------------------------------------------------------
    # Modeling consistency tests (based on explanation steps)
    # ------------------------------------------------------------
    def test_modeling_Upsilon_prime_energy(self):
        """Check that E_Upsilon_prime = E_Upsilon + delta_E_b."""
        if None in (self.E_Upsilon_prime, self.delta_E_b, self.E_Upsilon):
            return (False, "SKIP: missing required values")
        expected = self.E_Upsilon + self.delta_E_b
        if math.isclose(self.E_Upsilon_prime, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_Upsilon' = {self.E_Upsilon_prime:.3f} GeV ≈ {expected:.3f} GeV")
        else:
            return (False, f"E_Upsilon' = {self.E_Upsilon_prime:.3f} GeV != {expected:.3f} GeV (tolerance ±1%)")

    def test_modeling_psi_double_prime_energy(self):
        """Check that E_psi'' = E_psi' + delta_E_psi2."""
        if None in (self.E_psi_double_prime, self.delta_E_psi2, self.E_psi_prime):
            return (False, "SKIP: missing required values")
        expected = self.E_psi_prime + self.delta_E_psi2
        if math.isclose(self.E_psi_double_prime, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_psi'' = {self.E_psi_double_prime:.3f} GeV ≈ {expected:.3f} GeV")
        else:
            return (False, f"E_psi'' = {self.E_psi_double_prime:.3f} GeV != {expected:.3f} GeV (tolerance ±1%)")

    def test_modeling_delta_E_psi2_from_R(self):
        """Check that delta_E_psi2 = R * delta_E_c."""
        if None in (self.delta_E_psi2, self.R, self.delta_E_c):
            return (False, "SKIP: missing required values")
        expected = self.R * self.delta_E_c
        if math.isclose(self.delta_E_psi2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"delta_E_psi2 = {self.delta_E_psi2:.4f} GeV ≈ {expected:.4f} GeV")
        else:
            return (False, f"delta_E_psi2 = {self.delta_E_psi2:.4f} GeV != {expected:.4f} GeV (tolerance ±1%)")

    # ------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------
    def test_monotonicity_splitting_decreases_with_mass(self):
        """
        If the bottom mass were larger, the Upsilon splitting should be smaller.
        Perturb m_b upward by +10% and check that the predicted delta_E_b decreases.
        (Uses the derived scaling delta_E_b ∝ m_b^{-1/3})
        """
        if self.delta_E_b is None:
            return (False, "SKIP: missing delta_E_b")
        # baseline scaling: delta_E_b = delta_E_c * (m_c / m_b)^(1/3)
        # Perturb m_b -> m_b_pert = m_b * 1.1
        m_b_pert = self.m_b * 1.1
        delta_E_b_pert = self.delta_E_c * (self.m_c / m_b_pert) ** (1/3)
        if delta_E_b_pert < self.delta_E_b:
            return (True, f"Monotonicity holds: m_b increased by 10% → delta_E_b decreases from {self.delta_E_b:.4f} to {delta_E_b_pert:.4f}")
        else:
            return (False, f"Monotonicity fails: m_b increased but delta_E_b did not decrease (baseline {self.delta_E_b:.4f}, perturbed {delta_E_b_pert:.4f})")

    # ------------------------------------------------------------
    # Weak answer sanity (single method, no derivation, no math.isclose)
    # ------------------------------------------------------------
    def test_answer_sanity(self):
        """
        Minimal positivity and plausible range check.
        No math.isclose, no re-derivation.
        """
        outputs = [
            self.delta_E_b, self.E_Upsilon_prime,
            self.R, self.delta_E_psi2, self.E_psi_double_prime
        ]
        for val in outputs:
            if val is None:
                return (False, "SKIP: some output is None")
            if not (math.isfinite(val) and val > 0):
                return (False, f"Non-positive or infinite value encountered: {val}")
        # Very loose plausible ranges (not derived, just safe bounds)
        checks = [
            (0.3 < self.delta_E_b < 0.6, f"delta_E_b = {self.delta_E_b} GeV"),
            (9.5 < self.E_Upsilon_prime < 10.5, f"E_Upsilon' = {self.E_Upsilon_prime} GeV"),
            (0.5 < self.R < 1.0, f"R = {self.R}"),
            (0.3 < self.delta_E_psi2 < 0.8, f"delta_E_psi2 = {self.delta_E_psi2} GeV"),
            (4.0 < self.E_psi_double_prime < 4.5, f"E_psi'' = {self.E_psi_double_prime} GeV")
        ]
        for ok, desc in checks:
            if not ok:
                return (False, f"Out of bounds: {desc}")
        return (True, "All outputs are positive, finite, and within plausible ranges")

    # ------------------------------------------------------------
    # Master runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)