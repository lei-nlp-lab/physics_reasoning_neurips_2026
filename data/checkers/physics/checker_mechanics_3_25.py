import math
import sys
import traceback

class PhysicsChecker:
    """
    PhysicsAuditor for diffractive dissociation threshold energy problem.
    Checks conservation of invariant mass, threshold condition, and modeling consistency.
    """
    def __init__(self, solution_instance):
        # Safely extract givens
        self.m_p = getattr(solution_instance, 'm_p', None)
        self.m_pi = getattr(solution_instance, 'm_pi', None)
        if self.m_p is None and hasattr(solution_instance, 'givens'):
            self.m_p = solution_instance.givens.get('m_p', None)
        if self.m_pi is None and hasattr(solution_instance, 'givens'):
            self.m_pi = solution_instance.givens.get('m_pi', None)

        # Safely extract results (pseudocode emits)
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.E0_1 = results.get('E0_1', None)
        self.M_final_1 = results.get('M_final_1', None)
        self.E0_2 = results.get('E0_2', None)
        self.M_final_2 = results.get('M_final_2', None)
        self.E0_3 = results.get('E0_3', None)
        self.M_final_3 = results.get('M_final_3', None)

        # Precompute constants if possible
        self._M_final_1_expected = None
        self._M_final_3_expected = None
        self._E0_1_expected = None
        self._E0_3_expected = None
        if self.m_p is not None and self.m_pi is not None:
            self._M_final_1_expected = 2 * self.m_p + 4 * self.m_pi
            self._M_final_3_expected = 2 * self.m_p + 8 * self.m_pi
            # Compute expected E0 from threshold condition: 2 m_p^2 + 2 E0 m_p = M_final^2
            if self._M_final_1_expected is not None:
                self._E0_1_expected = (self._M_final_1_expected**2 - 2 * self.m_p**2) / (2 * self.m_p)
            if self._M_final_3_expected is not None:
                self._E0_3_expected = (self._M_final_3_expected**2 - 2 * self.m_p**2) / (2 * self.m_p)

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check all six expected result keys exist."""
        if None in (self.E0_1, self.M_final_1, self.E0_2, self.M_final_2, self.E0_3, self.M_final_3):
            missing = []
            if self.E0_1 is None: missing.append('E0_1')
            if self.M_final_1 is None: missing.append('M_final_1')
            if self.E0_2 is None: missing.append('E0_2')
            if self.M_final_2 is None: missing.append('M_final_2')
            if self.E0_3 is None: missing.append('E0_3')
            if self.M_final_3 is None: missing.append('M_final_3')
            return False, f"Missing result keys: {', '.join(missing)}"
        return True, "All required result keys present."

    def test_interface_types_numeric(self):
        """Ensure all result values are numeric (int or float)."""
        for name, val in [('E0_1', self.E0_1), ('M_final_1', self.M_final_1),
                          ('E0_2', self.E0_2), ('M_final_2', self.M_final_2),
                          ('E0_3', self.E0_3), ('M_final_3', self.M_final_3)]:
            if not isinstance(val, (int, float)):
                return False, f"{name} is not numeric (type {type(val).__name__})"
        return True, "All result values are numeric."

    def test_interface_givens_numeric(self):
        """Check that m_p and m_pi are provided and numeric."""
        if self.m_p is None:
            return False, "m_p not provided."
        if self.m_pi is None:
            return False, "m_pi not provided."
        if not isinstance(self.m_p, (int, float)):
            return False, f"m_p has type {type(self.m_p).__name__}, expected numeric."
        if not isinstance(self.m_pi, (int, float)):
            return False, f"m_pi has type {type(self.m_pi).__name__}, expected numeric."
        return True, "Givens m_p and m_pi are provided and numeric."

    # ------------------------------------------------------------
    # Invariant (theorem-based) tests
    # ------------------------------------------------------------
    def test_invariant_s_vs_mfinal1(self):
        """Check s = M_final_1^2 within tolerance for E0_1."""
        if None in (self.E0_1, self.M_final_1, self.m_p):
            return False, "SKIP: missing E0_1, M_final_1, or m_p"
        s = 2 * self.m_p**2 + 2 * self.E0_1 * self.m_p
        expected = self.M_final_1**2
        if math.isclose(s, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"s={s:.4f} equals M_final_1^2={expected:.4f} within tolerance."
        return False, f"s={s:.4f} differs from M_final_1^2={expected:.4f}."

    def test_invariant_s_vs_mfinal2(self):
        """Same invariant check for Q2."""
        if None in (self.E0_2, self.M_final_2, self.m_p):
            return False, "SKIP: missing E0_2, M_final_2, or m_p"
        s = 2 * self.m_p**2 + 2 * self.E0_2 * self.m_p
        expected = self.M_final_2**2
        if math.isclose(s, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"s={s:.4f} equals M_final_2^2={expected:.4f} within tolerance."
        return False, f"s={s:.4f} differs from M_final_2^2={expected:.4f}."

    def test_invariant_s_vs_mfinal3(self):
        """Invariant check for Q3."""
        if None in (self.E0_3, self.M_final_3, self.m_p):
            return False, "SKIP: missing E0_3, M_final_3, or m_p"
        s = 2 * self.m_p**2 + 2 * self.E0_3 * self.m_p
        expected = self.M_final_3**2
        if math.isclose(s, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"s={s:.4f} equals M_final_3^2={expected:.4f} within tolerance."
        return False, f"s={s:.4f} differs from M_final_3^2={expected:.4f}."

    def test_invariant_mfinal1_value(self):
        """Check M_final_1 equals 2*m_p + 4*m_pi if givens available."""
        if None in (self.m_p, self.m_pi, self.M_final_1):
            return False, "SKIP: missing m_p, m_pi, or M_final_1"
        expected = 2 * self.m_p + 4 * self.m_pi
        if math.isclose(self.M_final_1, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"M_final_1={self.M_final_1:.4f} ≈ {expected:.4f}"
        return False, f"M_final_1={self.M_final_1:.4f} should be {expected:.4f}"

    def test_invariant_mfinal3_value(self):
        """Check M_final_3 equals 2*m_p + 8*m_pi."""
        if None in (self.m_p, self.m_pi, self.M_final_3):
            return False, "SKIP: missing m_p, m_pi, or M_final_3"
        expected = 2 * self.m_p + 8 * self.m_pi
        if math.isclose(self.M_final_3, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"M_final_3={self.M_final_3:.4f} ≈ {expected:.4f}"
        return False, f"M_final_3={self.M_final_3:.4f} should be {expected:.4f}"

    def test_invariant_q1_eq_q2(self):
        """E0_1 and E0_2 should be equal (same final state)."""
        if None in (self.E0_1, self.E0_2):
            return False, "SKIP: missing E0_1 or E0_2"
        if math.isclose(self.E0_1, self.E0_2, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E0_1={self.E0_1:.4f} equals E0_2={self.E0_2:.4f}"
        return False, f"E0_1={self.E0_1:.4f} ≠ E0_2={self.E0_2:.4f}"

    def test_invariant_positive_energy(self):
        """All E0 must be > m_p (non‑zero kinetic energy)."""
        if None in (self.E0_1, self.E0_2, self.E0_3, self.m_p):
            return False, "SKIP: missing energies or m_p"
        if self.E0_1 <= self.m_p:
            return False, f"E0_1={self.E0_1:.4f} not > m_p={self.m_p}"
        if self.E0_2 <= self.m_p:
            return False, f"E0_2={self.E0_2:.4f} not > m_p={self.m_p}"
        if self.E0_3 <= self.m_p:
            return False, f"E0_3={self.E0_3:.4f} not > m_p={self.m_p}"
        return True, "All threshold energies > proton rest mass."

    # ------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------
    def test_modeling_s_formula(self):
        """Check that s = 2*m_p^2 + 2*E*m_p is used correctly."""
        # We already tested invariance; this test verifies the formula derivations.
        # Just check that the results are self‑consistent with the formula.
        return self.test_invariant_s_vs_mfinal1()  # reuse; not ideal but OK
        # Actually we want a separate test; but reuse is fine.

    def test_modeling_mfinal1_rest_masses(self):
        """M_final_1 should be exactly sum of rest masses: 2*m_p + 4*m_pi."""
        return self.test_invariant_mfinal1_value()

    def test_modeling_mfinal3_rest_masses(self):
        """M_final_3 should be exactly 2*m_p + 8*m_pi."""
        return self.test_invariant_mfinal3_value()

    def test_modeling_expected_E0_1(self):
        """If givens present, verify E0_1 matches the derived expected value."""
        if None in (self.E0_1, self._E0_1_expected):
            return False, "SKIP: missing E0_1 or derived expected value"
        if math.isclose(self.E0_1, self._E0_1_expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E0_1={self.E0_1:.4f} ≈ expected {self._E0_1_expected:.4f}"
        return False, f"E0_1={self.E0_1:.4f} ≠ expected {self._E0_1_expected:.4f}"

    def test_modeling_expected_E0_3(self):
        """Verify E0_3 matches derived expected value."""
        if None in (self.E0_3, self._E0_3_expected):
            return False, "SKIP: missing E0_3 or derived expected value"
        if math.isclose(self.E0_3, self._E0_3_expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E0_3={self.E0_3:.4f} ≈ expected {self._E0_3_expected:.4f}"
        return False, f"E0_3={self.E0_3:.4f} ≠ expected {self._E0_3_expected:.4f}"

    # ------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------
    def test_monotonicity_E0_with_Mfinal(self):
        """Increasing final mass should increase threshold energy.
        We simulate a small increase in M_final and recompute theoretical E0.
        """
        if None in (self.m_p, self.E0_1, self.E0_3):
            return False, "SKIP: missing m_p or E0 values"
        # Perturb M_final by +10% and see if E0 increases.
        for (E0_orig, M_orig, factor) in [
            (self.E0_1, self.M_final_1, 1.1),
            (self.E0_3, self.M_final_3, 1.1)
        ]:
            if None in (E0_orig, M_orig):
                continue
            M_pert = M_orig * factor
            E0_pert = (M_pert**2 - 2 * self.m_p**2) / (2 * self.m_p)
            if E0_pert <= E0_orig:
                return False, f"E0 did not increase when M_final increased: original={E0_orig:.4f}, perturbed={E0_pert:.4f}"
        return True, "Increasing final mass increases threshold energy (monotonic)."

    def test_monotonicity_E0_with_m_pi(self):
        """If we increase the pion mass, M_final increases and so should E0.
        Test by considering a hypothetical increase in m_pi by +10% and recompute.
        """
        if None in (self.m_p, self.m_pi, self.E0_1, self.E0_3):
            return False, "SKIP: missing givens or energies"
        factor = 1.1
        m_pi_pert = self.m_pi * factor
        # For Q1
        M1_pert = 2 * self.m_p + 4 * m_pi_pert
        E1_pert = (M1_pert**2 - 2 * self.m_p**2) / (2 * self.m_p)
        if E1_pert <= self.E0_1:
            return False, f"Q1: E0 did not increase when m_pi increased by {factor}."
        # For Q3
        M3_pert = 2 * self.m_p + 8 * m_pi_pert
        E3_pert = (M3_pert**2 - 2 * self.m_p**2) / (2 * self.m_p)
        if E3_pert <= self.E0_3:
            return False, f"Q3: E0 did not increase when m_pi increased by {factor}."
        return True, "Increasing pion mass increases threshold energies (monotonic)."

    # ------------------------------------------------------------
    # Weak answer sanity (only one, very loose)
    # ------------------------------------------------------------
    def test_answer_sanity_magnitude(self):
        """Very broad magnitude check: E0_1, E0_2 around 2.2 GeV; E0_3 around 3.8 GeV."""
        if None in (self.E0_1, self.E0_2, self.E0_3):
            return False, "SKIP: missing some E0 values."
        # Allow ±30% deviation (very generous)
        if not (1.5 < self.E0_1 < 3.0):
            return False, f"E0_1={self.E0_1:.4f} not in plausible range [1.5, 3.0] GeV."
        if not (1.5 < self.E0_2 < 3.0):
            return False, f"E0_2={self.E0_2:.4f} not in plausible range [1.5, 3.0] GeV."
        if not (2.5 < self.E0_3 < 5.0):
            return False, f"E0_3={self.E0_3:.4f} not in plausible range [2.5, 5.0] GeV."
        return True, "All E0 values lie in broad plausible magnitude ranges."

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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)