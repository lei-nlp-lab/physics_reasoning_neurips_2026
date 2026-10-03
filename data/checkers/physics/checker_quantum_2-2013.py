import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the positronium problem.
    Audits interface completeness, physical invariants, modeling trace consistency,
    monotonicity, and a single weak sanity check.
    """

    def __init__(self, solution_instance):
        # Safe extraction of result dictionary
        self.results = getattr(solution_instance, 'results', None) or {}
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract result quantities with None defaults
        self.E_ps = self.results.get('E_ps', None)
        self.a_ps = self.results.get('a_ps', None)
        self.delta_E = self.results.get('delta_E', None)
        self.J = self.results.get('J', None)

        # Extract given constants (may be None)
        self.m_e = getattr(solution_instance, 'm_e', None)
        self.e = getattr(solution_instance, 'e', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.c = getattr(solution_instance, 'c', None)

        # Also check a 'givens' dictionary attribute
        if hasattr(solution_instance, 'givens'):
            givens = solution_instance.givens
            if isinstance(givens, dict):
                self.m_e = givens.get('m_e', self.m_e)
                self.e = givens.get('e', self.e)
                self.hbar = givens.get('hbar', self.hbar)
                self.c = givens.get('c', self.c)

        # Physical constants and conversion factors
        self.erg_to_eV = 1.602e-12          # 1 eV = 1.602e-12 erg
        self.erg_to_MeV = 1.602e-6          # 1 MeV = 1.602e-6 erg

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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        required_keys = ['E_ps', 'a_ps', 'delta_E', 'J']
        missing = [k for k in required_keys if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_results_not_none(self):
        keys = ['E_ps', 'a_ps', 'delta_E', 'J']
        none_keys = [k for k in keys if getattr(self, k) is None]
        if none_keys:
            return (False, f"Result keys with None value: {none_keys}")
        return (True, "All results are non-None")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_energy_formula(self):
        """Check E_ps = - m_e * e^4 / (4 ħ^2) (in erg), converted to eV."""
        if None in (self.m_e, self.e, self.hbar, self.E_ps):
            return (False, "SKIP: missing constants (m_e, e, hbar) or E_ps")
        # Compute expected in erg and convert to eV
        expected_erg = -self.m_e * (self.e**4) / (4 * self.hbar**2)
        expected_eV = expected_erg / self.erg_to_eV
        # Interpret user value – if very small it's in eV, else assume erg
        if abs(self.E_ps) < 0.1:
            user_eV = self.E_ps
        else:
            # Likely in erg, convert to eV
            user_eV = self.E_ps / self.erg_to_eV
        if math.isclose(user_eV, expected_eV, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_ps = {user_eV} eV matches expected {expected_eV} eV")
        return (False, f"E_ps = {user_eV} eV, expected ~{expected_eV} eV")

    def test_invariant_bohr_radius_formula(self):
        """Check a_ps = 2 ħ^2 / (m_e e^2) (in cm)."""
        if None in (self.m_e, self.e, self.hbar, self.a_ps):
            return (False, "SKIP: missing constants or a_ps")
        expected_cm = 2 * self.hbar**2 / (self.m_e * self.e**2)
        if math.isclose(self.a_ps, expected_cm, rel_tol=1e-2, abs_tol=1e-3):
            return (True, f"a_ps = {self.a_ps} cm matches expected {expected_cm} cm")
        return (False, f"a_ps = {self.a_ps} cm, expected ~{expected_cm} cm")

    def test_invariant_annihilation_energy_formula(self):
        """Check delta_E ≈ 2 m_e c^2 (in MeV)."""
        if None in (self.m_e, self.c, self.delta_E):
            return (False, "SKIP: missing m_e, c, or delta_E")
        expected_erg = 2 * self.m_e * self.c**2
        expected_MeV = expected_erg / self.erg_to_MeV
        # Interpret user value
        if abs(self.delta_E) > 1e6:
            user_MeV = self.delta_E / 1e6          # assume eV -> MeV
        elif abs(self.delta_E) > 1e-6:
            user_MeV = self.delta_E                 # assume MeV
        else:
            user_MeV = self.delta_E / self.erg_to_MeV  # assume erg -> MeV
        if math.isclose(user_MeV, expected_MeV, rel_tol=1e-2, abs_tol=1e-3):
            return (True, f"delta_E ≈ {user_MeV} MeV matches expected {expected_MeV} MeV")
        return (False, f"delta_E = {user_MeV} MeV, expected ~{expected_MeV} MeV")

    def test_invariant_angular_momentum_formula(self):
        """Check J = sqrt(2) ħ (for triplet ground state)."""
        if None in (self.hbar, self.J):
            return (False, "SKIP: missing hbar or J")
        expected = math.sqrt(2) * self.hbar
        if math.isclose(self.J, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"J = {self.J} matches sqrt(2)*hbar = {expected}")
        return (False, f"J = {self.J}, expected ~{expected}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (check relations from explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_energy_ratio_to_hydrogen(self):
        """
        Check E_ps ≈ E_0_H / 2, where E_0_H = -m_e e^4/(2 ħ^2).
        This traces the explanation step that the reduced mass halves the energy.
        """
        if None in (self.m_e, self.e, self.hbar, self.E_ps):
            return (False, "SKIP: missing constants or E_ps")
        # Compute hydrogen ground state energy in eV
        E0_H_erg = -self.m_e * (self.e**4) / (2 * self.hbar**2)
        E0_H_eV = E0_H_erg / self.erg_to_eV
        # Expected positronium energy = half that
        expected_E_ps_eV = E0_H_eV / 2
        # Interpret user value as before
        if abs(self.E_ps) < 0.1:
            user_eV = self.E_ps
        else:
            user_eV = self.E_ps / self.erg_to_eV
        if math.isclose(user_eV, expected_E_ps_eV, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_ps = {user_eV} eV is half of hydrogen ground state ({E0_H_eV} eV)")
        return (False, f"E_ps = {user_eV} eV, expected half of hydrogen ~{expected_E_ps_eV} eV")

    def test_modeling_bohr_radius_ratio_to_hydrogen(self):
        """
        Check a_ps ≈ 2 * a_0_H, where a_0_H = ħ^2/(m_e e^2).
        This traces the explanation step that the radius doubles.
        """
        if None in (self.m_e, self.e, self.hbar, self.a_ps):
            return (False, "SKIP: missing constants or a_ps")
        a0_H = self.hbar**2 / (self.m_e * self.e**2)
        expected_a_ps = 2 * a0_H
        if math.isclose(self.a_ps, expected_a_ps, rel_tol=1e-2, abs_tol=1e-3):
            return (True, f"a_ps = {self.a_ps} cm is twice hydrogen Bohr radius ({a0_H} cm)")
        return (False, f"a_ps = {self.a_ps} cm, expected twice hydrogen ~{expected_a_ps} cm")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb an input and compare output change)
    # ------------------------------------------------------------------
    def test_monotonicity_energy_decreases_with_mass(self):
        """
        Under a +10% increase in m_e, the ground state energy should become more negative.
        We compute the expected new energy and check that it is less than the reported energy.
        """
        if None in (self.m_e, self.e, self.hbar, self.E_ps):
            return (False, "SKIP: missing constants or E_ps")
        # Baseline energy in eV (reported)
        if abs(self.E_ps) < 0.1:
            baseline_eV = self.E_ps
        else:
            baseline_eV = self.E_ps / self.erg_to_eV
        # New mass increased by 10%
        m_e_new = self.m_e * 1.10
        # Compute new energy expectation
        new_expected_erg = -m_e_new * (self.e**4) / (4 * self.hbar**2)
        new_expected_eV = new_expected_erg / self.erg_to_eV
        # Check that new energy is more negative (smaller)
        if new_expected_eV < baseline_eV:
            return (True, f"Increasing m_e makes E_ps more negative: baseline {baseline_eV} eV -> new {new_expected_eV} eV")
        return (False, f"Monotonicity failed: baseline {baseline_eV} eV, expected new {new_expected_eV} eV")

    def test_monotonicity_radius_decreases_with_mass(self):
        """
        Under a +10% increase in m_e, the Bohr radius should decrease.
        """
        if None in (self.m_e, self.e, self.hbar, self.a_ps):
            return (False, "SKIP: missing constants or a_ps")
        # Baseline radius
        baseline = self.a_ps
        # New mass increased by 10%
        m_e_new = self.m_e * 1.10
        new_expected = 2 * self.hbar**2 / (m_e_new * self.e**2)
        # Check that new radius is smaller
        if new_expected < baseline:
            return (True, f"Increasing m_e decreases a_ps: baseline {baseline} cm -> new {new_expected} cm")
        return (False, f"Monotonicity failed: baseline {baseline} cm, expected new {new_expected} cm")

    # ------------------------------------------------------------------
    # Weak sanity test (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_plausibility(self):
        """
        A single weak sanity check for all four values: basic sign, finite, plausible magnitude.
        No math.isclose, no re-derivation of expected values.
        """
        # Check each quantity quickly
        issues = []
        # Energy
        if self.E_ps is None:
            issues.append("E_ps missing")
        elif self.E_ps >= 0:
            issues.append(f"E_ps positive ({self.E_ps}) — should be negative")
        elif abs(self.E_ps) < 1e-6 or abs(self.E_ps) > 1000:
            issues.append(f"E_ps magnitude implausible: {self.E_ps}")
        # Radius
        if self.a_ps is None:
            issues.append("a_ps missing")
        elif self.a_ps <= 0 or self.a_ps > 1e-6 or self.a_ps < 1e-12:
            issues.append(f"a_ps implausible: {self.a_ps}")
        # Annihilation energy
        if self.delta_E is None:
            issues.append("delta_E missing")
        elif self.delta_E <= 0:
            issues.append(f"delta_E non-positive ({self.delta_E})")
        elif abs(self.delta_E) < 1e-12 or abs(self.delta_E) > 1e9:
            issues.append(f"delta_E magnitude implausible: {self.delta_E}")
        # Angular momentum
        if self.J is None:
            issues.append("J missing")
        elif self.J <= 0:
            issues.append(f"J non-positive ({self.J})")
        elif self.J < 1e-32 or self.J > 1e-22:
            issues.append(f"J magnitude implausible: {self.J}")
        if issues:
            return (False, f"Sanity issues: {'; '.join(issues)}")
        return (True, "All four results pass basic sanity (sign, finite, plausible magnitude)")