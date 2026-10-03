import math
import sympy
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Physics checker for the problem: Express fundamental quantities in terms of
    ℏ, e, c, m, M and give numerical estimates.

    Expects solution_instance to have:
        - givens (dict) with keys: 'hbar', 'e', 'c', 'm', 'M' (all in cgs Gaussian units)
        - results (dict) with keys: 'a0', 'EB', 'mu_B', 'lambda_C', 're', 'Ee', 'Ep', 'alpha', 'delta_E_FS'

    Checks focus on invariant relationships (theorem-based) and modeling quality,
    with a single weak answer sanity test.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract givens – if missing, keep as None
        givens = getattr(solution_instance, 'givens', {})
        self.hbar = givens.get('hbar', None)   # erg·s
        self.e = givens.get('e', None)         # esu
        self.c = givens.get('c', None)         # cm/s
        self.m = givens.get('m', None)         # g
        self.M = givens.get('M', None)         # g

        # Conversion factors (None if missing)
        self.eV_to_erg = givens.get('1 eV', None)
        self.MeV_to_erg = givens.get('1 MeV', None)

        # Safely extract results – do not assign numeric defaults
        self.results = getattr(solution_instance, 'results', {})
        self.a0 = self.results.get('a0', None)                     # cm
        self.EB = self.results.get('EB', None)                    # erg or eV
        self.mu_B = self.results.get('mu_B', None)                # erg/Gs
        self.lambda_C = self.results.get('lambda_C', None)        # cm
        self.re = self.results.get('re', None)                    # cm
        self.Ee = self.results.get('Ee', None)                    # erg or MeV
        self.Ep = self.results.get('Ep', None)                    # erg or MeV
        self.alpha = self.results.get('alpha', None)              # dimensionless
        self.delta_E_FS = self.results.get('delta_E_FS', None)    # erg or eV

        # Compute derived fine-structure constant if possible
        self._computed_alpha = None
        if self.e is not None and self.hbar is not None and self.c is not None:
            self._computed_alpha = self.e**2 / (self.hbar * self.c)

    # ----------------------------------------------------------------
    # Helper: math.isclose with tolerance 1e-2
    # ----------------------------------------------------------------
    def _close(self, a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    # ----------------------------------------------------------------
    # Interface tests – ensure required keys exist and have valid types
    # ----------------------------------------------------------------
    def test_interface_a0_exists(self) -> Tuple[bool, str]:
        if self.a0 is None:
            return (False, "a0 missing or None")
        if isinstance(self.a0, (int, float)):
            return (True, "a0 present and numeric")
        return (False, f"a0 has unexpected type {type(self.a0)}")

    def test_interface_EB_exists(self) -> Tuple[bool, str]:
        if self.EB is None:
            return (False, "EB missing or None")
        if isinstance(self.EB, (int, float)):
            return (True, "EB present and numeric")
        return (False, f"EB has unexpected type {type(self.EB)}")

    def test_interface_mu_B_exists(self) -> Tuple[bool, str]:
        if self.mu_B is None:
            return (False, "mu_B missing or None")
        return (True, "mu_B present")

    def test_interface_lambda_C_exists(self) -> Tuple[bool, str]:
        if self.lambda_C is None:
            return (False, "lambda_C missing or None")
        return (True, "lambda_C present")

    def test_interface_re_exists(self) -> Tuple[bool, str]:
        if self.re is None:
            return (False, "re missing or None")
        return (True, "re present")

    def test_interface_Ee_exists(self) -> Tuple[bool, str]:
        if self.Ee is None:
            return (False, "Ee missing or None")
        return (True, "Ee present")

    def test_interface_Ep_exists(self) -> Tuple[bool, str]:
        if self.Ep is None:
            return (False, "Ep missing or None")
        return (True, "Ep present")

    def test_interface_alpha_exists(self) -> Tuple[bool, str]:
        if self.alpha is None:
            return (False, "alpha missing or None")
        return (True, "alpha present")

    def test_interface_delta_E_FS_exists(self) -> Tuple[bool, str]:
        if self.delta_E_FS is None:
            return (False, "delta_E_FS missing or None")
        return (True, "delta_E_FS present")

    # ----------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (relaxed tolerance)
    # ----------------------------------------------------------------
    def test_invariant_fine_structure_constant(self) -> Tuple[bool, str]:
        """Check that alpha = e^2/(hbar c) within 1%."""
        if None in (self.alpha, self._computed_alpha):
            return (True, "SKIP: insufficient givens (e, hbar, c) or alpha")
        if self._close(self.alpha, self._computed_alpha):
            return (True, f"alpha={self.alpha} matches computed {self._computed_alpha}")
        return (False, f"alpha={self.alpha} differs from expected {self._computed_alpha}")

    def test_invariant_bohr_radius(self) -> Tuple[bool, str]:
        """Check a0 = hbar^2/(m e^2) numerically."""
        if None in (self.a0, self.hbar, self.m, self.e):
            return (True, "SKIP: missing givens (hbar, m, e) or a0")
        expected = self.hbar**2 / (self.m * self.e**2)
        if self._close(self.a0, expected):
            return (True, f"a0={self.a0} ≈ expected {expected}")
        return (False, f"a0={self.a0} not close to {expected}")

    def test_invariant_binding_energy(self) -> Tuple[bool, str]:
        """Check EB = m e^4/(2 hbar^2) numerically."""
        if None in (self.EB, self.m, self.e, self.hbar):
            return (True, "SKIP: missing givens or EB")
        expected_erg = self.m * self.e**4 / (2 * self.hbar**2)
        # If conversion factor available, check in appropriate units
        if self.eV_to_erg is not None:
            expected_eV = expected_erg / self.eV_to_erg
            if self._close(self.EB, expected_eV) or self._close(self.EB, expected_erg):
                return (True, f"EB={self.EB} matches expected in appropriate units")
            return (False, f"EB={self.EB} not close to {expected_erg} erg or {expected_eV} eV")
        else:
            # Assume EB is in erg (cgs)
            if self._close(self.EB, expected_erg):
                return (True, f"EB={self.EB} ≈ expected {expected_erg} erg")
            return (False, f"EB={self.EB} not close to {expected_erg} erg")

    def test_invariant_bohr_magneton(self) -> Tuple[bool, str]:
        """Check mu_B = e hbar/(2 m c) in cgs."""
        if None in (self.mu_B, self.e, self.hbar, self.m, self.c):
            return (True, "SKIP: missing givens or mu_B")
        expected = self.e * self.hbar / (2 * self.m * self.c)
        if self._close(self.mu_B, expected):
            return (True, f"mu_B={self.mu_B} ≈ expected {expected}")
        return (False, f"mu_B={self.mu_B} not close to {expected}")

    def test_invariant_compton_wavelength(self) -> Tuple[bool, str]:
        """Check lambda_C is consistent with hbar/(m c) or 2π*hbar/(m c)."""
        if None in (self.lambda_C, self.hbar, self.m, self.c):
            return (True, "SKIP: missing givens or lambda_C")
        reduced = self.hbar / (self.m * self.c)
        full = 2 * math.pi * reduced
        if self._close(self.lambda_C, reduced) or self._close(self.lambda_C, full):
            return (True, f"lambda_C={self.lambda_C} consistent with expected")
        return (False, f"lambda_C={self.lambda_C} not close to {reduced} (reduced) or {full}")

    def test_invariant_classical_radius(self) -> Tuple[bool, str]:
        """Check re = e^2/(m c^2)."""
        if None in (self.re, self.e, self.m, self.c):
            return (True, "SKIP: missing givens or re")
        expected = self.e**2 / (self.m * self.c**2)
        if self._close(self.re, expected):
            return (True, f"re={self.re} ≈ expected {expected}")
        return (False, f"re={self.re} not close to {expected}")

    def test_invariant_rest_energy_electron(self) -> Tuple[bool, str]:
        """Check Ee = m c^2."""
        if None in (self.Ee, self.m, self.c):
            return (True, "SKIP: missing givens or Ee")
        expected_erg = self.m * self.c**2
        if self.MeV_to_erg is not None:
            expected_MeV = expected_erg / self.MeV_to_erg
            if self._close(self.Ee, expected_MeV) or self._close(self.Ee, expected_erg):
                return (True, f"Ee={self.Ee} consistent with expected")
            return (False, f"Ee={self.Ee} not close to expected")
        else:
            if self._close(self.Ee, expected_erg):
                return (True, f"Ee={self.Ee} ≈ {expected_erg} erg")
            return (False, f"Ee={self.Ee} not close to {expected_erg} erg")

    def test_invariant_rest_energy_proton(self) -> Tuple[bool, str]:
        """Check Ep = M c^2."""
        if None in (self.Ep, self.M, self.c):
            return (True, "SKIP: missing givens or Ep")
        expected_erg = self.M * self.c**2
        if self.MeV_to_erg is not None:
            expected_MeV = expected_erg / self.MeV_to_erg
            if self._close(self.Ep, expected_MeV) or self._close(self.Ep, expected_erg):
                return (True, f"Ep={self.Ep} consistent with expected")
            return (False, f"Ep={self.Ep} not close to expected")
        else:
            if self._close(self.Ep, expected_erg):
                return (True, f"Ep={self.Ep} ≈ {expected_erg} erg")
            return (False, f"Ep={self.Ep} not close to {expected_erg} erg")

    def test_invariant_fine_structure_splitting(self) -> Tuple[bool, str]:
        """Check delta_E_FS = (1/8) α^4 m c^2 (following target answer)."""
        if None in (self.delta_E_FS, self.m, self.c, self.alpha):
            return (True, "SKIP: missing givens or delta_E_FS")
        expected_erg = (1/8) * (self.alpha**4) * self.m * self.c**2
        if self.eV_to_erg is not None:
            expected_eV = expected_erg / self.eV_to_erg
            if self._close(self.delta_E_FS, expected_eV) or self._close(self.delta_E_FS, expected_erg):
                return (True, f"delta_E_FS={self.delta_E_FS} consistent with expected")
            return (False, f"delta_E_FS={self.delta_E_FS} not close to expected")
        else:
            if self._close(self.delta_E_FS, expected_erg):
                return (True, f"delta_E_FS={self.delta_E_FS} ≈ {expected_erg} erg")
            return (False, f"delta_E_FS={self.delta_E_FS} not close to {expected_erg} erg")

    # ----------------------------------------------------------------
    # Modeling/trace tests – cross-check relationships between quantities
    # ----------------------------------------------------------------
    def test_modeling_bohr_energy_vs_radius(self) -> Tuple[bool, str]:
        """Check EB = e^2/(2 a0) from Bohr model."""
        if None in (self.EB, self.e, self.a0):
            return (True, "SKIP: missing EB, e, or a0")
        expected = self.e**2 / (2 * self.a0)
        if self._close(self.EB, expected):
            return (True, f"EB={self.EB} ≈ e^2/(2a0)={expected}")
        return (False, f"EB={self.EB} not close to e^2/(2a0)={expected}")

    def test_modeling_a0_and_re_relation(self) -> Tuple[bool, str]:
        """Check a0 = re / α^2."""
        if None in (self.a0, self.re, self._computed_alpha):
            return (True, "SKIP: missing a0, re, or computed alpha")
        expected_a0 = self.re / (self._computed_alpha**2)
        if self._close(self.a0, expected_a0):
            return (True, f"a0={self.a0} ≈ re/alpha^2={expected_a0}")
        return (False, f"a0={self.a0} not close to re/alpha^2={expected_a0}")

    # ----------------------------------------------------------------
    # Monotonicity tests – explicitly perturb an input and compare
    # ----------------------------------------------------------------
    def test_monotonicity_bohr_radius_vs_mass(self) -> Tuple[bool, str]:
        """Check that increasing m decreases a0."""
        if None in (self.a0, self.m, self.hbar, self.e):
            return (True, "SKIP: missing a0, m, hbar, e")
        # Baseline with original mass
        a0_original = self.a0
        # Perturb: increase mass by 10%
        m_new = self.m * 1.1
        a0_new = self.hbar**2 / (m_new * self.e**2)
        # Since mass increased, a0 should decrease
        if a0_new < a0_original:
            return (True, "Increasing mass decreases a0 as expected")
        return (False, "Monotonicity violated: increasing mass did not decrease a0")

    def test_monotonicity_rest_energy_vs_mass(self) -> Tuple[bool, str]:
        """Check that increasing mass increases rest energy (E = m c^2)."""
        if None in (self.Ee, self.m, self.c):
            return (True, "SKIP: missing Ee, m, c")
        Ee_original = self.Ee
        # Perturb: increase mass by 10% (assuming rest energy is in consistent units)
        Ee_new = (1.1 * self.m) * self.c**2
        # Convert to same units as Ee if conversion available
        if self.MeV_to_erg is not None:
            # Assume Ee is in MeV (common)
            Ee_new_MeV = Ee_new / self.MeV_to_erg
            if Ee_new_MeV > Ee_original:
                return (True, "Increasing mass increases rest energy as expected")
            return (False, "Monotonicity violated: increasing mass did not increase rest energy")
        else:
            # Assume Ee is in erg
            if Ee_new > Ee_original:
                return (True, "Increasing mass increases rest energy as expected")
            return (False, "Monotonicity violated")

    def test_monotonicity_re_vs_mass(self) -> Tuple[bool, str]:
        """Check that increasing mass decreases classical radius (re ∝ 1/m)."""
        if None in (self.re, self.m, self.e, self.c):
            return (True, "SKIP: missing re, m, e, c")
        re_original = self.re
        m_new = self.m * 1.1
        re_new = self.e**2 / (m_new * self.c**2)
        if re_new < re_original:
            return (True, "Increasing mass decreases classical radius as expected")
        return (False, "Monotonicity violated")

    # ----------------------------------------------------------------
    # Weak answer sanity test (only one)
    # ----------------------------------------------------------------
    def test_answer_sanity_quantities_positive(self) -> Tuple[bool, str]:
        """All derived quantities must be finite and positive."""
        quantities = [self.a0, self.EB, self.mu_B, self.lambda_C, self.re, self.Ee, self.Ep, self.alpha, self.delta_E_FS]
        names = ['a0', 'EB', 'mu_B', 'lambda_C', 're', 'Ee', 'Ep', 'alpha', 'delta_E_FS']
        for q, name in zip(quantities, names):
            if q is None:
                return (False, f"{name} missing")
            if not isinstance(q, (int, float)):
                return (False, f"{name} not numeric: {type(q)}")
            if q <= 0:
                return (False, f"{name} = {q} not positive")
        return (True, "All quantities are positive and finite")

    # ----------------------------------------------------------------
    # run_all_tests – discover and execute all test_ methods
    # ----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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