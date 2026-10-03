import math
import sympy as sp
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Audits a solution instance for the spherical cavity photon/electron problem.
    Extracts all relevant variables safely (default None), then runs unit-test-like checks.
    """

    def __init__(self, solution_instance: Any):
        # -- Extract target results, defaulting to None --
        self.results = getattr(solution_instance, 'results', {})
        self.N_gamma = self.results.get('N_gamma', None)
        self.E_gamma = self.results.get('E_gamma', None)
        self.p_F = self.results.get('p_F', None)

        # -- Extract givens, constants, and intermediate variables --
        # Givens (from problem statement)
        self.R = getattr(solution_instance, 'R', None)  # cm
        self.T = getattr(solution_instance, 'T', None)  # K
        self.N_e = getattr(solution_instance, 'N_e', None)  # number of electrons

        # Constants (should match standard values; we store for later checks)
        self.k_B = getattr(solution_instance, 'k_B', None)      # erg/K
        self.hbar = getattr(solution_instance, 'hbar', None)    # erg·s
        self.c = getattr(solution_instance, 'c', None)          # cm/s
        self.pi = getattr(solution_instance, 'pi', None)
        self.zeta3 = getattr(solution_instance, 'zeta3', None)  # ζ(3)

        # Derived/intermediate quantities (the solution may store them)
        self.V = self.results.get('V', None)                     # volume
        self.n_gamma = self.results.get('n_gamma', None)        # photon number density
        self.u_gamma = self.results.get('u_gamma', None)        # photon energy density
        self.n_e = self.results.get('n_e', None)                # electron number density

        # Explanation steps may also be available
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Helper: safe recomputation of intermediate quantities from givens
    # ------------------------------------------------------------------
    def _compute_volume(self) -> float:
        """Return V = (4/3)πR^3 if R is valid, else None."""
        if self.R is None:
            return None
        return (4.0/3.0) * math.pi * self.R**3

    def _compute_n_gamma(self) -> float:
        """n_gamma = (2*ζ(3)/π^2)*(k_B*T/(hbar*c))^3"""
        if any(x is None for x in [self.T, self.k_B, self.hbar, self.c, self.pi, self.zeta3]):
            return None
        coeff = (2.0 * self.zeta3) / (self.pi**2)
        factor = (self.k_B * self.T) / (self.hbar * self.c)
        return coeff * factor**3

    def _compute_u_gamma(self) -> float:
        """u_gamma = (π^2/15)*(k_B*T)^4/(hbar^3*c^3)"""
        if any(x is None for x in [self.T, self.k_B, self.hbar, self.c, self.pi]):
            return None
        coeff = (self.pi**2) / 15.0
        numer = (self.k_B * self.T)**4
        denom = (self.hbar**3) * (self.c**3)
        return coeff * numer / denom

    def _compute_n_e(self) -> float:
        """n_e = N_e / V"""
        if self.N_e is None or self.V is None:
            return None
        return self.N_e / self.V

    def _compute_p_F(self) -> float:
        """p_F = hbar * (3*π^2 * n_e)^(1/3)"""
        n_e = self._compute_n_e()
        if any(x is None for x in [self.hbar, self.pi, n_e]):
            return None
        return self.hbar * (3.0 * self.pi**2 * n_e)**(1.0/3.0)

    # ------------------------------------------------------------------
    # Test interface: required keys exist and have valid numeric types
    # ------------------------------------------------------------------
    def test_interface_required_keys_present(self) -> Tuple[bool, str]:
        required = ['N_gamma', 'E_gamma', 'p_F']
        missing = [k for k in required if k not in self.results or self.results[k] is None]
        if missing:
            return (False, f"Missing required keys: {missing}")
        return (True, "All required keys present")

    def test_interface_numeric_types(self) -> Tuple[bool, str]:
        for name, val in [('N_gamma', self.N_gamma), ('E_gamma', self.E_gamma), ('p_F', self.p_F)]:
            if val is not None and not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric, type={type(val).__name__}")
        return (True, "All result values are numeric")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_photon_quantities_positive_finite(self) -> Tuple[bool, str]:
        for name, val in [('N_gamma', self.N_gamma), ('E_gamma', self.E_gamma), ('p_F', self.p_F)]:
            if val is None:
                return (False, f"SKIP: {name} is None")
            if not (math.isfinite(val) and val > 0):
                return (False, f"{name} not positive finite: {val}")
        return (True, "All quantities positive and finite")

    def test_invariant_photon_energy_density_relation(self) -> Tuple[bool, str]:
        """
        Check that u_gamma (energy density) is consistent with N_gamma * average photon energy.
        This is a loose invariance: for a blackbody, E_gamma / V ~ 2.701 * k_B * T * n_gamma.
        Relatively large tolerance because it's approximate.
        """
        if any(x is None for x in [self.u_gamma, self.n_gamma, self.T, self.k_B]):
            return (False, "SKIP: missing u_gamma, n_gamma, T, or k_B")
        # approximate ratio: u_gamma / (n_gamma * k_B * T) ≈ 2.701 (from Planck distribution)
        expected_ratio = 2.701
        actual_ratio = self.u_gamma / (self.n_gamma * self.k_B * self.T)
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=5e-2):
            return (False, f"Photon energy density ratio {actual_ratio:.3f} differs from expected {expected_ratio}")
        return (True, f"Photon energy density ratio {actual_ratio:.3f} agrees with expectation")

    def test_invariant_fermi_momentum_reasonable(self) -> Tuple[bool, str]:
        """
        Fermi momentum for electrons at 1e80 in the given volume should be extremely small
        (order 10^-26 g·cm/s). Check that it's between 1e-27 and 1e-25.
        """
        if self.p_F is None:
            return (False, "SKIP: p_F is None")
        lower = 1e-27
        upper = 1e-25
        if lower <= self.p_F <= upper:
            return (True, f"p_F = {self.p_F:.2e} within plausible range")
        return (False, f"p_F = {self.p_F:.2e} outside plausible range [{lower:.0e}, {upper:.0e}]")

    # ------------------------------------------------------------------
    # Modeling / trace tests: check that the formulas used are correctly implemented
    # ------------------------------------------------------------------
    def test_modeling_volume_formula(self) -> Tuple[bool, str]:
        """
        Check that V (if provided) matches V = (4/3)πR^3.
        """
        if self.V is None or self.R is None:
            return (False, "SKIP: V or R not available")
        expected_V = self._compute_volume()
        if expected_V is None:
            return (False, "SKIP: could not compute expected volume")
        if not math.isclose(self.V, expected_V, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Volume mismatch: reported {self.V:.4e}, expected {expected_V:.4e}")
        return (True, f"Volume matches: {self.V:.4e}")

    def test_modeling_photon_number_density(self) -> Tuple[bool, str]:
        """
        Check that n_gamma (if present) matches the blackbody formula.
        """
        if self.n_gamma is None:
            return (False, "SKIP: n_gamma not available")
        expected_n = self._compute_n_gamma()
        if expected_n is None:
            return (False, "SKIP: could not compute expected n_gamma due to missing constants")
        if not math.isclose(self.n_gamma, expected_n, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"n_gamma mismatch: reported {self.n_gamma:.4e}, expected {expected_n:.4e}")
        return (True, f"n_gamma agrees: {self.n_gamma:.4e}")

    def test_modeling_photon_energy_density(self) -> Tuple[bool, str]:
        """
        Check that u_gamma (if present) matches the blackbody formula.
        """
        if self.u_gamma is None:
            return (False, "SKIP: u_gamma not available")
        expected_u = self._compute_u_gamma()
        if expected_u is None:
            return (False, "SKIP: could not compute expected u_gamma")
        if not math.isclose(self.u_gamma, expected_u, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"u_gamma mismatch: reported {self.u_gamma:.4e}, expected {expected_u:.4e}")
        return (True, f"u_gamma agrees: {self.u_gamma:.4e}")

    def test_modeling_N_gamma_from_density(self) -> Tuple[bool, str]:
        """
        Check that N_gamma = n_gamma * V (or at least close).
        """
        if any(x is None for x in [self.N_gamma, self.n_gamma, self.V]):
            return (False, "SKIP: N_gamma, n_gamma, or V not available")
        expected_N = self.n_gamma * self.V
        if not math.isclose(self.N_gamma, expected_N, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"N_gamma mismatch: reported {self.N_gamma:.4e}, expected n_gamma*V = {expected_N:.4e}")
        return (True, f"N_gamma from density matches")

    def test_modeling_E_gamma_from_density(self) -> Tuple[bool, str]:
        """
        Check that E_gamma = u_gamma * V.
        """
        if any(x is None for x in [self.E_gamma, self.u_gamma, self.V]):
            return (False, "SKIP: E_gamma, u_gamma, or V not available")
        expected_E = self.u_gamma * self.V
        if not math.isclose(self.E_gamma, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_gamma mismatch: reported {self.E_gamma:.4e}, expected u_gamma*V = {expected_E:.4e}")
        return (True, f"E_gamma from density matches")

    def test_modeling_electron_density(self) -> Tuple[bool, str]:
        """
        Check that n_e (if present) equals N_e / V.
        """
        if self.n_e is None or self.N_e is None or self.V is None:
            return (False, "SKIP: n_e, N_e, or V not available")
        expected_n_e = self._compute_n_e()
        if expected_n_e is None:
            return (False, "SKIP: could not compute expected n_e")
        if not math.isclose(self.n_e, expected_n_e, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"n_e mismatch: reported {self.n_e:.4e}, expected {expected_n_e:.4e}")
        return (True, f"n_e agrees: {self.n_e:.4e}")

    def test_modeling_fermi_momentum_from_density(self) -> Tuple[bool, str]:
        """
        Check that p_F = hbar * (3π^2 n_e)^(1/3), either from reported n_e or computed n_e.
        """
        if self.p_F is None:
            return (False, "SKIP: p_F not available")
        # Try to use reported n_e first, else compute from N_e/V
        if self.n_e is not None and all(x is not None for x in [self.hbar, self.pi]):
            computed_p_F = self.hbar * (3.0 * self.pi**2 * self.n_e)**(1.0/3.0)
        elif self.N_e is not None and self.V is not None and all(x is not None for x in [self.hbar, self.pi]):
            n_e = self._compute_n_e()
            if n_e is None:
                return (False, "SKIP: cannot compute n_e")
            computed_p_F = self.hbar * (3.0 * self.pi**2 * n_e)**(1.0/3.0)
        else:
            return (False, "SKIP: insufficient data to compute expected p_F")
        if not math.isclose(self.p_F, computed_p_F, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"p_F mismatch: reported {self.p_F:.4e}, expected {computed_p_F:.4e}")
        return (True, f"p_F agrees: {self.p_F:.4e}")

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb an input and check the direction
    # ------------------------------------------------------------------
    def test_monotonicity_photon_number_increases_with_volume(self) -> Tuple[bool, str]:
        """
        If we increase R by 1%, volume increases by ~3%, so N_gamma should increase.
        Compare using the formula.
        """
        if self.R is None or self.n_gamma is None:
            return (False, "SKIP: R or n_gamma not available")
        # Increase R by 1%
        R_pert = self.R * 1.01
        V_orig = (4.0/3.0) * math.pi * self.R**3
        V_pert = (4.0/3.0) * math.pi * R_pert**3
        N_orig = self.n_gamma * V_orig if self.N_gamma is None else self.N_gamma
        # If N_gamma is given, use it; else compute from n_gamma.
        # For a true monotonicity check, we need a baseline. Use reported N_gamma if available.
        baseline = self.N_gamma if self.N_gamma is not None else N_orig
        if baseline is None:
            return (False, "SKIP: no baseline N_gamma")
        expected_pert = self.n_gamma * V_pert
        # Ensure expected_pert > baseline (positive derivative)
        if expected_pert <= baseline:
            return (False, f"Expected N_gamma to increase with volume: orig={baseline:.4e}, perturbed={expected_pert:.4e}")
        return (True, f"N_gamma increases with volume (baseline {baseline:.4e} < perturbed {expected_pert:.4e})")

    def test_monotonicity_photon_energy_increases_with_temperature(self) -> Tuple[bool, str]:
        """
        Check that u_gamma (and thus E_gamma) increases with T.
        Since u_gamma ∝ T^4, a 1% rise in T should give ~4% rise in energy density.
        """
        if any(x is None for x in [self.T, self.k_B, self.hbar, self.c, self.pi]):
            return (False, "SKIP: missing constants or T")
        # Compute baseline u_gamma (we'll use reported or compute)
        baseline = self.u_gamma if self.u_gamma is not None else self._compute_u_gamma()
        if baseline is None:
            return (False, "SKIP: cannot get baseline u_gamma")
        T_pert = self.T * 1.01
        u_pert = self._compute_u_gamma()  # T from instance is self.T, so recompute with new T? We need to temporarily change self.T.
        # Actually it's easier to compute perturbed u manually:
        coeff = (self.pi**2) / 15.0
        u_pert_val = coeff * (self.k_B * T_pert)**4 / (self.hbar**3 * self.c**3)
        if u_pert_val <= baseline:
            return (False, f"Expected u_gamma to increase with T: baseline={baseline:.4e}, perturbed={u_pert_val:.4e}")
        # Check approximate scaling: ratio should be ~ (1.01)^4 ≈ 1.0406
        ratio = u_pert_val / baseline
        if not math.isclose(ratio, 1.0406, rel_tol=1e-2):
            return (False, f"Scaling ratio {ratio:.4f} not close to expected 1.0406")
        return (True, f"u_gamma increases with T (baseline {baseline:.4e} < perturbed {u_pert_val:.4e})")

    def test_monotonicity_fermi_momentum_increases_with_electron_number(self) -> Tuple[bool, str]:
        """
        Check that p_F increases with N_e for fixed volume.
        """
        if any(x is None for x in [self.N_e, self.V, self.hbar, self.pi]):
            return (False, "SKIP: missing N_e, V, or constants")
        baseline_p = self.p_F if self.p_F is not None else self._compute_p_F()
        if baseline_p is None:
            return (False, "SKIP: cannot get baseline p_F")
        # Increase N_e by 10%
        N_e_pert = self.N_e * 1.10
        n_e_pert = N_e_pert / self.V
        p_pert = self.hbar * (3.0 * self.pi**2 * n_e_pert)**(1.0/3.0)
        if p_pert <= baseline_p:
            return (False, f"Expected p_F to increase with N_e: baseline={baseline_p:.4e}, perturbed={p_pert:.4e}")
        # Ratio: n_e increases by 1.1, so p_F increases by (1.1)^(1/3) ≈ 1.0323
        expected_ratio = (1.10)**(1.0/3.0)
        actual_ratio = p_pert / baseline_p
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            return (False, f"Scaling ratio {actual_ratio:.4f} not close to expected {expected_ratio:.4f}")
        return (True, f"p_F increases with N_e (baseline {baseline_p:.4e} < perturbed {p_pert:.4e})")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, use broad constraints)
    # ------------------------------------------------------------------
    def test_answer_sanity_magnitude_orders(self) -> Tuple[bool, str]:
        """
        Weak magnitude checks: N_gamma ~1e87, E_gamma ~1e72 erg, p_F ~2e-26 g·cm/s.
        """
        checks = []
        if self.N_gamma is not None:
            if not (1e86 < self.N_gamma < 1e88):
                checks.append(f"N_gamma={self.N_gamma:.2e} out of plausible range [1e86,1e88]")
        else:
            checks.append("N_gamma missing")
        if self.E_gamma is not None:
            if not (1e71 < self.E_gamma < 1e73):
                checks.append(f"E_gamma={self.E_gamma:.2e} out of plausible range [1e71,1e73]")
        else:
            checks.append("E_gamma missing")
        if self.p_F is not None:
            if not (1e-27 < self.p_F < 1e-25):
                checks.append(f"p_F={self.p_F:.2e} out of plausible range [1e-27,1e-25]")
        else:
            checks.append("p_F missing")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All answers within expected orders of magnitude")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
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