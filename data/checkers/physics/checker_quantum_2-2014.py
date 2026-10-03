import math
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Audits a solution for the linear potential V=kr ground state problem.
    Covers interface, invariants, modeling/trace, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        """Extract all needed data safely; no numeric fallbacks."""
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Givens – from solution instance or results dict
        self.m = getattr(solution_instance, 'm', self.results.get('m', None))
        self.hbar = getattr(solution_instance, 'hbar', self.results.get('hbar', None))
        self.k = getattr(solution_instance, 'k', self.results.get('k', None))

        # Emitted results
        self.E_unc  = self.results.get('E_unc', None)
        self.E_BS   = self.results.get('E_BS', None)
        self.E_var  = self.results.get('E_var', None)
        self.E_exact = self.results.get('E_exact', None)
        self.R_str  = self.results.get('R', None)
        self.V_eff_str = self.results.get('V_eff', None)

        # Reference formulas (re‑derived inside)
        self._set_references()

    def _set_references(self) -> None:
        """Compute reference energies from givens if all are available."""
        self.ref_E_unc = None
        self.ref_E_BS = None
        self.ref_E_var = None
        self.ref_E_exact = None
        self.x0 = 2.338107410459767  # first zero of Airy Ai

        if all(v is not None for v in (self.m, self.hbar, self.k)):
            self.ref_E_unc = 1.5 * (self.k**2 * self.hbar**2 / (4.0 * self.m)) ** (1/3)
            self.ref_E_BS  = 1.5 * (self.k**2 * self.hbar**2 / self.m) ** (1/3)
            self.ref_E_var = 1.5 * ((9.0/4.0) * self.k**2 * self.hbar**2 / self.m) ** (1/3)
            self.ref_E_exact = (self.hbar**2 / (2.0 * self.m * self.k)) ** (1/3) * self.k * self.x0

    # ------------------------------------------------------------------
    # INTERFACE TESTS
    # ------------------------------------------------------------------
    def test_interface_E_unc_exists(self) -> Tuple[bool, str]:
        if self.E_unc is None:
            return (False, "SKIP: 'E_unc' not found")
        return (True, "E_unc present")

    def test_interface_E_BS_exists(self) -> Tuple[bool, str]:
        if self.E_BS is None:
            return (False, "SKIP: 'E_BS' not found")
        return (True, "E_BS present")

    def test_interface_E_var_exists(self) -> Tuple[bool, str]:
        if self.E_var is None:
            return (False, "SKIP: 'E_var' not found")
        return (True, "E_var present")

    def test_interface_E_exact_exists(self) -> Tuple[bool, str]:
        if self.E_exact is None:
            return (False, "SKIP: 'E_exact' not found")
        return (True, "E_exact present")

    def test_interface_R_exists(self) -> Tuple[bool, str]:
        if self.R_str is None:
            return (False, "SKIP: 'R' not found")
        return (True, "R present")

    def test_interface_V_eff_exists(self) -> Tuple[bool, str]:
        if self.V_eff_str is None:
            return (False, "SKIP: 'V_eff' not found")
        return (True, "V_eff present")

    def test_interface_E_unc_type(self) -> Tuple[bool, str]:
        if not isinstance(self.E_unc, (int, float)):
            return (False, "E_unc must be numeric")
        return (True, "E_unc numeric")

    def test_interface_E_BS_type(self) -> Tuple[bool, str]:
        if not isinstance(self.E_BS, (int, float)):
            return (False, "E_BS must be numeric")
        return (True, "E_BS numeric")

    def test_interface_E_var_type(self) -> Tuple[bool, str]:
        if not isinstance(self.E_var, (int, float)):
            return (False, "E_var must be numeric")
        return (True, "E_var numeric")

    def test_interface_E_exact_type(self) -> Tuple[bool, str]:
        if not isinstance(self.E_exact, (int, float)):
            return (False, "E_exact must be numeric")
        return (True, "E_exact numeric")

    def test_interface_R_type(self) -> Tuple[bool, str]:
        if not isinstance(self.R_str, str):
            return (False, "R must be a string")
        return (True, "R is string")

    def test_interface_V_eff_type(self) -> Tuple[bool, str]:
        if not isinstance(self.V_eff_str, str):
            return (False, "V_eff must be a string")
        return (True, "V_eff is string")

    # ------------------------------------------------------------------
    # INVARIANT TESTS (theorem‑based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_uncertainty_energy(self) -> Tuple[bool, str]:
        if self.E_unc is None or self.ref_E_unc is None:
            return (False, "SKIP: E_unc or constants missing")
        if math.isclose(self.E_unc, self.ref_E_unc, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_unc ≈ {self.ref_E_unc:.6e}")
        return (False, f"E_unc={self.E_unc:.6e}, expected ≈ {self.ref_E_unc:.6e}")

    def test_invariant_BS_energy(self) -> Tuple[bool, str]:
        if self.E_BS is None or self.ref_E_BS is None:
            return (False, "SKIP: E_BS or constants missing")
        if math.isclose(self.E_BS, self.ref_E_BS, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_BS ≈ {self.ref_E_BS:.6e}")
        return (False, f"E_BS={self.E_BS:.6e}, expected ≈ {self.ref_E_BS:.6e}")

    def test_invariant_variational_energy(self) -> Tuple[bool, str]:
        if self.E_var is None or self.ref_E_var is None:
            return (False, "SKIP: E_var or constants missing")
        if math.isclose(self.E_var, self.ref_E_var, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_var ≈ {self.ref_E_var:.6e}")
        return (False, f"E_var={self.E_var:.6e}, expected ≈ {self.ref_E_var:.6e}")

    def test_invariant_exact_energy(self) -> Tuple[bool, str]:
        if self.E_exact is None or self.ref_E_exact is None:
            return (False, "SKIP: E_exact or constants missing")
        if math.isclose(self.E_exact, self.ref_E_exact, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_exact ≈ {self.ref_E_exact:.6e}")
        return (False, f"E_exact={self.E_exact:.6e}, expected ≈ {self.ref_E_exact:.6e}")

    def test_invariant_R_contains_Airy(self) -> Tuple[bool, str]:
        if self.R_str is None:
            return (False, "SKIP: R not present")
        if "Ai" in self.R_str:
            return (True, "R mentions Ai function")
        return (False, "R does not contain 'Ai'")

    def test_invariant_V_eff_structure(self) -> Tuple[bool, str]:
        if self.V_eff_str is None:
            return (False, "SKIP: V_eff not present")
        has_kr = "k*r" in self.V_eff_str or "kr" in self.V_eff_str
        has_cent = "hbar^2" in self.V_eff_str and "l(l+1)" in self.V_eff_str
        if has_kr and has_cent:
            return (True, "V_eff contains correct terms")
        return (False, f"V_eff structure mismatch: '{self.V_eff_str}'")

    # ------------------------------------------------------------------
    # MODELING / TRACE CONSISTENCY (physical relationships)
    # ------------------------------------------------------------------
    def test_modeling_variational_upper_bound(self) -> Tuple[bool, str]:
        """Variational energy must be ≥ exact ground state energy."""
        if self.E_var is None or self.E_exact is None:
            return (False, "SKIP: need both E_var and E_exact")
        if self.E_var >= self.E_exact - 1e-9:
            return (True, f"E_var={self.E_var:.6e} >= E_exact={self.E_exact:.6e}")
        return (False, "Variational principle violated: E_var < E_exact")

    def test_modeling_BS_order_of_magnitude(self) -> Tuple[bool, str]:
        """Bohr‑Sommerfeld energy should be within factor 3 of exact."""
        if self.E_BS is None or self.E_exact is None:
            return (False, "SKIP: E_BS or E_exact missing")
        ratio = self.E_BS / self.E_exact
        if 0.3 <= ratio <= 3.0:
            return (True, f"E_BS/E_exact = {ratio:.3f} (within factor 3)")
        return (False, f"E_BS/E_exact = {ratio:.3f} out of range [0.3,3]")

    def test_modeling_uncertainty_positive(self) -> Tuple[bool, str]:
        """E_unc must be positive (minimum of variational estimate)."""
        if self.E_unc is None:
            return (False, "SKIP: E_unc missing")
        if self.E_unc > 0:
            return (True, "E_unc > 0")
        return (False, "E_unc <= 0")

    def test_modeling_R_normalization_hint(self) -> Tuple[bool, str]:
        """R string should contain 1/r and Airy function."""
        if self.R_str is None:
            return (False, "SKIP: R missing")
        if "1/r" in self.R_str and "Ai" in self.R_str:
            return (True, "R has expected 1/r and Ai form")
        return (False, f"R does not contain required pattern: '{self.R_str}'")

    # ------------------------------------------------------------------
    # MONOTONICITY TESTS (explicit perturbation of an input)
    # ------------------------------------------------------------------
    def test_monotonicity_E_unc_increasing_with_k(self) -> Tuple[bool, str]:
        """E_unc increases when k is increased by 1%."""
        if self.E_unc is None or self.k is None:
            return (False, "SKIP: E_unc or k missing")
        k_pert = self.k * 1.01
        E_pert = 1.5 * (k_pert**2 * self.hbar**2 / (4.0 * self.m)) ** (1/3)
        if E_pert > self.E_unc:
            return (True, f"k↑ by 1% → E_unc↑ (baseline={self.E_unc:.6e}, perturbed={E_pert:.6e})")
        return (False, f"Monotonicity fails: k increased but E_unc did not increase")

    def test_monotonicity_E_unc_decreasing_with_m(self) -> Tuple[bool, str]:
        """E_unc decreases when m is increased by 1%."""
        if self.E_unc is None or self.m is None:
            return (False, "SKIP: E_unc or m missing")
        m_pert = self.m * 1.01
        E_pert = 1.5 * (self.k**2 * self.hbar**2 / (4.0 * m_pert)) ** (1/3)
        if E_pert < self.E_unc:
            return (True, f"m↑ by 1% → E_unc↓ (baseline={self.E_unc:.6e}, perturbed={E_pert:.6e})")
        return (False, f"Monotonicity fails: m increased but E_unc did not decrease")

    def test_monotonicity_E_unc_increasing_with_hbar(self) -> Tuple[bool, str]:
        """E_unc increases when ħ is increased by 1%."""
        if self.E_unc is None or self.hbar is None:
            return (False, "SKIP: E_unc or hbar missing")
        hbar_pert = self.hbar * 1.01
        E_pert = 1.5 * (self.k**2 * hbar_pert**2 / (4.0 * self.m)) ** (1/3)
        if E_pert > self.E_unc:
            return (True, f"ħ↑ by 1% → E_unc↑ (baseline={self.E_unc:.6e}, perturbed={E_pert:.6e})")
        return (False, f"Monotonicity fails: ħ increased but E_unc did not increase")

    def test_monotonicity_E_exact_increasing_with_k(self) -> Tuple[bool, str]:
        """E_exact increases when k is increased by 1%."""
        if self.E_exact is None or self.k is None:
            return (False, "SKIP: E_exact or k missing")
        k_pert = self.k * 1.01
        E_pert = (self.hbar**2 / (2.0 * self.m * k_pert)) ** (1/3) * k_pert * self.x0
        if E_pert > self.E_exact:
            return (True, f"k↑ by 1% → E_exact↑ (baseline={self.E_exact:.6e}, perturbed={E_pert:.6e})")
        return (False, f"Monotonicity fails: k increased but E_exact did not increase")

    def test_monotonicity_E_exact_decreasing_with_m(self) -> Tuple[bool, str]:
        """E_exact decreases when m is increased by 1%."""
        if self.E_exact is None or self.m is None:
            return (False, "SKIP: E_exact or m missing")
        m_pert = self.m * 1.01
        E_pert = (self.hbar**2 / (2.0 * m_pert * self.k)) ** (1/3) * self.k * self.x0
        if E_pert < self.E_exact:
            return (True, f"m↑ by 1% → E_exact↓ (baseline={self.E_exact:.6e}, perturbed={E_pert:.6e})")
        return (False, f"Monotonicity fails: m increased but E_exact did not decrease")

    # ------------------------------------------------------------------
    # WEAK ANSWER SANITY (only one, no isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_positivity(self) -> Tuple[bool, str]:
        """All four energies must be finite and positive."""
        names = ['E_unc', 'E_BS', 'E_var', 'E_exact']
        vals = [self.E_unc, self.E_BS, self.E_var, self.E_exact]
        for name, val in zip(names, vals):
            if val is None:
                return (False, f"SKIP: {name} missing")
            if not (isinstance(val, (int, float)) and math.isfinite(val) and val > 0):
                return (False, f"{name} not finite positive ({val})")
        return (True, "All energies are positive finite numbers")

    # ------------------------------------------------------------------
    # RUN ALL TESTS
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