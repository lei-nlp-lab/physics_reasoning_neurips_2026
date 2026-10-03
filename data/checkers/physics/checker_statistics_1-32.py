import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    PhysicsChecker for a Carnot cycle problem.
    Checks interface completeness, invariant relations, modeling consistency,
    monotonicity (where possible), and a weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all relevant data from the solution_instance.
        Use safe defaults (None) for missing values.
        """
        # Store the solution instance for auxiliary data access
        self.solution = solution_instance

        # Main results (expected from [emit] contract)
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        self.Qh = self.results.get('Qh', None)
        self.Qc = self.results.get('Qc', None)
        self.eta = self.results.get('eta', None)

        # Additional values: try results, then direct attribute, then givens dict
        self.Th = self._find_value('Th')
        self.Tc = self._find_value('Tc')
        self.n = self._find_value('n')
        self.R = self._find_value('R')
        self.Va = self._find_value('Va')
        self.Vb = self._find_value('Vb')
        self.Vc = self._find_value('Vc')
        self.Vd = self._find_value('Vd')

        # Derived volume ratio (if available)
        self.ratio_V = self._find_value('ratio_V')
        if self.ratio_V is None:
            if self.Vb is not None and self.Va is not None:
                try:
                    self.ratio_V = self.Vb / self.Va
                except:
                    pass
            elif self.Vc is not None and self.Vd is not None:
                try:
                    self.ratio_V = self.Vc / self.Vd
                except:
                    pass

        # Store a flag for gamma (not required, but used in adiabatic relation)
        self.gamma = self._find_value('gamma')  # may be given or not

    def _find_value(self, key: str) -> Any:
        """
        Search for a value in order: results dict, direct attribute, givens dict.
        Returns None if not found.
        """
        # 1. Try results
        val = self.results.get(key)
        if val is not None:
            return val

        # 2. Try direct attribute
        val = getattr(self.solution, key, None)
        if val is not None:
            return val

        # 3. Try givens dict (may exist)
        givens = getattr(self.solution, 'givens', None)
        if isinstance(givens, dict):
            val = givens.get(key)
            if val is not None:
                return val

        return None

    # ------------------------------------------------------------------
    # Helper: check if two floats are close (for invariant/modeling tests)
    # ------------------------------------------------------------------
    def _isclose(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # ------------------------------------------------------------------
    # Interface tests: required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_Qh_Qc_eta_present(self) -> Tuple[bool, str]:
        """Check that Qh, Qc, and eta are not None."""
        if self.Qh is None:
            return (False, "Missing result: Qh")
        if self.Qc is None:
            return (False, "Missing result: Qc")
        if self.eta is None:
            return (False, "Missing result: eta")
        return (True, "All required results (Qh, Qc, eta) are present")

    def test_interface_required_types(self) -> Tuple[bool, str]:
        """Check that Qh, Qc, eta are numeric (int or float) and not NaN."""
        for name, val in [('Qh', self.Qh), ('Qc', self.Qc), ('eta', self.eta)]:
            if val is None:
                continue  # will be caught by interface test above
            if not isinstance(val, (int, float)):
                return (False, f"{name} has invalid type {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                return (False, f"{name} is NaN or Inf")
        return (True, "Result types are numeric and finite")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_efficiency_formula(self) -> Tuple[bool, str]:
        """
        Check that eta = 1 - Qc/Qh (assuming Qc and Qh positive magnitudes).
        If Qh is missing, skip.
        """
        if self.Qh is None or self.Qc is None or self.eta is None:
            return (False, "SKIP: Qh, Qc, or eta missing")
        if self.Qh <= 0:
            return (False, f"Qh should be positive, got {self.Qh}")
        if self.Qc <= 0:
            return (False, f"Qc should be positive, got {self.Qc}")
        computed_eta = 1.0 - self.Qc / self.Qh
        if self._isclose(computed_eta, self.eta):
            return (True, f"eta = 1 - Qc/Qh holds (computed {computed_eta:.6f}, given {self.eta:.6f})")
        else:
            return (False, f"eta mismatch: expected {computed_eta:.6f}, got {self.eta:.6f}")

    def test_invariant_efficiency_range(self) -> Tuple[bool, str]:
        """
        Carnot efficiency must be strictly between 0 and 1 (exclusive).
        (Reversible cycle with Th > Tc > 0)
        """
        if self.eta is None:
            return (False, "SKIP: eta missing")
        if not (0.0 < self.eta < 1.0):
            return (False, f"Efficiency outside (0,1): {self.eta}")
        return (True, f"Efficiency {self.eta:.6f} lies in (0,1)")

    def test_invariant_carnot_limit(self) -> Tuple[bool, str]:
        """
        If both Th and Tc are available, check eta = 1 - Tc/Th.
        This is the Carnot formula derivation.
        """
        if self.Th is None or self.Tc is None or self.eta is None:
            return (False, "SKIP: Th, Tc, or eta missing")
        if self.Th <= self.Tc:
            return (False, f"Th must be > Tc, got Th={self.Th}, Tc={self.Tc}")
        expected_eta = 1.0 - self.Tc / self.Th
        if self._isclose(expected_eta, self.eta):
            return (True, f"eta = 1 - Tc/Th holds (expected {expected_eta:.6f}, given {self.eta:.6f})")
        else:
            return (False, f"Carnot limit mismatch: expected {expected_eta:.6f}, got {self.eta:.6f}")

    def test_invariant_heat_positive(self) -> Tuple[bool, str]:
        """Both Qh and Qc must be positive (magnitudes)."""
        if self.Qh is None or self.Qc is None:
            return (False, "SKIP: Qh or Qc missing")
        if self.Qh <= 0:
            return (False, f"Qh not positive: {self.Qh}")
        if self.Qc <= 0:
            return (False, f"Qc not positive: {self.Qc}")
        return (True, f"Qh={self.Qh:.6f}, Qc={self.Qc:.6f} both positive")

    # ------------------------------------------------------------------
    # Modeling / trace tests: check internal consistency of derived quantities
    # ------------------------------------------------------------------
    def test_modeling_isothermal_formula(self) -> Tuple[bool, str]:
        """
        If n, R, Th, and volume ratio are available,
        check that Qh ≈ n*R*Th*ln(ratio_V) (isothermal expansion).
        """
        if any(v is None for v in [self.n, self.R, self.Th, self.ratio_V, self.Qh]):
            return (False, "SKIP: insufficient data for isothermal check")
        computed_Qh = self.n * self.R * self.Th * math.log(self.ratio_V)
        if computed_Qh <= 0:
            return (False, f"Isothermal formula gives non-positive Qh: {computed_Qh}")
        if self._isclose(computed_Qh, self.Qh):
            return (True, f"Isothermal Qh matches formula (computed {computed_Qh:.6f}, given {self.Qh:.6f})")
        else:
            return (False, f"Isothermal Qh mismatch: computed {computed_Qh:.6f}, given {self.Qh:.6f}")

    def test_modeling_adiabatic_volume_ratio(self) -> Tuple[bool, str]:
        """
        The volume ratio at isothermal expansion (Vb/Va) and at isothermal
        compression (Vc/Vd) should be equal (from adiabatic relations).
        """
        ratio1 = None
        ratio2 = None
        if self.Vb is not None and self.Va is not None:
            try:
                ratio1 = self.Vb / self.Va
            except:
                pass
        if self.Vc is not None and self.Vd is not None:
            try:
                ratio2 = self.Vc / self.Vd
            except:
                pass
        if ratio1 is None and ratio2 is None:
            return (False, "SKIP: no volume ratio data")
        # If we have the stored ratio_V from results, use that as reference
        if self.ratio_V is not None:
            if ratio1 is not None:
                if not self._isclose(ratio1, self.ratio_V):
                    return (False, f"Vb/Va ({ratio1:.6f}) ≠ ratio_V ({self.ratio_V:.6f})")
            if ratio2 is not None:
                if not self._isclose(ratio2, self.ratio_V):
                    return (False, f"Vc/Vd ({ratio2:.6f}) ≠ ratio_V ({self.ratio_V:.6f})")
            return (True, f"Volume ratios consistent with ratio_V={self.ratio_V:.6f}")
        else:
            # Compare the two ratios directly if both exist
            if ratio1 is not None and ratio2 is not None:
                if self._isclose(ratio1, ratio2):
                    return (True, f"Vb/Va ({ratio1:.6f}) ≈ Vc/Vd ({ratio2:.6f})")
                else:
                    return (False, f"Volume ratios differ: Vb/Va={ratio1:.6f}, Vc/Vd={ratio2:.6f}")
            else:
                return (False, "SKIP: insufficient data to compare ratios")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of an input parameter)
    # ------------------------------------------------------------------
    def test_monotonicity_efficiency_increases_with_Th(self) -> Tuple[bool, str]:
        """
        For fixed Tc, increasing Th should increase efficiency.
        Use a 1% increase in Th and verify the Carnot formula gives higher eta.
        Requires Th, Tc.
        """
        if self.Th is None or self.Tc is None:
            return (False, "SKIP: Th or Tc missing")
        if self.Th <= self.Tc:
            return (False, "Th ≤ Tc, cannot test monotonicity of efficiency")
        # Baseline efficiency from formula
        eta0 = 1.0 - self.Tc / self.Th
        # Perturb Th up by 1%
        Th_new = self.Th * 1.01
        eta1 = 1.0 - self.Tc / Th_new
        if eta1 > eta0:
            return (True, f"Increasing Th from {self.Th} to {Th_new:.6f} increases eta ({eta0:.6f} -> {eta1:.6f})")
        else:
            return (False, f"Monotonicity violation: eta(Th) should increase but eta1={eta1:.6f} ≤ eta0={eta0:.6f}")

    def test_monotonicity_efficiency_decreases_with_Tc(self) -> Tuple[bool, str]:
        """
        For fixed Th, increasing Tc should decrease efficiency.
        Use a 1% increase in Tc and verify the Carnot formula gives lower eta.
        """
        if self.Th is None or self.Tc is None:
            return (False, "SKIP: Th or Tc missing")
        if self.Th <= self.Tc:
            return (False, "Th ≤ Tc, cannot test monotonicity of efficiency")
        eta0 = 1.0 - self.Tc / self.Th
        # Perturb Tc up by 1%
        Tc_new = self.Tc * 1.01
        eta1 = 1.0 - Tc_new / self.Th
        if eta1 < eta0:
            return (True, f"Increasing Tc from {self.Tc} to {Tc_new:.6f} decreases eta ({eta0:.6f} -> {eta1:.6f})")
        else:
            return (False, f"Monotonicity violation: eta(Tc) should decrease but eta1={eta1:.6f} ≥ eta0={eta0:.6f}")

    # ------------------------------------------------------------------
    # Answer sanity (weak – only sign, range, finiteness)
    # ------------------------------------------------------------------
    def test_answer_sanity_efficiency_sign_and_bound(self) -> Tuple[bool, str]:
        """
        Very weak check: eta must be a finite number between 0 and 1 (exclusive).
        Do NOT use math.isclose here.
        """
        if self.eta is None:
            return (False, "SKIP: eta missing")
        if not isinstance(self.eta, (int, float)):
            return (False, f"eta is not numeric: {type(self.eta)}")
        if math.isnan(self.eta) or math.isinf(self.eta):
            return (False, f"eta is NaN or Inf: {self.eta}")
        if self.eta <= 0.0 or self.eta >= 1.0:
            return (False, f"eta out of (0,1) range: {self.eta}")
        return (True, f"eta = {self.eta} is a finite number in (0,1)")

    # ------------------------------------------------------------------
    # Main runner – discovers all test_ methods, executes them, collects results
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass = True
        # Iterate over all bound methods starting with "test_"
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}"
                traceback.print_exc()  # for debugging; can be removed
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)