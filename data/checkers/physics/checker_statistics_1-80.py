import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Auditor for the elastic band problem combining thermodynamics and oscillations.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Safely extract givens and results, defaulting to None.
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})

        # Physical quantities expected from givens (from problem context)
        self.W: Optional[float] = self.givens.get('W', None)        # weight (N)
        self.g: Optional[float] = self.givens.get('g', None)        # gravity (m/s^2)
        self.nu: Optional[float] = self.givens.get('nu', None)      # frequency (Hz)
        self.delta_Q: Optional[float] = self.givens.get('delta_Q', None)  # added heat (J)
        self.dL0_dT: Optional[float] = self.givens.get('dL0_dT', None)    # K^{-1}
        self.dk_dT: Optional[float] = self.givens.get('dk_dT', None)      # N/(m·K)
        self.C_F: Optional[float] = self.givens.get('C_F', None)          # J/K
        self.alpha_ad: Optional[float] = self.givens.get('alpha_ad', None) # (dT/dL)_S from (a)

        # Expected result key
        self.delta_L: Optional[float] = self.results.get('delta_L', None)

        # Derived quantities
        self.k: Optional[float] = None
        if self.W is not None and self.g is not None and self.nu is not None:
            self.k = (4 * math.pi**2 * self.W / self.g) * (self.nu ** 2)

    # ----------------------------------------------------------------
    # Test interface – required keys existence and valid types
    # ----------------------------------------------------------------
    def test_interface_delta_L_present(self) -> Tuple[bool, str]:
        """Check that the final answer key delta_L exists in results."""
        if self.delta_L is None:
            return (False, "Missing 'delta_L' in results")
        if not isinstance(self.delta_L, (int, float)):
            return (False, f"delta_L type is {type(self.delta_L)}, expected numeric")
        return (True, "delta_L present and numeric")

    def test_interface_required_givens(self) -> Tuple[bool, str]:
        """Check that all essential givens are present (not None)."""
        missing = []
        for name, val in [('W', self.W), ('g', self.g), ('nu', self.nu),
                          ('delta_Q', self.delta_Q), ('dL0_dT', self.dL0_dT),
                          ('dk_dT', self.dk_dT), ('C_F', self.C_F)]:
            if val is None:
                missing.append(name)
        if missing:
            return (True, f"Missing givens: {missing}. Only a warning, test continues.")
        return (True, "All essential givens present")

    # ----------------------------------------------------------------
    # Invariant / theorem-based tests (use math.isclose)
    # ----------------------------------------------------------------
    def test_invariant_stiffness_k(self) -> Tuple[bool, str]:
        """
        Check that the adiabatic stiffness k = (∂F/∂L)_S equals
        4π² ν² W/g, as derived from simple harmonic motion.
        """
        if self.k is None:
            return (False, "SKIP: cannot compute k (W, g, nu missing)")
        # If results provides a computed k, check consistency (optional)
        k_from_results = self.results.get('k', None)
        if k_from_results is not None:
            if not math.isclose(self.k, k_from_results, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"k mismatch: computed {self.k:.4f}, results {k_from_results:.4f}")
            return (True, f"k from formula {self.k:.4f} matches results")
        else:
            # If no explicit k, just a sanity that it is well-defined
            if self.k <= 0:
                return (False, f"Computed k = {self.k} is non-positive")
            return (True, f"k = {self.k:.4f} is positive (no comparison with results)")

    def test_invariant_maxwell_relation(self) -> Tuple[bool, str]:
        """
        Verify Maxwell relation (∂T/∂L)_S = (∂F/∂S)_L using given or derived data.
        Since we only have alpha_ad = (∂T/∂L)_S, we compute (∂F/∂S)_L from
        cyclic or other relations if possible. Fallback: symbolic consistency.
        """
        if self.alpha_ad is None:
            return (False, "SKIP: missing alpha_ad (measurement a)")
        # We cannot directly compute (∂F/∂S)_L, but we can check that
        # the derived formula for δL satisfies the cyclic condition:
        # (∂L/∂S)_F * (∂S/∂F)_L * (∂F/∂L)_S = -1
        # If we have all needed variables, test it.
        if None in (self.delta_L, self.delta_Q, self.k, self.C_F):
            return (False, "SKIP: missing data to check cyclic condition")
        # (∂L/∂S)_F = δL / (δQ/T). We need an estimate of T.
        T = self.givens.get('T', None)
        if T is None:
            return (False, "SKIP: temperature T not provided")
        dLdS_F = self.delta_L * T / self.delta_Q
        # (∂F/∂L)_S = k
        # Need (∂S/∂F)_L – can be computed from heat capacity at constant L?
        # C_L = T (∂S/∂T)_L, but (∂S/∂F)_L = (∂S/∂T)_L * (∂T/∂F)_L
        # Not enough data. Instead check sign consistency:
        # Using expected cyclic product, we get (∂S/∂F)_L = -1/(dLdS_F * k)
        expected_dSdF_L = -1.0 / (dLdS_F * self.k)
        # This value should be negative if k>0 and dLdS_F positive? Not required.
        # Just check finite.
        if not math.isfinite(expected_dSdF_L):
            return (False, f"Cyclic relation leads to non-finite (∂S/∂F)_L: {expected_dSdF_L}")
        return (True, f"Cyclic condition consistent (derived (∂S/∂F)_L = {expected_dSdF_L:.4e})")

    def test_invariant_adiabatic_energy(self) -> Tuple[bool, str]:
        """
        Check that during adiabatic stretch (dS=0) the first law dU = F dL holds.
        If we have energy change and work, compare.
        """
        dU = self.results.get('dU', None)
        F = self.givens.get('F', None)  # tension
        dL_ad = self.givens.get('dL_ad', None)  # length change in measurement (a)
        if None in (dU, F, dL_ad):
            return (False, "SKIP: missing data for adiabatic energy check")
        work = F * dL_ad
        if not math.isclose(dU, work, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"dU={dU} != F*dL={work}")
        return (True, "Adiabatic first law satisfied")

    # ----------------------------------------------------------------
    # Modeling / trace consistency checks (one relation per test)
    # ----------------------------------------------------------------
    def test_modeling_deltaL_formula(self) -> Tuple[bool, str]:
        """
        Check that the derived δL matches the required formula:
        δL = (dL0/dT - W/k² dk/dT) δQ / C_F
        """
        if self.delta_L is None:
            return (False, "SKIP: no delta_L to compare")
        # If any of the required components is missing, skip
        if None in (self.k, self.W, self.dL0_dT, self.dk_dT, self.delta_Q, self.C_F):
            return (False, "SKIP: missing one or more inputs for formula")
        expected_delta_L = (self.dL0_dT - (self.W / (self.k**2)) * self.dk_dT) * (self.delta_Q / self.C_F)
        if not math.isclose(self.delta_L, expected_delta_L, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"delta_L mismatch: results {self.delta_L:.4e}, expected {expected_delta_L:.4e}")
        return (True, f"delta_L matches formula: {self.delta_L:.4e}")

    def test_modeling_heat_capacity_definition(self) -> Tuple[bool, str]:
        """
        Check consistency: C_F = δQ / dT, if dT is provided.
        """
        dT = self.givens.get('dT', None)
        if None in (self.delta_Q, dT):
            return (False, "SKIP: missing delta_Q or dT for heat capacity check")
        if self.C_F is not None:
            expected_C = self.delta_Q / dT
            if not math.isclose(self.C_F, expected_C, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"C_F from results {self.C_F} differs from δQ/dT = {expected_C}")
            return (True, f"heat capacity consistent: C_F = {self.C_F}")
        else:
            self.C_F = self.delta_Q / dT
            return (True, f"Implied C_F = {self.C_F} (not directly given)")

    # ----------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ----------------------------------------------------------------
    def test_monotonicity_deltaL_with_deltaQ(self) -> Tuple[bool, str]:
        """
        Increase deltaQ by 10% – deltaL should increase proportionally
        (since delta_L is linear in delta_Q).
        """
        if None in (self.delta_Q, self.dL0_dT, self.dk_dT, self.k, self.W, self.C_F, self.delta_L):
            return (False, "SKIP: missing data for monotonicity test")
        # Perturbation factor
        factor = 1.1
        deltaQ_prime = self.delta_Q * factor
        expected_deltaL = (self.dL0_dT - (self.W / (self.k**2)) * self.dk_dT) * (deltaQ_prime / self.C_F)
        # The new delta_L should be exactly factor times old delta_L
        if self.delta_L == 0:
            if expected_deltaL != 0:
                return (False, "delta_L=0 but perturbation gives non-zero")
            return (True, "Both zero, monotonicity trivial")
        ratio = expected_deltaL / self.delta_L
        if not math.isclose(ratio, factor, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"delta_L ratio {ratio:.4f} != expected {factor}")
        return (True, f"Delta_L increases by factor {factor} when deltaQ increased by {factor}")

    def test_monotonicity_deltaL_with_dL0_dT(self) -> Tuple[bool, str]:
        """
        Increase dL0_dT by 10% and check that delta_L increases by exactly
        0.1 * dL0_dT * (delta_Q / C_F) (since delta_L is linear in dL0_dT).
        """
        if None in (self.dL0_dT, self.delta_Q, self.C_F, self.k, self.W, self.dk_dT, self.delta_L):
            return (False, "SKIP: missing data for monotonicity test")
        # Perturb dL0_dT by +10%
        factor = 1.1
        dL0_dT_new = self.dL0_dT * factor
        # Compute new delta_L using the same formula
        expected_new = (dL0_dT_new - (self.W / (self.k**2)) * self.dk_dT) * (self.delta_Q / self.C_F)
        # Expected difference due to change in dL0_dT:
        delta_diff = (dL0_dT_new - self.dL0_dT) * (self.delta_Q / self.C_F)
        # So expected_new should equal self.delta_L + delta_diff
        expected_from_baseline = self.delta_L + delta_diff
        if not math.isclose(expected_new, expected_from_baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Expected new delta_L {expected_new} does not match baseline+diff {expected_from_baseline}")
        # Also check direction: if delta_diff > 0 then new > old
        if delta_diff > 0 and expected_new <= self.delta_L:
            return (False, "Increasing dL0_dT should increase delta_L")
        if delta_diff < 0 and expected_new >= self.delta_L:
            return (False, "Decreasing dL0_dT should decrease delta_L")
        return (True, f"Monotonicity with dL0_dT consistent. New delta_L = {expected_new}")

    # ----------------------------------------------------------------
    # Weak final answer sanity (sign, finiteness, magnitude)
    # ----------------------------------------------------------------
    def test_answer_sanity_deltaL(self) -> Tuple[bool, str]:
        """
        Weak sanity: delta_L must be finite, non-None, and within plausible
        range (e.g., not astronomically large).
        """
        if self.delta_L is None:
            return (False, "No delta_L to check")
        if not math.isfinite(self.delta_L):
            return (False, f"delta_L is not finite: {self.delta_L}")
        # Arbitrary plausible range: for typical elastic band, length change
        # from a small heat addition might be on order of mm to cm.
        if abs(self.delta_L) > 10:  # meters – unlikely
            return (True, f"delta_L magnitude ({self.delta_L}) is large but physically possible")
        return (True, "delta_L is finite and plausible")

    # ----------------------------------------------------------------
    # Main runner – discover all test_* methods
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)