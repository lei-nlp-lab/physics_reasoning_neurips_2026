import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audit class for entropy change problem: 1 kg water heated from 0°C to 100°C.
    Focuses on thermodynamic consistency, modeling quality, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results with None defaults (safe access)
        results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}

        self.dS_water: Optional[float] = results.get('dS_water', None)
        self.dS_univ: Optional[float] = results.get('dS_univ', None)
        self.dS_univ_rev: Optional[float] = results.get('dS_univ_rev', None)

        # Problem constants (hardcoded – part of the problem definition, not fallbacks)
        self.m = 1.0           # kg
        self.cp = 4186.0       # J/(kg·K)
        self.Ti = 273.0        # K
        self.Tf = 373.0        # K
        self.Tres = 373.0      # K

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_results_present(self) -> Tuple[bool, str]:
        """Check that all three required result keys are present and numeric."""
        keys = ['dS_water', 'dS_univ', 'dS_univ_rev']
        for key in keys:
            val = getattr(self, key, None)
            if val is None:
                return (False, f"Required key '{key}' is missing or None")
            if not isinstance(val, (int, float)):
                return (False, f"Key '{key}' is not numeric (type: {type(val)})")
        return (True, "All required result keys present and numeric")

    def test_interface_water_entropy_positive(self) -> Tuple[bool, str]:
        """Sanity: dS_water must be positive and finite."""
        if self.dS_water is None:
            return (False, "SKIP: dS_water not available")
        if self.dS_water <= 0:
            return (False, f"dS_water = {self.dS_water} is not positive")
        if math.isinf(self.dS_water) or math.isnan(self.dS_water):
            return (False, f"dS_water = {self.dS_water} is not a finite number")
        return (True, "dS_water is positive and finite")

    def test_interface_universe_entropy_positive(self) -> Tuple[bool, str]:
        """Sanity: dS_univ must be positive and finite."""
        if self.dS_univ is None:
            return (False, "SKIP: dS_univ not available")
        if self.dS_univ <= 0:
            return (False, f"dS_univ = {self.dS_univ} is not positive")
        if math.isinf(self.dS_univ) or math.isnan(self.dS_univ):
            return (False, f"dS_univ = {self.dS_univ} is not a finite number")
        return (True, "dS_univ is positive and finite")

    def test_interface_reversible_entropy_zero_or_close(self) -> Tuple[bool, str]:
        """Sanity: dS_univ_rev should be 0 (or very close due to rounding)."""
        if self.dS_univ_rev is None:
            return (False, "SKIP: dS_univ_rev not available")
        if not math.isfinite(self.dS_univ_rev):
            return (False, f"dS_univ_rev = {self.dS_univ_rev} is not finite")
        if abs(self.dS_univ_rev) > 1e-2:
            return (False, f"dS_univ_rev = {self.dS_univ_rev} is not approximately zero")
        return (True, "dS_univ_rev is zero (or negligible)")

    # ------------------------------------------------------------
    # Invariant checks (theorem-based)
    # ------------------------------------------------------------
    def test_invariant_entropy_water_formula(self) -> Tuple[bool, str]:
        """
        Check that ΔS_water = m cp ln(Tf/Ti) within 2% relative/absolute tolerance.
        Uses fuzzy matching as per instruction.
        """
        if self.dS_water is None:
            return (False, "SKIP: dS_water not available")
        expected = self.m * self.cp * math.log(self.Tf / self.Ti)
        if math.isclose(self.dS_water, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dS_water = {self.dS_water:.1f} matches expected {expected:.1f}")
        else:
            return (False, f"dS_water = {self.dS_water:.1f} != expected {expected:.1f} (tolerance 2%)")

    def test_invariant_universe_entropy_calculation(self) -> Tuple[bool, str]:
        """
        Check that ΔS_univ = ΔS_water - Q/Tres, where Q = m cp (Tf - Ti).
        Uses fuzzy matching.
        """
        if self.dS_water is None or self.dS_univ is None:
            return (False, "SKIP: dS_water or dS_univ not available")
        Q = self.m * self.cp * (self.Tf - self.Ti)
        expected_univ = self.dS_water - Q / self.Tres
        if math.isclose(self.dS_univ, expected_univ, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dS_univ = {self.dS_univ:.1f} matches expected {expected_univ:.1f}")
        else:
            return (False, f"dS_univ = {self.dS_univ:.1f} != expected {expected_univ:.1f} (tolerance 2%)")

    def test_invariant_reversible_universe_zero(self) -> Tuple[bool, str]:
        """
        For a reversible process, ΔS_universe = 0. Check that dS_univ_rev is within 1 J/K of zero.
        """
        if self.dS_univ_rev is None:
            return (False, "SKIP: dS_univ_rev not available")
        if math.isclose(self.dS_univ_rev, 0.0, abs_tol=1.0):
            return (True, f"dS_univ_rev = {self.dS_univ_rev:.6f} is effectively zero")
        else:
            return (False, f"dS_univ_rev = {self.dS_univ_rev:.6f} is not close to zero")

    # ------------------------------------------------------------
    # Modeling/trace consistency checks
    # ------------------------------------------------------------
    def test_trace_entropy_via_integral(self) -> Tuple[bool, str]:
        """
        Check that dS_water can be obtained by integration: ∫ m cp dT/T from Ti to Tf.
        """
        if self.dS_water is None:
            return (False, "SKIP: dS_water not available")
        integral_value = self.m * self.cp * math.log(self.Tf / self.Ti)
        if math.isclose(self.dS_water, integral_value, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "dS_water matches integral ∫ m cp dT/T")
        else:
            return (False, f"dS_water ({self.dS_water:.1f}) differs from integral ({integral_value:.1f})")

    def test_trace_reservoir_entropy_sign(self) -> Tuple[bool, str]:
        """
        Check that ΔS_res = -Q/Tres is negative (reservoir loses entropy).
        """
        if self.dS_water is None or self.dS_univ is None:
            return (False, "SKIP: dS_water or dS_univ not available")
        Q = self.m * self.cp * (self.Tf - self.Ti)
        dS_res = -Q / self.Tres
        if dS_res >= 0:
            return (False, f"Reservoir entropy change {dS_res:.1f} is not negative (should be negative)")
        return (True, f"Reservoir entropy change is negative ({dS_res:.1f}) as expected")

    # ------------------------------------------------------------
    # Monotonicity checks
    # ------------------------------------------------------------
    def test_monotonicity_dS_water_increases_with_Tf(self) -> Tuple[bool, str]:
        """
        Perturb Tf upward by +1% and verify that the water entropy increases.
        """
        if self.dS_water is None:
            return (False, "SKIP: dS_water not available")
        delta_factor = 0.01
        perturbed_Tf = self.Tf * (1 + delta_factor)
        perturbed_dS_water = self.m * self.cp * math.log(perturbed_Tf / self.Ti)
        if perturbed_dS_water <= self.dS_water:
            return (False, f"dS_water does not increase when Tf is increased by 1% (baseline {self.dS_water:.1f}, perturbed {perturbed_dS_water:.1f})")
        return (True, f"dS_water increases from {self.dS_water:.1f} to {perturbed_dS_water:.1f} when Tf increases by 1%")

    def test_monotonicity_dS_univ_increases_with_water_entropy(self) -> Tuple[bool, str]:
        """
        If dS_water is larger (due to larger Tf), then dS_univ should also be larger.
        Check by perturbing Tf by +1% and recomputing expected dS_univ.
        """
        if self.dS_water is None or self.dS_univ is None:
            return (False, "SKIP: dS_water or dS_univ not available")
        delta_factor = 0.01
        perturbed_Tf = self.Tf * (1 + delta_factor)
        perturbed_dS_water = self.m * self.cp * math.log(perturbed_Tf / self.Ti)
        Q_perturbed = self.m * self.cp * (perturbed_Tf - self.Ti)
        perturbed_dS_univ = perturbed_dS_water - Q_perturbed / self.Tres
        if perturbed_dS_univ <= self.dS_univ:
            return (False, f"dS_univ does not increase when Tf is increased (baseline {self.dS_univ:.1f}, perturbed {perturbed_dS_univ:.1f})")
        return (True, f"dS_univ increases from {self.dS_univ:.1f} to {perturbed_dS_univ:.1f} when Tf increases by 1%")

    # ------------------------------------------------------------
    # Single weak answer sanity check (combined)
    # ------------------------------------------------------------
    def test_answer_sanity_magnitudes(self) -> Tuple[bool, str]:
        """
        Weak broad magnitude check: water entropy should be between 1000 and 2000 J/K,
        universe entropy between 0 and 500 J/K. No isclose or re-derivation.
        """
        if self.dS_water is None or self.dS_univ is None:
            return (False, "SKIP: dS_water or dS_univ not available")
        # Check water entropy range
        if not (1000 <= self.dS_water <= 2000):
            return (False, f"dS_water = {self.dS_water:.1f} J/K outside plausible range [1000,2000]")
        # Check universe entropy range
        if not (0 < self.dS_univ < 500):
            return (False, f"dS_univ = {self.dS_univ:.1f} J/K outside plausible range (0,500)")
        return (True, f"dS_water = {self.dS_water:.1f} and dS_univ = {self.dS_univ:.1f} within plausible ranges")

    # ------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound methods starting with 'test_', run them sequentially,
        collect results and messages. Returns (overall_pass, list_of_logs).
        """
        logs: List[str] = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
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