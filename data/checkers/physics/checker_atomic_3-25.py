import sympy as sp
import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    """
    Audits solutions for the problem:
    Determine allowed/forbidden reactions and compute σ(pp→π⁺d)/σ(np→π⁰d).
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely; default to empty dict
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract the key entries with None defaults (NO numeric fallbacks)
        self.sigma_pp_prop = self.results.get('sigma_pp_prop', None)
        self.sigma_np_prop = self.results.get('sigma_np_prop', None)
        self.ratio = self.results.get('ratio', None)

    # ------------------------------------------------------------------
    # Interface checks: required result keys exist and have correct types
    # ------------------------------------------------------------------
    def test_interface_sigma_pp_prop(self) -> Tuple[bool, str]:
        """Check that sigma_pp_prop is present and numeric."""
        if self.sigma_pp_prop is None:
            return False, "sigma_pp_prop missing or None"
        if not isinstance(self.sigma_pp_prop, (int, float)):
            return False, f"sigma_pp_prop type {type(self.sigma_pp_prop)} not numeric"
        return True, "sigma_pp_prop present and numeric"

    def test_interface_sigma_np_prop(self) -> Tuple[bool, str]:
        """Check that sigma_np_prop is present and numeric."""
        if self.sigma_np_prop is None:
            return False, "sigma_np_prop missing or None"
        if not isinstance(self.sigma_np_prop, (int, float)):
            return False, f"sigma_np_prop type {type(self.sigma_np_prop)} not numeric"
        return True, "sigma_np_prop present and numeric"

    def test_interface_ratio(self) -> Tuple[bool, str]:
        """Check that ratio is present and numeric."""
        if self.ratio is None:
            return False, "ratio missing or None"
        if not isinstance(self.ratio, (int, float)):
            return False, f"ratio type {type(self.ratio)} not numeric"
        return True, "ratio present and numeric"

    # ------------------------------------------------------------------
    # Modeling / trace consistency checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_modeling_ratio_from_props(self) -> Tuple[bool, str]:
        """Check that ratio equals sigma_pp_prop / sigma_np_prop (within tolerance)."""
        if any(x is None for x in (self.sigma_pp_prop, self.sigma_np_prop, self.ratio)):
            return False, "SKIP: missing one or more required values"
        if math.isclose(self.sigma_np_prop, 0.0, abs_tol=1e-12):
            return False, "sigma_np_prop is zero, cannot compute ratio"
        expected_ratio = self.sigma_pp_prop / self.sigma_np_prop
        if math.isclose(self.ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"ratio ({self.ratio}) ≈ {expected_ratio} OK"
        else:
            return False, f"ratio ({self.ratio}) differs from {expected_ratio}"

    def test_modeling_amplitude_squared_ratio(self) -> Tuple[bool, str]:
        """
        Verify that sigma_pp_prop / sigma_np_prop ≈ 2, as required by physics.
        This directly checks the theoretical result from isospin.
        """
        if any(x is None for x in (self.sigma_pp_prop, self.sigma_np_prop)):
            return False, "SKIP: sigma_pp_prop or sigma_np_prop missing"
        if math.isclose(self.sigma_np_prop, 0.0, abs_tol=1e-12):
            return False, "sigma_np_prop is zero"
        computed_ratio = self.sigma_pp_prop / self.sigma_np_prop
        if math.isclose(computed_ratio, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"sigma_pp/sigma_np ratio ≈ 2 (got {computed_ratio})"
        else:
            return False, f"sigma_pp/sigma_np ratio = {computed_ratio}, expected ~2"

    def test_modeling_sigma_pp_positive(self) -> Tuple[bool, str]:
        """sigma_pp_prop must be positive (>0)."""
        if self.sigma_pp_prop is None:
            return False, "SKIP: sigma_pp_prop missing"
        if self.sigma_pp_prop > 0:
            return True, "sigma_pp_prop > 0"
        else:
            return False, f"sigma_pp_prop = {self.sigma_pp_prop} not positive"

    def test_modeling_sigma_np_positive(self) -> Tuple[bool, str]:
        """sigma_np_prop must be positive (>0)."""
        if self.sigma_np_prop is None:
            return False, "SKIP: sigma_np_prop missing"
        if self.sigma_np_prop > 0:
            return True, "sigma_np_prop > 0"
        else:
            return False, f"sigma_np_prop = {self.sigma_np_prop} not positive"

    # ------------------------------------------------------------------
    # Theorem-based invariant tests (at least two)
    # ------------------------------------------------------------------
    def test_invariant_ratio_under_common_scaling(self) -> Tuple[bool, str]:
        """
        The ratio should be invariant under multiplying both cross-section 
        proportionality constants by the same positive factor.
        We simulate by scaling both variables by a factor of 2 and checking 
        that the ratio remains unchanged.
        """
        if any(x is None for x in (self.sigma_pp_prop, self.sigma_np_prop)):
            return False, "SKIP: sigma_pp_prop or sigma_np_prop missing"
        if math.isclose(self.sigma_np_prop, 0.0, abs_tol=1e-12):
            return False, "sigma_np_prop is zero"
        original_ratio = self.sigma_pp_prop / self.sigma_np_prop
        scale_factor = 2.0
        scaled_pp = self.sigma_pp_prop * scale_factor
        scaled_np = self.sigma_np_prop * scale_factor
        scaled_ratio = scaled_pp / scaled_np
        if math.isclose(original_ratio, scaled_ratio, rel_tol=1e-12, abs_tol=1e-12):
            return True, f"Ratio invariant under scaling (original {original_ratio}, scaled {scaled_ratio})"
        else:
            return False, f"Ratio changed after scaling: {original_ratio} vs {scaled_ratio}"

    def test_invariant_isospin_amplitude_relation(self) -> Tuple[bool, str]:
        """
        Theorem: The amplitude for pp -> π⁺d is M (I=1 only); for np -> π⁰d it is M/√2.
        Therefore sigma_pp_prop = |M|², sigma_np_prop = |M|²/2.
        Invariant: sigma_pp_prop * 0.5 should equal sigma_np_prop (within tolerance).
        """
        if any(x is None for x in (self.sigma_pp_prop, self.sigma_np_prop)):
            return False, "SKIP: sigma_pp_prop or sigma_np_prop missing"
        if math.isclose(self.sigma_np_prop, 0.0, abs_tol=1e-12):
            return False, "sigma_np_prop is zero"
        # From isospin theorem: sigma_np_prop = sigma_pp_prop / 2
        expected_np = self.sigma_pp_prop / 2.0
        if math.isclose(expected_np, self.sigma_np_prop, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"sigma_np_prop ({self.sigma_np_prop}) ≈ sigma_pp_prop/2 ({expected_np}) satisfies isospin"
        else:
            return False, f"sigma_np_prop ({self.sigma_np_prop}) ≠ sigma_pp_prop/2 ({expected_np}); isospin violated"

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb one cross-section and compare ratio
    # ------------------------------------------------------------------
    def test_monotonicity_ratio_increases_with_sigma_pp(self) -> Tuple[bool, str]:
        """
        If sigma_pp_prop increases by 10% (while sigma_np_prop unchanged), 
        the ratio should increase.
        """
        if any(x is None for x in (self.sigma_pp_prop, self.sigma_np_prop)):
            return False, "SKIP: needed values missing"
        if math.isclose(self.sigma_np_prop, 0.0, abs_tol=1e-12):
            return False, "sigma_np_prop is zero"
        baseline_ratio = self.sigma_pp_prop / self.sigma_np_prop
        # 10% increase in sigma_pp
        perturbed_pp = self.sigma_pp_prop * 1.1
        new_ratio = perturbed_pp / self.sigma_np_prop
        if new_ratio > baseline_ratio:
            return True, f"Ratio increased from {baseline_ratio} to {new_ratio} (+10% sigma_pp)"
        else:
            return False, f"Ratio did NOT increase: baseline {baseline_ratio}, new {new_ratio}"

    def test_monotonicity_ratio_decreases_with_sigma_np(self) -> Tuple[bool, str]:
        """
        If sigma_np_prop increases by 10% (while sigma_pp_prop unchanged), 
        the ratio should decrease.
        """
        if any(x is None for x in (self.sigma_pp_prop, self.sigma_np_prop)):
            return False, "SKIP: needed values missing"
        if math.isclose(self.sigma_np_prop, 0.0, abs_tol=1e-12):
            return False, "sigma_np_prop is zero"
        baseline_ratio = self.sigma_pp_prop / self.sigma_np_prop
        # 10% increase in sigma_np
        perturbed_np = self.sigma_np_prop * 1.1
        new_ratio = self.sigma_pp_prop / perturbed_np
        if new_ratio < baseline_ratio:
            return True, f"Ratio decreased from {baseline_ratio} to {new_ratio} (+10% sigma_np)"
        else:
            return False, f"Ratio did NOT decrease: baseline {baseline_ratio}, new {new_ratio}"

    # ------------------------------------------------------------------
    # Weak answer sanity check (no re-derivation, no math.isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_ratio(self) -> Tuple[bool, str]:
        """Ratio should be a finite positive number (loose bounds)."""
        if self.ratio is None:
            return False, "SKIP: ratio missing"
        if not isinstance(self.ratio, (int, float)):
            return False, "ratio not numeric"
        if math.isfinite(self.ratio) and self.ratio > 0:
            return True, f"ratio = {self.ratio} is finite and positive"
        else:
            return False, f"ratio = {self.ratio} not finite/positive"

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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs