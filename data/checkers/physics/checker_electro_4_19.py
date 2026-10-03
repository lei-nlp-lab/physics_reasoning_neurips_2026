import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    """
    Auditor for the reflection of linearly polarized light off a birefringent material.
    Checks interface, invariants, modeling, monotonicity, and answer sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract used inputs
        self.n_R = getattr(solution_instance, 'n_R', None)
        self.n_L = getattr(solution_instance, 'n_L', None)
        self.E0 = getattr(solution_instance, 'E0', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.c = getattr(solution_instance, 'c', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)

        # Safely extract results
        results = getattr(solution_instance, 'results', {}) or {}
        self.I_r_ratio = results.get('I_r_ratio', None)
        self.polarization = results.get('polarization', None)

        # Derived quantities (computed on the fly when needed)
        self._r_R = None
        self._r_L = None
        if self.n_R is not None and self.n_L is not None:
            try:
                self._r_R = (1.0 - self.n_R) / (1.0 + self.n_R)
                self._r_L = (1.0 - self.n_L) / (1.0 + self.n_L)
            except (ZeroDivisionError, TypeError):
                pass

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_I_r_ratio_key(self) -> Tuple[bool, str]:
        if self.I_r_ratio is None:
            return (False, "Missing key 'I_r_ratio' in results")
        return (True, "Key 'I_r_ratio' present")

    def test_interface_polarization_key(self) -> Tuple[bool, str]:
        if self.polarization is None:
            return (False, "Missing key 'polarization' in results")
        return (True, "Key 'polarization' present")

    def test_interface_polarization_type(self) -> Tuple[bool, str]:
        if self.polarization is None:
            return (False, "SKIP: polarization not available")
        if not isinstance(self.polarization, str):
            return (False, f"polarization should be a string, got {type(self.polarization)}")
        return (True, "polarization is a string")

    def test_interface_I_r_ratio_type(self) -> Tuple[bool, str]:
        if self.I_r_ratio is None:
            return (False, "SKIP: I_r_ratio not available")
        if not isinstance(self.I_r_ratio, (int, float)):
            return (False, f"I_r_ratio should be numeric, got {type(self.I_r_ratio)}")
        return (True, "I_r_ratio is numeric")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use isclose)
    # ------------------------------------------------------------------
    def test_invariant_reflection_coefficient_r_R(self) -> Tuple[bool, str]:
        if self._r_R is None:
            return (False, "SKIP: n_R missing, cannot compute r_R")
        # Not exported by solution, skip
        return (True, "SKIP: solution does not export r_R")

    def test_invariant_I_r_ratio_formula(self) -> Tuple[bool, str]:
        if self._r_R is None or self._r_L is None:
            return (False, "SKIP: n_R/n_L missing, cannot compute expected ratio")
        if self.I_r_ratio is None:
            return (False, "SKIP: I_r_ratio not available")
        expected = (self._r_R**2 + self._r_L**2) / 2.0
        if not math.isclose(self.I_r_ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"I_r_ratio {self.I_r_ratio} != expected {expected} "
                    f"(r_R={self._r_R:.4f}, r_L={self._r_L:.4f})")
        return (True, f"I_r_ratio = {self.I_r_ratio:.4f} matches expected")

    def test_invariant_reflected_intensity_nonnegative(self) -> Tuple[bool, str]:
        if self.I_r_ratio is None:
            return (False, "SKIP: I_r_ratio not available")
        if self.I_r_ratio < 0:
            return (False, f"I_r_ratio negative: {self.I_r_ratio}")
        return (True, "I_r_ratio is non-negative")

    def test_invariant_I_r_ratio_finite(self) -> Tuple[bool, str]:
        if self.I_r_ratio is None:
            return (False, "SKIP: I_r_ratio not available")
        if not math.isfinite(self.I_r_ratio):
            return (False, f"I_r_ratio is not finite: {self.I_r_ratio}")
        return (True, "I_r_ratio is finite")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_polarization_elliptical(self) -> Tuple[bool, str]:
        """
        If n_R != n_L, the reflected polarization should be elliptical.
        Check that the complex ratio i*(r_L - r_R)/(r_R + r_L) has magnitude != 1
        (unless r_R + r_L = 0, which gives linear polarisation, but that's fine).
        """
        if self._r_R is None or self._r_L is None:
            return (False, "SKIP: cannot compute ratio")
        if self.polarization is None:
            return (False, "SKIP: polarization not available")
        # Compute the complex ratio (without c=1 for simplicity)
        denom = self._r_R + self._r_L
        if abs(denom) < 1e-12:
            # Then E_x = 0 -> linear along y, but that's still linear, not elliptical.
            # Accept if solution says linear, but problem expects elliptical.
            # Since n_R != n_L usually, skip.
            return (True, "SKIP: degenerate case r_R + r_L = 0")
        ratio_imag = (self._r_L - self._r_R) / denom  # i times this gives E_y/E_x
        magnitude = abs(ratio_imag)  # ratio = i * ratio_imag, magnitude = |ratio_imag|
        # If magnitude ≈ 1, it could be circular; else elliptical.
        if abs(magnitude - 1.0) < 0.01:
            # Could be circular; then polarization label should be 'circular'
            # But problem expects elliptical unless n_R = n_L
            return (True, "SKIP: borderline case magnitude close to 1")
        pol_lower = self.polarization.lower().replace('-', '').replace('_', '').replace(' ', '')
        if 'elliptical' not in pol_lower:
            return (False,
                    f"Polarization should be 'elliptical', but got '{self.polarization}'")
        return (True, "Polarization correctly identified as elliptical")

    def test_modeling_polarization_ratio_consistency(self) -> Tuple[bool, str]:
        """
        If the solution provides E_xr and E_yr (not in contract), we could check the ratio.
        Since not required, skip.
        """
        return (True, "SKIP: solution does not export field components")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_I_r_ratio_decreases_with_n_R(self) -> Tuple[bool, str]:
        """
        For fixed n_L, increasing n_R should decrease |r_R| and therefore decrease I_r_ratio.
        """
        if self.n_R is None or self.n_L is None or self.I_r_ratio is None:
            return (False, "SKIP: missing n_R, n_L, or I_r_ratio")
        try:
            n_R_base = float(self.n_R)
            n_L_base = float(self.n_L)
        except (TypeError, ValueError):
            return (False, "SKIP: n_R/n_L not convertible to float")

        # Perturb n_R upward by 10%
        n_R_perturbed = n_R_base * 1.10
        r_R_pert = (1.0 - n_R_perturbed) / (1.0 + n_R_perturbed)
        r_L_orig = (1.0 - n_L_base) / (1.0 + n_L_base)
        new_ratio = (r_R_pert**2 + r_L_orig**2) / 2.0

        if new_ratio > self.I_r_ratio + 1e-9:
            return (False,
                    f"Increasing n_R from {n_R_base:.4f} to {n_R_perturbed:.4f} "
                    f"should decrease I_r_ratio, but it increased from {self.I_r_ratio:.4f} "
                    f"to {new_ratio:.4f}")
        return (True,
                f"Monotonicity holds: I_r_ratio decreased when n_R increased")

    # ------------------------------------------------------------------
    # Answer sanity tests (weak, no isclose, no re-derivation; at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic_range(self) -> Tuple[bool, str]:
        """
        Weak sanity: check that I_r_ratio is in [0,1] and polarization string is non-empty.
        No hardcoded formula or re-derivation.
        """
        if self.I_r_ratio is None and self.polarization is None:
            return (False, "SKIP: both results missing")
        # Check I_r_ratio if available
        if self.I_r_ratio is not None:
            if self.I_r_ratio < 0.0 or self.I_r_ratio > 1.0:
                return (False, f"I_r_ratio out of [0,1] range: {self.I_r_ratio}")
        # Check polarization if available
        if self.polarization is not None:
            if len(self.polarization.strip()) == 0:
                return (False, "polarization string is empty")
        return (True, "Answer-sanity checks passed")

    # ------------------------------------------------------------------
    # Master test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_* methods, run them, collect logs.
        Returns (all_pass, logs_list).
        """
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
                import traceback
                tb = traceback.format_exc()
                passed, msg = False, f"{name} CRASHED: {e}\nTraceback:\n{tb}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)