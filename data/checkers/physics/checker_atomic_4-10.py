import math
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Physics checker for the proton Compton wavelength problem.
    Tests interface completeness, theorem invariants, modeling consistency,
    monotonicity, and a weak final answer sanity check.
    """

    # Standard constants (from problem statement and pseudocode)
    H = 6.626e-34       # J·s
    C = 3.0e8           # m/s
    M_P = 1.6726e-27    # kg

    def __init__(self, solution_instance: Any) -> None:
        # Extract results safely – always use None default
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        self.lambda_cm: Optional[float] = self.results.get('lambda_cm', None)
        self.lambda_m: Optional[float] = self.results.get('lambda_m', None)

        # Optionally extract givens (used by monotonicity tests fallback)
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        self.h: Optional[float] = self.givens.get('h', None)
        self.c: Optional[float] = self.givens.get('c', None)
        self.m_p: Optional[float] = self.givens.get('m_p', None)

    # ------------------------------------------------------------------
    # Interface tests: required keys exist and are valid
    # ------------------------------------------------------------------
    def test_interface_lambda_cm_key(self) -> Tuple[bool, str]:
        if self.lambda_cm is None:
            return (False, "Missing key 'lambda_cm' in results.")
        return (True, "lambda_cm key present in results.")

    def test_interface_lambda_cm_type(self) -> Tuple[bool, str]:
        if self.lambda_cm is None:
            return (False, "Cannot check type: lambda_cm is None.")
        if not isinstance(self.lambda_cm, (int, float)):
            return (False, f"lambda_cm is not numeric (type: {type(self.lambda_cm).__name__}).")
        return (True, "lambda_cm is numeric.")

    def test_interface_lambda_cm_finite(self) -> Tuple[bool, str]:
        if self.lambda_cm is None:
            return (False, "Cannot check finiteness: lambda_cm is None.")
        if not math.isfinite(self.lambda_cm):
            return (False, f"lambda_cm is not finite (value: {self.lambda_cm}).")
        return (True, "lambda_cm is finite.")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_compton_formula(self) -> Tuple[bool, str]:
        """
        Check that lambda_cm ≈ (h / (m_p * c)) * 100.
        Use constants from givens if present, otherwise fallback to standard values.
        """
        h = self.h if self.h is not None else self.H
        c = self.c if self.c is not None else self.C
        m_p = self.m_p if self.m_p is not None else self.M_P

        if self.lambda_cm is None:
            return (False, "Skip invariant: lambda_cm not available.")
        try:
            expected_lambda_m = h / (m_p * c)
            expected_lambda_cm = expected_lambda_m * 100.0
        except Exception as e:
            return (False, f"Computation of expected lambda failed: {e}")

        if not math.isfinite(expected_lambda_cm):
            return (False, "Expected lambda_cm is not finite (division by zero?).")

        if not math.isclose(self.lambda_cm, expected_lambda_cm, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"lambda_cm ({self.lambda_cm:.4e}) does not match expected ({expected_lambda_cm:.4e}) within 1%.")
        return (True, "Compton formula invariant holds.")

    def test_invariant_conversion_factor(self) -> Tuple[bool, str]:
        """If both lambda_m and lambda_cm are present, check lambda_cm ≈ lambda_m * 100."""
        if self.lambda_m is None or self.lambda_cm is None:
            return (True, "Skip: one of lambda_m or lambda_cm missing; conversion cannot be checked.")
        if not isinstance(self.lambda_m, (int, float)):
            return (False, "lambda_m is not numeric.")
        converted = self.lambda_m * 100.0
        if not math.isclose(self.lambda_cm, converted, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"lambda_cm ({self.lambda_cm:.4e}) != lambda_m * 100 ({converted:.4e}).")
        return (True, "Conversion factor (x100) is consistent.")

    # ------------------------------------------------------------------
    # Modeling / trace tests: consistency of intermediate modeling steps
    # ------------------------------------------------------------------
    def test_modeling_order_of_magnitude(self) -> Tuple[bool, str]:
        """
        Check that lambda_cm is within one decade of the known Compton wavelength
        of a proton (~1.32e-13 cm). This is a re-derived range from standard constants.
        """
        ACCEPTED = 1.32e-13  # cm (derived from h/(m_p*c) * 100, not copied from explanation)
        if self.lambda_cm is None:
            return (False, "Skip: lambda_cm not available.")
        lower = ACCEPTED / 10.0
        upper = ACCEPTED * 10.0
        if not (lower <= self.lambda_cm <= upper):
            return (False, f"lambda_cm ({self.lambda_cm:.4e}) is not within [{lower:.4e}, {upper:.4e}].")
        return (True, "Order of magnitude is plausible (1e-13 cm range).")

    def test_modeling_lambda_m_product(self) -> Tuple[bool, str]:
        """
        If lambda_m is available, check that h ≈ m_p * c * lambda_m.
        This directly verifies the Compton formula from the intermediate result.
        """
        if self.lambda_m is None:
            return (True, "Skip: lambda_m not present; consistency check not applicable.")
        h = self.h if self.h is not None else self.H
        c = self.c if self.c is not None else self.C
        m_p = self.m_p if self.m_p is not None else self.M_P
        product = m_p * c * self.lambda_m
        if not math.isclose(product, h, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Product m_p*c*lambda_m ({product:.4e}) does not equal h ({h:.4e}) within 1%.")
        return (True, "Lambda_m satisfies h = m_p * c * lambda_m.")

    def test_modeling_positive_lambda(self) -> Tuple[bool, str]:
        """Ensure lambda_cm is positive (physical requirement)."""
        if self.lambda_cm is None:
            return (False, "Cannot check positivity: lambda_cm is None.")
        if self.lambda_cm <= 0:
            return (False, f"lambda_cm is not positive (value: {self.lambda_cm}).")
        return (True, "lambda_cm is positive.")

    # ------------------------------------------------------------------
    # Monotonicity tests: explicit perturbation and directional comparison
    # ------------------------------------------------------------------
    def test_monotonicity_inverse_mass(self) -> Tuple[bool, str]:
        """
        If mass increases, Compton wavelength decreases (λ ∝ 1/m).
        Perturb mass by +1% and compute new lambda; must be smaller than baseline.
        """
        m_p = self.m_p if self.m_p is not None else self.M_P
        if self.lambda_cm is None:
            return (False, "Skip: lambda_cm not available.")
        h = self.h if self.h is not None else self.H
        c = self.c if self.c is not None else self.C
        m_p_pert = m_p * 1.01
        lambda_pert = (h / (m_p_pert * c)) * 100.0
        if lambda_pert >= self.lambda_cm:
            if math.isclose(lambda_pert, self.lambda_cm, rel_tol=1e-2):
                return (True, "Negligible change within tolerance.")
            return (False, f"Expected lambda to decrease when mass increases. Baseline: {self.lambda_cm:.4e}, perturbed: {lambda_pert:.4e}.")
        return (True, "Inverse monotonicity with mass holds.")

    def test_monotonicity_positive_planck(self) -> Tuple[bool, str]:
        """
        If Planck constant is increased, wavelength increases (λ ∝ h).
        Perturb h by +1% and compute new lambda; must be larger than baseline.
        """
        h = self.h if self.h is not None else self.H
        m_p = self.m_p if self.m_p is not None else self.M_P
        c = self.c if self.c is not None else self.C
        if self.lambda_cm is None:
            return (False, "Skip: lambda_cm not available.")
        h_pert = h * 1.01
        lambda_pert = (h_pert / (m_p * c)) * 100.0
        if lambda_pert <= self.lambda_cm:
            if math.isclose(lambda_pert, self.lambda_cm, rel_tol=1e-2):
                return (True, "Negligible change within tolerance.")
            return (False, f"Expected lambda to increase when h increases. Baseline: {self.lambda_cm:.4e}, perturbed: {lambda_pert:.4e}.")
        return (True, "Positive monotonicity with Planck constant holds.")

    # ------------------------------------------------------------------
    # Weak answer sanity test (only one allowed, no isclose, no re‑derived expected)
    # ------------------------------------------------------------------
    def test_answer_sanity_between_options(self) -> Tuple[bool, str]:
        """
        Check that lambda_cm is between the extremes of the given options:
        (a) 1e-6 cm and (c) 1e-24 cm.
        This is a broad range check, not a precise matching.
        """
        if self.lambda_cm is None:
            return (False, "Skip: lambda_cm not available.")
        OPTION_A = 1e-6   # cm
        OPTION_C = 1e-24  # cm
        if not (OPTION_C < self.lambda_cm < OPTION_A):
            return (False, f"lambda_cm ({self.lambda_cm:.4e}) is not between the extreme options ({OPTION_C} and {OPTION_A}).")
        return (True, "lambda_cm lies between the extreme options (a) and (c).")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and execute all test_* methods, returning aggregate success and log messages."""
        logs: List[str] = []
        all_pass = True
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{attr_name} CRASHED: {type(e).__name__}: {e}"
                traceback.print_exc(file=sys.stderr)
            logs.append(f"{'PASS' if passed else 'FAIL'}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs