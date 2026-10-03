import math
import cmath
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Robust physics checker for coherent states of the harmonic oscillator.
    Checks interface completeness, theorem-based invariants, modeling quality,
    and monotonic trends.
    """

    def __init__(self, solution_instance: Any):
        # Safe extraction of givens (use None defaults)
        givens = getattr(solution_instance, 'givens', None) or {}
        self.m = givens.get('m', None)          # mass
        self.omega = givens.get('omega', None)  # angular frequency
        self.hbar = givens.get('hbar', None)    # reduced Planck constant
        # alpha (complex) may be given as real and imag parts or complex
        alpha = givens.get('alpha', None)
        if alpha is None:
            alpha_real = givens.get('alpha_real', None)
            alpha_imag = givens.get('alpha_imag', None)
            if alpha_real is not None and alpha_imag is not None:
                alpha = complex(alpha_real, alpha_imag)
        self.alpha = alpha   # may be None
        # time t (optional)
        self.t = givens.get('t', None)          # time for part e

        # Safe extraction of results (use None defaults)
        results = getattr(solution_instance, 'results', None) or {}
        self.a_plus_eigenfunction = results.get('a_plus_eigenfunction', None)
        self.a_minus_eigenfunction = results.get('a_minus_eigenfunction', None)
        self.x_expect = results.get('x_expect', None)
        self.x2_expect = results.get('x2_expect', None)
        self.p_expect = results.get('p_expect', None)
        self.p2_expect = results.get('p2_expect', None)
        self.sigma_x_sq = results.get('sigma_x_sq', None)
        self.sigma_p_sq = results.get('sigma_p_sq', None)
        self.sigma_x = results.get('sigma_x', None)
        self.sigma_p = results.get('sigma_p', None)
        self.product = results.get('product', None)
        self.c_n_form = results.get('c_n_form', None)
        self.c_0 = results.get('c_0', None)
        self.alpha_t = results.get('alpha_t', None)
        self.ground_eigenvalue = results.get('ground_eigenvalue', None)

        # Optional: explanation_steps list (for modeling checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []

    # ----------------------------------------------------------------
    # Interface tests (keys exist and have correct types)
    # ----------------------------------------------------------------
    def test_interface_a_plus_eigenfunction(self) -> Tuple[bool, str]:
        if self.a_plus_eigenfunction is None:
            return (False, "SKIP: a_plus_eigenfunction not provided")
        if not isinstance(self.a_plus_eigenfunction, bool):
            return (False, "a_plus_eigenfunction must be bool")
        if self.a_plus_eigenfunction != False:
            return (False, "a_plus_eigenfunction should be False (no normalizable eigenfunctions)")
        return (True, "a_plus_eigenfunction correctly False")

    def test_interface_a_minus_eigenfunction(self) -> Tuple[bool, str]:
        if self.a_minus_eigenfunction is None:
            return (False, "SKIP: a_minus_eigenfunction not provided")
        if not isinstance(self.a_minus_eigenfunction, bool):
            return (False, "a_minus_eigenfunction must be bool")
        if self.a_minus_eigenfunction != True:
            return (False, "a_minus_eigenfunction should be True (has eigenfunctions)")
        return (True, "a_minus_eigenfunction correctly True")

    def test_interface_expectation_types(self) -> Tuple[bool, str]:
        """Check all expectation values are numeric (complex or float)."""
        needed = {
            'x_expect': self.x_expect,
            'x2_expect': self.x2_expect,
            'p_expect': self.p_expect,
            'p2_expect': self.p2_expect,
        }
        for name, val in needed.items():
            if val is None:
                return (False, f"SKIP: {name} not provided")
            if not isinstance(val, (int, float, complex)):
                return (False, f"{name} must be numeric, got {type(val)}")
        return (True, "Expectation values have valid types")

    def test_interface_sigma_types(self) -> Tuple[bool, str]:
        needed = {
            'sigma_x_sq': self.sigma_x_sq,
            'sigma_p_sq': self.sigma_p_sq,
            'sigma_x': self.sigma_x,
            'sigma_p': self.sigma_p,
            'product': self.product,
        }
        for name, val in needed.items():
            if val is None:
                return (False, f"SKIP: {name} not provided")
            if not isinstance(val, (int, float, complex)):
                return (False, f"{name} must be numeric, got {type(val)}")
        return (True, "Uncertainty quantities have valid types")

    def test_interface_c_n_form(self) -> Tuple[bool, str]:
        if self.c_n_form is None:
            return (False, "SKIP: c_n_form not provided")
        if not isinstance(self.c_n_form, str):
            return (False, "c_n_form must be a string")
        # Check it contains expected pattern
        if "alpha" not in self.c_n_form or "sqrt" not in self.c_n_form:
            return (False, "c_n_form does not look correct")
        return (True, "c_n_form looks reasonable")

    def test_interface_c_0(self) -> Tuple[bool, str]:
        if self.c_0 is None:
            return (False, "SKIP: c_0 not provided")
        if not isinstance(self.c_0, (int, float)):
            return (False, "c_0 must be a real number")
        if self.c_0 < 0:
            return (False, "c_0 should be positive (conventional phase)")
        return (True, "c_0 positive and numeric")

    def test_interface_alpha_t(self) -> Tuple[bool, str]:
        if self.alpha_t is None:
            return (False, "SKIP: alpha_t not provided")
        if not isinstance(self.alpha_t, (int, float, complex)):
            return (False, "alpha_t must be complex")
        return (True, "alpha_t has valid type")

    def test_interface_ground_eigenvalue(self) -> Tuple[bool, str]:
        if self.ground_eigenvalue is None:
            return (False, "SKIP: ground_eigenvalue not provided")
        if not isinstance(self.ground_eigenvalue, (int, float)):
            return (False, "ground_eigenvalue must be numeric")
        if self.ground_eigenvalue != 0:
            return (False, "ground_eigenvalue should be 0")
        return (True, "ground_eigenvalue correctly 0")

    # ----------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity, use math.isclose)
    # ----------------------------------------------------------------
    def test_invariant_uncertainty_product(self) -> Tuple[bool, str]:
        """
        Check sigma_x * sigma_p = hbar/2 (the uncertainty bound).
        This test uses isclose with 1e-2 tolerance for floating noise.
        """
        if self.sigma_x is None or self.sigma_p is None or self.hbar is None:
            return (False, "SKIP: sigma_x, sigma_p, or hbar missing")
        expected = self.hbar / 2.0
        computed_product = self.sigma_x * self.sigma_p
        if not math.isclose(computed_product, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma_x*sigma_p = {computed_product}, expected {expected}")
        return (True, f"Uncertainty product saturates bound: {computed_product} ~ {expected}")

    def test_invariant_sigma_x_sq_form(self) -> Tuple[bool, str]:
        """
        Check that sigma_x^2 = hbar/(2 m omega).
        """
        if self.sigma_x_sq is None or self.m is None or self.omega is None or self.hbar is None:
            return (False, "SKIP: sigma_x_sq, m, omega, or hbar missing")
        expected = self.hbar / (2.0 * self.m * self.omega)
        if not math.isclose(self.sigma_x_sq, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma_x^2 = {self.sigma_x_sq}, expected {expected}")
        return (True, f"sigma_x^2 correct: {self.sigma_x_sq}")

    def test_invariant_sigma_p_sq_form(self) -> Tuple[bool, str]:
        """
        Check that sigma_p^2 = hbar m omega / 2.
        """
        if self.sigma_p_sq is None or self.m is None or self.omega is None or self.hbar is None:
            return (False, "SKIP: sigma_p_sq, m, omega, or hbar missing")
        expected = self.hbar * self.m * self.omega / 2.0
        if not math.isclose(self.sigma_p_sq, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma_p^2 = {self.sigma_p_sq}, expected {expected}")
        return (True, f"sigma_p^2 correct: {self.sigma_p_sq}")

    def test_invariant_alpha_t_form(self) -> Tuple[bool, str]:
        """
        Check that alpha_t = alpha * exp(-i omega t)  (if t and alpha provided).
        """
        if self.alpha_t is None or self.alpha is None or self.omega is None or self.t is None:
            return (False, "SKIP: alpha_t, alpha, omega, or t missing")
        expected = self.alpha * cmath.exp(-1j * self.omega * self.t)
        # Compare complex with isclose on real and imag parts
        if not math.isclose(self.alpha_t.real, expected.real, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"alpha_t real part {self.alpha_t.real} != {expected.real}")
        if not math.isclose(self.alpha_t.imag, expected.imag, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"alpha_t imag part {self.alpha_t.imag} != {expected.imag}")
        return (True, f"alpha_t = {self.alpha_t} matches expected {expected}")

    def test_invariant_c_0_normalization(self) -> Tuple[bool, str]:
        """
        Check that |c_0|^2 = exp(-|alpha|^2) (the normalization condition).
        """
        if self.c_0 is None or self.alpha is None:
            return (False, "SKIP: c_0 or alpha missing")
        expected = math.exp(-abs(self.alpha)**2)
        if not math.isclose(self.c_0**2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"|c_0|^2 = {self.c_0**2}, expected {expected}")
        # Also check sign (c_0 should be positive real)
        if self.c_0 < 0:
            return (False, "c_0 is negative, conventionally positive")
        return (True, f"c_0 normalization satisfied: |c_0|^2 = {self.c_0**2} ~ {expected}")

    def test_invariant_ground_eigenvalue_zero(self) -> Tuple[bool, str]:
        """Check ground_eigenvalue is exactly 0 (integer)."""
        if self.ground_eigenvalue is None:
            return (False, "SKIP: ground_eigenvalue not provided")
        if self.ground_eigenvalue != 0:
            return (False, f"Ground eigenvalue should be 0, got {self.ground_eigenvalue}")
        return (True, "Ground eigenvalue correctly 0")

    # ----------------------------------------------------------------
    # Modeling / trace checks (checking specific relations from explanation_steps)
    # ----------------------------------------------------------------
    def test_modeling_expectation_relation(self) -> Tuple[bool, str]:
        """
        Check that <x^2> - <x>^2 = sigma_x^2 (variance relation).
        """
        if any(v is None for v in [self.x_expect, self.x2_expect, self.sigma_x_sq]):
            return (False, "SKIP: need x_expect, x2_expect, sigma_x_sq")
        # Compute variance
        variance = self.x2_expect - self.x_expect**2
        if not math.isclose(variance, self.sigma_x_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Var(x) = {variance}, but sigma_x^2 = {self.sigma_x_sq}")
        return (True, "Variance matches sigma_x^2")

    def test_modeling_p_expectation_relation(self) -> Tuple[bool, str]:
        """
        Check that <p^2> - <p>^2 = sigma_p^2.
        """
        if any(v is None for v in [self.p_expect, self.p2_expect, self.sigma_p_sq]):
            return (False, "SKIP: need p_expect, p2_expect, sigma_p_sq")
        variance = self.p2_expect - self.p_expect**2
        if not math.isclose(variance, self.sigma_p_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Var(p) = {variance}, but sigma_p^2 = {self.sigma_p_sq}")
        return (True, "Variance matches sigma_p^2")

    def test_modeling_x_expect_alpha(self) -> Tuple[bool, str]:
        """
        Check that <x> = sqrt(hbar/(2mω))*(α+α*), if α provided.
        This is a direct modeling check against the theoretical formula.
        """
        if any(v is None for v in [self.x_expect, self.alpha, self.m, self.omega, self.hbar]):
            return (False, "SKIP: missing parameters for x expectation")
        alpha_conj = self.alpha.conjugate()
        expected = math.sqrt(self.hbar/(2.0*self.m*self.omega)) * (self.alpha + alpha_conj)
        if not math.isclose(self.x_expect.real, expected.real, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"<x> real part {self.x_expect.real} != {expected.real}")
        if not math.isclose(self.x_expect.imag, expected.imag, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"<x> imag part {self.x_expect.imag} != {expected.imag}")
        return (True, "<x> matches theoretical expression")

    def test_modeling_p_expect_alpha(self) -> Tuple[bool, str]:
        """
        Check that <p> = -i sqrt(hbar m ω/2)*(α - α*).
        """
        if any(v is None for v in [self.p_expect, self.alpha, self.m, self.omega, self.hbar]):
            return (False, "SKIP: missing parameters for p expectation")
        alpha_conj = self.alpha.conjugate()
        factor = -1j * math.sqrt(self.hbar * self.m * self.omega / 2.0)
        expected = factor * (self.alpha - alpha_conj)
        if not math.isclose(self.p_expect.real, expected.real, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"<p> real part {self.p_expect.real} != {expected.real}")
        if not math.isclose(self.p_expect.imag, expected.imag, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"<p> imag part {self.p_expect.imag} != {expected.imag}")
        return (True, "<p> matches theoretical expression")

    def test_modeling_c0_from_cn_form(self) -> Tuple[bool, str]:
        """
        Weak check: if c_n_form is present and c_0, ensure they are consistent.
        This is more a trace check: the form should imply c_0 = exp(-|α|^2/2).
        """
        if self.c_n_form is None or self.c_0 is None or self.alpha is None:
            return (False, "SKIP: need c_n_form, c_0, alpha")
        # Check c_n_form contains "alpha^n" and "sqrt(n!)"
        if "alpha^n" in self.c_n_form and "sqrt(n!)" in self.c_n_form:
            # Derive expected c_0 from form and normalization:
            # |c_0|^2 = e^{-|α|^2} -> c_0 = e^{-|α|^2/2} (positive real)
            expected = math.exp(-0.5 * abs(self.alpha)**2)
            if not math.isclose(self.c_0, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"c_0 = {self.c_0}, expected {expected} from normalization")
            return (True, "c_0 consistent with c_n_form and normalization")
        else:
            return (True, "c_n_form present but not checked in detail")

    # ----------------------------------------------------------------
    # Monotonicity test (explicit perturbation)
    # ----------------------------------------------------------------
    def test_monotonicity_alpha_magnitude_normalization(self) -> Tuple[bool, str]:
        """
        Perturbation: increase |α| by 1% and check that |c_0| decreases
        (since c_0 = e^{-|α|^2/2}). This is a monotonic trend check.
        """
        if self.c_0 is None or self.alpha is None:
            return (False, "SKIP: missing c_0 or alpha")
        alpha_mag = abs(self.alpha)
        if alpha_mag == 0:
            return (True, "SKIP: alpha=0, no meaningful monotonicity")
        # Baseline: c_0(alpha)
        c0_baseline = self.c_0
        # Perturb: increase |α| by 1%
        new_mag = alpha_mag * 1.01
        # Direction: scale alpha by factor 1.01
        # theoretical new c_0 = exp(-|new_α|^2/2)
        new_c0 = math.exp(-new_mag**2/2)
        # Check that new_c0 is less than baseline
        if new_c0 >= c0_baseline:
            return (False, f"Monotonicity violated: increasing |α| should decrease |c_0|, but {new_c0} >= {c0_baseline}")
        return (True, "Monotonic trend verified: larger |α| gives smaller |c_0|")

    # ----------------------------------------------------------------
    # Weak answer sanity (only signs/finite/broad magnitude) - at most one
    # ----------------------------------------------------------------
    def test_answer_sanity_basic(self) -> Tuple[bool, str]:
        """
        Combined sanity: check all numerical outputs are finite and variances nonnegative.
        This is a single weak sanity test.
        """
        numerical_keys = [
            self.x_expect, self.x2_expect, self.p_expect, self.p2_expect,
            self.sigma_x_sq, self.sigma_p_sq, self.sigma_x, self.sigma_p,
            self.product, self.c_0
        ]
        for idx, val in enumerate(numerical_keys):
            if val is None:
                continue
            if not math.isfinite(val.real if isinstance(val, complex) else val):
                return (False, f"Non-finite value found at index {idx}: {val}")
        # Check variances (sigma_x_sq, sigma_p_sq) are nonnegative, if real
        for name, val in [('sigma_x_sq', self.sigma_x_sq), ('sigma_p_sq', self.sigma_p_sq)]:
            if val is None:
                continue
            if isinstance(val, complex):
                continue
            if val < 0:
                return (False, f"{name} is negative: {val}")
        return (True, "All numerical values finite and variances nonnegative")

    # ----------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_* methods and run them one by one.
        Returns (all_pass, logs).
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)