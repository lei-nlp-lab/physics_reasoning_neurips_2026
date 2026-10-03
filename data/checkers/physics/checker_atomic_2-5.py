import math
from typing import Dict, Any, Tuple, List

class PhysicsChecker:
    """
    Audits a solution instance for the problem:
    "The precession frequency of a nucleus in the magnetic field of the earth
     is $10^{-1}, 10^1, 10^3, 10^5 \text{ sec}^{-1}.$"
    Checks Larmor precession consistency, modeling quality, and monotonicity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Safely extract results dictionary
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        # Safely extract givens if they exist
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        if not isinstance(self.givens, dict):
            self.givens = {}

        # Extract required/optional quantities with safe defaults (None)
        self.f = self.get_result('f', None)
        self.gamma_p = self.get_given_or_result('gamma_p', None)
        self.B_earth = self.get_given_or_result('B_earth', None)
        self.omega = self.get_result('omega', None)  # optional

        # Explanation steps (for reference only, never hardcode numbers)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def get_result(self, key: str, default: Any = None) -> Any:
        return self.results.get(key, default)

    def get_given_or_result(self, key: str, default: Any = None) -> Any:
        """Try givens, then results, then None."""
        if isinstance(self.givens, dict) and key in self.givens:
            return self.givens[key]
        if isinstance(self.results, dict) and key in self.results:
            return self.results[key]
        return default

    def is_physical(self, x: Any) -> bool:
        """Check if x is a finite positive number."""
        if x is None:
            return False
        try:
            v = float(x)
            return math.isfinite(v) and v > 0
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_result_key_f(self) -> Tuple[bool, str]:
        """Check that the result contains key 'f'."""
        if self.f is None:
            return (False, "Missing result key 'f'")
        return (True, "Found result key 'f'")

    def test_interface_f_is_number(self) -> Tuple[bool, str]:
        """Check that f is numeric and positive."""
        if not self.is_physical(self.f):
            return (False, f"f = {self.f} is not a finite positive number")
        return (True, f"f = {self.f} is a finite positive number")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_larmor_formula(self) -> Tuple[bool, str]:
        """
        Verify the Larmor precession relation: f = (gamma/(2*pi)) * B
        if both gamma_p and B_earth are provided.
        """
        gamma = self.gamma_p
        B = self.B_earth
        if gamma is None or B is None:
            return (False, "SKIP: gamma_p and/or B_earth not available")
        try:
            gamma_f = float(gamma)
            B_f = float(B)
        except (TypeError, ValueError):
            return (False, "gamma_p or B_earth not convertible to float")
        if not math.isfinite(gamma_f) or not math.isfinite(B_f):
            return (False, "gamma_p or B_earth is non-finite")

        expected_f = gamma_f / (2 * math.pi) * B_f
        if not self.is_physical(expected_f):
            return (False, f"Computed expected frequency {expected_f} is not physical")
        if not self.is_physical(self.f):
            return (False, f"Given f = {self.f} is not physical")

        if math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"f = {self.f} matches Larmor formula: ~{expected_f:.4e}")
        else:
            return (False, f"f = {self.f} deviates from Larmor formula {expected_f:.4e}")

    def test_invariant_angular_frequency_consistency(self) -> Tuple[bool, str]:
        """
        If angular frequency omega is provided, check f = omega/(2*pi).
        """
        if self.omega is None or self.f is None:
            return (False, "SKIP: omega or f not both available")
        try:
            omega_f = float(self.omega)
            f_f = float(self.f)
        except (TypeError, ValueError):
            return (False, "Non-numeric omega or f")
        if not math.isfinite(omega_f) or not math.isfinite(f_f):
            return (False, "Non-finite omega or f")
        expected_f = omega_f / (2 * math.pi)
        if math.isclose(f_f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"f = {f_f} matches omega/(2*pi) = {expected_f:.4e}")
        else:
            return (False, f"f = {f_f} inconsistent with omega/(2*pi) = {expected_f:.4e}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (check modeling choices, no hardcoded constants)
    # ------------------------------------------------------------------
    def test_modeling_gyromagnetic_ratio_used(self) -> Tuple[bool, str]:
        """
        Check that gamma_p is provided; the explanation uses the proton gyromagnetic ratio.
        """
        if self.gamma_p is None:
            return (False, "gamma_p not found; modeling likely incomplete")
        try:
            gamma_val = float(self.gamma_p)
        except (TypeError, ValueError):
            return (False, "gamma_p not numeric")
        if not math.isfinite(gamma_val):
            return (False, "gamma_p non-finite")
        # Only verify existence and positivity; exact value is irrelevant here.
        if gamma_val > 0:
            return (True, f"gamma_p = {gamma_val:.3e} is present and positive")
        else:
            return (False, "gamma_p is not positive")

    def test_modeling_earth_field_used(self) -> Tuple[bool, str]:
        """
        Check that B_earth is provided; the explanation uses Earth's magnetic field.
        """
        if self.B_earth is None:
            return (False, "B_earth not found; modeling likely incomplete")
        try:
            B_val = float(self.B_earth)
        except (TypeError, ValueError):
            return (False, "B_earth not numeric")
        if not math.isfinite(B_val):
            return (False, "B_earth non-finite")
        if B_val > 0:
            return (True, f"B_earth = {B_val:.2e} T is present and positive")
        else:
            return (False, "B_earth is not positive")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation + comparison)
    # ------------------------------------------------------------------
    def test_monotonicity_f_increases_with_B(self) -> Tuple[bool, str]:
        """
        Perturb B_earth by +10% and verify that the computed f (using given gamma) increases.
        """
        if self.gamma_p is None or self.B_earth is None or self.f is None:
            return (False, "SKIP: missing required quantities for monotonicity test")
        try:
            gamma = float(self.gamma_p)
            B = float(self.B_earth)
            f_base = float(self.f)
        except (TypeError, ValueError):
            return (False, "Non-numeric quantities")
        B_pert = B * 1.10
        f_expected = gamma / (2 * math.pi) * B_pert
        if f_base < f_expected:
            return (True, "Monotonicity holds: increasing B increases f")
        else:
            return (False, f"Monotonicity violated: f_base={f_base:.4e} >= f_expected={f_expected:.4e}")

    def test_monotonicity_f_increases_with_gamma(self) -> Tuple[bool, str]:
        """
        Perturb gamma by +10% and verify that the computed f increases.
        """
        if self.gamma_p is None or self.B_earth is None or self.f is None:
            return (False, "SKIP: missing required quantities")
        try:
            gamma = float(self.gamma_p)
            B = float(self.B_earth)
            f_base = float(self.f)
        except (TypeError, ValueError):
            return (False, "Non-numeric quantities")
        gamma_pert = gamma * 1.10
        f_expected = gamma_pert / (2 * math.pi) * B
        if f_base < f_expected:
            return (True, "Monotonicity holds: increasing gamma increases f")
        else:
            return (False, f"Monotonicity violated: f_base={f_base:.4e} >= f_expected={f_expected:.4e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_positive(self) -> Tuple[bool, str]:
        """The output frequency must be a finite positive number."""
        if not self.is_physical(self.f):
            return (False, f"f = {self.f} is not finite positive")
        return (True, "f is finite positive")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, collect logs."""
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
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs