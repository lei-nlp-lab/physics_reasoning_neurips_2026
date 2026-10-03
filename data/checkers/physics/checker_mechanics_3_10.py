import math
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits a solution to the relativistic Doppler effect and aberration problem.
    Checks interface validity, physical invariants, modeling quality, monotonicity,
    and provides a weak final-answer sanity test.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Safe extraction of givens
        self.nu = self._safe_get(solution_instance, 'nu')
        self.c = self._safe_get(solution_instance, 'c')
        self.v_source = self._safe_get(solution_instance, 'v_source')
        self.theta_K = self._safe_get(solution_instance, 'theta_K')

        # Results from solution_instance.results (dict) or direct attributes
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}
        self.nu_prime = self.results.get('nu_prime', None)
        self.theta_prime = self.results.get('theta_prime', None)

        # Derived quantities (only if givens are present)
        self.beta: Optional[float] = None
        self.gamma: Optional[float] = None
        if self.c is not None and self.v_source is not None and self.c != 0:
            self.beta = self.v_source / self.c
            if abs(self.beta) < 1.0:
                self.gamma = 1.0 / math.sqrt(1.0 - self.beta * self.beta)

    def _safe_get(self, obj: Any, attr: str) -> Any:
        """Try to extract attribute from object, then from .givens dict, else return None."""
        val = getattr(obj, attr, None)
        if val is not None:
            return val
        givens = getattr(obj, 'givens', {})
        if givens and attr in givens:
            return givens[attr]
        return None

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that the solution emits nu_prime and theta_prime with valid types."""
        if self.nu_prime is None:
            return (False, "Missing required result: nu_prime")
        if self.theta_prime is None:
            return (False, "Missing required result: theta_prime")
        if not isinstance(self.nu_prime, (int, float)):
            return (False, f"nu_prime has invalid type: {type(self.nu_prime)}")
        if not isinstance(self.theta_prime, (int, float)):
            return (False, f"theta_prime has invalid type: {type(self.theta_prime)}")
        return (True, "Required keys present and numeric")

    def test_interface_givens(self) -> Tuple[bool, str]:
        """Check that all required givens are provided."""
        missing = []
        for name in ['nu', 'c', 'v_source', 'theta_K']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ----------------------------------------------------------------------
    # Invariant / theorem-based checks
    # ----------------------------------------------------------------------
    def test_invariant_lorentz_factor(self) -> Tuple[bool, str]:
        """Check that gamma = 5/3 from given v_source = 4c/5."""
        if self.gamma is None:
            return (False, "Cannot compute gamma: missing c or v_source")
        expected_gamma = 5.0 / 3.0
        if math.isclose(self.gamma, expected_gamma, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"gamma = {self.gamma:.3f} matches expected {expected_gamma:.3f}")
        return (False, f"gamma = {self.gamma:.3f} but expected ~{expected_gamma:.3f}")

    def test_invariant_frequency_formula(self) -> Tuple[bool, str]:
        """Check that nu_prime follows the relativistic Doppler formula."""
        if any(x is None for x in [self.nu, self.gamma, self.beta, self.theta_K, self.nu_prime]):
            return (False, "Missing data for frequency invariant")
        try:
            theta_rad = math.radians(self.theta_K)
        except TypeError:
            return (False, "theta_K must be numeric (degrees)")
        factor = self.gamma * (1 + self.beta * math.cos(theta_rad))
        expected_nu_prime = factor * self.nu
        if math.isclose(self.nu_prime, expected_nu_prime, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"nu_prime = {self.nu_prime:.3f} matches formula {expected_nu_prime:.3f}")
        return (False, f"nu_prime = {self.nu_prime:.3f} but formula gives {expected_nu_prime:.3f}")

    def test_invariant_angle_formula(self) -> Tuple[bool, str]:
        """Check that cos(theta_prime) = (cos(theta_K) + beta) / (1 + beta*cos(theta_K))."""
        if any(x is None for x in [self.beta, self.theta_K, self.theta_prime]):
            return (False, "Missing data for angle invariant")
        try:
            theta_rad = math.radians(self.theta_K)
        except TypeError:
            return (False, "theta_K must be numeric (degrees)")
        cos_theta = math.cos(theta_rad)
        cos_theta_prime = (cos_theta + self.beta) / (1 + self.beta * cos_theta)
        cos_theta_prime = max(-1.0, min(1.0, cos_theta_prime))
        expected_theta_prime_deg = math.degrees(math.acos(cos_theta_prime))
        if math.isclose(self.theta_prime, expected_theta_prime_deg, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"theta_prime = {self.theta_prime:.2f}° matches formula {expected_theta_prime_deg:.2f}°")
        return (False, f"theta_prime = {self.theta_prime:.2f}° but formula gives {expected_theta_prime_deg:.2f}°")

    # ----------------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------------
    def test_modeling_wave_vector_components(self) -> Tuple[bool, str]:
        """Check that the solution correctly computed k_x and k_y from the given angle."""
        if any(x is None for x in [self.nu, self.c, self.theta_K]):
            return (False, "Missing givens for wave vector check")
        try:
            theta_rad = math.radians(self.theta_K)
        except TypeError:
            return (False, "theta_K must be numeric (degrees)")
        omega = 2 * math.pi * self.nu
        k = omega / self.c
        expected_kx = k * math.cos(theta_rad)
        expected_ky = k * math.sin(theta_rad)
        if any(x is None for x in [self.beta, self.gamma, self.nu_prime]):
            return (False, "Missing beta/gamma/nu_prime for full check")
        omega_prime = 2 * math.pi * self.nu_prime
        expected_kx_prime = self.gamma * (expected_kx + self.beta * omega / self.c)
        expected_ky_prime = expected_ky
        if expected_kx_prime == 0:
            return (False, "k'_x is zero, cannot compute angle")
        tan_theta_prime = expected_ky_prime / expected_kx_prime
        theta_prime_expected = math.degrees(math.atan(tan_theta_prime))
        if self.theta_prime is None:
            return (False, "theta_prime not available for comparison")
        if math.isclose(self.theta_prime, theta_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"theta_prime from wave vector: {theta_prime_expected:.2f}° matches")
        return (False, f"theta_prime from wave vector gives {theta_prime_expected:.2f}° but solution has {self.theta_prime:.2f}°")

    def test_modeling_wave_invariance(self) -> Tuple[bool, str]:
        """
        Check that the solution respects the wave dispersion relation in the observer's frame:
        ω'^2 - c^2 (k'_x^2 + k'_y^2) = 0.
        This verifies that the Lorentz transformation of the wave four-vector is consistent.
        """
        if any(x is None for x in [self.nu, self.c, self.theta_K, self.beta, self.gamma, self.nu_prime]):
            return (False, "Missing data for wave invariance check")
        try:
            theta_rad = math.radians(self.theta_K)
        except TypeError:
            return (False, "theta_K must be numeric (degrees)")
        omega = 2 * math.pi * self.nu
        k = omega / self.c
        kx = k * math.cos(theta_rad)
        ky = k * math.sin(theta_rad)
        # Lorentz transform to K'
        kx_prime = self.gamma * (kx + self.beta * omega / self.c)  # v_boost = -v_source -> +beta
        ky_prime = ky
        omega_prime = 2 * math.pi * self.nu_prime
        # Compute left-hand side of dispersion relation
        lhs = omega_prime**2 - (self.c**2) * (kx_prime**2 + ky_prime**2)
        # In vacuum, it should be exactly 0, but we allow numerical tolerance
        if math.isclose(lhs, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ω'^2 - c^2 k'^2 = {lhs:.2e} (close to 0)")
        return (False, f"ω'^2 - c^2 k'^2 = {lhs:.2e} (should be 0)")

    # ----------------------------------------------------------------------
    # Monotonicity checks
    # ----------------------------------------------------------------------
    def test_monotonicity_frequency_increases_with_beta(self) -> Tuple[bool, str]:
        """
        Check that nu' increases with beta (blueshift stronger as relative speed increases).
        Since we have only one data point, we simulate by perturbing beta.
        """
        if any(x is None for x in [self.nu, self.gamma, self.beta, self.theta_K, self.nu_prime]):
            return (False, "Missing data for monotonicity")
        try:
            theta_rad = math.radians(self.theta_K)
        except TypeError:
            return (False, "theta_K must be numeric (degrees)")

        delta = 0.01  # 1% increase
        beta_new = self.beta * (1 + delta)
        if beta_new >= 1.0:
            return (False, "Perturbed beta exceeds c, cannot test")
        gamma_new = 1.0 / math.sqrt(1 - beta_new * beta_new)
        factor_new = gamma_new * (1 + beta_new * math.cos(theta_rad))
        nu_prime_new = factor_new * self.nu

        factor_orig = self.gamma * (1 + self.beta * math.cos(theta_rad))
        nu_prime_orig = factor_orig * self.nu

        if nu_prime_new > nu_prime_orig:
            return (True, f"Increasing beta from {self.beta:.3f} to {beta_new:.3f} increases nu' from {nu_prime_orig:.3f} to {nu_prime_new:.3f}")
        return (False, f"nu' does not increase with beta: original {nu_prime_orig:.3f}, new {nu_prime_new:.3f}")

    def test_monotonicity_angle_decreases_with_beta(self) -> Tuple[bool, str]:
        """
        Check that theta' decreases as beta increases (aberration makes the wave more forward).
        """
        if any(x is None for x in [self.beta, self.theta_K, self.theta_prime]):
            return (False, "Missing data for angle monotonicity")
        try:
            theta_rad = math.radians(self.theta_K)
        except TypeError:
            return (False, "theta_K must be numeric (degrees)")

        delta = 0.01
        beta_new = self.beta * (1 + delta)
        if beta_new >= 1.0:
            return (False, "Perturbed beta exceeds c, cannot test")
        cos_theta = math.cos(theta_rad)
        cos_theta_prime_new = (cos_theta + beta_new) / (1 + beta_new * cos_theta)
        cos_theta_prime_new = max(-1.0, min(1.0, cos_theta_prime_new))
        theta_prime_new = math.degrees(math.acos(cos_theta_prime_new))

        cos_theta_prime_orig = (cos_theta + self.beta) / (1 + self.beta * cos_theta)
        cos_theta_prime_orig = max(-1.0, min(1.0, cos_theta_prime_orig))
        theta_prime_orig = math.degrees(math.acos(cos_theta_prime_orig))

        if theta_prime_new < theta_prime_orig:
            return (True, f"Increasing beta from {self.beta:.3f} to {beta_new:.3f} decreases theta' from {theta_prime_orig:.2f}° to {theta_prime_new:.2f}°")
        return (False, f"theta' does not decrease: original {theta_prime_orig:.2f}°, new {theta_prime_new:.2f}°")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ----------------------------------------------------------------------
    def test_answer_sanity_plausibility(self) -> Tuple[bool, str]:
        """
        Check that the final answers are physically plausible:
        - nu_prime > nu (blueshift).
        - theta_prime between 0 and 90 degrees (first quadrant).
        - Both are finite and not NaN.
        """
        if self.nu_prime is None or self.theta_prime is None:
            return (False, "Missing results for sanity check")
        if not math.isfinite(self.nu_prime):
            return (False, f"nu_prime is not finite: {self.nu_prime}")
        if not math.isfinite(self.theta_prime):
            return (False, f"theta_prime is not finite: {self.theta_prime}")

        # Blueshift when source approaches observer
        if self.nu is not None and self.nu_prime <= self.nu:
            return (False, f"nu_prime = {self.nu_prime} should be > nu = {self.nu} for approaching source")
        # Angle in first quadrant
        if self.theta_prime <= 0 or self.theta_prime >= 90:
            return (False, f"theta_prime = {self.theta_prime}° should be between 0° and 90°")
        return (True, "Answers are finite, blueshifted, and angle in first quadrant")

    # ----------------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, collecting results."""
        logs: List[str] = []
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