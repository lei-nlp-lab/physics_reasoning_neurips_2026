import math
import traceback

class PhysicsChecker:
    """Checker for nuclear radius determination via electron scattering and muonic atoms."""

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        else:
            results = {}

        # Core physics quantities (None if missing)
        self.rms_radius = results.get('rms_radius', None)
        self.mean_square_radius = results.get('mean_square_radius', None)

        # Additional quantities that may be provided for detailed modeling checks
        self.F_q2 = results.get('F_q2', None)               # Form factor at a given q^2
        self.q_squared = results.get('q_squared', None)     # Momentum transfer squared
        self.delta_E_fs = results.get('delta_E_fs', None)   # Finite-size energy shift
        self.psi_sq_0 = results.get('psi_sq_0', None)       # |ψ(0)|^2 for the s-state
        self.Z = results.get('Z', None)                     # Nuclear charge
        self.alpha = results.get('alpha', None)             # Fine-structure constant
        self.m_r = results.get('m_r', None)                 # Reduced mass of muon

        # No numeric defaults – missing values stay None

    # ---------- interface tests ----------
    def test_interface_rms_radius_exists(self):
        """Check that rms_radius key is present and not None."""
        if self.rms_radius is None:
            return (False, "SKIP: rms_radius not provided")
        return (True, "rms_radius is present")

    def test_interface_mean_square_radius_exists(self):
        """Check that mean_square_radius key is present and not None."""
        if self.mean_square_radius is None:
            return (False, "SKIP: mean_square_radius not provided")
        return (True, "mean_square_radius is present")

    # ---------- invariant tests (physical sanity) ----------
    def test_invariant_rms_radius_positive(self):
        """rms radius must be positive."""
        if self.rms_radius is None:
            return (False, "SKIP: rms_radius not available")
        if isinstance(self.rms_radius, complex):
            return (False, "rms_radius is complex")
        if self.rms_radius <= 0:
            return (False, f"rms_radius = {self.rms_radius} is not positive")
        return (True, f"rms_radius = {self.rms_radius} > 0")

    def test_invariant_mean_square_radius_positive(self):
        """Mean-square radius must be positive."""
        if self.mean_square_radius is None:
            return (False, "SKIP: mean_square_radius not available")
        if isinstance(self.mean_square_radius, complex):
            return (False, "mean_square_radius is complex")
        if self.mean_square_radius <= 0:
            return (False, f"mean_square_radius = {self.mean_square_radius} is not positive")
        return (True, f"mean_square_radius = {self.mean_square_radius} > 0")

    def test_invariant_rms_finite(self):
        """rms radius must be finite (not NaN, not inf)."""
        if self.rms_radius is None:
            return (False, "SKIP: rms_radius not available")
        if math.isnan(self.rms_radius) or math.isinf(self.rms_radius):
            return (False, f"rms_radius is NaN or infinite")
        return (True, "rms_radius is finite")

    def test_invariant_mean_square_finite(self):
        """Mean-square radius must be finite."""
        if self.mean_square_radius is None:
            return (False, "SKIP: mean_square_radius not available")
        if math.isnan(self.mean_square_radius) or math.isinf(self.mean_square_radius):
            return (False, "mean_square_radius is NaN or infinite")
        return (True, "mean_square_radius is finite")

    # ---------- modeling / trace tests (quality checks) ----------
    def test_modeling_rms_to_mean_square(self):
        """
        Check that mean_square_radius equals (rms_radius)^2.
        This follows directly from the definition of rms radius.
        """
        if self.rms_radius is None or self.mean_square_radius is None:
            return (False, "SKIP: one or both radii missing")
        expected_ms = self.rms_radius ** 2
        if not math.isclose(self.mean_square_radius, expected_ms,
                            rel_tol=1e-2, abs_tol=1e-30):
            return (False, f"mean_square_radius {self.mean_square_radius} != (rms)^2 = {expected_ms}")
        return (True, "mean_square_radius = (rms_radius)^2 within tolerance")

    def test_modeling_small_q_form_factor(self):
        """
        If form factor and q^2 are given, verify the small-q expansion:
        F(q^2) ≈ 1 - (1/6) q^2 <r^2>
        """
        if any(x is None for x in [self.F_q2, self.q_squared, self.mean_square_radius]):
            return (False, "SKIP: insufficient data for form factor check")
        # Compute expected form factor
        expected_F = 1.0 - (1.0/6.0) * self.q_squared * self.mean_square_radius
        if not math.isclose(self.F_q2, expected_F, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Form factor {self.F_q2} != 1 - (1/6) q^2 <r^2> = {expected_F}")
        return (True, "Small-q form factor expansion holds within tolerance")

    def test_modeling_energy_shift(self):
        """
        If delta_E_fs, psi_sq_0, and mean_square_radius are provided,
        check the relation for s-state energy shift:
        ΔE = (2π/3) Z α |ψ(0)|^2 <r^2>
        """
        if any(x is None for x in [self.delta_E_fs, self.psi_sq_0, self.mean_square_radius, self.Z, self.alpha]):
            return (False, "SKIP: insufficient data for energy shift check")
        # Compute expected shift
        expected_shift = (2.0 * math.pi / 3.0) * self.Z * self.alpha * self.psi_sq_0 * self.mean_square_radius
        if not math.isclose(self.delta_E_fs, expected_shift, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"ΔE = {self.delta_E_fs} != expected = {expected_shift}")
        return (True, "Energy shift relation holds within tolerance")

    # ---------- monotonicity tests ----------
    def test_monotonicity_form_factor_decreases_with_q2(self):
        """
        For elastic electron scattering, the form factor F(q^2) decreases as q^2 increases,
        according to the small-q expansion: F ≈ 1 - (1/6) q^2 <r^2>.
        Perturb q^2 upward by 1% and verify F decreases.
        """
        if any(x is None for x in [self.F_q2, self.q_squared, self.mean_square_radius]):
            return (False, "SKIP: insufficient data for monotonicity check")
        # Baseline values
        F0 = self.F_q2
        q2_0 = self.q_squared
        # Perturb: increase q^2 by 1%
        q2_pert = q2_0 * 1.01
        # Compute perturbed form factor using small-q expansion (re-derived)
        F_pert = 1.0 - (1.0/6.0) * q2_pert * self.mean_square_radius
        if not F_pert < F0:
            return (False, f"Perturbed F(q^2) = {F_pert} is not less than original F = {F0}; expected decrease")
        return (True, "Form factor decreases with increasing q^2 as expected")

    # ---------- answer sanity test (weak) ----------
    def test_answer_sanity_radius_magnitude(self):
        """
        Weak sanity: rms radius of a nucleus should be in the 1-10 fm range,
        i.e., between 1e-16 m and 1e-13 m (broad range).
        """
        if self.rms_radius is None:
            return (False, "SKIP: rms_radius not available")
        # Typical nuclear radii: 1-10 fm = 1e-15 to 1e-14 m
        # Use a generous range
        lower = 1e-16
        upper = 1e-13
        if not (lower <= self.rms_radius <= upper):
            return (False, f"rms_radius = {self.rms_radius} is outside typical nuclear range [{lower}, {upper}]")
        return (True, "rms_radius within plausible nuclear range")

    # ---------- test runner ----------
    def run_all_tests(self):
        """Discover and run all test_* methods, collect results."""
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