import math
import traceback

class PhysicsChecker:
    """
    Physics audit for a geostationary satellite orbit height and ground
    receptor diffraction size problem.
    """

    def __init__(self, solution_instance):
        # --- Safe extraction of givens and results ---
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})

        # Required output keys (set to None if absent)
        self.h = self.results.get('h', None)
        self.d = self.results.get('d', None)

        # Input parameters (may be missing)
        self.T = self.givens.get('T', None) or self.results.get('T', None)
        self.R_E = self.givens.get('R_E', None)
        self.mu = self.givens.get('mu', None)
        self.lambda_ = self.givens.get('lambda', None)
        self.D = self.givens.get('D', None)

        # Standard constants (only used if input values are missing)
        self._g = 9.81          # m/s²
        self._R_E_default = 6.37e6   # m
        if self.R_E is None:
            self.R_E = self._R_E_default
        if self.mu is None:
            # Compute from g and R_E: μ = g R_E²
            self.mu = self._g * self.R_E ** 2

    # ============== Interface Tests ==============
    def test_interface_required_keys(self):
        """Check that both h and d are present and numeric finite values."""
        for key, val in [('h', self.h), ('d', self.d)]:
            if val is None:
                return (False, f"Required result '{key}' is None")
            if not isinstance(val, (int, float)):
                return (False, f"Required result '{key}' has type {type(val).__name__}, expected numeric")
            if not math.isfinite(val):
                return (False, f"Required result '{key}' is non-finite: {val}")
        return (True, "All required keys present and valid")

    # ============== Invariant (Theorem-Based) Tests ==============
    def test_invariant_geostationary_orbit(self):
        """Check that the orbit height satisfies Kepler's third law."""
        if self.h is None:
            return (False, "SKIP: h not available")
        if self.T is None:
            return (False, "SKIP: T not available")
        try:
            T_val = float(self.T)
            if T_val <= 0:
                return (False, "Period T must be positive")
            omega = 2.0 * math.pi / T_val        # rad/s
            R_expected = (self.mu / omega ** 2) ** (1.0 / 3.0)  # orbital radius
            h_expected = R_expected - self.R_E
            if not math.isfinite(h_expected):
                return (False, "Computed expected height is non-finite")
            if not math.isclose(self.h, h_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"h={self.h:.4e} does not match expected h≈{h_expected:.4e}, rel diff = {abs(self.h-h_expected)/max(abs(self.h),abs(h_expected)):.4f}")
            return (True, f"Orbit height consistent: h≈{h_expected:.4e}")
        except Exception as e:
            return (False, f"Exception in invariant check: {e}")

    def test_invariant_diffraction_relation(self):
        """Check that the receptor size follows d ≈ λ h / D."""
        if self.d is None or self.h is None or self.lambda_ is None or self.D is None:
            return (False, "SKIP: required inputs missing for diffraction check")
        try:
            lam = float(self.lambda_)
            D_val = float(self.D)
            if lam <= 0 or D_val <= 0:
                return (False, "Wavelength and aperture must be positive")
            d_approx = lam * self.h / D_val
            if not math.isfinite(d_approx):
                return (False, "Computed approximate d is non-finite")
            if not math.isclose(self.d, d_approx, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"d={self.d:.4e} does not match expected d≈{d_approx:.4e}, rel diff = {abs(self.d-d_approx)/max(abs(self.d),abs(d_approx)):.4f}")
            return (True, f"Diffraction relation consistent: d≈{d_approx:.4e}")
        except Exception as e:
            return (False, f"Exception in diffraction invariant: {e}")

    # ============== Modeling Quality Tests ==============
    def test_modeling_height_positive(self):
        """The orbit height must be positive."""
        if self.h is None:
            return (False, "SKIP: h not available")
        if self.h <= 0:
            return (False, f"Height h={self.h} must be positive")
        return (True, "Height is positive")

    def test_modeling_receptor_positive(self):
        """The ground receptor size must be positive."""
        if self.d is None:
            return (False, "SKIP: d not available")
        if self.d <= 0:
            return (False, f"Receptor size d={self.d} must be positive")
        return (True, "Receptor size is positive")

    # ============== Monotonicity Tests ==============
    def test_monotonicity_height_with_period(self):
        """Increasing orbital period increases height."""
        if self.h is None or self.T is None:
            return (False, "SKIP: h or T not available")
        T_orig = float(self.T)
        if T_orig <= 0:
            return (False, "Original period must be positive")
        # Compute expected height for original T
        omega_orig = 2.0 * math.pi / T_orig
        R_orig = (self.mu / omega_orig ** 2) ** (1.0 / 3.0)
        h_orig = R_orig - self.R_E
        if not math.isclose(self.h, h_orig, rel_tol=0.15):  # loose tolerance for stability
            # The solution may be approximate; but monotonicity test still relevant
            pass  # We still check direction
        # Perturb T upward by 1%
        T_pert = T_orig * 1.01
        omega_pert = 2.0 * math.pi / T_pert
        R_pert = (self.mu / omega_pert ** 2) ** (1.0 / 3.0)
        h_pert = R_pert - self.R_E
        if h_pert <= h_orig:
            return (False, f"Monotonicity violation: increasing T should increase h, but original h={h_orig:.4e} >= perturbed h={h_pert:.4e}")
        # Check that solution's h is consistent with this direction
        if self.h >= h_pert:
            return (False, f"Solution's h={self.h:.4e} is not less than perturbed h={h_pert:.4e}, contradicting positive monotonicity")
        return (True, "Height increases with period, consistent with geostationary condition")

    def test_monotonicity_receptor_with_wavelength(self):
        """Increasing wavelength increases receptor size."""
        if self.d is None or self.h is None or self.lambda_ is None or self.D is None:
            return (False, "SKIP: required inputs missing")
        lam_orig = float(self.lambda_)
        D_val = float(self.D)
        if lam_orig <= 0 or D_val <= 0:
            return (False, "Wavelength and aperture must be positive")
        d_orig = lam_orig * self.h / D_val
        # Perturb λ upward by 1%
        lam_pert = lam_orig * 1.01
        d_pert = lam_pert * self.h / D_val
        if d_pert <= d_orig:
            return (False, f"Monotonicity violation: increasing λ should increase d, but original d={d_orig:.4e} >= perturbed d={d_pert:.4e}")
        if self.d >= d_pert:
            return (False, f"Solution's d={self.d:.4e} is not less than perturbed d={d_pert:.4e}, contradicting positive monotonicity")
        return (True, "Receptor size increases with wavelength, consistent with diffraction")

    def test_monotonicity_receptor_with_aperture(self):
        """Increasing aperture decreases receptor size."""
        if self.d is None or self.h is None or self.lambda_ is None or self.D is None:
            return (False, "SKIP: required inputs missing")
        lam = float(self.lambda_)
        D_orig = float(self.D)
        if lam <= 0 or D_orig <= 0:
            return (False, "Wavelength and aperture must be positive")
        d_orig = lam * self.h / D_orig
        # Perturb D upward by 1%
        D_pert = D_orig * 1.01
        d_pert = lam * self.h / D_pert
        if d_pert >= d_orig:
            return (False, f"Monotonicity violation: increasing D should decrease d, but original d={d_orig:.4e} <= perturbed d={d_pert:.4e}")
        if self.d <= d_pert:
            return (False, f"Solution's d={self.d:.4e} is not greater than perturbed d={d_pert:.4e}, contradicting inverse monotonicity")
        return (True, "Receptor size decreases with aperture, consistent with diffraction")

    # ============== Answer Sanity (Weak) ==============
    def test_answer_sanity_bounds(self):
        """Weak sanity: height and size lie in plausible broad ranges."""
        if self.h is not None:
            if not (1e7 < self.h < 1e8):
                return (False, f"Height h={self.h:.4e} outside plausible range (1e7 .. 1e8) for geostationary orbit")
        if self.d is not None:
            if not (1e3 < self.d < 1e5):
                return (False, f"Receptor size d={self.d:.4e} outside plausible range (1e3 .. 1e5) for ground station")
        return (True, "Both values lie in physically plausible broad ranges")

    # ============== Test Runner ==============
    def run_all_tests(self):
        """
        Discover and execute all test_* methods.
        Returns (success_bool, logs_list).
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)