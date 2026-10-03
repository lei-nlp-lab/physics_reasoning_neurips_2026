import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Extract all givens and results safely (default None for missing items).
        """
        # Givens (try attributes or a 'givens' dict)
        self.L = getattr(solution_instance, 'L', None)
        if self.L is None and hasattr(solution_instance, 'givens'):
            self.L = solution_instance.givens.get('L', None)

        self.A_o = getattr(solution_instance, 'A_o', None)
        if self.A_o is None and hasattr(solution_instance, 'givens'):
            self.A_o = solution_instance.givens.get('A_o', None)

        self.A_i = getattr(solution_instance, 'A_i', None)
        if self.A_i is None and hasattr(solution_instance, 'givens'):
            self.A_i = solution_instance.givens.get('A_i', None)

        self.D_lens = getattr(solution_instance, 'D_lens', None)
        if self.D_lens is None and hasattr(solution_instance, 'givens'):
            self.D_lens = solution_instance.givens.get('D_lens', None)

        self.f = getattr(solution_instance, 'f', None)
        if self.f is None and hasattr(solution_instance, 'givens'):
            self.f = solution_instance.givens.get('f', None)

        self.n_lens = getattr(solution_instance, 'n_lens', None)
        if self.n_lens is None and hasattr(solution_instance, 'givens'):
            self.n_lens = solution_instance.givens.get('n_lens', None)

        # Results (try results dict first, then attributes)
        self.Phi_total = None
        self.s_o = None
        self.s_i = None
        self.Omega_lens = None
        self.Phi_image = None
        self.G_o = None
        self.G_i = None
        self.Phi_max = None

        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            res = solution_instance.results
            self.Phi_total = res.get('Phi_total', None)
            self.s_o = res.get('s_o', None)
            self.s_i = res.get('s_i', None)
            self.Omega_lens = res.get('Omega_lens', None)
            self.Phi_image = res.get('Phi_image', None)
            self.G_o = res.get('G_o', None)
            self.G_i = res.get('G_i', None)
            self.Phi_max = res.get('Phi_max', None)
        else:
            self.Phi_total = getattr(solution_instance, 'Phi_total', None)
            self.s_o = getattr(solution_instance, 's_o', None)
            self.s_i = getattr(solution_instance, 's_i', None)
            self.Omega_lens = getattr(solution_instance, 'Omega_lens', None)
            self.Phi_image = getattr(solution_instance, 'Phi_image', None)
            self.G_o = getattr(solution_instance, 'G_o', None)
            self.G_i = getattr(solution_instance, 'G_i', None)
            self.Phi_max = getattr(solution_instance, 'Phi_max', None)

        self.pi = math.pi

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all expected result keys are present."""
        keys_expected = ['Phi_total', 's_o', 's_i', 'Omega_lens', 'Phi_image', 'G_o', 'G_i', 'Phi_max']
        missing = [k for k in keys_expected if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing results: {missing}")
        return (True, "All required keys present")

    def test_interface_valid_types_positive(self):
        """Check that numeric results are positive finite numbers."""
        numeric_keys = ['Phi_total', 's_o', 's_i', 'Omega_lens', 'Phi_image', 'G_o', 'G_i', 'Phi_max']
        for key in numeric_keys:
            val = getattr(self, key, None)
            if val is None:
                return (False, f"{key} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{key} has invalid type {type(val).__name__}")
            if not math.isfinite(val) or val <= 0:
                return (False, f"{key} is not positive finite (value={val})")
        return (True, "All numeric results are positive finite numbers")

    # ----------------------------------------------------------------------
    # Invariant / theorem checks (use math.isclose with lenient tolerance)
    # ----------------------------------------------------------------------
    def test_invariant_lambert_total_power(self):
        """Check Φ_total = L * A_o * π."""
        if self.L is None or self.A_o is None:
            return (False, "SKIP: L or A_o missing")
        if self.Phi_total is None:
            return (False, "SKIP: Phi_total missing")
        expected = self.L * self.A_o * self.pi
        if not math.isclose(self.Phi_total, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Phi_total {self.Phi_total} != {expected} (L*A_o*pi)")
        return (True, "Lambert total power invariant holds")

    def test_invariant_lens_equation(self):
        """Check 1/f = 1/s_o + 1/s_i."""
        if self.f is None or self.s_o is None or self.s_i is None:
            return (False, "SKIP: f, s_o, or s_i missing")
        left = 1.0 / self.f
        right = 1.0 / self.s_o + 1.0 / self.s_i
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Lens equation fails: 1/f={left}, sum={right}")
        return (True, "Lens equation satisfied")

    def test_invariant_magnification(self):
        """Check m = sqrt(A_i/A_o) and m = s_i/s_o."""
        if self.A_o is None or self.A_i is None or self.s_o is None or self.s_i is None:
            return (False, "SKIP: A_o, A_i, s_o, or s_i missing")
        m_area = math.sqrt(self.A_i / self.A_o)
        m_dist = self.s_i / self.s_o
        if not math.isclose(m_dist, m_area, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Magnification mismatch: area sqrt={m_area}, dist ratio={m_dist}")
        return (True, "Magnification condition holds")

    def test_invariant_omega_lens_geometry(self):
        """Check Omega_lens = (π*(D_lens/2)^2) / s_o^2."""
        if self.D_lens is None or self.s_o is None or self.Omega_lens is None:
            return (False, "SKIP: D_lens, s_o, or Omega_lens missing")
        expected_omega = self.pi * (self.D_lens / 2.0)**2 / self.s_o**2
        if not math.isclose(self.Omega_lens, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Omega_lens {self.Omega_lens} != expected {expected_omega}")
        return (True, "Lens solid angle geometry consistent")

    def test_invariant_flux_via_etendue(self):
        """Check Phi_image = L * A_o * Omega_lens."""
        if self.L is None or self.A_o is None or self.Omega_lens is None or self.Phi_image is None:
            return (False, "SKIP: L, A_o, Omega_lens, or Phi_image missing")
        expected = self.L * self.A_o * self.Omega_lens
        if not math.isclose(self.Phi_image, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Phi_image {self.Phi_image} != {expected} (L*A_o*Omega)")
        return (True, "Flux via étendue consistent")

    def test_invariant_max_flux(self):
        """Check Phi_max = L * min(A_o, A_i) * π."""
        if self.L is None or self.A_o is None or self.A_i is None or self.Phi_max is None:
            return (False, "SKIP: L, A_o, A_i, or Phi_max missing")
        expected = self.L * min(self.A_o, self.A_i) * self.pi
        if not math.isclose(self.Phi_max, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Phi_max {self.Phi_max} != {expected} (L*min(A)*pi)")
        return (True, "Maximum flux invariant holds")

    # ----------------------------------------------------------------------
    # Modeling / trace checks (using explanation steps, not hardcoding)
    # ----------------------------------------------------------------------
    def test_modeling_image_distance_positive(self):
        """Check that s_o > f and s_i > 0 (real image)."""
        if self.f is None or self.s_o is None or self.s_i is None:
            return (False, "SKIP: f, s_o, or s_i missing")
        if not (self.s_o > self.f and self.s_i > 0):
            return (False, f"Impossible image distances: s_o={self.s_o}, s_i={self.s_i}, f={self.f}")
        return (True, "Real image distances consistent")

    def test_modeling_lens_diameter_small_angle(self):
        """Check that lens diameter is much smaller than s_o (paraxial)."""
        if self.D_lens is None or self.s_o is None:
            return (False, "SKIP: D_lens or s_o missing")
        if not self.s_o > 0:
            return (False, "s_o not positive")
        ratio = self.D_lens / self.s_o
        if ratio > 0.5:
            return (False, f"Paraxial approximation may be violated: D_lens/s_o = {ratio}")
        return (True, "Small-angle condition satisfied")

    # ----------------------------------------------------------------------
    # Monotonicity checks (perturb input, observe trend)
    # ----------------------------------------------------------------------
    def test_monotonicity_total_power_with_L(self):
        """Perturb L upward by 10% → Phi_total should increase."""
        if self.L is None or self.Phi_total is None:
            return (False, "SKIP: L or Phi_total missing")
        L_perturbed = self.L * 1.1
        expected_perturbed = L_perturbed * self.A_o * self.pi
        if expected_perturbed <= self.Phi_total:
            return (False, f"Monotonicity fails: perturbed total {expected_perturbed} <= original {self.Phi_total}")
        return (True, "Monotonicity of total power with L holds")

    def test_monotonicity_image_flux_with_lens_diameter(self):
        """Perturb D_lens upward by 10% → Phi_image should increase (if s_o fixed)."""
        if self.D_lens is None or self.s_o is None or self.L is None or self.A_o is None or self.Phi_image is None:
            return (False, "SKIP: D_lens, s_o, L, A_o, or Phi_image missing")
        D_perturbed = self.D_lens * 1.1
        Omega_perturbed = self.pi * (D_perturbed / 2.0)**2 / self.s_o**2
        perturbed_flux = self.L * self.A_o * Omega_perturbed
        if perturbed_flux <= self.Phi_image:
            return (False, f"Monotonicity fails: perturbed flux {perturbed_flux} <= original {self.Phi_image}")
        return (True, "Monotonicity of image flux with lens diameter holds")

    # ----------------------------------------------------------------------
    # Weak answer sanity (no isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Broad sanity checks: all fluxes are positive, finite, and lie in expected orders.
        """
        checks = []
        # Check individual fields
        values = {
            'Phi_total': self.Phi_total,
            'Phi_image': self.Phi_image,
            'Phi_max': self.Phi_max
        }
        for name, val in values.items():
            if val is None:
                checks.append(f"{name} is None")
                continue
            if not isinstance(val, (int, float)):
                checks.append(f"{name} not numeric")
                continue
            if not math.isfinite(val) or val <= 0:
                checks.append(f"{name} not positive finite")
                continue
        # Rough magnitude checks (not strict, just plausible)
        if self.Phi_total is not None:
            if not (0.1 < self.Phi_total < 10):
                checks.append(f"Phi_total magnitude suspicious: {self.Phi_total}")
        if self.Phi_image is not None:
            if not (1e-6 < self.Phi_image < 0.1):
                checks.append(f"Phi_image magnitude suspicious: {self.Phi_image}")
        if self.Phi_max is not None:
            if not (0.01 < self.Phi_max < 10):
                checks.append(f"Phi_max magnitude suspicious: {self.Phi_max}")
        if checks:
            return (False, "; ".join(checks))
        return (True, "Weak sanity passed (positive finite and plausible magnitudes)")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover and run every bound method starting with 'test_'.
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)