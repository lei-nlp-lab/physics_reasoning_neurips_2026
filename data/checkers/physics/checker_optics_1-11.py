import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        self.f = self.results.get('f', None)
        self.D = self.results.get('D', None)
        self.F = self.results.get('F', None)

        # Extract givens from various possible locations
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            givens = getattr(solution_instance, 'inputs', None)
        if givens is None:
            givens = {}
        # Object distance
        self.u = givens.get('object_distance', None)
        if self.u is None:
            self.u = givens.get('u', None)
        # Object length (linear dimension, either height or length)
        self.object_length = givens.get('object_length', None)
        if self.object_length is None:
            self.object_length = givens.get('object_height', None)
        # Image length
        self.image_length = givens.get('image_length', None)
        if self.image_length is None:
            self.image_length = givens.get('image_height', None)
        # Sensor shift (amount film plate is moved)
        self.shift = givens.get('sensor_shift', None)
        if self.shift is None:
            self.shift = givens.get('shift', None)
        # Blur circle diameter
        self.blur_c = givens.get('blur_circle_diameter', None)
        if self.blur_c is None:
            self.blur_c = givens.get('c', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all expected result keys are present and not None."""
        missing = []
        for key in ['f', 'D', 'F']:
            if key not in self.results or self.results[key] is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present and non-None")

    def test_interface_required_givens(self):
        """Check that all necessary givens are provided (skip if missing)."""
        missing = []
        checks = [
            ('u', self.u),
            ('object_length', self.object_length),
            ('image_length', self.image_length),
            ('shift', self.shift),
            ('blur_c', self.blur_c)
        ]
        for name, val in checks:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {missing} (physics tests will be skipped)")
        return (True, "All required givens present")

    # ------------------------------------------------------------------
    # Invariant / theorem-based checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_thin_lens(self):
        """Check thin lens equation and magnification: 1/f = 1/u + 1/v, with v = m*u."""
        if None in (self.u, self.object_length, self.image_length, self.f):
            return (True, "SKIP: missing givens or f")
        if self.object_length == 0:
            return (False, "Object length zero, cannot compute magnification")
        m = self.image_length / self.object_length
        v = m * self.u
        if v <= 0:
            return (False, "Computed image distance v <= 0")
        f_expected = 1.0 / (1.0/self.u + 1.0/v)
        if math.isclose(self.f, f_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Thin lens and magnification consistent: f={self.f}, expected ~{f_expected:.4f}")
        return (False, f"Thin lens mismatch: f={self.f}, expected ~{f_expected:.4f}")

    def test_invariant_defocus_blur(self):
        """Check defocus blur geometry: c = D * (v' - v) / v."""
        if None in (self.u, self.object_length, self.image_length, self.shift, self.blur_c, self.D):
            return (True, "SKIP: missing givens or D")
        if self.object_length == 0:
            return (False, "Object length zero")
        m = self.image_length / self.object_length
        v = m * self.u
        if v <= 0:
            return (False, "Image distance v <= 0")
        v_prime = v + self.shift
        c_expected = self.D * (v_prime - v) / v
        if math.isclose(self.blur_c, c_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Defocus blur consistent: c={self.blur_c}, expected ~{c_expected:.4f}")
        return (False, f"Defocus blur mismatch: c={self.blur_c}, expected ~{c_expected:.4f}")

    def test_invariant_F_number(self):
        """Check that F = f / D."""
        if None in (self.f, self.D, self.F):
            return (True, "SKIP: missing f, D or F")
        if self.D == 0:
            return (False, "Aperture diameter D is zero, F-number undefined")
        F_expected = self.f / self.D
        if math.isclose(self.F, F_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"F-number consistent: F={self.F}, expected ~{F_expected:.4f}")
        return (False, f"F-number mismatch: F={self.F}, expected ~{F_expected:.4f}")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_magnification_ratio(self):
        """Check that the user's magnification ratio matches the linear dimensions."""
        if None in (self.object_length, self.image_length):
            return (True, "SKIP: missing object/image lengths")
        if self.object_length == 0:
            return (False, "Object length zero")
        m = self.image_length / self.object_length
        # Expected m = 0.2 for this problem; we only check it is consistent with u and v if f is available
        if None not in (self.u, self.f):
            if self.f <= 0:
                return (False, "focal length non-positive")
            # v from thin lens: v = 1/(1/f - 1/u)
            v = 1.0 / (1.0/self.f - 1.0/self.u)
            m_from_v = v / self.u
            if math.isclose(m, m_from_v, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Magnification consistent: {m:.4f} vs v/u={m_from_v:.4f}")
            return (False, f"Magnification mismatch: object/image ratio {m:.4f}, but thin lens gives {m_from_v:.4f}")
        else:
            # Just check m is plausible (0 < m < 10 usually)
            if 0 < m < 10:
                return (True, f"Magnification ratio plausible: {m:.4f}")
            return (False, f"Magnification {m:.4f} out of plausible range")

    def test_modeling_aperture_positivity(self):
        """Check that aperture diameter D is positive."""
        if self.D is None:
            return (True, "SKIP: D not available")
        if self.D > 0:
            return (True, f"Aperture diameter D={self.D} positive")
        return (False, f"Aperture diameter D={self.D} non-positive")

    # ------------------------------------------------------------------
    # Monotonicity tests (direction/trend checks via explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_blur_increases_with_shift(self):
        """If the sensor shift increases, the blur circle diameter should increase (all else equal)."""
        if None in (self.u, self.object_length, self.image_length, self.shift, self.blur_c, self.D):
            return (True, "SKIP: missing data for monotonicity check")
        if self.object_length == 0:
            return (False, "Object length zero")
        m = self.image_length / self.object_length
        v = m * self.u
        if v <= 0:
            return (False, "Image distance v <= 0")
        # baseline blur
        original_blur = self.blur_c
        # increase shift by 10%
        increased_shift = self.shift * 1.1
        # recompute blur using same D and v
        increased_blur = self.D * (increased_shift) / v  # because v' - v = shift
        if increased_blur > original_blur:
            return (True, f"Monotonicity holds: 10% increase in shift leads to larger blur ({increased_blur:.4f} > {original_blur:.4f})")
        return (False, f"Monotonicity violation: increased shift ({self.shift} -> {increased_shift:.4f}) yields blur {increased_blur:.4f} <= {original_blur:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one such method; no isclose, no derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_plausibility(self):
        """Weak sanity: focal length positive, F-number in plausible range [1,32]."""
        failures = []
        if self.f is not None and self.f <= 0:
            failures.append(f"focal length f={self.f} non-positive")
        if self.F is not None and not (1 <= self.F <= 32):
            failures.append(f"F-number {self.F} out of typical range [1,32]")
        if failures:
            return (False, "; ".join(failures))
        return (True, "Focal length and F-number appear plausible")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Collect all bound methods starting with 'test_'
        test_methods = [name for name in dir(self) if name.startswith('test_')]
        # Sort for deterministic order
        test_methods.sort()
        for name in test_methods:
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