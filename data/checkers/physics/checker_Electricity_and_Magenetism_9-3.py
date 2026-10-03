import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results from solution_instance – mandatory keys listed in the contract
        self.results = getattr(solution_instance, 'results', {})
        # Results from the pseudocode emit
        self.f_x = self.results.get('f_x', None)
        self.f_y = self.results.get('f_y', None)
        self.k = self.results.get('k', None)
        self.p = self.results.get('p', None)
        self.f = self.results.get('f', None)
        self.f_numeric = self.results.get('f_numeric', None)

        # Attempt to extract input givens (problem data) – they may be stored as attributes or in a 'givens' dict
        self.givens = getattr(solution_instance, 'givens', {})
        self.B0 = self.givens.get('B0', getattr(solution_instance, 'B0', None))
        self.R = self.givens.get('R', getattr(solution_instance, 'R', None))
        self.L = self.givens.get('L', getattr(solution_instance, 'L', None))
        self.q = self.givens.get('q', getattr(solution_instance, 'q', None))
        self.E = self.givens.get('E', getattr(solution_instance, 'E', None))
        self.c = self.givens.get('c', getattr(solution_instance, 'c', None))
        self.m_e = self.givens.get('m_e', getattr(solution_instance, 'm_e', None))

    # ----------------------------------------------------------------------
    # Helper: safe numerical comparison using math.isclose
    # ----------------------------------------------------------------------
    def _safe_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Return True if both a and b are numbers and close, False otherwise."""
        if a is None or b is None:
            return False
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # ----------------------------------------------------------------------
    # Interface tests (required keys and types)
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all expected result keys exist."""
        required = ['f_x', 'f_y', 'k', 'p', 'f', 'f_numeric']
        missing = [key for key in required if key not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_results_types(self):
        """Check that each result value is a number (int or float)."""
        for key in ['f_x', 'f_y', 'k', 'p', 'f', 'f_numeric']:
            val = self.results.get(key)
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                return (False, f"{key} is not numeric (type: {type(val).__name__})")
        return (True, "All result values are numeric")

    # ----------------------------------------------------------------------
    # Invariant tests (physical relationships)
    # ----------------------------------------------------------------------
    def test_invariant_focal_length_signs(self):
        """Check that f_x and f_y have opposite signs and their sum is zero (within tolerance)."""
        if self.f_x is None or self.f_y is None:
            return (False, "SKIP: f_x or f_y missing")
        # f_x should be positive (focusing in x-plane for electrons), f_y negative
        if self.f_x <= 0:
            return (False, f"Expected f_x > 0 (focusing plane), got {self.f_x}")
        if self.f_y >= 0:
            return (False, f"Expected f_y < 0 (defocusing plane), got {self.f_y}")
        # Sum should be zero (magnitude equal)
        if not self._safe_close(self.f_x + self.f_y, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"f_x + f_y should be 0, got {self.f_x + self.f_y}")
        return (True, f"Sign and magnitude consistent: f_x={self.f_x}, f_y={self.f_y}")

    def test_invariant_focal_length_relation(self):
        """Check that f (magnitude) equals |f_x| and |f_y|."""
        if self.f is None or self.f_x is None or self.f_y is None:
            return (False, "SKIP: f, f_x or f_y missing")
        if not self._safe_close(self.f, abs(self.f_x)):
            return (False, f"f should equal |f_x|, got f={self.f}, |f_x|={abs(self.f_x)}")
        if not self._safe_close(self.f, abs(self.f_y)):
            return (False, f"f should equal |f_y|, got f={self.f}, |f_y|={abs(self.f_y)}")
        return (True, f"f={self.f} equals |f_x| and |f_y|")

    def test_invariant_momentum_definition(self):
        """If E and c given, check that p = E / c."""
        if self.p is None:
            return (False, "SKIP: p missing")
        if self.E is None or self.c is None:
            return (False, "SKIP: E or c not available")
        expected_p = self.E / self.c
        if not self._safe_close(self.p, expected_p):
            return (False, f"p should equal E/c = {expected_p}, got {self.p}")
        return (True, f"p = {self.p} matches E/c")

    def test_invariant_strength_definition(self):
        """Check that k = |q| B0 / (p R) if required givens exist."""
        if self.k is None or self.p is None:
            return (False, "SKIP: k or p missing")
        if self.B0 is None or self.R is None or self.q is None:
            return (False, "SKIP: B0, R, or q not available")
        qabs = abs(self.q)
        if qabs == 0:
            return (False, "SKIP: charge is zero (cannot compute k)")
        expected_k = qabs * self.B0 / (self.p * self.R)
        if not self._safe_close(self.k, expected_k):
            return (False, f"k should be {expected_k}, got {self.k}")
        return (True, f"k = {self.k} matches |q| B0 / (p R)")

    def test_invariant_focal_length_formula(self):
        """Check that f_x = p R / (|q| B0 L) and f_y = -f_x."""
        if self.f_x is None or self.f_y is None or self.p is None:
            return (False, "SKIP: f_x, f_y or p missing")
        if self.B0 is None or self.R is None or self.L is None or self.q is None:
            return (False, "SKIP: B0, R, L or q not available")
        qabs = abs(self.q)
        if qabs == 0:
            return (False, "SKIP: charge is zero (cannot compute focal length)")
        expected_f_x = self.p * self.R / (qabs * self.B0 * self.L)
        expected_f_y = -expected_f_x
        if not self._safe_close(self.f_x, expected_f_x):
            return (False, f"f_x should be {expected_f_x}, got {self.f_x}")
        if not self._safe_close(self.f_y, expected_f_y):
            return (False, f"f_y should be {expected_f_y}, got {self.f_y}")
        return (True, f"f_x and f_y correctly derived")

    # ----------------------------------------------------------------------
    # Modeling tests (quality of model implementation)
    # ----------------------------------------------------------------------
    def test_modeling_thin_lens_relation(self):
        """Check that f = 1/(k L) for the focusing plane."""
        if self.k is None or self.L is None:
            return (False, "SKIP: k or L missing")
        if self.f is None:
            return (False, "SKIP: f missing")
        expected_f = 1.0 / (self.k * self.L)
        if not self._safe_close(self.f, expected_f):
            return (False, f"Thin lens relation: f should be 1/(k L) = {expected_f}, got {self.f}")
        return (True, f"f = {self.f} ≈ 1/(k L) = {expected_f}")

    def test_modeling_focusing_plane_sign(self):
        """Verify focusing plane sign based on charge sign.

        For q < 0 (electron), Lorentz force gives focusing in x-plane (f_x positive, f_y negative).
        For q > 0, opposite: focusing in y-plane (f_y positive, f_x negative).
        The exact signs are derived from the physics of the quadrupole field.
        """
        if self.f_x is None or self.f_y is None:
            return (False, "SKIP: f_x or f_y missing")
        if self.q is None:
            return (False, "SKIP: charge q not available")
        # Derive expected sign of f_x from charge: for q<0, focusing plane is x, so f_x > 0, f_y < 0.
        if self.q < 0:
            focus_x_sign = 1   # f_x positive
            focus_y_sign = -1  # f_y negative
        else:
            focus_x_sign = -1  # f_x negative
            focus_y_sign = 1   # f_y positive
        # Check that signs match
        if self.f_x * focus_x_sign <= 0:
            return (False, f"Expected f_x sign {focus_x_sign}, got f_x={self.f_x}")
        if self.f_y * focus_y_sign <= 0:
            return (False, f"Expected f_y sign {focus_y_sign}, got f_y={self.f_y}")
        return (True, f"Focusing plane signs correct: f_x={self.f_x}, f_y={self.f_y}")

    def test_modeling_f_numeric_consistency(self):
        """If givens for numeric part exist, check that f_numeric matches the formula."""
        if self.f_numeric is None:
            return (False, "SKIP: f_numeric missing")
        if self.B0 is None or self.R is None or self.L is None or self.E is None or self.c is None:
            return (False, "SKIP: cannot compute expected f without B0, R, L, E, c")
        # Compute p = E/c, then f = p*R/(|q|*B0*L). Use charge magnitude if available, else default e.
        qabs = abs(self.q) if self.q is not None else 1.602176634e-19   # elementary charge
        p = self.E / self.c
        expected_f = p * self.R / (qabs * self.B0 * self.L)
        if not self._safe_close(self.f_numeric, expected_f):
            return (False, f"f_numeric should be {expected_f}, got {self.f_numeric}")
        return (True, f"f_numeric = {self.f_numeric} matches formula")

    # ----------------------------------------------------------------------
    # Monotonicity tests (explicitly perturb an input and compare perturbed output to baseline)
    # ----------------------------------------------------------------------
    def test_monotonicity_f_x_increases_with_p(self):
        """If p increases by 10%, f_x should increase by ~10%."""
        if self.f_x is None or self.p is None:
            return (False, "SKIP: f_x or p missing")
        if self.B0 is None or self.R is None or self.L is None or self.q is None:
            return (False, "SKIP: B0, R, L or q not available")
        qabs = abs(self.q)
        if qabs == 0:
            return (False, "SKIP: charge zero")
        # Baseline f_x using formula (should match stored value, but we recompute for consistency)
        f_x_baseline = self.p * self.R / (qabs * self.B0 * self.L)
        # Perturb p by +10%
        p_perturbed = self.p * 1.1
        f_x_perturbed = p_perturbed * self.R / (qabs * self.B0 * self.L)
        if not (f_x_perturbed > f_x_baseline):
            return (False, f"f_x should increase with p: baseline {f_x_baseline}, perturbed {f_x_perturbed}")
        # Also check that ratio matches
        expected_ratio = f_x_perturbed / f_x_baseline
        if not self._safe_close(expected_ratio, 1.1, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Expected ratio 1.1, got {expected_ratio}")
        return (True, f"f_x increases with p (baseline {f_x_baseline}, perturbed {f_x_perturbed})")

    def test_monotonicity_f_x_increases_with_R(self):
        """If R increases by 10%, f_x should increase by ~10%."""
        if self.f_x is None or self.p is None:
            return (False, "SKIP: f_x or p missing")
        if self.B0 is None or self.R is None or self.L is None or self.q is None:
            return (False, "SKIP: B0, R, L or q not available")
        qabs = abs(self.q)
        if qabs == 0:
            return (False, "SKIP: charge zero")
        f_x_baseline = self.p * self.R / (qabs * self.B0 * self.L)
        R_perturbed = self.R * 1.1
        f_x_perturbed = self.p * R_perturbed / (qabs * self.B0 * self.L)
        if not (f_x_perturbed > f_x_baseline):
            return (False, f"f_x should increase with R: baseline {f_x_baseline}, perturbed {f_x_perturbed}")
        expected_ratio = f_x_perturbed / f_x_baseline
        if not self._safe_close(expected_ratio, 1.1, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Expected ratio 1.1, got {expected_ratio}")
        return (True, f"f_x increases with R (baseline {f_x_baseline}, perturbed {f_x_perturbed})")

    def test_monotonicity_f_x_decreases_with_L(self):
        """If L increases by 10%, f_x should decrease to ~1/1.1 times."""
        if self.f_x is None or self.p is None:
            return (False, "SKIP: f_x or p missing")
        if self.B0 is None or self.R is None or self.L is None or self.q is None:
            return (False, "SKIP: B0, R, L or q not available")
        qabs = abs(self.q)
        if qabs == 0:
            return (False, "SKIP: charge zero")
        f_x_baseline = self.p * self.R / (qabs * self.B0 * self.L)
        L_perturbed = self.L * 1.1
        f_x_perturbed = self.p * self.R / (qabs * self.B0 * L_perturbed)
        if not (f_x_perturbed < f_x_baseline):
            return (False, f"f_x should decrease with L: baseline {f_x_baseline}, perturbed {f_x_perturbed}")
        expected_ratio = f_x_perturbed / f_x_baseline
        if not self._safe_close(expected_ratio, 1.0/1.1, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Expected ratio 1/1.1 ≈ {1.0/1.1}, got {expected_ratio}")
        return (True, f"f_x decreases with L (baseline {f_x_baseline}, perturbed {f_x_perturbed})")

    def test_monotonicity_k_decreases_with_p(self):
        """If p increases by 10%, k should decrease to ~1/1.1 times."""
        if self.k is None or self.p is None:
            return (False, "SKIP: k or p missing")
        if self.B0 is None or self.R is None or self.q is None:
            return (False, "SKIP: B0, R or q not available")
        qabs = abs(self.q)
        if qabs == 0:
            return (False, "SKIP: charge zero")
        k_baseline = qabs * self.B0 / (self.p * self.R)
        p_perturbed = self.p * 1.1
        k_perturbed = qabs * self.B0 / (p_perturbed * self.R)
        if not (k_perturbed < k_baseline):
            return (False, f"k should decrease with p: baseline {k_baseline}, perturbed {k_perturbed}")
        expected_ratio = k_perturbed / k_baseline
        if not self._safe_close(expected_ratio, 1.0/1.1, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Expected ratio 1/1.1 ≈ {1.0/1.1}, got {expected_ratio}")
        return (True, f"k decreases with p (baseline {k_baseline}, perturbed {k_perturbed})")

    # ----------------------------------------------------------------------
    # Answer sanity test (weak and secondary)
    # ----------------------------------------------------------------------
    def test_answer_sanity_f_numeric_range(self):
        """Check that f_numeric is a plausible positive number (no exact comparison)."""
        if self.f_numeric is None:
            return (False, "SKIP: f_numeric missing")
        if not isinstance(self.f_numeric, (int, float)):
            return (False, "f_numeric is not a number")
        if math.isnan(self.f_numeric) or math.isinf(self.f_numeric):
            return (False, f"f_numeric is NaN or infinite")
        if self.f_numeric <= 0:
            return (False, f"f_numeric should be positive, got {self.f_numeric}")
        # For a quadrupole with these parameters, expect a few meters (order 1-100)
        if self.f_numeric < 0.1 or self.f_numeric > 100.0:
            return (False, f"f_numeric outside expected range [0.1, 100]: {self.f_numeric}")
        return (True, f"f_numeric = {self.f_numeric} is a plausible positive value in [0.1,100]")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self):
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
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)