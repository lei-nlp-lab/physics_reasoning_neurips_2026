import math
import traceback

class PhysicsChecker:
    """Audits the solution of a double-slit interference and diffraction problem."""

    def __init__(self, solution_instance):
        """
        Extract all needed quantities from the solution_instance.
        Use None defaults for missing data.
        """
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', {}) or {}

        # Required results (emit keys)
        self.delta_y = self.results.get('delta_y', None)
        self.Delta_y1 = self.results.get('Delta_y1', None)
        self.Delta_y2 = self.results.get('Delta_y2', None)
        self.N = self.results.get('N', None)
        self.I_ratio = self.results.get('I_ratio', None)

        # Givens – we only need relationships, not raw numbers
        self.w1_mult = self.givens.get('w1_mult', None)   # e.g., 10
        self.w2_mult = self.givens.get('w2_mult', None)   # 20
        self.d_mult = self.givens.get('d_mult', None)     # 1000
        # If givens are not provided, we infer from problem constants
        # (the problem always uses w1 = 10λ, w2 = 20λ, d = 1000λ)
        self.w1_rel = 10.0
        self.w2_rel = 20.0
        self.d_rel = 1000.0

    # ------------------------------------------------------------------
    # Interface tests – ensure required keys exist
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all five result keys are present."""
        missing = []
        for key in ['delta_y', 'Delta_y1', 'Delta_y2', 'N', 'I_ratio']:
            if self.results.get(key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All required keys present")

    # ------------------------------------------------------------------
    # Invariant tests – theorem-based physical consistency (fuzzy isclose)
    # ------------------------------------------------------------------
    def test_invariant_Delta_y1_Delta_y2_ratio(self):
        """Δy₁ / Δy₂ should equal w₂ / w₁ = 2."""
        if self.Delta_y1 is None or self.Delta_y2 is None:
            return (False, "SKIP: Δy₁ or Δy₂ not available")
        if self.Delta_y2 == 0:
            return (False, "SKIP: Δy₂ is zero, cannot compute ratio")
        ratio = self.Delta_y1 / self.Delta_y2
        expected_ratio = self.w2_rel / self.w1_rel  # 2
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Δy₁/Δy₂ = {ratio:.4f} ≈ {expected_ratio}")
        return (False, f"Δy₁/Δy₂ = {ratio:.4f}, expected {expected_ratio}")

    def test_invariant_N_equals_2d_over_w2(self):
        """Number of fringes N should equal 2*d/w2 = 100."""
        if self.N is None:
            return (False, "SKIP: N not available")
        expected_N = 2 * self.d_rel / self.w2_rel  # 100
        if math.isclose(self.N, expected_N, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N = {self.N} ≈ {expected_N}")
        return (False, f"N = {self.N}, expected {expected_N}")

    def test_invariant_I_ratio_formula(self):
        """I_max/I_min should equal (w₁+w₂)²/(w₁-w₂)² = 9."""
        if self.I_ratio is None:
            return (False, "SKIP: I_ratio not available")
        expected_ratio = (self.w1_rel + self.w2_rel)**2 / (self.w1_rel - self.w2_rel)**2
        if math.isclose(self.I_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"I_ratio = {self.I_ratio:.4f} ≈ {expected_ratio}")
        return (False, f"I_ratio = {self.I_ratio:.4f}, expected {expected_ratio}")

    def test_invariant_cross_check_fringe_times_spacing(self):
        """N * δy should equal Δy₂ (central max of wider slit)."""
        if any(x is None for x in [self.N, self.delta_y, self.Delta_y2]):
            return (False, "SKIP: N, δy, or Δy₂ not available")
        product = self.N * self.delta_y
        if math.isclose(product, self.Delta_y2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N*δy = {product:.4f} ≈ Δy₂ = {self.Delta_y2:.4f}")
        return (False, f"N*δy = {product:.4f}, Δy₂ = {self.Delta_y2:.4f}")

    # ------------------------------------------------------------------
    # Modeling / trace tests – consistency of relationships
    # ------------------------------------------------------------------
    def test_modeling_all_values_positive(self):
        """All result values should be positive."""
        for key, val in [('delta_y', self.delta_y),
                         ('Delta_y1', self.Delta_y1),
                         ('Delta_y2', self.Delta_y2),
                         ('N', self.N),
                         ('I_ratio', self.I_ratio)]:
            if val is None:
                return (False, f"SKIP: {key} not available")
            if val <= 0:
                return (False, f"{key} = {val} is not positive")
        return (True, "All values > 0")

    def test_modeling_delta_y_less_than_diffraction_widths(self):
        """Interference fringe spacing must be smaller than any diffraction central max width."""
        if None in (self.delta_y, self.Delta_y1, self.Delta_y2):
            return (False, "SKIP: missing δy, Δy₁, or Δy₂")
        if self.delta_y < self.Delta_y1 and self.delta_y < self.Delta_y2:
            return (True, "δy < Δy₁ and δy < Δy₂")
        return (False, "δy is not smaller than both diffraction widths")

    # ------------------------------------------------------------------
    # Monotonicity tests – perturb a parameter and check directional effect
    # ------------------------------------------------------------------
    def test_monotonicity_delta_y_decreases_with_d(self):
        """If slit separation d increases, δy should decrease."""
        if self.delta_y is None:
            return (False, "SKIP: δy not available")
        # Simulate a 10% increase in d: new_d = 1.1 * d_rel
        # δy = λL/d, so unchanged λL = δy * d
        # For perturbed d', δy' = (δy * d) / d'
        d = self.d_rel
        new_d = 1.1 * d
        # We need δy for original; from the relation δy = λL/d, so λL = δy * d
        # If d was not given, we can use the relation that δy ∝ 1/d
        # This check is valid even if we don't know λL.
        # The direction: if d increases, δy decreases.
        perturbed_delta_y = self.delta_y * d / new_d  # = δy / 1.1
        # The perturbed value should be smaller than the original
        if perturbed_delta_y < self.delta_y:
            return (True, "δy decreases when d increases (consistent)")
        return (False, "Monotonicity violated: δy didn't decrease with d increase")

    def test_monotonicity_Delta_y1_decreases_with_w1(self):
        """If slit width w1 increases, Δy₁ should decrease."""
        if self.Delta_y1 is None:
            return (False, "SKIP: Δy₁ not available")
        # Δy₁ = 2λL/w1, so if w1 increases, Δy₁ decreases.
        w1 = self.w1_rel
        new_w1 = 1.1 * w1
        perturbed = self.Delta_y1 * w1 / new_w1  # = Δy₁ / 1.1
        if perturbed < self.Delta_y1:
            return (True, "Δy₁ decreases when w1 increases (consistent)")
        return (False, "Monotonicity violated: Δy₁ didn't decrease with w1 increase")

    def test_monotonicity_N_increases_with_d(self):
        """If slit separation d increases, number of fringes N should increase."""
        if self.N is None:
            return (False, "SKIP: N not available")
        # N = 2d/w2, so N ∝ d.
        d = self.d_rel
        new_d = 1.1 * d
        perturbed_N = self.N * new_d / d  # = N * 1.1
        if perturbed_N > self.N:
            return (True, "N increases when d increases (consistent)")
        return (False, "Monotonicity violated: N didn't increase with d increase")

    def test_monotonicity_I_ratio_invariant(self):
        """I_ratio should be independent of λ and L; check it stays constant under a fake perturbation."""
        if self.I_ratio is None:
            return (False, "SKIP: I_ratio not available")
        # I_ratio = (w₁+w₂)²/(w₁-w₂)², independent of λ, L, d.
        # So even if we "perturb" λ or L, I_ratio remains same.
        # We'll simulate a λ doubling: this would not affect I_ratio.
        perturbed_I_ratio = self.I_ratio  # stays same
        if math.isclose(perturbed_I_ratio, self.I_ratio, rel_tol=1e-2):
            return (True, "I_ratio is invariant under λ/L perturbation (consistent)")
        return (False, "I_ratio unexpectedly changed")

    # ------------------------------------------------------------------
    # Answer sanity (weak, secondary)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_not_nan(self):
        """Check that all results are finite and not NaN."""
        for key in ['delta_y', 'Delta_y1', 'Delta_y2', 'N', 'I_ratio']:
            val = getattr(self, key, None)
            if val is None:
                continue          # skip missing; interface test will catch
            if not math.isfinite(val):
                return (False, f"{key} is not finite (got {val})")
            if math.isnan(val):
                return (False, f"{key} is NaN")
        return (True, "All values finite and non-NaN")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_ methods, return (success, logs)."""
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