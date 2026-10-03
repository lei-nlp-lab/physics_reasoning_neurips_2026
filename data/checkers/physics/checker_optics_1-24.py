import math
import traceback

class PhysicsChecker:
    """
    Audits a solution_instance for the two-lens problem:
    find f2/f1 such that two converging lenses produce an inverted,
    virtual image at the object position and of the same size.
    """

    def __init__(self, solution_instance):
        # Extract results and other attributes safely
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        
        # Result keys expected from the pseudocode [emit] contract
        self.f2_div_f1 = self.results.get('f2_div_f1', None)
        
        # Optional additional values (may be present)
        self.f1 = self.results.get('f1', None) or self.givens.get('f1', None)
        self.f2 = self.results.get('f2', None) or self.givens.get('f2', None)
        
        self.d1 = self.results.get('d1', None) or self.givens.get('d1', None)
        self.d2 = self.results.get('d2', None) or self.givens.get('d2', None)
        
        self.s1 = self.results.get('s1', None) or self.givens.get('u1', None)
        self.s1_prime = self.results.get('s1_prime', None) or self.givens.get('v1', None)
        self.s2 = self.results.get('s2', None) or self.givens.get('u2', None)
        self.s2_prime = self.results.get('s2_prime', None) or self.givens.get('v2', None)
        
        self.m1 = self.results.get('m1', None)
        self.m2 = self.results.get('m2', None)
        self.m_total = self.results.get('m_total', None)

    # ------------------------------------------------------------------
    # Interface tests – required keys and types
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.f2_div_f1 is None:
            return (False, "Key 'f2_div_f1' missing from results.")
        return (True, "Key 'f2_div_f1' present.")

    def test_interface_key_f2_div_f1_type(self):
        if self.f2_div_f1 is None:
            return (False, "SKIP: f2_div_f1 not available.")
        if not isinstance(self.f2_div_f1, (int, float)):
            return (False, f"f2_div_f1 is not numeric, got {type(self.f2_div_f1)}.")
        return (True, "f2_div_f1 is numeric.")

    # ------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_thin_lens_lens1(self):
        """Thin lens equation for lens1: 1/f1 = 1/s1 + 1/s1'"""
        if any(x is None for x in (self.f1, self.s1, self.s1_prime)):
            return (False, "SKIP: missing f1, s1 or s1'.")
        if self.s1 == 0 or self.s1_prime == 0:
            return (False, "Division by zero: s1 or s1' is zero.")
        left = 1.0 / self.f1
        right = 1.0 / self.s1 + 1.0 / self.s1_prime
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Lens1 equation satisfied: 1/f1 ≈ {right:.3f}")
        return (False, f"Lens1 equation violated: 1/f1 = {left:.3f}, RHS = {right:.3f}")

    def test_invariant_thin_lens_lens2(self):
        """Thin lens equation for lens2: 1/f2 = 1/s2 + 1/s2'"""
        if any(x is None for x in (self.f2, self.s2, self.s2_prime)):
            return (False, "SKIP: missing f2, s2 or s2'.")
        if self.s2 == 0 or self.s2_prime == 0:
            return (False, "Division by zero: s2 or s2' is zero.")
        left = 1.0 / self.f2
        right = 1.0 / self.s2 + 1.0 / self.s2_prime
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Lens2 equation satisfied: 1/f2 ≈ {right:.3f}")
        return (False, f"Lens2 equation violated: 1/f2 = {left:.3f}, RHS = {right:.3f}")

    def test_invariant_total_magnification_minus_one(self):
        """Total magnification must equal -1."""
        if self.m_total is None:
            return (False, "SKIP: m_total not available.")
        if math.isclose(self.m_total, -1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"m_total = {self.m_total} ≈ -1")
        return (False, f"m_total = {self.m_total}, expected -1.")

    def test_invariant_magnification_product(self):
        """m1 * m2 must equal -1."""
        if self.m1 is None or self.m2 is None:
            return (False, "SKIP: missing m1 or m2.")
        product = self.m1 * self.m2
        if math.isclose(product, -1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"m1*m2 = {product} ≈ -1")
        return (False, f"m1*m2 = {product}, expected -1.")

    def test_invariant_final_image_at_object_position(self):
        """Final image position relative to lens2: s2' = -d2 (object at origin)."""
        if self.s2_prime is None or self.d2 is None:
            return (False, "SKIP: missing s2' or d2.")
        if self.d2 == 0:
            return (False, "d2 is zero, cannot compute ratio.")
        expected_s2_prime = -self.d2
        if math.isclose(self.s2_prime, expected_s2_prime, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"s2' = {self.s2_prime} ≈ -d2 = {expected_s2_prime}")
        return (False, f"s2' = {self.s2_prime}, expected -d2 = {expected_s2_prime}")

    # ------------------------------------------------------------------
    # Modeling / trace tests – check against explanation steps
    # ------------------------------------------------------------------
    def test_modeling_intermediate_image_distance(self):
        """s2 = d2 - d1 - s1' (distance from lens2 to intermediate image)."""
        if any(x is None for x in (self.d1, self.d2, self.s1_prime, self.s2)):
            return (False, "SKIP: missing d1, d2, s1' or s2.")
        expected_s2 = self.d2 - self.d1 - self.s1_prime
        if math.isclose(self.s2, expected_s2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"s2 = {self.s2} ≈ {expected_s2}")
        return (False, f"s2 = {self.s2}, expected {expected_s2}")

    def test_modeling_object_distance_lens1(self):
        """s1 should equal d1 (object at origin, lens1 at d1)."""
        if self.s1 is None or self.d1 is None:
            return (False, "SKIP: missing s1 or d1.")
        if math.isclose(self.s1, self.d1, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"s1 = {self.s1} ≈ d1 = {self.d1}")
        return (False, f"s1 = {self.s1}, expected d1 = {self.d1}")

    def test_trace_focal_length_ratio(self):
        """Check f2/f1 ≈ 9 from the results."""
        if self.f2 is not None and self.f1 is not None and self.f1 != 0:
            ratio = self.f2 / self.f1
            if math.isclose(ratio, 9.0, rel_tol=1e-2):
                return (True, f"f2/f1 = {ratio} ≈ 9")
            return (False, f"f2/f1 = {ratio}, expected ~9.")
        elif self.f2_div_f1 is not None:
            if math.isclose(self.f2_div_f1, 9.0, rel_tol=1e-2):
                return (True, f"f2_div_f1 = {self.f2_div_f1} ≈ 9")
            return (False, f"f2_div_f1 = {self.f2_div_f1}, expected ~9.")
        else:
            return (False, "SKIP: no data to check focal length ratio.")

    def test_trace_lens2_virtual_image_sign(self):
        """s2' must be negative (virtual image)."""
        if self.s2_prime is None:
            return (False, "SKIP: missing s2'.")
        if self.s2_prime < 0:
            return (True, f"s2' = {self.s2_prime} < 0 (virtual image)")
        return (False, f"s2' = {self.s2_prime} >= 0, expected negative.")

    def test_trace_lens2_object_inside_focal_length(self):
        """For virtual image, object must be inside focal length: s2 < f2."""
        if any(x is None for x in (self.s2, self.f2)):
            return (False, "SKIP: missing s2 or f2.")
        if self.s2 < self.f2:
            return (True, f"s2 = {self.s2} < f2 = {self.f2} (object inside focal length)")
        return (False, f"s2 = {self.s2} >= f2 = {self.f2}, object outside focal length.")

    # ------------------------------------------------------------------
    # Monotonicity test – perturb one parameter and check trend
    # ------------------------------------------------------------------
    def test_monotonicity_lens1_image_distance(self):
        """
        For lens1 with fixed f1, if object distance s1 increases,
        the image distance s1' decreases (real image case).
        We compute f1 from given s1 and s1', then artificially increase s1
        by 10% and recompute the new s1' using the same f1.
        """
        if any(x is None for x in (self.s1, self.s1_prime)):
            return (False, "SKIP: missing s1 or s1' to compute f1.")
        # Assume s1 != 0 and s1_prime != 0
        try:
            f1_derived = 1.0 / (1.0/self.s1 + 1.0/self.s1_prime)
        except ZeroDivisionError:
            return (False, "Division by zero when computing f1.")
        if f1_derived <= 0:
            return (False, f"Derived f1 = {f1_derived} not positive.")
        # Perturb s1 by +10%
        s1_new = self.s1 * 1.10
        # Solve thin lens for new s1'
        # 1/f1 = 1/s1_new + 1/s1' => 1/s1' = 1/f1 - 1/s1_new
        inv_rhs = 1.0/f1_derived - 1.0/s1_new
        if inv_rhs <= 0:
            return (False, f"Perturbation leads to non-positive 1/s1' ({inv_rhs}).")
        s1_prime_new = 1.0 / inv_rhs
        # For converging lens with real object (s1 > 0) and real image (s1'>0)
        # increasing s1 should decrease s1' (since s1' = f1 * s1 / (s1 - f1) if s1 > f1)
        # Check that new s1' < original s1'
        if s1_prime_new < self.s1_prime:
            return (True, f"After +10% s1, s1' decreased from {self.s1_prime} to {s1_prime_new}")
        else:
            return (False, f"After +10% s1, s1' increased from {self.s1_prime} to {s1_prime_new}, expected decrease.")

    # ------------------------------------------------------------------
    # Weak answer sanity – only sign, finiteness, broad range
    # ------------------------------------------------------------------
    def test_answer_sanity_f2_div_f1(self):
        if self.f2_div_f1 is None:
            return (False, "SKIP: f2_div_f1 not available.")
        if not math.isfinite(self.f2_div_f1):
            return (False, f"f2_div_f1 = {self.f2_div_f1} is not finite.")
        if self.f2_div_f1 <= 0:
            return (False, f"f2_div_f1 = {self.f2_div_f1} is non-positive.")
        if self.f2_div_f1 > 100:
            return (False, f"f2_div_f1 = {self.f2_div_f1} suspiciously large (>100).")
        return (True, f"f2_div_f1 = {self.f2_div_f1} is positive and finite.")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)