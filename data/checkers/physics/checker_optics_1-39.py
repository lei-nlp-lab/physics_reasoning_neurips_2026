import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract the results dictionary
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}

        # Required result keys per the pseudocode [emit] contract
        self.flux_ratio = results.get('flux_ratio', None)
        self.f_no_increase = results.get('f_no_increase', None)

        # Attempt to obtain input parameters (givens)
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            givens = {}

        self.r = givens.get('r', None)
        self.f = givens.get('f', None)
        self.theta = givens.get('theta', None)  # do NOT default to 0.01

        # Fallback: try to find r, f, theta directly on the solution instance
        if self.r is None:
            self.r = getattr(solution_instance, 'r', None)
        if self.f is None:
            self.f = getattr(solution_instance, 'f', None)
        if self.theta is None:
            self.theta = getattr(solution_instance, 'theta', None)

        # Also try from results (some solutions store givens there)
        if self.r is None:
            self.r = results.get('r', None)
        if self.f is None:
            self.f = results.get('f', None)
        if self.theta is None:
            self.theta = results.get('theta', None)

        # Explanation steps (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_flux_ratio_exists(self):
        """Check that 'flux_ratio' key is present in results."""
        if self.flux_ratio is None:
            return (False, "flux_ratio is missing from results")
        return (True, "flux_ratio present")

    def test_interface_f_no_increase_exists(self):
        """Check that 'f_no_increase' key is present in results."""
        if self.f_no_increase is None:
            return (False, "f_no_increase is missing from results")
        return (True, "f_no_increase present")

    def test_interface_flux_ratio_type(self):
        """Check that flux_ratio is numeric (int or float)."""
        if self.flux_ratio is None:
            return (False, "SKIP: flux_ratio not available")
        if not isinstance(self.flux_ratio, (int, float)):
            return (False, f"flux_ratio has type {type(self.flux_ratio).__name__}, expected int or float")
        if math.isnan(self.flux_ratio) or math.isinf(self.flux_ratio):
            return (False, "flux_ratio is NaN or Inf")
        return (True, "flux_ratio type and finiteness OK")

    def test_interface_f_no_increase_type(self):
        """Check that f_no_increase is numeric (int or float)."""
        if self.f_no_increase is None:
            return (False, "SKIP: f_no_increase not available")
        if not isinstance(self.f_no_increase, (int, float)):
            return (False, f"f_no_increase has type {type(self.f_no_increase).__name__}, expected int or float")
        if math.isnan(self.f_no_increase) or math.isinf(self.f_no_increase):
            return (False, "f_no_increase is NaN or Inf")
        return (True, "f_no_increase type and finiteness OK")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_flux_ratio_positive(self):
        """Flux ratio must be positive."""
        if self.flux_ratio is None:
            return (False, "SKIP: flux_ratio not available")
        if self.flux_ratio <= 0:
            return (False, f"flux_ratio = {self.flux_ratio} is not positive")
        return (True, "flux_ratio positive")

    def test_invariant_f_no_increase_positive(self):
        """f_no_increase must be positive."""
        if self.f_no_increase is None:
            return (False, "SKIP: f_no_increase not available")
        if self.f_no_increase <= 0:
            return (False, f"f_no_increase = {self.f_no_increase} is not positive")
        return (True, "f_no_increase positive")

    def test_invariant_flux_ratio_formula(self):
        """
        Check that flux_ratio ≈ 4 r^2 / (f^2 θ^2).
        Only runs if r, f, and theta are all available.
        """
        if any(x is None for x in (self.flux_ratio, self.r, self.f, self.theta)):
            return (False, "SKIP: missing r, f, or theta")
        if self.r <= 0 or self.f <= 0 or self.theta <= 0:
            return (False, "r, f, theta must be positive")
        expected = (4 * self.r ** 2) / (self.f ** 2 * self.theta ** 2)
        if not math.isclose(self.flux_ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"flux_ratio {self.flux_ratio} not close to expected {expected}")
        return (True, "flux_ratio matches 4r^2/(f^2 θ^2)")

    def test_invariant_f_no_increase_formula(self):
        """
        Check that f_no_increase ≈ 2 r / θ.
        Only runs if r and theta are available.
        """
        if any(x is None for x in (self.f_no_increase, self.r, self.theta)):
            return (False, "SKIP: missing r or theta")
        if self.r <= 0 or self.theta <= 0:
            return (False, "r and theta must be positive")
        expected = 2 * self.r / self.theta
        if not math.isclose(self.f_no_increase, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"f_no_increase {self.f_no_increase} not close to expected {expected}")
        return (True, "f_no_increase matches 2r/θ")

    def test_invariant_consistency_equal_flux(self):
        """
        If f equals f_no_increase, then flux_ratio should be 1.
        Only runs when both f and f_no_increase are given.
        """
        if any(x is None for x in (self.flux_ratio, self.f, self.f_no_increase)):
            return (False, "SKIP: missing one of flux_ratio, f, f_no_increase")
        if not math.isclose(self.f, self.f_no_increase, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "SKIP: f and f_no_increase are not equal")
        if not math.isclose(self.flux_ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"f ≈ f_no_increase but flux_ratio = {self.flux_ratio}, expected ~1")
        return (True, "When f = f_no_increase, flux_ratio ≈ 1")

    # ----------------------------------------------------------------------
    # Modelling / trace tests (from pseudocode or explanation_steps)
    # ----------------------------------------------------------------------
    def test_modeling_image_area_relation(self):
        """
        Check that the flux ratio equals the ratio of lens area to image area.
        This verifies the core modelling step of geometric optics.
        """
        if any(x is None for x in (self.flux_ratio, self.r, self.f, self.theta)):
            return (False, "SKIP: missing r, f, or theta")
        lens_area = math.pi * self.r ** 2
        image_area = math.pi * (self.f * self.theta / 2.0) ** 2
        area_ratio = lens_area / image_area
        if not math.isclose(self.flux_ratio, area_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"flux_ratio {self.flux_ratio} not consistent with area ratio {area_ratio}")
        return (True, "flux_ratio equals ratio of lens area to image area")

    def test_modeling_flux_ratio_product_form(self):
        """
        Check the identity: flux_ratio * (f^2 * θ^2) ≈ 4 r^2.
        This is derived from the formula, testing algebraic correctness.
        """
        if any(x is None for x in (self.flux_ratio, self.r, self.f, self.theta)):
            return (False, "SKIP: missing r, f, or theta")
        if self.f <= 0 or self.theta <= 0 or self.r <= 0:
            return (False, "r, f, theta must be positive")
        product = self.flux_ratio * (self.f ** 2 * self.theta ** 2)
        expected = 4 * self.r ** 2
        if not math.isclose(product, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"flux_ratio * f^2 * θ^2 = {product}, expected ~{expected}")
        return (True, "flux_ratio * f^2 * θ^2 ≈ 4 r^2")

    def test_modeling_no_increase_matches_formula(self):
        """
        Verify that the condition for no increase (f_no_increase) is derived
        correctly from the equality of lens area and image area.
        This is a modelling check because it uses the same physical principle.
        """
        if any(x is None for x in (self.f_no_increase, self.r, self.theta)):
            return (False, "SKIP: missing r or theta")
        if self.r <= 0 or self.theta <= 0:
            return (False, "r and theta must be positive")
        expected = 2 * self.r / self.theta
        if not math.isclose(self.f_no_increase, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"f_no_increase {self.f_no_increase} not equal to 2r/θ = {expected}")
        return (True, "f_no_increase correctly derived from area equality")

    # ----------------------------------------------------------------------
    # Monotonicity tests (trend checks via input perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_focal_length(self):
        """
        If flux_ratio, r, f, and theta are available, verify that
        increasing f decreases the flux ratio.
        Uses the theoretical formula for the perturbed case.
        """
        if any(x is None for x in (self.flux_ratio, self.r, self.f, self.theta)):
            return (False, "SKIP: missing r, f, or theta")
        if self.f <= 0:
            return (False, "f must be positive")
        f_perturbed = self.f * 1.1
        perturbed_ratio = (4 * self.r ** 2) / (f_perturbed ** 2 * self.theta ** 2)
        if not (self.flux_ratio > perturbed_ratio):
            return (False, f"flux_ratio {self.flux_ratio} is not greater than perturbed ratio {perturbed_ratio} with f increased by 10%")
        return (True, "flux_ratio decreases when f increases (monotonic)")

    def test_monotonicity_lens_radius(self):
        """
        If flux_ratio, r, f, and theta are available, verify that
        increasing r increases the flux ratio.
        """
        if any(x is None for x in (self.flux_ratio, self.r, self.f, self.theta)):
            return (False, "SKIP: missing r, f, or theta")
        if self.r <= 0:
            return (False, "r must be positive")
        r_perturbed = self.r * 1.1
        perturbed_ratio = (4 * r_perturbed ** 2) / (self.f ** 2 * self.theta ** 2)
        if not (self.flux_ratio < perturbed_ratio):
            return (False, f"flux_ratio {self.flux_ratio} is not less than perturbed ratio {perturbed_ratio} with r increased by 10%")
        return (True, "flux_ratio increases when r increases (monotonic)")

    # ----------------------------------------------------------------------
    # Weak final-answer sanity (only one method allowed)
    # ----------------------------------------------------------------------
    def test_answer_sanity_f_no_increase_plausible(self):
        """
        Very weak sanity: f_no_increase should be greater than r
        (since theta ~0.01, f_no_increase ≈ 200r > r).
        No exact computation, no isclose.
        """
        if any(x is None for x in (self.f_no_increase, self.r)):
            return (False, "SKIP: missing r or f_no_increase")
        if not math.isfinite(self.f_no_increase):
            return (False, "f_no_increase is not finite")
        if self.f_no_increase <= self.r:
            return (False, f"f_no_increase ({self.f_no_increase}) should be > r ({self.r})")
        return (True, "f_no_increase > r and finite (plausible)")

    # ----------------------------------------------------------------------
    # Test runner (discover all test_ methods)
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
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)