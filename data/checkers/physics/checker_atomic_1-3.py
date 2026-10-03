import math

class PhysicsChecker:
    """
    Physics auditor for the Bohr radius vs. Compton wavelength problem.
    Checks interface completeness, invariant plausibility, modeling consistency,
    and monotonicity, with a weak final-answer sanity.
    """

    def __init__(self, solution_instance):
        results = solution_instance.results if hasattr(solution_instance, 'results') else {}
        self.ratio = results.get('ratio', None)
        self.order = results.get('order', None)

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_ratio_exists(self):
        if self.ratio is None:
            return (False, "'ratio' is missing from results")
        if not isinstance(self.ratio, (int, float)):
            return (False, f"'ratio' should be numeric, got {type(self.ratio).__name__}")
        if math.isnan(self.ratio) or math.isinf(self.ratio):
            return (False, f"'ratio' is NaN or infinity")
        return (True, f"ratio = {self.ratio} is a finite number")

    def test_interface_order_exists(self):
        if self.order is None:
            return (False, "'order' is missing from results")
        if not isinstance(self.order, str):
            return (False, f"'order' should be a string, got {type(self.order).__name__}")
        return (True, f"order = '{self.order}' is a string")

    # ------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------
    def test_invariant_ratio_positive(self):
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if self.ratio <= 0:
            return (False, f"ratio = {self.ratio} is not positive")
        return (True, "ratio > 0")

    def test_invariant_ratio_order_of_magnitude(self):
        """The Bohr radius is ~100 times larger than the Compton wavelength
        (either full or reduced yields ~20 or ~137, both between 10 and 1000)."""
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if not (10 < self.ratio < 1000):
            return (False, f"ratio = {self.ratio} is outside plausible range (10, 1000)")
        return (True, f"ratio {self.ratio} lies in (10, 1000)")

    # ------------------------------------------------------------
    # Modeling / trace consistency tests
    # ------------------------------------------------------------
    def test_modeling_order_consistency_with_ratio(self):
        """If ratio ≈ 100 (20–150) then order should say '100 times larger';
        if ≈1000 (500–1500) then '1000 times larger'; otherwise 'about the same'."""
        if self.ratio is None or self.order is None:
            return (False, "SKIP: ratio or order missing")
        ratio = self.ratio
        if 20 <= ratio <= 150:
            expected = '100 times larger'
        elif 500 <= ratio <= 1500:
            expected = '1000 times larger'
        else:
            expected = 'about the same'
        if self.order == expected:
            return (True, f"order '{self.order}' matches ratio range")
        else:
            # Allow loose matches (e.g., "approximately 100 times larger")
            if '100' in self.order and 20 <= ratio <= 150:
                return (True, f"order '{self.order}' loosely consistent with ratio {ratio}")
            return (False, f"order '{self.order}' inconsistent with ratio {ratio} (expected '{expected}')")

    def test_modeling_ratio_consistent_with_fine_structure(self):
        """Check that the ratio is close to either 1/α (reduced Compton) or 1/(2πα) (full Compton)."""
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        alpha_approx = 1.0 / 137.0  # approximate fine-structure constant
        expected_reduced = 1.0 / alpha_approx        # ≈ 137
        expected_full = 1.0 / (2 * math.pi * alpha_approx)  # ≈ 21.8

        # Use relative tolerance of 10% to allow for rounding
        if (math.isclose(self.ratio, expected_reduced, rel_tol=0.1, abs_tol=1.0) or
            math.isclose(self.ratio, expected_full, rel_tol=0.1, abs_tol=1.0)):
            return (True, f"ratio {self.ratio} matches either reduced or full Compton definition")
        else:
            return (False, f"ratio {self.ratio} does not match expected values ({expected_reduced:.1f} or {expected_full:.1f})")

    # ------------------------------------------------------------
    # Monotonicity test (perturb an input and compare to baseline)
    # ------------------------------------------------------------
    def test_monotonicity_ratio_decreases_with_increased_alpha(self):
        """If the fine-structure constant α increases, the ratio a0/λ_C should decrease.
        We infer α from the ratio (assuming either reduced or full Compton definition)
        and perturb it by +5% to check monotonic direction."""
        if self.ratio is None:
            return (False, "SKIP: ratio not available")

        # Determine which definition was likely used
        if self.ratio > 50:
            # assume reduced Compton: ratio = 1/α
            alpha_inferred = 1.0 / self.ratio
            def ratio_of_alpha(alpha):
                return 1.0 / alpha
        elif self.ratio < 50:
            # assume full Compton: ratio = 1/(2π α)
            alpha_inferred = 1.0 / (2 * math.pi * self.ratio)
            def ratio_of_alpha(alpha):
                return 1.0 / (2 * math.pi * alpha)
        else:
            return (False, "SKIP: ratio ~50, ambiguous definition")

        # Perturb α upward by 5%
        perturb_factor = 1.05
        alpha_perturbed = alpha_inferred * perturb_factor
        new_ratio = ratio_of_alpha(alpha_perturbed)

        if new_ratio < self.ratio:
            return (True, f"Perturbing α by +5% reduces ratio from {self.ratio:.2f} to {new_ratio:.2f}")
        else:
            return (False, f"Monotonicity violated: α increased but ratio increased from {self.ratio:.2f} to {new_ratio:.2f}")

    # ------------------------------------------------------------
    # Weak answer sanity (only one allowed, no re‑derivation)
    # ------------------------------------------------------------
    def test_answer_sanity_order_choice(self):
        """The order must be one of the three given options (no numeric comparison)."""
        if self.order is None:
            return (False, "SKIP: order not available")
        valid_orders = ['100 times larger', '1000 times larger', 'about the same']
        if self.order in valid_orders:
            return (True, f"order '{self.order}' is a valid choice")
        else:
            return (False, f"order '{self.order}' is not one of {valid_orders}")

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
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