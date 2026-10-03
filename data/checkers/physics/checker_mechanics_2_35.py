import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely with None defaults
        results = getattr(solution_instance, 'results', {})
        self.d = results.get('d', None)
        self.l = results.get('l', None)
        self.g = results.get('g', None)
        self.m = results.get('m', None)
        self.alpha1 = results.get('alpha1', None)
        self.alpha2 = results.get('alpha2', None)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_d_exists(self):
        if self.d is None:
            return (False, "'d' key missing in results")
        return (True, "'d' key present")

    def test_interface_d_is_numeric(self):
        if self.d is None:
            return (False, "Cannot check type: 'd' missing")
        if not isinstance(self.d, (int, float)):
            return (False, f"'d' has type {type(self.d).__name__}, expected numeric")
        return (True, "'d' is numeric")

    # ----------------------------------------------------------------------
    # Weak answer-sanity (at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_d_positive(self):
        if self.d is None:
            return (False, "Cannot check: 'd' missing")
        if self.d <= 0:
            return (False, f"d = {self.d} is not positive")
        return (True, f"d = {self.d} > 0")

    # ----------------------------------------------------------------------
    # Theorem-based invariant checks (at least two)
    # ----------------------------------------------------------------------
    def test_invariant_d_within_rod_length(self):
        """The point must lie on the lower rod, i.e. 0 < d < 4l."""
        if self.d is None or self.l is None:
            return (False, "SKIP: Missing d or l")
        rod_length = 4 * self.l
        if not (0 < self.d < rod_length):
            return (False, f"d = {self.d} not in (0, {rod_length})")
        return (True, f"d = {self.d} in (0, 4l)")

    def test_invariant_mode_ratio_for_omega1(self):
        """
        For the normal mode with ω1 = sqrt(g/l), the mode amplitude ratio 
        θ2/θ1 = α1 should equal -1 (antisymmetric mode).
        """
        if self.alpha1 is None:
            return (False, "SKIP: alpha1 not provided")
        target = -1.0
        if not math.isclose(self.alpha1, target, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"alpha1 = {self.alpha1} does not match theoretical value -1")
        return (True, f"alpha1 = {self.alpha1} correctly equals -1")

    def test_invariant_d_times_alpha1_equals_minus_three_l(self):
        """
        From the zero horizontal motion condition: d * α = -3l.
        For mode 1 (α1 = -1), this implies d = 3l.
        Verify that d * α1 ≈ -3l.
        """
        if self.d is None or self.l is None or self.alpha1 is None:
            return (False, "SKIP: Missing d, l, or alpha1")
        target = -3 * self.l
        product = self.d * self.alpha1
        if not math.isclose(product, target, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"d * α1 = {product} does not equal -3l = {target}")
        return (True, f"d * α1 = {product} ≅ -3l")

    # ----------------------------------------------------------------------
    # Modeling / trace quality tests (at least two)
    # ----------------------------------------------------------------------
    def test_modeling_d_matches_three_l(self):
        """d should equal 3l (derived from the zero‑horizontal‑motion condition)."""
        if self.d is None or self.l is None:
            return (False, "SKIP: Missing d or l")
        expected = 3 * self.l
        if not math.isclose(self.d, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"d = {self.d} does not match 3l = {expected}")
        return (True, f"d = {self.d} ~ 3l = {expected}")

    def test_modeling_consistency_with_alpha1(self):
        """
        If alpha1 is provided, recompute d from d = -3l / α1 and check
        consistency with the reported d.
        """
        if self.alpha1 is None or self.l is None or self.d is None:
            return (False, "SKIP: Missing alpha1, l, or d")
        if math.isclose(self.alpha1, 0.0, abs_tol=1e-12):
            return (False, "alpha1 is zero, cannot compute d")
        computed_d = -3 * self.l / self.alpha1
        if not math.isclose(self.d, computed_d, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"d = {self.d} diverges from computed d = {computed_d} using alpha1 = {self.alpha1}")
        return (True, f"d = {self.d} consistent with alpha1 = {self.alpha1}")

    def test_modeling_second_mode_gives_negative_d(self):
        """
        For the second normal mode (α2 = 1.5), the condition d = -3l/α2
        yields a negative distance, which is unphysical.  Verify that the 
        reported d is positive, consistent with only the first mode being valid.
        """
        if self.alpha2 is None or self.l is None:
            return (False, "SKIP: Missing alpha2 or l")
        computed_d2 = -3 * self.l / self.alpha2
        if computed_d2 >= 0:
            return (False, f"Second mode predicts d = {computed_d2}, which is non‑negative; should be negative")
        if self.d is not None:
            if self.d > 0:
                return (True, f"Reported d = {self.d} positive, second mode gives negative d = {computed_d2}, consistent with rejection")
            else:
                return (False, "Reported d is not positive")
        return (True, "Second mode gives negative d, as expected")

    # ----------------------------------------------------------------------
    # Monotonicity check (explicit perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_d_increases_with_l(self):
        """
        If the length scale l is increased by 10%, the point location d = 3l should
        increase proportionally.  Perturb l by +10% and verify that the ratio d/l 
        remains constant (equal to 3).
        """
        if self.d is None or self.l is None:
            return (False, "SKIP: Missing d or l")
        if not math.isclose(self.d / self.l, 3.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "Original d/l not 3; cannot meaningfully test monotonicity")
        delta = 0.10  # +10%
        new_l = self.l * (1 + delta)
        # Under the physics, d = 3*l, so new_d should be 3*new_l
        expected_new_d = 3 * new_l
        # The reported d is the original; after scaling the original d by (1+delta) we get the predicted new d
        predicted_new_d = self.d * (1 + delta)
        if not math.isclose(expected_new_d, predicted_new_d, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Monotonicity fails: perturbing l by +{delta*100}% gives expected d = {expected_new_d}, but scaling original d gives {predicted_new_d}")
        return (True, f"d increases proportionally when l is increased by {delta*100}%")

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
                passed, msg = False, f"{name} raised exception: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)