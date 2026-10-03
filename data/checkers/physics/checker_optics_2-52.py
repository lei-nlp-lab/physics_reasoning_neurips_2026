import math
import sys
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for a 6-slit grating interference problem.
    Extracts results and givens from solution_instance and runs
    unit-test-like checks focusing on invariants, modeling quality,
    and a weak final answer sanity.
    """
    def __init__(self, solution_instance):
        # Safely extract results
        results = getattr(solution_instance, 'results', {})
        self.theta_sin = results.get('theta_sin', None)
        self.Delta_theta = results.get('Delta_theta', None)
        self.ratio = results.get('ratio', None)

        # Safely extract givens
        givens = getattr(solution_instance, 'givens', {})
        self.lambda_ = givens.get('lambda', givens.get('wavelength', None))
        self.d = givens.get('d', givens.get('grating_spacing', None))
        self.N = givens.get('N', givens.get('number_of_slits', None))
        # Also try from results if not in givens
        if self.N is None:
            self.N = results.get('N', None)
        # Explanation steps (optional, for additional context)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    def run_all_tests(self):
        """Execute all test methods and collect results."""
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
                passed = False
                msg = f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ------------------------------------------------------------------
    # Interface checks: do required keys exist and have valid type?
    # ------------------------------------------------------------------
    def test_interface_theta_sin_present(self):
        """Check that theta_sin result key exists (value may be None)."""
        if self.theta_sin is None:
            return (False, "theta_sin is missing or None")
        return (True, "theta_sin present")

    def test_interface_Delta_theta_present(self):
        """Check that Delta_theta result key exists."""
        if self.Delta_theta is None:
            return (False, "Delta_theta is missing or None")
        return (True, "Delta_theta present")

    def test_interface_ratio_present(self):
        """Check that ratio result key exists."""
        if self.ratio is None:
            return (False, "ratio is missing or None")
        return (True, "ratio present")

    # ------------------------------------------------------------------
    # Invariant checks: physics law sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_ratio_equals_N_squared(self):
        """
        For N identical, narrow slits, intensity ratio = N^2.
        This is a derived invariant from the formula.
        """
        if self.ratio is None or self.N is None:
            return (False, "SKIP: ratio or N missing")
        expected = float(self.N) ** 2
        if math.isclose(self.ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ratio {self.ratio} equals N² = {expected}")
        else:
            return (False, f"ratio {self.ratio} != N² = {expected}")

    def test_invariant_Delta_theta_positive(self):
        """Angular half‑width must be positive."""
        if self.Delta_theta is None:
            return (False, "SKIP: Delta_theta missing")
        if self.Delta_theta > 0:
            return (True, f"Delta_theta = {self.Delta_theta} > 0")
        else:
            return (False, f"Delta_theta = {self.Delta_theta} is not positive")

    def test_invariant_Delta_theta_finite(self):
        """Angular half‑width must be finite (not inf or nan)."""
        if self.Delta_theta is None:
            return (False, "SKIP: Delta_theta missing")
        if math.isfinite(self.Delta_theta):
            return (True, "Delta_theta is finite")
        else:
            return (False, "Delta_theta is not finite")

    # ------------------------------------------------------------------
    # Modeling quality checks: consistency with explanation steps/pseudocode
    # ------------------------------------------------------------------
    def test_modeling_Delta_theta_formula(self):
        """
        Check that DELTA_theta = lambda / (N*d) using provided givens.
        This re‑derives the expected value inside the checker.
        """
        if self.Delta_theta is None:
            return (False, "SKIP: Delta_theta missing")
        if None in (self.lambda_, self.d, self.N):
            return (False, "SKIP: missing lambda, d, or N")
        expected = self.lambda_ / (self.N * self.d)
        if math.isclose(self.Delta_theta, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Delta_theta {self.Delta_theta} ≈ λ/(Nd) = {expected}")
        else:
            return (False, f"Delta_theta {self.Delta_theta} != λ/(Nd) = {expected}")

    def test_modeling_ratio_consistent_with_N_six(self):
        """
        Given that the problem states N=6, the ratio must be 36.
        This is a modeling consistency check (not a final answer test).
        """
        if self.ratio is None:
            return (False, "SKIP: ratio missing")
        # Assume N=6 from problem context (but we check against actual N if given)
        if self.N is not None and self.N != 6:
            return (False, f"Expected N=6 from problem context, but got N={self.N}")
        expected = 36.0
        if math.isclose(self.ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ratio {self.ratio} ≈ 36")
        else:
            return (False, f"ratio {self.ratio} != 36")

    # ------------------------------------------------------------------
    # Monotonicity checks: perturb an input and observe the trend
    # ------------------------------------------------------------------
    def test_monotonicity_Delta_theta_increases_with_lambda(self):
        """
        If lambda increases, Delta_theta = λ/(Nd) increases.
        We simulate a 10% increase in lambda while keeping d,N fixed,
        and verify that the reported Delta_theta would increase accordingly.
        (Since we only have one data point, we recompute the perturbed value
         and check if it is larger than the original.)
        """
        if self.Delta_theta is None or self.lambda_ is None or self.d is None or self.N is None:
            return (False, "SKIP: missing data for monotonicity check")
        # Perturb lambda by +10%
        lambda_new = self.lambda_ * 1.10
        Delta_theta_new = lambda_new / (self.N * self.d)
        if Delta_theta_new > self.Delta_theta:
            return (True, f"10% increase in λ → Δθ increases (new={Delta_theta_new} > old={self.Delta_theta})")
        else:
            return (False, f"Monotonicity failed: Δθ did not increase with λ")

    # ------------------------------------------------------------------
    # Weak final answer sanity (at most one such method)
    # ------------------------------------------------------------------
    def test_answer_sanity_ratio_range(self):
        """
        Very weak check: ratio must be finite, positive, and not obviously unrealistic.
        No math.isclose, no derived expected values.
        """
        if self.ratio is None:
            return (False, "SKIP: ratio missing")
        if not math.isfinite(self.ratio):
            return (False, f"ratio is not finite: {self.ratio}")
        if self.ratio <= 0:
            return (False, f"ratio must be positive, got {self.ratio}")
        # Typical values for N=6: ratio=36; but we only check a broad range
        if self.ratio < 1 or self.ratio > 1000:
            return (False, f"ratio {self.ratio} outside plausible range [1,1000]")
        return (True, f"ratio {self.ratio} is plausible")