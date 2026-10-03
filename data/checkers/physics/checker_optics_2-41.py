import math
import traceback

class PhysicsChecker:
    """
    Audits physics consistency for an optical apodisation problem.
    Expected instance attributes (via getattr with None defaults):
        - b: slit width (float)
        - lambda_: wavelength (float)
        - n: integer order (>=1)
        - results: dict with keys:
            'theta_min_apod' : sin(min_angle) for apodized slit
            'theta_min_unapod': sin(min_angle) for unapodized slit
            'I_ratio_apod'   : I_apod / I_0 at halfway point (unitless)
            'I_ratio_unapod' : I_unapod / I_0 at halfway point (unitless)
    """

    def __init__(self, solution_instance):
        # Extract givens – never force numeric defaults
        self.b = getattr(solution_instance, 'b', None)
        self.lambda_ = getattr(solution_instance, 'lambda', None)  # protect keyword
        self.n = getattr(solution_instance, 'n', None)

        # Extract results
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}
        self.theta_min_apod = self.results.get('theta_min_apod', None)
        self.theta_min_unapod = self.results.get('theta_min_unapod', None)
        self.I_ratio_apod = self.results.get('I_ratio_apod', None)
        self.I_ratio_unapod = self.results.get('I_ratio_unapod', None)

    # ------------------------------------------------------------------
    # Helper methods for theoretical expectation (re‑derived from givens)
    # ------------------------------------------------------------------

    def _compute_expected_I_ratio_apod(self, n):
        """Expected I_apod/I_0 from target formula."""
        if n is None or n < 1:
            return None
        return 1.0 / ((4.0 * n**2 - 1.0) ** 2)

    def _compute_expected_I_ratio_unapod(self, n):
        """Expected I_unapod/I_0 from target formula."""
        if n is None or n < 1:
            return None
        return 4.0 / (((2.0 * n - 1.0) * math.pi) ** 2)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------

    def test_interface_required_keys_present(self):
        """Check that all four expected result keys exist and are not None."""
        if self.theta_min_apod is None:
            return (False, "theta_min_apod missing")
        if self.theta_min_unapod is None:
            return (False, "theta_min_unapod missing")
        if self.I_ratio_apod is None:
            return (False, "I_ratio_apod missing")
        if self.I_ratio_unapod is None:
            return (False, "I_ratio_unapod missing")
        return (True, "All required result keys present")

    def test_interface_inputs_non_null(self):
        """Check that the three required input parameters are provided."""
        missing = []
        if self.b is None:
            missing.append("b")
        if self.lambda_ is None:
            missing.append("lambda")
        if self.n is None:
            missing.append("n")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All givens present")

    def test_interface_n_type(self):
        """Check that n is an integer >= 1 (as required in problem)."""
        if self.n is None:
            return (True, "SKIP: n not provided")
        if not isinstance(self.n, int) and not (isinstance(self.n, float) and self.n.is_integer()):
            return (False, f"n should be integer, got {type(self.n)}")
        if self.n < 1:
            return (False, f"n should be >=1, got {self.n}")
        return (True, f"n={self.n} passes type/range check")

    # ------------------------------------------------------------------
    # Invariant (theorem‑based) checks – use isclose
    # ------------------------------------------------------------------

    def test_invariant_minima_difference(self):
        """
        Check that sinθ_apod - sinθ_unapod = λ/(2b) for the same n.
        Derived from explanation step: sinθ_apod - sinθ_unapod = λ/(2b).
        """
        if any(x is None for x in [self.theta_min_apod, self.theta_min_unapod,
                                    self.b, self.lambda_]):
            return (True, "SKIP: missing data for difference test")
        expected_diff = self.lambda_ / (2.0 * self.b)
        actual_diff = self.theta_min_apod - self.theta_min_unapod
        if math.isclose(actual_diff, expected_diff, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Δsinθ = {actual_diff:.5e} ≈ {expected_diff:.5e}")
        return (False, f"Δsinθ = {actual_diff:.5e}, expected {expected_diff:.5e}")

    def test_invariant_apod_minima_greater_than_unapod(self):
        """
        For n>=1, apodized minima should be further from centre: sinθ_apod > sinθ_unapod.
        (Since λ/(2b) > 0)
        """
        if any(x is None for x in [self.theta_min_apod, self.theta_min_unapod]):
            return (True, "SKIP: missing minima for ordering")
        if self.theta_min_apod > self.theta_min_unapod:
            return (True, "apodized minima are farther (θ_apod > θ_unapod)")
        return (False, "apodized minima should be farther (θ_apod > θ_unapod)")

    # ------------------------------------------------------------------
    # Modelling / trace consistency checks – use isclose
    # ------------------------------------------------------------------

    def test_modeling_apod_intensity_ratio(self):
        """Check apodized intensity ratio at halfway point matches theoretical expression."""
        if self.n is None or self.I_ratio_apod is None:
            return (True, "SKIP: missing n or I_ratio_apod")
        expected = self._compute_expected_I_ratio_apod(self.n)
        if expected is None:
            return (True, "SKIP: cannot compute expected I_ratio_apod")
        if math.isclose(self.I_ratio_apod, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"I_ratio_apod ≈ {expected:.6f}")
        return (False, f"I_ratio_apod = {self.I_ratio_apod:.6f}, expected {expected:.6f}")

    def test_modeling_unapod_intensity_ratio(self):
        """Check unapodized intensity ratio at halfway point matches theoretical expression."""
        if self.n is None or self.I_ratio_unapod is None:
            return (True, "SKIP: missing n or I_ratio_unapod")
        expected = self._compute_expected_I_ratio_unapod(self.n)
        if expected is None:
            return (True, "SKIP: cannot compute expected I_ratio_unapod")
        if math.isclose(self.I_ratio_unapod, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"I_ratio_unapod ≈ {expected:.6f}")
        return (False, f"I_ratio_unapod = {self.I_ratio_unapod:.6f}, expected {expected:.6f}")

    # ------------------------------------------------------------------
    # Monotonicity checks – explicitly perturb an input, compare to baseline
    # ------------------------------------------------------------------

    def test_monotonicity_apod_intensity_decreases(self):
        """
        Perturb n by +1 and check that the theoretical I_ratio_apod decreases.
        Baseline: given I_ratio_apod for original n.
        Perturbed output: computed theoretical value for n+1.
        Verify baseline > perturbed (monotonic decreasing).
        """
        if self.n is None or self.I_ratio_apod is None:
            return (True, "SKIP: missing data")
        n = self.n
        perturbed_n = int(n) + 1  # integer perturbation
        expected_perturbed = self._compute_expected_I_ratio_apod(perturbed_n)
        if expected_perturbed is None:
            return (True, "SKIP: cannot compute perturbed ratio")
        # Baseline is the given result (should match theory, but we compare directly)
        if self.I_ratio_apod > expected_perturbed:
            return (True, f"I_ratio_apod({n}) > I_ratio_apod({perturbed_n}) (decreasing)")
        return (False, f"I_ratio_apod({n}) = {self.I_ratio_apod:.6e} is not > perturbed value {expected_perturbed:.6e}")

    # ------------------------------------------------------------------
    # Weak answer sanity – must NOT use math.isclose or re‑derive final answer
    # ------------------------------------------------------------------

    def test_answer_sanity_positive_finite(self):
        """All four output quantities are positive and finite."""
        checks = [
            ("theta_min_apod", self.theta_min_apod),
            ("theta_min_unapod", self.theta_min_unapod),
            ("I_ratio_apod", self.I_ratio_apod),
            ("I_ratio_unapod", self.I_ratio_unapod),
        ]
        for name, val in checks:
            if val is None:
                continue
            if not math.isfinite(val) or val <= 0:
                return (False, f"{name} not positive/finite: {val}")
        return (True, "All required results are positive finite numbers")

    # ------------------------------------------------------------------
    # Full test suite runner
    # ------------------------------------------------------------------

    def run_all_tests(self):
        """
        Discover all bound methods starting with 'test_', execute them,
        and collect results.  Never crashes – wraps each in try/except.
        """
        logs = []
        all_pass = True
        for method_name in sorted(dir(self)):
            if not method_name.startswith("test_"):
                continue
            method = getattr(self, method_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {method_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)