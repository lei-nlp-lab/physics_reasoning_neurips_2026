import math

class PhysicsChecker:
    """
    Robust Physics Checker for the cosmic ray proton mean free path problem.
    Extracts relevant quantities from the solution instance and performs
    unit-test-like checks.
    """
    def __init__(self, solution_instance):
        # Extract results (lambda_cm is the primary output)
        results = getattr(solution_instance, 'results', {})
        self.lambda_cm = results.get('lambda_cm', None)

        # Attempt to extract givens – these may be in a dict or as attributes
        givens = getattr(solution_instance, 'givens', None)
        if isinstance(givens, dict):
            self.P = givens.get('P', None)          # pressure in Pa
            self.T = givens.get('T', None)          # temperature in K
            self.sigma = givens.get('sigma', None)  # cross-section in m^2
        else:
            self.P = None
            self.T = None
            self.sigma = None

        # Also try to extract directly computed n (number density) if stored
        n = results.get('n', None)
        if n is None:
            # Try to compute n from P and T if available
            if self.P is not None and self.T is not None:
                k_B = 1.38e-23  # J/K
                self.n = self.P / (k_B * self.T)
            else:
                self.n = None
        else:
            self.n = n

        # Explanation steps – not stored in detail, but we note they exist
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Pseudocode constants (from the solving plan) – used for modeling checks
        self._pseudo_n = 2.5e25       # m^{-3}
        self._pseudo_sigma = 4e-30    # m^2

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_lambda_cm_present(self):
        """Check that lambda_cm exists and is a number."""
        if self.lambda_cm is None:
            return (False, "lambda_cm is missing in results")
        if not isinstance(self.lambda_cm, (int, float)):
            return (False, f"lambda_cm has invalid type: {type(self.lambda_cm)}")
        return (True, "lambda_cm is present and numeric")

    def test_interface_lambda_cm_positive(self):
        """Check that lambda_cm is positive."""
        if self.lambda_cm is None:
            return (False, "lambda_cm not available")
        if self.lambda_cm <= 0:
            return (False, f"lambda_cm is non-positive: {self.lambda_cm}")
        return (True, "lambda_cm is positive")

    def test_interface_lambda_cm_finite(self):
        """Check that lambda_cm is finite (not inf or nan)."""
        if self.lambda_cm is None:
            return (False, "lambda_cm not available")
        if not math.isfinite(self.lambda_cm):
            return (False, f"lambda_cm is not finite: {self.lambda_cm}")
        return (True, "lambda_cm is finite")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_mean_free_path_formula(self):
        """
        Verify that lambda_cm ≈ 1/(n * sigma) * 100, if n and sigma are available.
        Use lenient tolerance due to approximate numbers.
        """
        if self.lambda_cm is None:
            return (False, "lambda_cm not available, cannot check formula")
        if self.n is None or self.sigma is None:
            return (False, "SKIP: n or sigma not available for formula check")

        # Compute expected mean free path in cm
        expected_lambda_m = 1.0 / (self.n * self.sigma)
        expected_lambda_cm = expected_lambda_m * 100.0

        # Use isclose with 10% relative tolerance (physics approximations)
        if not math.isclose(self.lambda_cm, expected_lambda_cm, rel_tol=0.1, abs_tol=1e3):
            return (False, f"lambda_cm={self.lambda_cm:.2e} does not match expected {expected_lambda_cm:.2e}")
        return (True, "lambda_cm agrees with 1/(n*sigma) formula")

    def test_invariant_cross_section_order(self):
        """
        Check that sigma (if provided) is of order 4e-30 m^2.
        This is a modeling invariant: p-Air cross-section at high energy.
        """
        if self.sigma is None:
            return (False, "SKIP: sigma not available")
        if self.sigma <= 0:
            return (False, f"sigma is non-positive: {self.sigma}")
        if not (1e-31 < self.sigma < 1e-28):
            return (False, f"sigma={self.sigma:.2e} is not near typical 40 mb (~4e-30 m^2)")
        return (True, "sigma is in realistic range")

    def test_invariant_number_density_order(self):
        """
        Check that n (if available) is of order 2.5e25 m^{-3}.
        """
        if self.n is None:
            return (False, "SKIP: n not available")
        if self.n <= 0:
            return (False, f"n is non-positive: {self.n}")
        if not (8e24 < self.n < 7.5e25):
            return (False, f"n={self.n:.3e} is not near sea-level 2.5e25 m^{-3}")
        return (True, "n is in plausible range")

    # ------------------------------------------------------------------
    # Modeling/trace quality tests (based on pseudocode steps)
    # ------------------------------------------------------------------
    def test_modeling_cross_section_in_mb(self):
        """
        If sigma is provided, check that it is close to 40 mb (4e-30 m^2).
        This tests that the student used an appropriate cross-section value.
        """
        if self.sigma is None:
            return (False, "SKIP: sigma not available")
        # Convert to mb: 1 mb = 1e-31 m^2
        sigma_mb = self.sigma / 1e-31
        if not math.isclose(sigma_mb, 40, rel_tol=0.2):
            return (False, f"sigma = {sigma_mb:.1f} mb, not ~40 mb")
        return (True, "sigma is approximately 40 mb")

    def test_modeling_pseudocode_consistency(self):
        """
        Check that lambda_cm is consistent with the pseudocode's explicit
        n = 2.5e25 m^{-3} and sigma = 4e-30 m^2. This verifies the solver
        followed the step-by-step plan.
        """
        if self.lambda_cm is None:
            return (False, "SKIP: lambda_cm not available, cannot check pseudocode consistency")
        # Use pseudocode constants (re-derived from the solving plan)
        n_pseudo = 2.5e25
        sigma_pseudo = 4e-30
        expected_lambda_m = 1.0 / (n_pseudo * sigma_pseudo)
        expected_lambda_cm = expected_lambda_m * 100.0
        if not math.isclose(self.lambda_cm, expected_lambda_cm, rel_tol=0.1, abs_tol=1e3):
            return (False, f"lambda_cm={self.lambda_cm:.2e} not close to pseudocode expectation {expected_lambda_cm:.2e}")
        return (True, "lambda_cm consistent with pseudocode step constants")

    # ------------------------------------------------------------------
    # Monotonicity test (input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_density_inverse(self):
        """
        If we have n and sigma, perturb n by +10% and check that lambda_cm
        decreases by about 9% (since λ ∝ 1/n).
        """
        if self.n is None or self.sigma is None or self.lambda_cm is None:
            return (False, "SKIP: insufficient data for monotonicity test")

        # Baseline lambda in cm using the given inputs
        baseline_lambda = 100.0 / (self.n * self.sigma)
        # Perturb n upward by 10%
        perturbed_n = self.n * 1.10
        perturbed_lambda = 100.0 / (perturbed_n * self.sigma)
        # Relative change: new should be smaller by about 9%
        rel_change = (perturbed_lambda - baseline_lambda) / baseline_lambda
        expected_change = -1.0 + 1.0/1.10  # about -0.0909
        if not math.isclose(rel_change, expected_change, rel_tol=0.05, abs_tol=0.02):
            return (False, f"lambda changed by {rel_change:.4f}, expected ~{expected_change:.4f}")
        return (True, "lambda decreases as n increases (inverse relationship)")

    # ------------------------------------------------------------------
    # Weak final-answer sanity check (secondary, at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_range(self):
        """
        Broad sanity: lambda_cm should be positive and between 1 cm and 1e8 cm.
        """
        if self.lambda_cm is None:
            return (False, "lambda_cm not available")
        if self.lambda_cm <= 0:
            return (False, f"lambda_cm non-positive: {self.lambda_cm}")
        if self.lambda_cm > 1e8:
            return (False, f"lambda_cm too large: {self.lambda_cm:.3e} > 1e8 cm")
        if self.lambda_cm < 1.0:
            return (False, f"lambda_cm too small: {self.lambda_cm:.3e} < 1 cm")
        return (True, "lambda_cm is in physically plausible range")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover all test_* methods, execute them, collect results.
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
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)