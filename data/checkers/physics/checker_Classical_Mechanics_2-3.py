import math
import traceback

class PhysicsChecker:
    """
    Auditor for the Capstan equation problem: minimum number of rope wraps.
    Extracts givens and results, then runs a suite of unit tests.
    """

    def __init__(self, solution_instance):
        # Safe extraction of required givens and results.
        self.alpha = None
        self.mu = None
        self.n_min = None

        # Try to get from solution_instance.results dict
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.alpha = solution_instance.results.get('alpha', None)
            self.mu = solution_instance.results.get('mu', None)
            self.n_min = solution_instance.results.get('n_min', None)

        # Also try direct attributes
        if self.alpha is None:
            self.alpha = getattr(solution_instance, 'alpha', None)
        if self.mu is None:
            self.mu = getattr(solution_instance, 'mu', None)
        if self.n_min is None:
            self.n_min = getattr(solution_instance, 'n_min', None)

        # Additionally, check for theta_min in case it's emitted
        self.theta_min = getattr(solution_instance, 'theta_min', None)
        if self.theta_min is None and hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.theta_min = solution_instance.results.get('theta_min', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------

    def test_interface_required_keys_present(self):
        """Check that alpha, mu, n_min are present and numeric."""
        if self.alpha is None or self.mu is None or self.n_min is None:
            missing = []
            if self.alpha is None:
                missing.append('alpha')
            if self.mu is None:
                missing.append('mu')
            if self.n_min is None:
                missing.append('n_min')
            return (False, f"Missing required keys: {', '.join(missing)}")
        if not isinstance(self.alpha, (int, float)):
            return (False, f"alpha is not numeric: {type(self.alpha)}")
        if not isinstance(self.mu, (int, float)):
            return (False, f"mu is not numeric: {type(self.mu)}")
        if not isinstance(self.n_min, (int, float)):
            return (False, f"n_min is not numeric: {type(self.n_min)}")
        return (True, "All required keys present and numeric")

    def test_interface_theta_min_present_if_emitted(self):
        """Optional: if theta_min is emitted, check it's numeric."""
        if self.theta_min is None:
            return (True, "theta_min not emitted, skip")
        if not isinstance(self.theta_min, (int, float)):
            return (False, f"theta_min is not numeric: {type(self.theta_min)}")
        return (True, "theta_min is numeric")

    # ------------------------------------------------------------------
    # Invariant / theorem-based sanity checks
    # ------------------------------------------------------------------

    def test_invariant_capstan_inequality(self):
        """
        Check that the product mu * (2π * n_min) is at least alpha.
        Derived from capstan equation: T_high/T_low = e^(μθ), with θ = 2π n_min.
        For no-slip, need μθ >= α.
        """
        if any(x is None for x in [self.alpha, self.mu, self.n_min]):
            return (False, "SKIP: missing givens/result for capstan check")
        if self.mu == 0:
            return (False, "mu cannot be zero")
        exponent = self.mu * (2 * math.pi * self.n_min)
        if exponent < self.alpha - 1e-2:
            return (False, f"mu * 2π n_min = {exponent} is too small relative to alpha = {self.alpha}")
        return (True, "Capstan inequality holds: mu * 2π n_min >= alpha")

    def test_invariant_n_min_non_negative(self):
        """n_min must be positive (real winding count)."""
        if self.n_min is None:
            return (False, "SKIP: n_min not available")
        if self.n_min < 0:
            return (False, f"n_min is negative: {self.n_min}")
        if math.isnan(self.n_min):
            return (False, "n_min is NaN")
        if math.isinf(self.n_min):
            return (False, "n_min is infinite")
        return (True, "n_min is non-negative and finite")

    def test_invariant_mu_positive(self):
        """Friction coefficient must be positive."""
        if self.mu is None:
            return (False, "SKIP: mu not available")
        if self.mu <= 0:
            return (False, f"mu must be positive, got {self.mu}")
        return (True, "mu is positive")

    def test_invariant_alpha_positive(self):
        """Alpha (exponent) must be positive."""
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        if self.alpha <= 0:
            return (False, f"alpha must be positive, got {self.alpha}")
        return (True, "alpha is positive")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------

    def test_modeling_exponential_consistency(self):
        """
        Check that the exponential ratio exp(μ * 2π * n_min) ≈ exp(α).
        This tests the core capstan relation directly, using the theorem.
        """
        if any(x is None for x in [self.alpha, self.mu, self.n_min]):
            return (False, "SKIP: missing data for exponential consistency")
        ratio = math.exp(self.mu * 2 * math.pi * self.n_min)
        expected_ratio = math.exp(self.alpha)
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Capstan ratio mismatch: computed {ratio}, expected {expected_ratio}")
        return (True, "Exponential ratio consistent with capstan equation")

    def test_modeling_wrap_angle_conversion(self):
        """
        If theta_min is emitted, verify n_min = theta_min / (2π).
        This is a modeling consistency check from the definition of turns.
        """
        if self.theta_min is None or self.n_min is None:
            return (False, "SKIP: theta_min or n_min not available")
        expected_n_min = self.theta_min / (2 * math.pi)
        if not math.isclose(self.n_min, expected_n_min, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"n_min from theta_min mismatch: expected {expected_n_min}, got {self.n_min}")
        return (True, "n_min correctly derived from theta_min")

    # ------------------------------------------------------------------
    # Monotonicity checks
    # ------------------------------------------------------------------

    def test_monotonicity_n_min_increases_with_alpha(self):
        """
        Perturb alpha by +10% and check that the formula predicts a larger n_min.
        Compare the perturbed formula result to the actual n_min to ensure monotonic trend.
        """
        if any(x is None for x in [self.alpha, self.mu, self.n_min]):
            return (False, "SKIP: missing data for monotonicity alpha")
        n_min_base = self.n_min
        alpha_pert = 1.1 * self.alpha
        # Recompute expected n_min for perturbed alpha using capstan theory
        n_min_pert = alpha_pert / (2 * math.pi * self.mu)
        if n_min_pert <= n_min_base:
            return (False, f"n_min did not increase with alpha: base {n_min_base}, perturbed formula {n_min_pert}")
        return (True, "n_min increases with alpha (confirmed via perturbed formula)")

    def test_monotonicity_n_min_decreases_with_mu(self):
        """
        Perturb mu upward by +10% and check that the formula predicts a smaller n_min.
        """
        if any(x is None for x in [self.alpha, self.mu, self.n_min]):
            return (False, "SKIP: missing data for monotonicity mu")
        n_min_base = self.n_min
        mu_pert = 1.1 * self.mu
        n_min_pert = self.alpha / (2 * math.pi * mu_pert)
        if n_min_pert >= n_min_base:
            return (False, f"n_min did not decrease with mu: base {n_min_base}, perturbed formula {n_min_pert}")
        return (True, "n_min decreases when mu increases")

    # ------------------------------------------------------------------
    # Weak final-answer sanity (at most one, no isclose, no re‑derivation)
    # ------------------------------------------------------------------

    def test_answer_sanity_weak(self):
        """
        Very weak sanity: n_min should be positive, finite, and not astronomically large.
        No re‑derivation of the formula; no isclose.
        """
        if self.n_min is None:
            return (False, "SKIP: n_min not available")
        if self.n_min <= 0:
            return (False, f"n_min is not positive: {self.n_min}")
        if math.isnan(self.n_min):
            return (False, "n_min is NaN")
        if math.isinf(self.n_min):
            return (False, "n_min is infinite")
        if self.n_min > 1e6:
            return (False, f"n_min unreasonably large: {self.n_min}")
        return (True, "n_min passes weak sanity checks")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def run_all_tests(self):
        """
        Discover all test_* methods and run them.
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
                tb = traceback.format_exc()
                passed = False
                msg = f"{name} CRASHED: {e}. Traceback:\n{tb}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)