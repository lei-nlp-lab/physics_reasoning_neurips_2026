import math
import traceback

class PhysicsChecker:
    """
    Physics checker for the N-slit interference problem.
    Audits results: ratio (weakest/most intense peak intensity) and
    theta_first (angular position of first secondary peak).
    """

    def __init__(self, solution_instance):
        # ---------- extract givens ----------
        self.N = getattr(solution_instance, 'N', None)
        self.d_over_lambda = getattr(solution_instance, 'd_over_lambda', None)
        if self.N is None or self.d_over_lambda is None:
            givens = getattr(solution_instance, 'givens', None)
            if isinstance(givens, dict):
                self.N = givens.get('N', self.N)
                self.d_over_lambda = givens.get('d_over_lambda', self.d_over_lambda)

        # ---------- extract results ----------
        results = getattr(solution_instance, 'results', None)
        if results is not None:
            self.ratio = results.get('ratio', None)
            self.theta_first = results.get('theta_first', None)
        else:
            # fallback: direct attributes
            self.ratio = getattr(solution_instance, 'ratio', None)
            self.theta_first = getattr(solution_instance, 'theta_first', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that both ratio and theta_first are present and numeric."""
        if self.ratio is None:
            return (False, "Missing required result: 'ratio'")
        if self.theta_first is None:
            return (False, "Missing required result: 'theta_first'")
        if not isinstance(self.ratio, (int, float)):
            return (False, f"'ratio' is not numeric: {type(self.ratio)}")
        if not isinstance(self.theta_first, (int, float)):
            return (False, f"'theta_first' is not numeric: {type(self.theta_first)}")
        return (True, "All required keys present and numeric")

    def test_interface_givens_available(self):
        """Check that givens N and d_over_lambda are available (needed for derived tests)."""
        missing = []
        if self.N is None:
            missing.append('N')
        if self.d_over_lambda is None:
            missing.append('d_over_lambda')
        if missing:
            return (True, f"SKIPPING: givens {', '.join(missing)} not provided – derived tests will be skipped")
        return (True, "All needed givens present")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_ratio_vs_N_squared(self):
        """Check ratio ≈ 1/N² (weakest secondary max / principal max)."""
        if self.N is None or self.ratio is None:
            return (True, "SKIP: N or ratio missing")
        expected = 1.0 / (self.N ** 2)
        if math.isclose(self.ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ratio {self.ratio} ≈ 1/{self.N**2} = {expected}")
        else:
            return (False, f"ratio {self.ratio} not close to expected {expected} (1/N²)")

    def test_invariant_theta_first_secondary_max(self):
        """Check theta_first = 3/(2*N*d_over_lambda) rad (first secondary maximum)."""
        if self.N is None or self.d_over_lambda is None or self.theta_first is None:
            return (True, "SKIP: missing givens or theta_first")
        # For N=5, d_over_lambda=10 → 3/(2*5*10)=3/100=0.03
        expected = 3.0 / (2.0 * self.N * self.d_over_lambda)
        if math.isclose(self.theta_first, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"theta_first {self.theta_first} ≈ {expected} rad")
        else:
            return (False, f"theta_first {self.theta_first} not close to expected {expected} rad")

    # ------------------------------------------------------------------
    # Modeling / trace consistency tests
    # ------------------------------------------------------------------
    def test_modeling_small_angle_sanity(self):
        """Check that theta_first is small (< 0.1 rad, consistent with small-angle approximation)."""
        if self.theta_first is None:
            return (True, "SKIP: theta_first missing")
        if self.theta_first > 0.0 and self.theta_first < 0.1:
            return (True, f"theta_first = {self.theta_first} rad is in small-angle range")
        else:
            return (False, f"theta_first = {self.theta_first} rad is not plausibly small (<0.1)")

    def test_modeling_ratio_positive_less_than_one(self):
        """Check ratio is positive and less than 1 (physically plausible)."""
        if self.ratio is None:
            return (True, "SKIP: ratio missing")
        if 0 < self.ratio < 1:
            return (True, f"ratio = {self.ratio} is between 0 and 1")
        else:
            return (False, f"ratio = {self.ratio} is not between 0 and 1")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_ratio_decreases_with_N(self):
        """Check that increasing N decreases ratio (ratio ≈ 1/N²). 
        Uses baseline N and ratio, then computes expected ratio for N+1."""
        if self.N is None or self.ratio is None:
            return (True, "SKIP: N or ratio missing")
        if not isinstance(self.N, (int, float)):
            return (True, "SKIP: N not numeric")
        baseline_N = self.N
        perturbed_N = baseline_N + 1   # increase N by 1
        # Expected ratio for perturbed N = 1/(perturbed_N^2)
        expected_perturbed_ratio = 1.0 / (perturbed_N ** 2)
        # Baseline ratio should be greater than perturbed ratio (since 1/N² decreases with N)
        if self.ratio > expected_perturbed_ratio:
            return (True, f"Monotonicity holds: ratio({baseline_N})={self.ratio} > ratio({perturbed_N})={expected_perturbed_ratio}")
        else:
            return (False, f"Monotonicity violated: ratio({baseline_N})={self.ratio} <= ratio({perturbed_N})={expected_perturbed_ratio}")

    def test_monotonicity_theta_first_decreases_with_d_over_lambda(self):
        """Check that increasing d/λ decreases theta_first.
        Uses baseline d_over_lambda and theta_first, computes expected theta_first for increased value."""
        if self.d_over_lambda is None or self.theta_first is None or self.N is None:
            return (True, "SKIP: missing givens or theta_first")
        if not isinstance(self.d_over_lambda, (int, float)):
            return (True, "SKIP: d_over_lambda not numeric")
        baseline_dol = self.d_over_lambda
        perturbed_dol = baseline_dol * 1.1   # increase by 10%
        # Expected theta_first for perturbed d_over_lambda = 3/(2*N*perturbed_dol)
        expected_perturbed_theta = 3.0 / (2.0 * self.N * perturbed_dol)
        # Baseline theta_first should be larger than perturbed (since denominator larger)
        if self.theta_first > expected_perturbed_theta:
            return (True, f"Monotonicity holds: theta_first({baseline_dol})={self.theta_first} > theta_first({perturbed_dol})={expected_perturbed_theta}")
        else:
            return (False, f"Monotonicity violated: theta_first({baseline_dol})={self.theta_first} <= theta_first({perturbed_dol})={expected_perturbed_theta}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one, no math.isclose, no derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_range(self):
        """Very weak sanity: ratio and theta_first positive and not NaN/inf."""
        if self.ratio is None or self.theta_first is None:
            return (True, "SKIP: missing results")
        if math.isnan(self.ratio) or math.isinf(self.ratio):
            return (False, "ratio is NaN or inf")
        if math.isnan(self.theta_first) or math.isinf(self.theta_first):
            return (False, "theta_first is NaN or inf")
        if self.ratio <= 0 or self.theta_first <= 0:
            return (False, f"non-positive values: ratio={self.ratio}, theta_first={self.theta_first}")
        return (True, "Results are finite and positive")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Get all bound methods starting with "test_"
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
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)