import math
import traceback

class PhysicsChecker:
    """
    Auditor for the Zipper model (DNA unwinding) solution.
    Checks interface completeness, physical invariants, modeling consistency,
    monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract results dict
        self.results = getattr(solution_instance, 'results', {})
        # Safely extract givens from either direct attributes or results dict
        self.beta = (
            getattr(solution_instance, 'beta', None)
            or self.results.get('beta', None)
        )
        self.epsilon = (
            getattr(solution_instance, 'epsilon', None)
            or self.results.get('epsilon', None)
        )
        self.N = (
            getattr(solution_instance, 'N', None)
            or self.results.get('N', None)
        )
        # Extracted result quantities
        self.Z = self.results.get('Z', None)
        self.avg_n = self.results.get('avg_n', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_Z_exists(self):
        """Check that the partition function Z is present and numeric."""
        if self.Z is None:
            return (False, "Z is missing from results")
        if not isinstance(self.Z, (int, float)):
            return (False, f"Z has non‑numeric type: {type(self.Z)}")
        return (True, "Z exists and is numeric")

    def test_interface_avg_n_exists(self):
        """Check that the average number of open links is present and numeric."""
        if self.avg_n is None:
            return (False, "avg_n is missing from results")
        if not isinstance(self.avg_n, (int, float)):
            return (False, f"avg_n has non‑numeric type: {type(self.avg_n)}")
        return (True, "avg_n exists and is numeric")

    def test_interface_givens_present(self):
        """Check that required givens (beta, epsilon, N) are present and positive."""
        missing = []
        for name, val in [("beta", self.beta), ("epsilon", self.epsilon), ("N", self.N)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Required givens missing: {', '.join(missing)}")
        # positivity
        problems = []
        if self.beta <= 0:
            problems.append("beta must be >0")
        if self.epsilon <= 0:
            problems.append("epsilon must be >0")
        if self.N < 0 or not isinstance(self.N, int):
            problems.append("N must be a non‑negative integer")
        if problems:
            return (False, "; ".join(problems))
        return (True, "All required givens present and have valid positivity")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_Z_positive(self):
        """Partition function must be strictly positive."""
        if self.Z is None:
            return (False, "SKIP: Z not available")
        if self.Z <= 0:
            return (False, f"Z = {self.Z} should be >0")
        return (True, f"Z = {self.Z} > 0")

    def test_invariant_Z_range(self):
        """Z must lie in [1, N+1] because each term is ≤1 and there are N+1 states."""
        if self.Z is None or self.N is None:
            return (False, "SKIP: Z or N not available")
        if not (1 <= self.Z <= self.N + 1):
            return (False, f"Z = {self.Z} outside expected range [1, {self.N+1}]")
        return (True, f"Z = {self.Z} in [1, {self.N+1}]")

    def test_invariant_avg_n_nonnegative(self):
        """Average number of open links must be non‑negative."""
        if self.avg_n is None:
            return (False, "SKIP: avg_n not available")
        if self.avg_n < 0:
            return (False, f"avg_n = {self.avg_n} < 0")
        return (True, f"avg_n = {self.avg_n} ≥ 0")

    def test_invariant_avg_n_less_than_N(self):
        """Average number of open links cannot exceed N."""
        if self.avg_n is None or self.N is None:
            return (False, "SKIP: avg_n or N not available")
        if self.avg_n > self.N:
            return (False, f"avg_n = {self.avg_n} > N = {self.N}")
        return (True, f"avg_n = {self.avg_n} ≤ N = {self.N}")

    # ------------------------------------------------------------------
    # Modeling/Trace tests (consistency with physics relations)
    # ------------------------------------------------------------------
    def test_modeling_Z_geometric_formula(self):
        """
        Re‑derive Z from the closed‑form geometric series using the given
        givens and compare with the reported Z.
        """
        if any(v is None for v in [self.beta, self.epsilon, self.N, self.Z]):
            return (False, "SKIP: missing data for geometric re‑derivation")
        # compute Z from the known formula
        beta_eps = self.beta * self.epsilon
        exponent = -beta_eps * (self.N + 1)
        try:
            numerator = 1 - math.exp(exponent)
            denominator = 1 - math.exp(-beta_eps)
            if denominator == 0:
                return (False, "Denominator zero in geometric formula (beta=0?)")
            Z_derived = numerator / denominator
        except OverflowError:
            Z_derived = 1.0  # exp(-large) -> 0
        if math.isclose(Z_derived, self.Z, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Geometric Z formula matches reported Z ({self.Z})")
        else:
            return (False, f"Derived Z = {Z_derived}, reported Z = {self.Z}")

    def test_modeling_avg_n_partition_sum_relation(self):
        """
        Check that avg_n * Z equals the explicit sum Σ m·exp(-βε m).
        This verifies the definition of average.
        """
        if any(v is None for v in [self.beta, self.epsilon, self.N, self.Z, self.avg_n]):
            return (False, "SKIP: missing data for sum relation")
        beta_eps = self.beta * self.epsilon
        total = 0.0
        for m in range(self.N + 1):
            total += m * math.exp(-beta_eps * m)
        lhs = self.avg_n * self.Z
        if math.isclose(lhs, total, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "avg_n * Z ≈ explicit sum")
        else:
            return (False, f"avg_n * Z = {lhs}, explicit sum = {total}")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb input, compare perturbed vs baseline)
    # ------------------------------------------------------------------
    def _compute_avg_n(self, beta, epsilon, N):
        """
        Helper: compute the average number of open links for given parameters.
        Uses the exact geometric sum formula.
        """
        beta_eps = beta * epsilon
        # compute Z
        try:
            exponent = -beta_eps * (N + 1)
            numerator = 1 - math.exp(exponent)
            denominator = 1 - math.exp(-beta_eps)
            if denominator == 0:
                return None  # ill‑defined
            Z = numerator / denominator
        except (OverflowError, ZeroDivisionError):
            return None
        # compute sum_{m} m * exp(-beta_eps * m)
        sum_m = 0.0
        for m in range(N + 1):
            sum_m += m * math.exp(-beta_eps * m)
        if Z <= 0:
            return None
        return sum_m / Z

    def test_monotonicity_avg_n_decreases_with_beta(self):
        """
        Increasing beta (decreasing temperature) should reduce the average number
        of open links. Perturb beta by +5% and check the sign.
        """
        if any(v is None for v in [self.beta, self.epsilon, self.N, self.avg_n]):
            return (False, "SKIP: missing data for monotonicity beta")
        baseline = self.avg_n
        perturbed_beta = self.beta * 1.05
        perturbed_avg = self._compute_avg_n(perturbed_beta, self.epsilon, self.N)
        if perturbed_avg is None:
            return (False, "Could not compute perturbed avg_n (overflow/division)")
        if perturbed_avg < baseline - 1e-9:
            return (True, f"Increasing beta from {self.beta} to {perturbed_beta} reduces avg_n from {baseline} to {perturbed_avg}")
        else:
            return (False, f"Monotonicity violated: beta increased but avg_n did not decrease (baseline {baseline}, perturbed {perturbed_avg})")

    def test_monotonicity_avg_n_decreases_with_epsilon(self):
        """
        Increasing epsilon (energy cost) should reduce the average number
        of open links. Perturb epsilon by +5% and check sign.
        """
        if any(v is None for v in [self.beta, self.epsilon, self.N, self.avg_n]):
            return (False, "SKIP: missing data for monotonicity epsilon")
        baseline = self.avg_n
        perturbed_eps = self.epsilon * 1.05
        perturbed_avg = self._compute_avg_n(self.beta, perturbed_eps, self.N)
        if perturbed_avg is None:
            return (False, "Could not compute perturbed avg_n (overflow/division)")
        if perturbed_avg < baseline - 1e-9:
            return (True, f"Increasing epsilon from {self.epsilon} to {perturbed_eps} reduces avg_n from {baseline} to {perturbed_avg}")
        else:
            return (False, f"Monotonicity violated: epsilon increased but avg_n did not decrease (baseline {baseline}, perturbed {perturbed_avg})")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Sanity check: avg_n must be finite, non‑NaN, and non‑negative.
        (No numerical comparison with target answers.)
        """
        if self.avg_n is None:
            return (False, "SKIP: avg_n not available")
        if math.isnan(self.avg_n) or math.isinf(self.avg_n):
            return (False, f"avg_n is invalid (nan/inf)")
        return (True, f"avg_n = {self.avg_n} is finite and non‑negative")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover all test_* methods and run them, collecting results."""
        logs = []
        all_pass = True
        test_names = [name for name in dir(self) if name.startswith("test_")]
        for name in sorted(test_names):
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