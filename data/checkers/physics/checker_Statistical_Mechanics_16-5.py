import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the cylindrical radiation shield heat transfer problem.
    Checks interface, physical invariants, modeling quality, monotonicity, and
    performs a weak final-answer sanity check.
    """

    # Known physical constants (from problem statement)
    SIGMA = 5.67e-8          # W/(m^2 K^4)
    INNER_RADIUS = 0.1       # m
    LENGTH = 2.0             # m
    DENSITY = 807            # kg/m^3
    SPECIFIC_HEAT = 2042     # J/(kg K)
    T0 = 67.0                # K
    TF = 77.0                # K
    TOUT = 300.0             # K
    N_SHIELDS = 99

    def __init__(self, solution_instance):
        """
        Extract results and optional givens from the solution instance.
        Always use None as default for missing data.
        """
        # Results: first try dictionary, then fallback to attribute access
        if hasattr(solution_instance, 'results'):
            res = solution_instance.results
            self.t0 = res.get('t0', None)
            self.t99 = res.get('t99', None)
        else:
            self.t0 = getattr(solution_instance, 't0', None)
            self.t99 = getattr(solution_instance, 't99', None)

        # Optional givens (for extended checks, not required)
        self.given_r_i = getattr(solution_instance, 'r_i', None)
        self.given_L = getattr(solution_instance, 'L', None)
        self.given_rho = getattr(solution_instance, 'rho', None)
        self.given_c = getattr(solution_instance, 'c', None)
        self.given_T0 = getattr(solution_instance, 'T0', None)
        self.given_Tf = getattr(solution_instance, 'Tf', None)
        self.given_Tout = getattr(solution_instance, 'Tout', None)
        self.given_N = getattr(solution_instance, 'N_shields', None)

    @staticmethod
    def _is_numeric(x):
        """Return True if x is a finite int or float."""
        if isinstance(x, (int, float)):
            return math.isfinite(x)
        return False

    # ─── Interface tests ───────────────────────────────────────────
    def test_interface_t0_exists(self):
        if self.t0 is None:
            return (False, "t0 is missing (None)")
        if not self._is_numeric(self.t0):
            return (False, f"t0 is not a finite numeric value: {self.t0}")
        return (True, "t0 exists and is numeric")

    def test_interface_t99_exists(self):
        if self.t99 is None:
            return (False, "t99 is missing (None)")
        if not self._is_numeric(self.t99):
            return (False, f"t99 is not a finite numeric value: {self.t99}")
        return (True, "t99 exists and is numeric")

    # ─── Invariant tests (use math.isclose) ────────────────────────
    def test_invariant_t99_is_multiple_of_t0(self):
        """With 99 shields, t99 should be 100 * t0."""
        if self.t0 is None or self.t99 is None:
            return (False, "SKIP: t0 or t99 missing")
        expected = 100.0 * self.t0
        if math.isclose(self.t99, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t99 ≈ 100 * t0: {self.t99} vs {expected}")
        return (False, f"t99 = {self.t99} but expected {expected}")

    def test_invariant_t0_positive(self):
        if self.t0 is None:
            return (False, "SKIP: t0 missing")
        if self.t0 <= 0:
            return (False, f"t0 is not positive: {self.t0}")
        return (True, "t0 is positive")

    def test_invariant_t99_positive(self):
        if self.t99 is None:
            return (False, "SKIP: t99 missing")
        if self.t99 <= 0:
            return (False, f"t99 is not positive: {self.t99}")
        return (True, "t99 is positive")

    # ─── Modeling/trace tests (use math.isclose) ──────────────────
    def test_modeling_ratio_matches_shield_count(self):
        """Check that t99 / t0 is approximately N+1 = 100."""
        if self.t0 is None or self.t99 is None:
            return (False, "SKIP: t0 or t99 missing")
        if self.t0 == 0:
            return (False, "t0 is zero, cannot compute ratio")
        ratio = self.t99 / self.t0
        if math.isclose(ratio, 100.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t99/t0 = {ratio} ≈ 100")
        return (False, f"t99/t0 = {ratio} but expected ~100")

    def test_modeling_t0_order_of_magnitude(self):
        """t0 should be on the order of 0.01‑10 seconds (based on known physics)."""
        if self.t0 is None:
            return (False, "SKIP: t0 missing")
        if 0.01 <= self.t0 <= 10:
            return (True, f"t0 = {self.t0} is in plausible range [0.01, 10]")
        return (False, f"t0 = {self.t0} outside plausible range [0.01, 10]")

    # ─── Monotonicity tests (perturb an input, compare to baseline) ──
    def test_monotonicity_t0_decreases_with_increasing_Tout(self):
        """
        Perturb Tout upward by 5% and use approximate scaling (t0 ∝ 1/Tout^4)
        to estimate the new time. Check that the estimated new time is smaller.
        """
        if self.t0 is None:
            return (False, "SKIP: t0 missing")
        # Use provided Tout if available, otherwise problem constant
        Tout_old = self.given_Tout if self.given_Tout is not None else self.TOUT
        # Small fractional increase
        fraction = 0.05
        Tout_new = Tout_old * (1 + fraction)
        # Approximate scaling: t0 ∝ 1/Tout^4 when T << Tout (valid for 67-77 K vs 300 K)
        scaling_factor = (Tout_old ** 4) / (Tout_new ** 4)
        estimated_new_t0 = self.t0 * scaling_factor
        # If Tout increased, scaling_factor < 1, so estimated_new_t0 < t0
        if estimated_new_t0 < self.t0:
            return (True, f"Perturbing Tout by +{fraction*100}% gives estimated t0 = {estimated_new_t0:.4e} < original {self.t0:.4e}")
        else:
            return (False, f"Perturbing Tout by +{fraction*100}% gives estimated t0 = {estimated_new_t0:.4e}, which is not smaller than original {self.t0:.4e}")

    def test_monotonicity_t99_increases_with_increasing_N(self):
        """
        Perturb N_shields from 99 to 100 (small increase) and estimate t100.
        Since t ∝ (N+1), t100 = (101/100)*t99. Verify that estimated t100 > t99.
        """
        if self.t99 is None:
            return (False, "SKIP: t99 missing")
        N_old = self.given_N if self.given_N is not None else self.N_SHIELDS
        N_new = N_old + 1   # small perturbation
        # t(N) ∝ (N+1)
        ratio = (N_new + 1) / (N_old + 1)
        estimated_new_t99 = self.t99 * ratio
        if estimated_new_t99 > self.t99:
            return (True, f"Perturbing N from {N_old} to {N_new} gives estimated t = {estimated_new_t99:.4e} > original {self.t99:.4e}")
        else:
            return (False, f"Perturbing N from {N_old} to {N_new} gives estimated t = {estimated_new_t99:.4e}, which is not greater than original {self.t99:.4e}")

    # ─── Weak answer sanity (no isclose, no re‑derivation) ────────
    def test_answer_sanity_broad_range(self):
        """
        Very loose sanity: both times must be within physically plausible
        broad ranges. This test does NOT match against known target values.
        """
        if self.t0 is None or self.t99 is None:
            return (False, "SKIP: t0 or t99 missing")
        if not (0.001 <= self.t0 <= 1000.0):
            return (False, f"t0 = {self.t0} outside broad range [0.001, 1000]")
        if not (0.01 <= self.t99 <= 10000.0):
            return (False, f"t99 = {self.t99} outside broad range [0.01, 10000]")
        return (True, "Both times are within broad physically reasonable range")

    # ─── Test runner ──────────────────────────────────────────────
    def run_all_tests(self):
        """
        Discover, execute, and collect all test methods.
        Returns (all_pass, list_of_log_strings).
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)