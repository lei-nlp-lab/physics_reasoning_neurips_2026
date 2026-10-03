import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to the problem of resolving two stars via telescope
    and diffraction grating. Checks interface, physical invariants,
    modeling quality, monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance):
        # ----- Extract givens (angular separation, wavelengths) -----
        # Allow attribute or 'results' dict; default None if missing.
        self.theta = getattr(solution_instance, 'theta', None)
        self.lambda1 = getattr(solution_instance, 'lambda1', None)
        self.lambda2 = getattr(solution_instance, 'lambda2', None)

        # Also try results dict as fallback
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.theta = self.theta or solution_instance.results.get('theta')
            self.lambda1 = self.lambda1 or solution_instance.results.get('lambda1')
            self.lambda2 = self.lambda2 or solution_instance.results.get('lambda2')

        # ----- Extract results (D_m, D_cm, N) -----
        self.D_m = getattr(solution_instance, 'D_m', None)
        self.D_cm = getattr(solution_instance, 'D_cm', None)
        self.N = getattr(solution_instance, 'N', None)

        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.D_m = self.D_m or solution_instance.results.get('D_m')
            self.D_cm = self.D_cm or solution_instance.results.get('D_cm')
            self.N = self.N or solution_instance.results.get('N')

    # ------------------------------------------------------------------
    #   INTERFACE TESTS
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that three required result keys are present."""
        if self.D_m is None:
            return (False, "SKIP: D_m not found")
        if self.D_cm is None:
            return (False, "SKIP: D_cm not found")
        if self.N is None:
            return (False, "SKIP: N not found")
        return (True, "All required keys (D_m, D_cm, N) are present")

    def test_interface_positive_numeric(self):
        """Check that D_m, D_cm, N are positive numbers."""
        for name, val in [("D_m", self.D_m), ("D_cm", self.D_cm), ("N", self.N)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric ({type(val).__name__})")
            if val <= 0:
                return (False, f"{name} is not positive ({val})")
        return (True, "All quantities are positive numbers")

    # ------------------------------------------------------------------
    #   INVARIANT TESTS (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_rayleigh_criterion(self):
        """
        Rayleigh criterion: D_m * theta == 1.22 * lambda2 (with units).
        Also checks that D_cm = 100 * D_m.
        """
        if any(x is None for x in [self.D_m, self.D_cm, self.theta, self.lambda2]):
            return (False, "SKIP: one or more required values missing (D_m, D_cm, theta, lambda2)")
        # expected D_m from Rayleigh
        expected_D_m = 1.22 * self.lambda2 / self.theta
        if not math.isclose(self.D_m, expected_D_m, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"D_m ({self.D_m}) does not match Rayleigh criterion (expected {expected_D_m:.6f})")
        # D_cm should be 100 * D_m
        expected_D_cm = 100.0 * self.D_m
        if not math.isclose(self.D_cm, expected_D_cm, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"D_cm ({self.D_cm}) does not equal 100*D_m ({expected_D_cm:.2f})")
        return (True, "Rayleigh criterion and unit conversion are consistent")

    def test_invariant_grating_resolving_power(self):
        """
        For m=1: N = lambda_avg / (lambda2 - lambda1).
        """
        if any(x is None for x in [self.N, self.lambda1, self.lambda2]):
            return (False, "SKIP: one or more required values missing (N, lambda1, lambda2)")
        delta = self.lambda2 - self.lambda1
        if delta <= 0:
            return (False, "lambda2 must be > lambda1")
        lambda_avg = (self.lambda1 + self.lambda2) / 2.0
        expected_N = lambda_avg / delta
        if not math.isclose(self.N, expected_N, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"N ({self.N}) does not match resolving power formula (expected {expected_N:.2f})")
        return (True, "Grating resolving power formula is satisfied (m=1)")

    # ------------------------------------------------------------------
    #   MODELING / TRACE TESTS
    # ------------------------------------------------------------------
    def test_modeling_larger_wavelength_chosen(self):
        """
        The telescope diameter should be computed using the larger wavelength
        (5790 Å) to guarantee resolution for both. Check that D is at least
        as large as the value needed for the smaller wavelength.
        """
        if any(x is None for x in [self.D_m, self.theta, self.lambda1, self.lambda2]):
            return (False, "SKIP: missing givens or D_m")
        # D_min for lambda1
        D_min_lambda1 = 1.22 * self.lambda1 / self.theta
        # D_m should be >= D_min_lambda1 (with small tolerance for floating point)
        if self.D_m < D_min_lambda1 - 1e-12:
            return (False, f"D_m ({self.D_m}) is smaller than required for lambda1 ({D_min_lambda1:.6f})")
        return (True, "D_m is at least as large as required for the smaller wavelength")

    def test_modeling_N_positive_integer(self):
        """
        N should be a positive integer (or very close). Check that the
        fractional part is negligible.
        """
        if self.N is None:
            return (False, "SKIP: N not available")
        if self.N < 1:
            return (False, f"N is not positive ({self.N})")
        # allow tolerance 0.1 for floating point error
        if abs(self.N - round(self.N)) > 0.1:
            return (False, f"N ({self.N}) is not close to an integer")
        return (True, f"N = {self.N} is effectively an integer")

    # ------------------------------------------------------------------
    #   MONOTONICITY TESTS (perturb a given, check trend)
    # ------------------------------------------------------------------
    def test_monotonicity_angular_separation(self):
        """
        If theta is reduced by 10%, D should increase by factor ~1/0.9.
        """
        if any(x is None for x in [self.D_m, self.theta, self.lambda2]):
            return (False, "SKIP: missing D_m, theta, or lambda2")
        perturbation = 0.9  # reduce theta by 10%
        new_theta = self.theta * perturbation
        new_D = 1.22 * self.lambda2 / new_theta
        ratio = new_D / self.D_m
        expected_ratio = 1.0 / perturbation  # = 1.1111...
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Monotonicity fails: D ratio {ratio:.4f} (expected ~{expected_ratio:.4f})")
        return (True, f"D scales as 1/theta (ratio = {ratio:.4f})")

    def test_monotonicity_wavelength_difference(self):
        """
        If delta_lambda is reduced by 10%, N should increase by factor ~1/0.9.
        """
        if any(x is None for x in [self.N, self.lambda1, self.lambda2]):
            return (False, "SKIP: missing N, lambda1, or lambda2")
        delta = self.lambda2 - self.lambda1
        if delta <= 0:
            return (False, "lambda2 must be > lambda1")
        perturbation = 0.9  # reduce delta by 10%
        new_delta = delta * perturbation
        lambda_avg = (self.lambda1 + self.lambda2) / 2.0
        new_N = lambda_avg / new_delta
        ratio = new_N / self.N
        expected_ratio = 1.0 / perturbation  # = 1.1111...
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Monotonicity fails: N ratio {ratio:.4f} (expected ~{expected_ratio:.4f})")
        return (True, f"N scales inversely with delta_lambda (ratio = {ratio:.4f})")

    # ------------------------------------------------------------------
    #   WEAK ANSWER SANITY TEST (only one, no derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Check that D_cm is in a plausible range (50-90 cm) and N is in a
        plausible range (200-400). No exact comparison.
        """
        failures = []
        if self.D_cm is not None:
            if not (50 < self.D_cm < 90):
                failures.append(f"D_cm={self.D_cm} outside (50,90)")
        if self.N is not None:
            if not (200 < self.N < 400):
                failures.append(f"N={self.N} outside (200,400)")
        if failures:
            return (False, "; ".join(failures))
        return (True, "Answer magnitudes are plausible")

    # ------------------------------------------------------------------
    #   MAIN RUNNER
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover all bound methods whose names start with 'test_',
        execute each wrapped in try/except, and return (all_pass, logs).
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)