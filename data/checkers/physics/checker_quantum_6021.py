import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens (safe defaults)
        self.m = getattr(solution_instance, 'm', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.A = getattr(solution_instance, 'A', None)
        self.a = getattr(solution_instance, 'a', None)
        self.k = getattr(solution_instance, 'k', None)
        self.theta = getattr(solution_instance, 'theta', None)
        # Extract results dictionary
        results = getattr(solution_instance, 'results', {})
        self.f = results.get('f', None)
        self.dsigma = results.get('dsigma_dOmega', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_results_keys(self):
        """Check that required result keys are present and not None."""
        if self.f is None:
            return False, "Missing 'f' in results"
        if self.dsigma is None:
            return False, "Missing 'dsigma_dOmega' in results"
        return True, "All required result keys present"

    def test_interface_givens_types(self):
        """Check that all given parameters are present and numeric."""
        for name, val in [("m", self.m), ("hbar", self.hbar),
                          ("A", self.A), ("a", self.a),
                          ("k", self.k), ("theta", self.theta)]:
            if val is None:
                return False, f"Missing given: {name}"
            if not isinstance(val, (int, float)):
                return False, f"{name} must be numeric, got {type(val)}"
        return True, "All givens present and numeric"

    # ------------------------------------------------------------------
    # Invariant (theorem-based) checks
    # ------------------------------------------------------------------
    def test_invariant_dsigma_from_f(self):
        """dsigma_dOmega must equal |f|^2."""
        if self.f is None or self.dsigma is None:
            return False, "SKIP: missing f or dsigma"
        expected = abs(self.f) ** 2
        if math.isclose(self.dsigma, expected, rel_tol=1e-2):
            return True, "dsigma = |f|^2 within tolerance"
        return False, f"dsigma = {self.dsigma}, expected |f|^2 = {expected}"

    def test_invariant_sign_f(self):
        """For a repulsive potential (A>0), f must be negative."""
        if self.f is None or self.A is None:
            return False, "SKIP: missing f or A"
        if self.A <= 0:
            return False, "SKIP: A not positive (repulsive assumed)"
        if self.f < 0:
            return True, "f is negative as expected for repulsive potential"
        return False, f"f = {self.f} should be negative for A>0"

    def test_invariant_positive_dsigma(self):
        """Cross section must be positive."""
        if self.dsigma is None:
            return False, "SKIP: missing dsigma"
        if self.dsigma > 0:
            return True, "dsigma > 0"
        return False, f"dsigma = {self.dsigma} is not positive"

    def test_invariant_finite(self):
        """All computed quantities must be finite (not NaN, not inf)."""
        for name, val in [("m", self.m), ("hbar", self.hbar),
                          ("A", self.A), ("a", self.a),
                          ("k", self.k), ("theta", self.theta),
                          ("f", self.f), ("dsigma", self.dsigma)]:
            if val is None:
                continue  # skip missing
            if not math.isfinite(val):
                return False, f"{name} is not finite: {val}"
        return True, "All available quantities are finite"

    # ------------------------------------------------------------------
    # Modeling quality checks (deriving expected values from physics)
    # ------------------------------------------------------------------
    def test_modeling_amplitude_expr(self):
        """Check that the amplitude f matches the Born formula."""
        if any(v is None for v in [self.m, self.hbar, self.A, self.a,
                                   self.k, self.theta, self.f]):
            return False, "SKIP: missing data for modeling amplitude"

        q = 2 * self.k * math.sin(self.theta / 2)
        expected_f = - (self.m * self.A * self.a**3 * math.sqrt(math.pi)) / (2 * self.hbar**2)
        expected_f *= math.exp(-q**2 * self.a**2 / 4)

        if math.isclose(self.f, expected_f, rel_tol=1e-2):
            return True, "f matches Born formula"
        return False, f"f = {self.f}, expected {expected_f}"

    def test_modeling_cross_section_expr(self):
        """Check that dsigma_dOmega matches the Born formula."""
        if any(v is None for v in [self.m, self.hbar, self.A, self.a,
                                   self.k, self.theta, self.dsigma]):
            return False, "SKIP: missing data for modeling cross section"

        q = 2 * self.k * math.sin(self.theta / 2)
        expected_dsigma = (self.m**2 * self.A**2 * self.a**6 * math.pi) / (4 * self.hbar**4)
        expected_dsigma *= math.exp(-q**2 * self.a**2 / 2)

        if math.isclose(self.dsigma, expected_dsigma, rel_tol=1e-2):
            return True, "dsigma matches Born formula"
        return False, f"dsigma = {self.dsigma}, expected {expected_dsigma}"

    # ------------------------------------------------------------------
    # Monotonicity check(s)
    # ------------------------------------------------------------------
    def test_monotonicity_cross_section_with_theta(self):
        """Cross section must decrease when scattering angle theta increases."""
        if any(v is None for v in [self.m, self.hbar, self.A, self.a,
                                   self.k, self.theta, self.dsigma]):
            return False, "SKIP: missing data for monotonicity test"

        # Baseline values
        q0 = 2 * self.k * math.sin(self.theta / 2)
        prefactor = (self.m**2 * self.A**2 * self.a**6 * math.pi) / (4 * self.hbar**4)
        dsigma0_re = prefactor * math.exp(-q0**2 * self.a**2 / 2)

        # Confirm baseline matches given result (should, but just in case)
        if not math.isclose(self.dsigma, dsigma0_re, rel_tol=1e-2):
            return False, f"Baseline dsigma {self.dsigma} does not match recomputed {dsigma0_re}"

        # Perturb theta upward by a small amount (0.1 rad)
        theta1 = self.theta + 0.1
        q1 = 2 * self.k * math.sin(theta1 / 2)
        dsigma1_re = prefactor * math.exp(-q1**2 * self.a**2 / 2)

        if dsigma1_re < dsigma0_re - 1e-12:
            return True, f"dsigma decreases from {dsigma0_re} to {dsigma1_re} when theta increases"
        else:
            return False, f"dsigma does not decrease: baseline {dsigma0_re}, perturbed {dsigma1_re}"

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
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
        return all_pass, logs