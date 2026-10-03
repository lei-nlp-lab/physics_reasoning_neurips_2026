import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits a solution for the comet-sun cross section problem.
    Checks interface compliance, invariant conservation laws,
    modeling correctness, monotonicity properties, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract givens and results safely, default to None
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.results = getattr(solution_instance, 'results', {}) or {}

        # Givens
        self.v0 = self.givens.get('v0', None)    # initial comet speed
        self.M = self.givens.get('M', None)      # Sun mass
        self.R = self.givens.get('R', None)      # Sun radius
        self.G = self.givens.get('G', None)      # gravitational constant

        # Results (from pseudocode [emit])
        self.b_crit = self.results.get('b_crit', None)
        self.sigma = self.results.get('sigma', None)

        # Optional: store solution_instance for extra data
        self.sol = solution_instance

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_givens_exist(self):
        """Check that required givens are present and non-None."""
        required = [('v0', self.v0), ('M', self.M), ('R', self.R), ('G', self.G)]
        missing = []
        for name, val in required:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    def test_interface_results_exist(self):
        """Check that required result keys are present and non-None."""
        required = [('b_crit', self.b_crit), ('sigma', self.sigma)]
        missing = []
        for name, val in required:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing results: {', '.join(missing)}")
        return (True, "All required results present")

    def test_interface_types_are_numeric(self):
        """All numeric values should be finite floats or ints."""
        vals = [self.v0, self.M, self.R, self.G, self.b_crit, self.sigma]
        names = ['v0','M','R','G','b_crit','sigma']
        for name, val in zip(names, vals):
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric: {type(val)}")
            if math.isnan(val) or math.isinf(val):
                return (False, f"{name} is NaN or infinite")
        return (True, "All numeric types valid")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based sanity)
    # ------------------------------------------------------------------
    def test_invariant_energy_and_angular_momentum(self):
        """
        Symbolic check: derive b_crit from conservation laws and compare.
        Use sympy to re-derive and numerically verify with isclose.
        """
        if any(v is None for v in [self.v0, self.M, self.R, self.G, self.b_crit]):
            return (False, "SKIP: missing required values")

        # Define symbols
        v0, M, R, G, b, v_peri = sp.symbols('v0 M R G b v_peri', positive=True)
        # Energy: ½ m v0² = ½ m v_peri² - G M m / R
        # Cancel mass m
        eq_energy = sp.Eq(sp.Rational(1,2)*v0**2, sp.Rational(1,2)*v_peri**2 - G*M/R)
        # Angular momentum: m v0 b = m v_peri R   => v_peri = v0 b / R
        eq_ang = sp.Eq(v0*b, v_peri*R)

        # Substitute v_peri from angular momentum into energy
        subs_vperi = sp.solve(eq_ang, v_peri)[0]
        eq_energy_sub = eq_energy.subs(v_peri, subs_vperi)

        # Solve for b^2
        sol_b2 = sp.solve(eq_energy_sub, b**2)[0]
        # Expected expression: R^2 * (1 + 2G M / (R v0^2))
        expected_b2 = R**2 * (1 + 2*G*M/(R*v0**2))
        # Simplify difference
        diff = sp.simplify(sol_b2 - expected_b2)
        if diff != 0:
            return (False, f"Symbolic derivation mismatch: diff = {diff}")

        # Numerical fallback: compute expected b_crit from formula
        expected_bcrit = self.R * math.sqrt(1 + 2*self.G*self.M/(self.R*self.v0**2))
        if not math.isclose(self.b_crit, expected_bcrit, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"b_crit {self.b_crit} != expected {expected_bcrit}")
        return (True, "Energy & angular momentum conservation verified")

    def test_invariant_sigma_equals_pi_b_crit_squared(self):
        """σ = π b_crit² must hold."""
        if self.b_crit is None or self.sigma is None:
            return (False, "SKIP: b_crit or sigma missing")
        expected_sigma = math.pi * self.b_crit**2
        if not math.isclose(self.sigma, expected_sigma, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma {self.sigma} != pi*b_crit^2 = {expected_sigma}")
        return (True, "sigma = pi * b_crit^2 holds")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_b_crit_formula(self):
        """
        Check that b_crit matches the derived formula.
        This is a modeling quality test: recompute from givens.
        """
        if any(v is None for v in [self.v0, self.M, self.R, self.G, self.b_crit]):
            return (False, "SKIP: missing values")
        expected = self.R * math.sqrt(1 + 2*self.G*self.M/(self.R*self.v0**2))
        if not math.isclose(self.b_crit, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"b_crit {self.b_crit} != derived {expected}")
        return (True, "b_crit formula correct")

    def test_modeling_sigma_positive_and_gt_geometric(self):
        """
        Cross section should be > πR² because gravitational focusing.
        Also positive.
        """
        if self.sigma is None or self.R is None:
            return (False, "SKIP: sigma or R missing")
        if self.sigma <= 0:
            return (False, f"sigma not positive: {self.sigma}")
        geometric = math.pi * self.R**2
        if self.sigma < geometric - 1e-12:  # allow numerical fuzz
            return (False, f"sigma {self.sigma} < geometric cross section {geometric}")
        return (True, "sigma > πR² (gravitational focusing) and positive")

    def test_modeling_b_crit_gt_R(self):
        """Critical impact parameter > R due to gravity."""
        if self.b_crit is None or self.R is None:
            return (False, "SKIP: b_crit or R missing")
        if self.b_crit <= self.R:
            return (False, f"b_crit {self.b_crit} <= R {self.R}")
        return (True, "b_crit > R (as expected)")

    # ------------------------------------------------------------------
    # Monotonicity tests (trend checks via perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_b_crit_decreases_with_v0(self):
        """
        If v0 increases, b_crit should decrease.
        Perturb v0 up by 10%, recalculate expected b_crit, check trend.
        """
        if any(v is None for v in [self.v0, self.M, self.R, self.G, self.b_crit]):
            return (False, "SKIP: missing values")
        # baseline b_crit from result
        b_base = self.b_crit
        # perturb v0 upward
        v0_perturbed = self.v0 * 1.1
        b_perturbed = self.R * math.sqrt(1 + 2*self.G*self.M/(self.R*v0_perturbed**2))
        if b_perturbed >= b_base - 1e-12:
            return (False, f"Increasing v0 did not decrease b_crit: base {b_base} perturbed {b_perturbed}")
        return (True, "b_crit decreases with increasing v0 (monotonic)")

    def test_monotonicity_b_crit_increases_with_M(self):
        """If M increases, b_crit should increase."""
        if any(v is None for v in [self.v0, self.M, self.R, self.G, self.b_crit]):
            return (False, "SKIP: missing values")
        b_base = self.b_crit
        M_perturbed = self.M * 1.2
        b_perturbed = self.R * math.sqrt(1 + 2*self.G*M_perturbed/(self.R*self.v0**2))
        if b_perturbed <= b_base + 1e-12:
            return (False, f"Increasing M did not increase b_crit: base {b_base} perturbed {b_perturbed}")
        return (True, "b_crit increases with increasing M (monotonic)")

    # ------------------------------------------------------------------
    # Weak answer sanity test (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_cross_section_sign_and_magnitude(self):
        """
        Very weak: cross section should be positive and roughly of order πR²*(1+...).
        Do NOT compute exact expected value.
        """
        if self.sigma is None or self.R is None:
            return (False, "SKIP: sigma or R missing")
        # Must be positive
        if self.sigma <= 0:
            return (False, f"sigma not positive: {self.sigma}")
        # Must be at least geometric (true from physics)
        geometric = math.pi * self.R**2
        if self.sigma < geometric:
            return (False, f"sigma {self.sigma} < geometric {geometric}")
        # No further numeric check; just sign/order
        return (True, "sigma positive and at least geometric (sanity)")

    # ------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)