import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits a solution for a two-level system problem in statistical mechanics.
    Checks Boltzmann distribution, average energy, specific heat, and limits.
    """

    def __init__(self, solution_instance):
        # --- Extract givens from solution_instance ---
        self.E0 = None
        self.E1 = None
        self.N = None
        self.kB = None
        self.T = None

        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = solution_instance if isinstance(solution_instance, dict) else {}

        # Attempt to extract givens (may be in results or as attributes)
        for attr in ['E0', 'E1', 'N', 'kB', 'T']:
            val = None
            if isinstance(results, dict):
                val = results.get(attr, None)
            if val is None:
                val = getattr(solution_instance, attr, None)
            setattr(self, attr, val)

        # --- Extract computed results ---
        self.u = None
        self.u_low = None
        self.u_high = None
        self.C = None
        self.C_low = None
        self.C_high = None

        if isinstance(results, dict):
            self.u = results.get('u', None)
            self.u_low = results.get('u_low', None)
            self.u_high = results.get('u_high', None)
            self.C = results.get('C', None)
            self.C_low = results.get('C_low', None)
            self.C_high = results.get('C_high', None)

        # Fallback to attributes
        if self.u is None:
            self.u = getattr(solution_instance, 'u', None)
        if self.u_low is None:
            self.u_low = getattr(solution_instance, 'u_low', None)
        if self.u_high is None:
            self.u_high = getattr(solution_instance, 'u_high', None)
        if self.C is None:
            self.C = getattr(solution_instance, 'C', None)
        if self.C_low is None:
            self.C_low = getattr(solution_instance, 'C_low', None)
        if self.C_high is None:
            self.C_high = getattr(solution_instance, 'C_high', None)

        self.has_givens = all(v is not None for v in [self.E0, self.E1, self.N, self.kB, self.T])
        self.Delta = None
        self.beta = None
        if self.E0 is not None and self.E1 is not None:
            self.Delta = self.E1 - self.E0
        if self.kB is not None and self.T is not None and self.T > 0:
            self.beta = 1.0 / (self.kB * self.T)

    def run_all_tests(self):
        """Execute all test methods and return (success, logs)."""
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

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys exist and are not None."""
        keys = ['u', 'u_low', 'u_high', 'C', 'C_low', 'C_high']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_givens_available(self):
        """Check that essential givens are provided."""
        if not self.has_givens:
            return (False, "Missing required givens (E0, E1, N, kB, T)")
        return (True, "Givens available")

    # ------------------------------------------------------------------
    # Invariant (theorem-based) tests
    # ------------------------------------------------------------------
    def test_invariant_energy_range(self):
        """Average energy must lie between E0 and E1."""
        if not self.has_givens or self.u is None:
            return (False, "SKIP: missing data")
        if self.u < self.E0 - 1e-12 or self.u > self.E1 + 1e-12:
            return (False, f"u = {self.u} not in [{self.E0}, {self.E1}]")
        return (True, "Energy within bounds")

    def test_invariant_low_t_limit(self):
        """As T->0, u should approach E0 + Delta*exp(-beta*Delta)."""
        if self.u_low is None or self.beta is None or self.Delta is None or self.E0 is None:
            return (False, "SKIP: missing data")
        expected = self.E0 + self.Delta * math.exp(-self.beta * self.Delta)
        if not math.isclose(self.u_low, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"u_low = {self.u_low}, expected ~ {expected}")
        return (True, "Low-T limit correct")

    def test_invariant_high_t_limit(self):
        """As T->infinity, u should approach (E0+E1)/2 - beta*Delta^2/4."""
        if self.u_high is None or self.beta is None or self.Delta is None or self.E0 is None or self.E1 is None:
            return (False, "SKIP: missing data")
        expected = (self.E0 + self.E1) / 2.0 - (self.beta * self.Delta**2) / 4.0
        if not math.isclose(self.u_high, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"u_high = {self.u_high}, expected ~ {expected}")
        return (True, "High-T limit correct")

    def test_invariant_specific_heat_positive(self):
        """Specific heat should be positive for finite T."""
        if self.C is None or self.T is None or self.T <= 0:
            return (False, "SKIP: missing data")
        if self.C <= 0:
            return (False, f"C = {self.C} is not positive")
        return (True, "C positive")

    def test_invariant_low_t_C_limit(self):
        """Low-T specific heat ~ N*kB*(beta*Delta)^2 * exp(-beta*Delta)."""
        if self.C_low is None or self.beta is None or self.Delta is None or self.N is None or self.kB is None:
            return (False, "SKIP: missing data")
        expected = self.N * self.kB * (self.beta * self.Delta)**2 * math.exp(-self.beta * self.Delta)
        if not math.isclose(self.C_low, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"C_low = {self.C_low}, expected ~ {expected}")
        return (True, "Low-T C limit correct")

    def test_invariant_high_t_C_limit(self):
        """High-T specific heat ~ N*kB*(beta*Delta)^2 /4."""
        if self.C_high is None or self.beta is None or self.Delta is None or self.N is None or self.kB is None:
            return (False, "SKIP: missing data")
        expected = self.N * self.kB * (self.beta * self.Delta)**2 / 4.0
        if not math.isclose(self.C_high, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"C_high = {self.C_high}, expected ~ {expected}")
        return (True, "High-T C limit correct")

    def test_invariant_energy_level_order(self):
        """E1 must be greater than E0 for a meaningful two-level system."""
        if self.E0 is None or self.E1 is None:
            return (False, "SKIP: missing E0 or E1")
        if self.E1 <= self.E0:
            return (False, f"E1 = {self.E1} is not greater than E0 = {self.E0}")
        return (True, "E1 > E0")

    # ------------------------------------------------------------------
    # Modeling tests (check derived expressions against Boltzmann)
    # ------------------------------------------------------------------
    def test_modeling_u_expression(self):
        """Verify u = (E0 + E1*exp(-beta*Delta))/(1+exp(-beta*Delta))."""
        if not self.has_givens or self.u is None:
            return (False, "SKIP: missing data")
        if self.beta is None or self.Delta is None:
            return (False, "SKIP: cannot compute beta or Delta")
        expected = (self.E0 + self.E1 * math.exp(-self.beta * self.Delta)) / (1 + math.exp(-self.beta * self.Delta))
        if not math.isclose(self.u, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"u = {self.u}, expected ~ {expected}")
        return (True, "u expression matches Boltzmann")

    def test_modeling_C_expression(self):
        """Verify C = N*kB*(beta*Delta)^2 * exp(-beta*Delta)/(1+exp(-beta*Delta))^2."""
        if not self.has_givens or self.C is None:
            return (False, "SKIP: missing data")
        if self.beta is None or self.Delta is None:
            return (False, "SKIP: cannot compute beta or Delta")
        eb = math.exp(-self.beta * self.Delta)
        expected = self.N * self.kB * (self.beta * self.Delta)**2 * eb / ((1 + eb)**2)
        if not math.isclose(self.C, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"C = {self.C}, expected ~ {expected}")
        return (True, "C expression matches derivative")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb input, compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_u_increases_with_T(self):
        """Perturb T upward and check that u increases."""
        if not self.has_givens or self.u is None or self.T <= 0:
            return (False, "SKIP: missing data")
        T2 = self.T * 1.1
        beta2 = 1.0 / (self.kB * T2)
        eb2 = math.exp(-beta2 * self.Delta)
        u2 = (self.E0 + self.E1 * eb2) / (1 + eb2)
        if u2 <= self.u - 1e-12:
            return (False, f"u decreased: u={self.u}, u(T*1.1)={u2}")
        return (True, "u increases with T")

    def test_monotonicity_C_increases_for_small_T_increase(self):
        """
        For a small increase in T starting from a very low temperature,
        the specific heat should increase (because we are in the low-T regime).
        We only apply this test if beta*Delta > 5 (i.e., very low T).
        """
        if not self.has_givens or self.C is None:
            return (False, "SKIP: missing data")
        if self.beta is None or self.Delta is None:
            return (False, "SKIP: cannot compute beta or Delta")
        # Check that we are in the low-T regime
        if self.beta * self.Delta < 5:
            return (False, "SKIP: not in low-T regime (beta*Delta too small)")
        # Baseline C from solution
        C_base = self.C
        # Increase T by 5%
        T2 = self.T * 1.05
        beta2 = 1.0 / (self.kB * T2)
        eb2 = math.exp(-beta2 * self.Delta)
        C2 = self.N * self.kB * (beta2 * self.Delta)**2 * eb2 / ((1 + eb2)**2)
        if C2 <= C_base - 1e-12:
            return (False, f"C decreased after T increase: baseline {C_base}, perturbed {C2}")
        return (True, "C increases for small T increase in low-T regime")

    # ------------------------------------------------------------------
    # Answer sanity (single weak test)
    # ------------------------------------------------------------------
    def test_answer_sanity_all_finite(self):
        """Check that all result values are finite numbers."""
        keys = ['u', 'u_low', 'u_high', 'C', 'C_low', 'C_high']
        for k in keys:
            val = getattr(self, k)
            if val is not None and (not math.isfinite(val)):
                return (False, f"{k} is non-finite: {val}")
        return (True, "All results finite")