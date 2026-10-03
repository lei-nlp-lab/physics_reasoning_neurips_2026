import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for kinetic theory problem:
    number of collisions per second on one wall of a cube of He gas at STP.
    """

    def __init__(self, solution_instance):
        # Extract givens (use None as default)
        self.L = getattr(solution_instance, 'L', None)           # side length in m
        self.T = getattr(solution_instance, 'T', None)           # temperature in K
        self.P = getattr(solution_instance, 'P', None)           # pressure in Pa
        self.k = getattr(solution_instance, 'k', None)           # Boltzmann constant in J/K
        self.m_He = getattr(solution_instance, 'm_He', None)     # mass of He atom in kg

        # Extract intermediate variables (optional)
        self.n = getattr(solution_instance, 'n', None)           # number density
        self.v_avg = getattr(solution_instance, 'v_avg', None)   # mean speed
        self.A = getattr(solution_instance, 'A', None)           # wall area

        # Extract result
        self.R = solution_instance.results.get('R', None) if hasattr(solution_instance, 'results') else None

    # ---------- Interface tests ----------
    def test_interface_has_givens(self):
        """Check that all required givens are provided."""
        missing = []
        for var, name in [(self.L, 'L'), (self.T, 'T'), (self.P, 'P'), (self.k, 'k'), (self.m_He, 'm_He')]:
            if var is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All required givens present")

    def test_interface_has_result_R(self):
        """Check that the collision rate R is provided."""
        if self.R is None:
            return (False, "Missing result R")
        return (True, "Result R present")

    def test_interface_given_types(self):
        """Check that givens are numeric (int or float)."""
        non_numeric = []
        for var, name in [(self.L, 'L'), (self.T, 'T'), (self.P, 'P'), (self.k, 'k'), (self.m_He, 'm_He')]:
            if var is not None and not isinstance(var, (int, float)):
                non_numeric.append(name)
        if non_numeric:
            return (False, f"Non-numeric givens: {non_numeric}")
        return (True, "All givens are numeric")

    # ---------- Invariant tests (physical relationships) ----------
    def test_invariant_n_from_ideal_gas_law(self):
        """Check n = P/(k*T) if n is provided."""
        if self.n is None:
            return (False, "SKIP: n not available")
        if any(var is None for var in [self.P, self.k, self.T]):
            return (False, "SKIP: missing P, k, or T")
        n_expected = self.P / (self.k * self.T)
        if math.isclose(self.n, n_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "n matches P/(kT)")
        else:
            return (False, f"n={self.n} but P/(kT)={n_expected}")

    def test_invariant_v_avg_from_formula(self):
        """Check v_avg = sqrt(8*k*T/(pi*m_He)) if v_avg provided."""
        if self.v_avg is None:
            return (False, "SKIP: v_avg not available")
        if any(var is None for var in [self.k, self.T, self.m_He]):
            return (False, "SKIP: missing k, T, or m_He")
        v_expected = math.sqrt(8 * self.k * self.T / (math.pi * self.m_He))
        if math.isclose(self.v_avg, v_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "v_avg matches sqrt(8kT/(πm))")
        else:
            return (False, f"v_avg={self.v_avg} but expected={v_expected}")

    def test_invariant_area_from_L(self):
        """Check A = L^2 if A provided."""
        if self.A is None:
            return (False, "SKIP: A not available")
        if self.L is None:
            return (False, "SKIP: L not available")
        A_expected = self.L ** 2
        if math.isclose(self.A, A_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "A matches L^2")
        else:
            return (False, f"A={self.A} but expected={A_expected}")

    def test_invariant_R_from_formula(self):
        """Check R = (1/4)*n*v_avg*A if all provided."""
        if self.R is None:
            return (False, "SKIP: R not available")
        if any(var is None for var in [self.n, self.v_avg, self.A]):
            return (False, "SKIP: missing n, v_avg, or A")
        R_expected = 0.25 * self.n * self.v_avg * self.A
        if math.isclose(self.R, R_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "R matches (1/4)*n*v_avg*A")
        else:
            return (False, f"R={self.R} but expected={R_expected}")

    def test_invariant_positivity_n_v_avg(self):
        """Check that n and v_avg are positive (if available)."""
        issues = []
        if self.n is not None and self.n <= 0:
            issues.append("n <= 0")
        if self.v_avg is not None and self.v_avg <= 0:
            issues.append("v_avg <= 0")
        if issues:
            return (False, f"Non-positive values: {issues}")
        return (True, "n and v_avg are positive")

    # ---------- Modeling tests ----------
    def test_modeling_R_order_of_magnitude(self):
        """Check that R is in the ballpark of 10^25 to 10^26 s^-1."""
        if self.R is None:
            return (False, "SKIP: R not available")
        if not isinstance(self.R, (int, float)):
            return (False, "R is not numeric")
        if math.isinf(self.R) or math.isnan(self.R):
            return (False, "R is inf or NaN")
        # Order of magnitude: 1e24 <= R <= 1e26 (broad)
        if 1e24 <= self.R <= 1e26:
            return (True, f"R = {self.R} is within expected order of magnitude 1e24-1e26")
        else:
            return (False, f"R = {self.R} is outside typical order of magnitude")

    def test_modeling_R_positive_finite(self):
        """R must be positive and finite."""
        if self.R is None:
            return (False, "SKIP: R not available")
        if self.R > 0 and math.isfinite(self.R):
            return (True, "R is positive and finite")
        else:
            return (False, f"R invalid: value={self.R}, isfinite={math.isfinite(self.R)}")

    # ---------- Monotonicity test ----------
    def test_monotonicity_R_increases_with_T(self):
        """
        Perturb T by +10% and recompute R using the kinetic formula.
        Expect R' > R.
        """
        if any(var is None for var in [self.L, self.T, self.P, self.k, self.m_He]):
            return (False, "SKIP: missing givens for recomputation")
        # Baseline R using given givens (ignore any provided n,v_avg,A,R)
        n_base = self.P / (self.k * self.T)
        v_base = math.sqrt(8 * self.k * self.T / (math.pi * self.m_He))
        A_base = self.L ** 2
        R_base = 0.25 * n_base * v_base * A_base

        # Perturb T by +10%
        T_new = self.T * 1.1
        n_new = self.P / (self.k * T_new)
        v_new = math.sqrt(8 * self.k * T_new / (math.pi * self.m_He))
        R_new = 0.25 * n_new * v_new * A_base

        if R_new > R_base:
            return (True, f"Monotonicity OK: T increased 10% -> R increased (R_base={R_base:.2e}, R_new={R_new:.2e})")
        else:
            return (False, f"Monotonicity violation: T increased but R did not increase (base={R_base:.2e}, new={R_new:.2e})")

    # ---------- Weak answer sanity (only one allowed) ----------
    def test_answer_sanity_R_broad_range(self):
        """
        Weak sanity: R should be positive and less than 10^27 (upper bound)
        (no math.isclose, no re-derivation)
        """
        if self.R is None:
            return (False, "SKIP: R not provided")
        if self.R <= 0:
            return (False, "R is not positive")
        if self.R >= 1e27:
            return (False, "R is unreasonably large (>1e27)")
        return (True, "R passes broad range sanity")

    # ---------- Test runner ----------
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)