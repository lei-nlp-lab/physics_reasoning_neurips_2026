import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely – never use numeric defaults
        results = getattr(solution_instance, 'results', {})
        self.S0 = results.get('S0', None)
        self.Smax = results.get('Smax', None)
        self.Umin = results.get('Umin', None)
        self.Z = results.get('Z', None)
        self.U_mp = results.get('U_mp', None)
        self.integral = results.get('integral', None)

        # Extract givens (assumed stored in a dict or directly as attributes)
        givens = getattr(solution_instance, 'givens', {})
        self.N = givens.get('N', None)
        self.E = givens.get('E', None)
        self.k_B = givens.get('k_B', None)
        self.beta = givens.get('beta', None)   # Optional, may be missing

    # ------------------------------------------------------------
    # INTERFACE TESTS
    # ------------------------------------------------------------
    def test_interface_required_keys_present(self):
        """All five result keys (S0, Smax, Umin, Z, U_mp, integral) must be present."""
        keys = ['S0', 'Smax', 'Umin', 'Z', 'U_mp', 'integral']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"Missing results: {missing}")
        return (True, "All required keys present")

    def test_interface_given_parameters_present(self):
        """N, E, k_B must be given."""
        if self.N is None:
            return (False, "N missing")
        if self.E is None:
            return (False, "E missing")
        if self.k_B is None:
            return (False, "k_B missing")
        return (True, "All required givens present")

    # ------------------------------------------------------------
    # INVARIANT TESTS (use math.isclose)
    # ------------------------------------------------------------
    def test_invariant_S0_equals_zero(self):
        """S(0) must be exactly 0 (ground state degeneracy Ω=1)."""
        if self.S0 is None:
            return (False, "S0 not available")
        expected = 0.0
        if not math.isclose(self.S0, expected, rel_tol=1e-12, abs_tol=1e-12):
            return (False, f"S0 = {self.S0}, expected {expected}")
        return (True, "S0 = 0")

    def test_invariant_Smax_formula(self):
        """Smax must equal N * k_B * ln(3)."""
        if self.Smax is None or self.N is None or self.k_B is None:
            return (False, "Required data for Smax missing")
        expected = self.N * self.k_B * math.log(3)
        if not math.isclose(self.Smax, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Smax = {self.Smax}, expected = {expected}")
        return (True, "Smax consistent with N k ln 3")

    def test_invariant_Umin_formula(self):
        """Umin must equal - N * E."""
        if self.Umin is None or self.N is None or self.E is None:
            return (False, "Required data for Umin missing")
        expected = -self.N * self.E
        if not math.isclose(self.Umin, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Umin = {self.Umin}, expected = {expected}")
        return (True, "Umin = -N E")

    def test_invariant_integral_equals_entropy_difference(self):
        """∫ C/T dT = S(∞) - S(0) = Smax - S0."""
        if None in (self.S0, self.Smax, self.integral):
            return (False, "Missing S0, Smax, or integral")
        expected = self.Smax - self.S0
        if not math.isclose(self.integral, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"integral = {self.integral}, expected Smax - S0 = {expected}")
        return (True, "Integral equals entropy difference")

    def test_invariant_U_mp_at_zero_temperature(self):
        """U_mp should approach -N E as T→0 (β→∞). Perform check if beta given."""
        if self.U_mp is None or self.N is None or self.E is None:
            return (False, "Data missing for U_mp check")
        if self.beta is None:
            return (False, "Skip: beta not provided")
        # Parse U_mp as sympy expression (assuming it is)
        try:
            expr = sp.sympify(self.U_mp)
        except (TypeError, sp.SympifyError):
            return (False, "U_mp not a sympy expression; skip numeric check")
        # Large beta approximation
        large_beta = 1e10
        subs = {sp.Symbol('N'): self.N, sp.Symbol('E'): self.E, sp.Symbol('beta'): large_beta}
        try:
            val = expr.subs(subs).evalf()
        except Exception:
            return (False, "Cannot substitute symbolic values")
        expected = -self.N * self.E
        if not math.isclose(float(val), expected, rel_tol=1e-1, abs_tol=1e-1):
            return (False, f"U_mp at low T ≈ {val}, expected ≈ {expected}")
        return (True, "U_mp tends to -N E at low T")

    # ------------------------------------------------------------
    # MODELING / TRACE TESTS (at least two)
    # ------------------------------------------------------------
    def test_modeling_partition_function_product_form(self):
        """Z must equal (1 + 2 cosh(βE))^N."""
        if self.Z is None or self.N is None or self.E is None:
            return (False, "Data missing for Z check")
        if self.beta is None:
            return (False, "Skip: beta not provided")
        try:
            Z_sym = sp.sympify(self.Z)
        except:
            return (False, "Z not sympy expression")
        expected = (1 + 2*sp.cosh(self.beta * self.E))**self.N
        diff = sp.simplify(Z_sym - expected)
        if diff != 0:
            # numerical test as fallback
            test_beta = 1.0
            test_E = 1.0
            test_N = int(self.N)
            ref = (1 + 2*math.cosh(test_beta * test_E))**test_N
            try:
                val = Z_sym.subs({sp.Symbol('beta'): test_beta,
                                  sp.Symbol('E'): test_E,
                                  sp.Symbol('N'): test_N}).evalf()
            except:
                return (False, "Cannot evaluate Z numerically")
            if not math.isclose(float(val), ref, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Z numeric mismatch: {val} vs {ref}")
        return (True, "Z matches product form")

    def test_modeling_U_mp_from_lnZ_derivative(self):
        """U_mp must equal -d(ln Z)/dβ, derived from Z."""
        if self.Z is None or self.U_mp is None or self.N is None or self.E is None:
            return (False, "Data missing for U_mp derivative check")
        if self.beta is None:
            return (False, "Skip: beta not provided")
        try:
            Z_sym = sp.sympify(self.Z)
            lnZ = sp.log(Z_sym)
            dlnZ_dbeta = sp.diff(lnZ, sp.Symbol('beta'))
            expected_U = -sp.simplify(dlnZ_dbeta)
            # Compare with provided U_mp
            provided_U = sp.sympify(self.U_mp)
            diff = sp.simplify(provided_U - expected_U)
            if diff != 0:
                # numeric fallback
                subs = {sp.Symbol('N'): self.N, sp.Symbol('E'): self.E, sp.Symbol('beta'): self.beta}
                try:
                    num_provided = float(provided_U.subs(subs).evalf())
                    num_expected = float(expected_U.subs(subs).evalf())
                except:
                    return (False, "Cannot evaluate U_mp numerically")
                if not math.isclose(num_provided, num_expected, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"U_mp {num_provided} != -d ln Z/dβ = {num_expected}")
        except Exception as e:
            return (False, f"Derivative check crashed: {e}")
        return (True, "U_mp matches -d ln Z / dβ")

    # ------------------------------------------------------------
    # MONOTONICITY TESTS (each must perturb an input and compare)
    # ------------------------------------------------------------
    def test_monotonicity_Smax_increases_with_N(self):
        """If N increases by 1, Smax should increase by exactly k_B ln(3)."""
        if self.N is None or self.k_B is None or self.Smax is None:
            return (False, "Data missing for Smax monotonicity")
        # Baseline: Smax from given N
        baseline = self.N * self.k_B * math.log(3)
        if not math.isclose(self.Smax, baseline, rel_tol=1e-2):
            return (False, "Baseline Smax deviates from expected")
        # Perturb N -> N+1
        perturbed_N = self.N + 1
        perturbed_Smax = perturbed_N * self.k_B * math.log(3)
        if self.Smax >= perturbed_Smax:
            return (False, f"Smax ({self.Smax}) not less than perturbed Smax for N+1 ({perturbed_Smax})")
        return (True, "Smax increases with N (perturbation test passed)")

    def test_monotonicity_Umin_decreases_with_N(self):
        """If N increases by 1, Umin becomes more negative (decreases)."""
        if self.N is None or self.E is None or self.Umin is None:
            return (False, "Data missing for Umin monotonicity")
        baseline = -self.N * self.E
        if not math.isclose(self.Umin, baseline, rel_tol=1e-2):
            return (False, "Baseline Umin deviates from expected")
        perturbed_N = self.N + 1
        perturbed_Umin = -perturbed_N * self.E
        # For N+1, Umin is more negative (smaller)
        if self.Umin <= perturbed_Umin:
            return (False, f"Umin ({self.Umin}) not greater than perturbed Umin for N+1 ({perturbed_Umin})")
        return (True, "Umin decreases (more negative) as N increases (perturbation test passed)")

    # ------------------------------------------------------------
    # WEAK ANSWER SANITY TEST (max one)
    # ------------------------------------------------------------
    def test_answer_sanity_basic_properties(self):
        """Weak sanity: finite, non-NaN, sign constraints for numeric results."""
        numeric = {
            'S0': self.S0, 'Smax': self.Smax,
            'Umin': self.Umin, 'integral': self.integral
        }
        for name, val in numeric.items():
            if val is None:
                return (False, f"{name} missing")
            if not isinstance(val, (int, float)):
                return (False, f"{name} has non-numeric type")
            if math.isnan(val) or math.isinf(val):
                return (False, f"{name} is NaN or Inf")
        # Physics sign constraints
        if self.S0 < 0:
            return (False, "S0 should be non-negative")
        if self.Smax < 0:
            return (False, "Smax should be non-negative")
        if self.Umin > 0:
            return (False, "Umin should be ≤ 0")
        if self.integral < 0:
            return (False, "Integral should be non-negative")
        return (True, "All numeric results pass basic sanity")

    # ------------------------------------------------------------
    # TEST RUNNER
    # ------------------------------------------------------------
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
                passed = False
                msg = f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)