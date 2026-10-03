import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})
        # Required result keys
        self.P2 = self.results.get('P2', None)
        self.P3 = self.results.get('P3', None)
        self.cond_tau = self.results.get('cond_tau', None)
        self.cond_pert = self.results.get('cond_pert', None)
        # Physical constants (try givens, then results)
        self.hbar = self._get_constant('hbar')
        self.e = self._get_constant('e')
        self.m = self._get_constant('m')
        self.a = self._get_constant('a')
        self.E_field = self._get_constant('E')
        self.tau = self._get_constant('tau')

    def _get_constant(self, name):
        """Retrieve a constant from givens or results; return None if missing."""
        val = self.givens.get(name, None)
        if val is None:
            val = self.results.get(name, None)
        return val

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
        return (all_pass, logs)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        missing = []
        if self.P2 is None:
            missing.append('P2')
        if self.P3 is None:
            missing.append('P3')
        if self.cond_tau is None:
            missing.append('cond_tau')
        if self.cond_pert is None:
            missing.append('cond_pert')
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_P2_type(self):
        if self.P2 is None:
            return (False, "SKIP: P2 not available")
        if isinstance(self.P2, (int, float)):
            return (True, "P2 is numeric")
        elif isinstance(self.P2, str):
            return (True, "P2 is a string expression")
        else:
            return (False, f"P2 has unexpected type {type(self.P2)}")

    def test_interface_P3_type(self):
        if self.P3 is None:
            return (False, "SKIP: P3 not available")
        if isinstance(self.P3, (int, float)):
            return (True, "P3 is numeric")
        elif isinstance(self.P3, str):
            return (True, "P3 is a string expression")
        else:
            return (False, f"P3 has unexpected type {type(self.P3)}")

    # ----------------------------------------------------------------------
    # Invariant tests (physics principles)
    # ----------------------------------------------------------------------
    def test_invariant_P3_zero(self):
        if self.P3 is None:
            return (False, "SKIP: P3 not available")
        # P3 must be zero because <3|x|1>=0 by parity
        if isinstance(self.P3, (int, float)):
            if math.isclose(self.P3, 0, abs_tol=1e-10):
                return (True, "P3 is zero (numerically)")
            else:
                return (False, f"P3 = {self.P3}, expected 0")
        elif isinstance(self.P3, str):
            try:
                expr = sp.sympify(self.P3)
                if sp.simplify(expr) == 0:
                    return (True, "P3 expression simplifies to 0")
                else:
                    return (False, f"P3 expression '{self.P3}' does not simplify to 0")
            except Exception:
                return (False, "Cannot parse P3 string, but P3 should be 0")
        else:
            return (False, "P3 type not handled")

    def test_invariant_short_time_condition(self):
        # τ << ħ/(E₂–E₁) must hold for the short-time approximation
        if None in [self.hbar, self.m, self.a, self.tau]:
            return (False, "SKIP: need numeric ħ, m, a, τ")
        # E₂–E₁ = (4-1)π²ħ²/(2ma²) = 3π²ħ²/(2ma²)
        # τ_limit = ħ/(E₂–E₁) = 2ma²/(3π²ħ)
        tau_limit = 2 * self.m * self.a**2 / (3 * math.pi**2 * self.hbar)
        # A conservative check: τ < 0.1 * τ_limit is "much less"
        if self.tau < 0.1 * tau_limit:
            return (True, f"τ = {self.tau} << τ_limit = {tau_limit}")
        elif self.tau < tau_limit:
            return (True, f"τ = {self.tau} < τ_limit = {tau_limit} (not strictly <<)")
        else:
            return (False, f"τ = {self.tau} is not less than τ_limit = {tau_limit}")

    def test_invariant_perturbation_small(self):
        # For first-order perturbation theory, |c_n|² << 1, so P2 << 1
        if self.P2 is None:
            return (False, "SKIP: P2 not available")
        if isinstance(self.P2, (int, float)):
            if self.P2 < 1:
                return (True, f"P2 = {self.P2} < 1 (valid perturbation)")
            else:
                return (False, f"P2 = {self.P2} is not < 1 (first-order theory may break)")
        else:
            return (False, "SKIP: P2 not numeric, cannot check magnitude")

    # ----------------------------------------------------------------------
    # Modeling tests (expression correctness)
    # ----------------------------------------------------------------------
    def test_modeling_P2_expression(self):
        # P2 must match (16 e E a τ / (9π² ħ))²
        if self.P2 is None:
            return (False, "SKIP: P2 not available")
        # Define symbolic constants
        hbar_sym, e_sym, E_sym, a_sym, tau_sym, pi_sym = sp.symbols('hbar e E a tau pi', positive=True)
        expected = (16*e_sym*E_sym*a_sym*tau_sym/(9*pi_sym**2*hbar_sym))**2

        # Case 1: P2 is a string expression
        if isinstance(self.P2, str):
            try:
                expr = sp.sympify(self.P2)
            except Exception as e:
                return (False, f"Cannot sympify P2 string: {e}")
            diff = sp.simplify(expr - expected)
            if diff == 0:
                return (True, "P2 symbolic expression matches expected form")
            # Try ratio check
            ratio = sp.simplify(expr / expected)
            if ratio == 1:
                return (True, "P2 symbolic expression matches expected (ratio=1)")
            else:
                return (False, f"P2 expression differs from expected. Difference: {diff}")

        # Case 2: P2 is numeric; require numeric constants to recompute
        elif isinstance(self.P2, (int, float)):
            if None in [self.hbar, self.e, self.a, self.tau, self.E_field]:
                return (False, "SKIP: numeric constants needed to verify numeric P2")
            expected_num = (16*self.e*self.E_field*self.a*self.tau/(9*math.pi**2*self.hbar))**2
            if math.isclose(self.P2, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Numerical P2 = {self.P2} agrees with expected ≈ {expected_num}")
            else:
                return (False, f"Numerical P2 = {self.P2}, expected approx {expected_num}")
        else:
            return (False, "P2 type not handled")

    def test_modeling_cond_tau_correctness(self):
        # Check that cond_tau mentions the essential inequality
        if self.cond_tau is None:
            return (False, "SKIP: cond_tau not available")
        msg = self.cond_tau
        if 'tau' in msg and 'hbar' in msg and '<<' in msg and ('E2' in msg or 'E1' in msg):
            return (True, "cond_tau correctly references τ, ħ, and energy levels")
        else:
            return (False, f"cond_tau '{msg}' does not seem correct")

    def test_modeling_cond_pert_correctness(self):
        # Check that cond_pert mentions the smallness condition from P2<<1
        if self.cond_pert is None:
            return (False, "SKIP: cond_pert not available")
        msg = self.cond_pert
        if 'eE' in msg and 'tau' in msg and 'hbar' in msg and '<<' in msg:
            # Optionally check for the numerical factor (9π²/16)
            if '9' in msg and 'pi' in msg:
                return (True, "cond_pert looks correct")
            else:
                return (False, "cond_pert mentions eEτ/ħ but may miss the numerical factor")
        else:
            return (False, f"cond_pert '{msg}' does not contain necessary quantities")

    # ----------------------------------------------------------------------
    # Monotonicity tests (behaviour under parametric variation)
    # ----------------------------------------------------------------------
    def test_monotonicity_P2_increases_with_tau(self):
        # P2 ∝ τ², so increasing τ should increase P2
        if not isinstance(self.P2, (int, float)):
            return (False, "SKIP: need numeric P2 for monotonicity test")
        if not isinstance(self.tau, (int, float)):
            return (False, "SKIP: need numeric τ")
        if None in [self.hbar, self.e, self.a, self.E_field]:
            return (False, "SKIP: need numeric constants to compute new P2")
        # Compute P2 at τ_new = τ * 1.1
        tau_new = self.tau * 1.1
        new_P2 = (16*self.e*self.E_field*self.a*tau_new/(9*math.pi**2*self.hbar))**2
        if new_P2 > self.P2:
            return (True, f"Increasing τ increases P2 (baseline {self.P2}, new {new_P2})")
        else:
            return (False, "P2 does not increase with τ")

    def test_monotonicity_P2_increases_with_E(self):
        # P2 ∝ E², so increasing E should increase P2
        if not isinstance(self.P2, (int, float)):
            return (False, "SKIP: need numeric P2 for monotonicity test")
        if not isinstance(self.E_field, (int, float)):
            return (False, "SKIP: need numeric E")
        if None in [self.hbar, self.e, self.a, self.tau]:
            return (False, "SKIP: need numeric constants to compute new P2")
        # Compute P2 at E_new = E * 1.1
        E_new = self.E_field * 1.1
        new_P2 = (16*self.e*E_new*self.a*self.tau/(9*math.pi**2*self.hbar))**2
        if new_P2 > self.P2:
            return (True, f"Increasing E increases P2 (baseline {self.P2}, new {new_P2})")
        else:
            return (False, "P2 does not increase with E")

    # ----------------------------------------------------------------------
    # Answer sanity test (weak, at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_nonnegative(self):
        # Only check that P2 and P3 are nonnegative if numeric
        if self.P2 is not None and isinstance(self.P2, (int, float)):
            if self.P2 < -1e-12:
                return (False, f"P2 = {self.P2} is negative")
        if self.P3 is not None and isinstance(self.P3, (int, float)):
            if self.P3 < -1e-12:
                return (False, f"P3 = {self.P3} is negative")
        if self.P2 is None and self.P3 is None:
            return (False, "SKIP: neither P2 nor P3 available")
        return (True, "P2 and P3 (if numeric) are nonnegative")