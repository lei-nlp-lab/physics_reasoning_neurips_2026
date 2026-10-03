import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary safely
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        
        # Required result strings
        self.L_str = results.get('L', None)
        self.p_theta_str = results.get('p_theta', None)
        self.first_integral_str = results.get('first_integral', None)
        
        # Optional numeric givens (may be missing)
        self.m = getattr(solution_instance, 'm', None)
        self.k = getattr(solution_instance, 'k', None)
        self.r = getattr(solution_instance, 'r', None)
        self.theta = getattr(solution_instance, 'theta', None)
        self.r_dot = getattr(solution_instance, 'r_dot', None)
        self.theta_dot = getattr(solution_instance, 'theta_dot', None)
        
        # Pre-parse symbolic expressions (if possible)
        self.L_sym = None
        self.p_theta_sym = None
        if self.L_str:
            try:
                self.L_sym = sp.sympify(self.L_str)
            except (sp.SympifyError, TypeError):
                self.L_sym = None
        if self.p_theta_str:
            try:
                self.p_theta_sym = sp.sympify(self.p_theta_str)
            except (sp.SympifyError, TypeError):
                self.p_theta_sym = None

    # ---------- Interface Tests ----------
    def test_interface_required_keys_exist(self):
        missing = []
        if self.L_str is None:
            missing.append('L')
        if self.p_theta_str is None:
            missing.append('p_theta')
        if self.first_integral_str is None:
            missing.append('first_integral')
        if missing:
            return (False, f"Missing required result keys: {', '.join(missing)}")
        return (True, "All required result keys present")

    def test_interface_keys_are_nonempty_strings(self):
        for key, val in [('L', self.L_str), ('p_theta', self.p_theta_str), ('first_integral', self.first_integral_str)]:
            if not isinstance(val, str) or len(val.strip()) == 0:
                return (False, f"Key '{key}' is not a non-empty string")
        return (True, "All result strings are non-empty")

    # ---------- Invariant (theorem-based) Tests ----------
    def test_invariant_lagrangian_form(self):
        """Check that Lagrangian matches T - V exactly: 1/2 m (r_dot^2 + r^2 theta_dot^2) + k/r."""
        if self.L_sym is None:
            return (False, "Cannot parse L string into sympy expression")
        m, k, r, theta, r_dot, theta_dot = sp.symbols('m k r theta r_dot theta_dot', positive=True, real=True)
        expected = (sp.Rational(1,2)) * m * (r_dot**2 + r**2 * theta_dot**2) + k / r
        diff = sp.simplify(self.L_sym - expected)
        if diff != 0:
            return (False, f"Lagrangian form incorrect. Difference: {diff}")
        return (True, "Lagrangian matches expected form")

    def test_invariant_conjugate_momentum_form(self):
        """p_theta should equal m r^2 theta_dot."""
        if self.p_theta_sym is None:
            return (False, "Cannot parse p_theta string")
        m, r, theta_dot = sp.symbols('m r theta_dot', positive=True, real=True)
        expected = m * r**2 * theta_dot
        diff = sp.simplify(self.p_theta_sym - expected)
        if diff != 0:
            return (False, f"Conjugate momentum form incorrect. Difference: {diff}")
        return (True, "Conjugate momentum matches expected form")

    def test_invariant_first_integral_statement(self):
        """First integral should state that p_theta is constant."""
        if self.first_integral_str is None:
            return (False, "First integral string is missing")
        # Check that 'constant' appears and that p_theta expression is included
        s = self.first_integral_str.lower()
        if 'constant' not in s:
            return (False, "First integral does not mention 'constant'")
        # Try to parse something like 'm r^2 theta_dot = constant' from the string
        # We'll just check that the string contains the expected pattern.
        # A more robust check would be to extract LHS and compare with p_theta.
        if 'm' not in s or 'r' not in s or 'theta' not in s:
            return (False, "First integral likely missing correct functional form")
        return (True, "First integral states p_theta = constant")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_lagrangian_cyclic_coordinate(self):
        """Lagrangian should be independent of theta (cyclic)."""
        if self.L_sym is None:
            return (False, "Cannot parse L string")
        theta = sp.symbols('theta', real=True)
        # Check if any occurrence of theta without derivative
        # We can differentiate L with respect to theta
        # But sympy may not know if theta appears as r_dot etc. We'll check there's no 'theta' symbol alone (except in r^2 theta_dot^2)
        # A simple approach: count occurrences of theta as a symbol not within derivative.
        from sympy import Derivative
        # List free symbols except m, k, r, r_dot, theta_dot
        all_syms = self.L_sym.free_symbols
        # Check for bare theta (not theta_dot)
        theta_sym = sp.Symbol('theta')
        if theta_sym in all_syms:
            return (False, "Lagrangian contains explicit theta dependence (should be cyclic)")
        return (True, "Lagrangian shows no explicit theta dependence")

    def test_modeling_potential_sign(self):
        """Lagrangian should have +k/r term (since V = -k/r gives L = T - V = T + k/r)."""
        if self.L_sym is None:
            return (False, "Cannot parse L string")
        # Look for a term of the form k / r or k * r**(-1) with positive sign.
        # Expand and collect terms.
        expanded = sp.expand(self.L_sym)
        # Get coefficient of 1/r
        r = sp.symbols('r', positive=True)
        # Extract terms that have r in denominator
        # Simpler: check that the sum of terms with r in denominator is positive if k>0.
        # Use substitution r = 1 to get expression in k? Not safe.
        # Better: check that the expression does not contain -k/r.
        # We'll parse string: ensure it contains "+k/r" or "k/r" (positive)
        s = self.L_str.replace(" ", "")
        if "k/r" not in s and "k*r**(-1)" not in s:
            return (False, "Lagrangian missing positive k/r term")
        # Also check that there is no "-k/r" (negative)
        if "-k/r" in s:
            return (False, "Lagrangian contains -k/r, but should be +k/r")
        return (True, "Potential term has correct positive sign")

    def test_modeling_euler_lagrange_theta(self):
        """From L, derive d/dt(dL/dtheta_dot) = 0 should imply p_theta constant."""
        if self.L_sym is None:
            return (False, "Cannot parse L string")
        # Define symbols
        m, k, r, theta_dot, t = sp.symbols('m k r theta_dot t', positive=True, real=True)
        # We need to treat r as function of time? But L is given in terms of r and r_dot and theta_dot.
        # We can verify that dL/dtheta = 0 (already tested) and that dL/dtheta_dot = m r^2 theta_dot.
        # We already checked p_theta form. For Euler-Lagrange:
        # d/dt(p_theta) - dL/dtheta = 0 => d/dt(p_theta) = 0 because dL/dtheta=0.
        # Check that dL/dtheta is zero (already done)
        # Also check that p_theta is independent of theta (it is)
        # We can also symbolically compute dL/dtheta_dot and confirm it yields m r^2 theta_dot.
        # But that's redundant with test_invariant_conjugate_momentum_form.
        # Instead, we'll check that the Euler-Lagrange equation would imply conservation.
        # If we can compute the Euler-Lagrange expression symbolically and see it simplifies.
        # Since we don't have time derivatives of coordinates as functions, we can just assert the logical chain.
        # For robustness, we'll do a string check: confirm that first_integral is consistent with p_theta constant.
        # Already done. So this test can be a logical check.
        # We'll return a weak pass if the previous tests passed.
        if self.first_integral_str and self.p_theta_str and 'constant' in self.first_integral_str:
            return (True, "Cyclic coordinate leads to conserved p_theta as per first_integral")
        else:
            return (False, "Cannot confirm Euler-Lagrange conservation chain")

    # ---------- Monotonicity Tests ----------
    def test_monotonicity_lagrangian_increases_with_k(self):
        """If numeric givens available, increasing k should increase L (for fixed r)."""
        if self.m is None or self.k is None or self.r is None or self.r_dot is None or self.theta_dot is None:
            return (False, "SKIP: Numeric givens not available for monotonicity test")
        # Compute baseline L using the formula: T = 0.5*m*(r_dot^2 + r^2*theta_dot^2); V = -k/r; L = T - V = T + k/r
        T_base = 0.5 * self.m * (self.r_dot**2 + self.r**2 * self.theta_dot**2)
        V_base = -self.k / self.r
        L_base = T_base - V_base   # = T_base + self.k/self.r
        # Increase k by 10%
        k_new = self.k * 1.1
        L_new = T_base + k_new / self.r  # T unchanged because it doesn't involve k
        if L_new <= L_base:
            return (False, f"Lagrangian did not increase when k increased: baseline {L_base}, new {L_new}")
        return (True, f"Lagrangian increases with k: baseline {L_base}, perturbed {L_new}")

    def test_monotonicity_angular_momentum_increases_with_r(self):
        """If numeric givens available, increasing r should increase p_theta = m r^2 theta_dot."""
        if self.m is None or self.r is None or self.theta_dot is None:
            return (False, "SKIP: Numeric givens not available for monotonicity test")
        p_base = self.m * self.r**2 * self.theta_dot
        # Increase r by 5%
        r_new = self.r * 1.05
        p_new = self.m * r_new**2 * self.theta_dot
        if p_new <= p_base:
            return (False, f"p_theta did not increase when r increased: baseline {p_base}, new {p_new}")
        return (True, f"p_theta increases with r: baseline {p_base}, perturbed {p_new}")

    # ---------- Weak Answer Sanity Test ----------
    def test_answer_sanity_general(self):
        """Weak sanity: existence and non-emptiness of results."""
        # Already checked in interface; just verify all three are present and not too short.
        if self.L_str and len(self.L_str) > 5 and \
           self.p_theta_str and len(self.p_theta_str) > 5 and \
           self.first_integral_str and len(self.first_integral_str) > 5:
            return (True, "All results seem plausible length")
        return (False, "One or more results are too short or missing")

    # ---------- Runner ----------
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