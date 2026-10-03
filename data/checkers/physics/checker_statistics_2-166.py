import sympy as sp
import math

class PhysicsChecker:
    """
    PhysicsChecker for thermodynamics problem: equation of state and heat capacity.
    Checks interface, invariants (exactness, first law), modeling (integration, ideal gas relation),
    monotonicity (by explicit perturbation of T), and weak sanity of answers.
    """
    
    def __init__(self, solution_instance):
        """
        Extract inputs and results safely, defaulting to None for missing values.
        Expects solution_instance to have:
          - results: dict with keys ['dU', 'a', 'd', 'e', 'U', 'ideal_relation']
          - givens: dict with keys ['p', 'Cv'] (optional) and constants a-f (optional)
        """
        # Results (required)
        self.results = getattr(solution_instance, 'results', {})
        self.dU_expr = self.results.get('dU', None)
        self.a_val = self.results.get('a', None)
        self.d_val = self.results.get('d', None)
        self.e_val = self.results.get('e', None)
        self.U_expr = self.results.get('U', None)
        self.ideal_relation = self.results.get('ideal_relation', None)
        
        # Givens (optional, for physics checks)
        self.givens = getattr(solution_instance, 'givens', {})
        self.p_expr = self.givens.get('p', None)
        self.Cv_expr = self.givens.get('Cv', None)
        # Constants (if provided separately)
        self.constants = {}
        for c in ['a','b','c','d','e','f']:
            self.constants[c] = self.givens.get(c, None)
        
        # Explanation steps (optional, for trace)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        
        # Define symbolic variables for computation
        self.T, self.V = sp.symbols('T V', positive=True)
        # Constants as symbols
        self.a, self.b, self.c, self.d, self.e, self.f = sp.symbols('a b c d e f')
    
    # ----- Interface Tests -----
    def test_interface_required_keys(self):
        """Check that all expected result keys exist and are not None."""
        required = ['dU', 'a', 'd', 'e', 'U', 'ideal_relation']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing results keys: {missing}")
        return (True, "All required keys present and non-None")
    
    def test_interface_dU_type(self):
        """Check dU is a string or sympy expression."""
        dU = self.dU_expr
        if dU is None:
            return (False, "SKIP: dU not available")
        if not isinstance(dU, (str, sp.Expr)):
            return (False, f"dU has unexpected type {type(dU)}")
        return (True, "dU type okay")
    
    def test_interface_constants_types(self):
        """Check a, d, e are numeric (int or float)."""
        for name, val in [('a', self.a_val), ('d', self.d_val), ('e', self.e_val)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            if not isinstance(val, (int, float)):
                return (False, f"{name} should be numeric, got {type(val)}")
        return (True, "a, d, e are numeric")
    
    # ----- Invariant Tests (physics laws) -----
    def test_invariant_exactness_condition(self):
        """
        Verify the exactness condition: ∂/∂V (C_v) = ∂/∂T ( T (∂p/∂T)_V - p ).
        If givens are available, derive constraints a=0, d=0, e=6b.
        """
        if self.p_expr is None or self.Cv_expr is None:
            return (False, "SKIP: givens p and Cv not available")
        try:
            p = sp.sympify(self.p_expr) if isinstance(self.p_expr, str) else self.p_expr
            Cv = sp.sympify(self.Cv_expr) if isinstance(self.Cv_expr, str) else self.Cv_expr
        except:
            return (False, "Could not parse p or Cv expressions")
        
        dp_dT = sp.diff(p, self.T)
        T_dp_dT_minus_p = self.T * dp_dT - p
        dCv_dV = sp.diff(Cv, self.V)
        dCoef_dT = sp.diff(T_dp_dT_minus_p, self.T)
        eq = sp.simplify(dCv_dV - dCoef_dT)
        
        if self.a_val is not None and self.d_val is not None and self.e_val is not None:
            subs_dict = {self.a: self.a_val, self.d: self.d_val, self.e: self.e_val,
                         self.b: self.b, self.c: self.c, self.f: self.f}
            eq_subbed = eq.subs(subs_dict)
            if sp.simplify(eq_subbed) == 0:
                return (True, "Exactness condition satisfied with given constants")
            else:
                return (False, f"Exactness condition not satisfied: {eq_subbed} != 0")
        else:
            return (False, "SKIP: cannot verify exactness without a,d,e from results")
    
    def test_invariant_first_law_expression(self):
        """
        Check that the dU expression in results matches the form:
        dU = Cv dT + ( T*(∂p/∂T)_V - p ) dV
        """
        if self.dU_expr is None:
            return (False, "SKIP: dU not available")
        if self.p_expr is None or self.Cv_expr is None:
            return (False, "SKIP: givens p and Cv not available")
        try:
            p = sp.sympify(self.p_expr) if isinstance(self.p_expr, str) else self.p_expr
            Cv = sp.sympify(self.Cv_expr) if isinstance(self.Cv_expr, str) else self.Cv_expr
        except:
            return (False, "Could not parse p or Cv")
        
        dp_dT = sp.diff(p, self.T)
        T_dp_dT_minus_p = self.T * dp_dT - p
        dU_expected = Cv * sp.Symbol('dT') + T_dp_dT_minus_p * sp.Symbol('dV')
        
        try:
            dU_actual = sp.sympify(self.dU_expr) if isinstance(self.dU_expr, str) else self.dU_expr
        except:
            return (False, "Could not parse dU expression")
        
        diff = sp.simplify(dU_expected - dU_actual)
        if diff == 0:
            return (True, "dU matches first law expression")
        else:
            return (False, f"dU does not match first law; difference: {diff}")
    
    # ----- Modeling Tests (quality checks) -----
    def test_modeling_U_integration(self):
        """
        Verify that U from results satisfies dU = (∂U/∂T)_V dT + (∂U/∂V)_T dV,
        and that ∂U/∂T matches Cv, ∂U/∂V matches T*(∂p/∂T)_V - p.
        """
        if self.U_expr is None:
            return (False, "SKIP: U not available")
        if self.p_expr is None or self.Cv_expr is None:
            return (False, "SKIP: givens p and Cv not available")
        try:
            U = sp.sympify(self.U_expr) if isinstance(self.U_expr, str) else self.U_expr
            p = sp.sympify(self.p_expr) if isinstance(self.p_expr, str) else self.p_expr
            Cv = sp.sympify(self.Cv_expr) if isinstance(self.Cv_expr, str) else self.Cv_expr
        except:
            return (False, "Could not parse expressions")
        
        dU_dT = sp.diff(U, self.T)
        dU_dV = sp.diff(U, self.V)
        expected_dU_dT = Cv
        expected_dU_dV = self.T * sp.diff(p, self.T) - p
        
        diff_T = sp.simplify(dU_dT - expected_dU_dT)
        diff_V = sp.simplify(dU_dV - expected_dU_dV)
        
        if diff_T == 0 and diff_V == 0:
            return (True, "U correctly integrates to correct partial derivatives")
        else:
            msg = ""
            if diff_T != 0:
                msg += f"∂U/∂T mismatch: {diff_T}; "
            if diff_V != 0:
                msg += f"∂U/∂V mismatch: {diff_V}"
            return (False, msg)
    
    def test_modeling_ideal_relation_consistency(self):
        """
        Weak check: ensure ideal_relation string is non-empty and contains plausible symbols.
        """
        if self.ideal_relation is None:
            return (False, "SKIP: ideal_relation not available")
        if not isinstance(self.ideal_relation, str) or len(self.ideal_relation) == 0:
            return (False, "ideal_relation is empty or not a string")
        for sym in ['b', 'f', 'c', 'T', 'V']:
            if sym not in self.ideal_relation:
                return (False, f"ideal_relation missing expected symbol '{sym}'")
        return (True, "ideal_relation contains expected symbols")
    
    # ----- Monotonicity Test (explicit perturbation of T) -----
    def test_monotonicity_pressure_increases_with_temperature(self):
        """
        Perturb temperature upward at multiple scales (1%, 5%, 10%) and verify
        that pressure is non-decreasing and that larger perturbations yield
        larger pressure (monotonically increasing).
        Requires numeric constants from givens.
        """
        # Need p expression and numeric constants
        if self.p_expr is None:
            return (False, "SKIP: p not available")
        numeric_constants = {}
        for c in ['a','b','c','d','e','f']:
            val = self.constants.get(c, None)
            if val is None:
                return (False, f"SKIP: numeric constant {c} not available")
            numeric_constants[c] = val
        
        # Parse p expression and substitute numeric constants
        try:
            p_sym = sp.sympify(self.p_expr) if isinstance(self.p_expr, str) else self.p_expr
            subs_dict = {self.a: numeric_constants['a'],
                         self.b: numeric_constants['b'],
                         self.c: numeric_constants['c'],
                         self.d: numeric_constants['d'],
                         self.e: numeric_constants['e'],
                         self.f: numeric_constants['f']}
            p_func = sp.lambdify((self.T, self.V), p_sym.subs(subs_dict), 'math')
        except Exception as e:
            return (False, f"Could not prepare numeric p function: {e}")
        
        # Base point
        T0 = 300.0  # K
        V0 = 0.001  # m^3
        try:
            p0 = float(p_func(T0, V0))
        except Exception as e:
            return (False, f"Error evaluating base pressure: {e}")
        
        # Perturbations
        percentages = [1, 5, 10]  # percent increase
        p_values = [p0]
        T_values = [T0]
        for perc in percentages:
            delta = T0 * perc / 100.0
            T_new = T0 + delta
            try:
                p_new = float(p_func(T_new, V0))
            except Exception as e:
                return (False, f"Error at {perc}% perturbation: {e}")
            p_values.append(p_new)
            T_values.append(T_new)
        
        # Check non-decreasing: each subsequent p >= previous
        for i in range(len(p_values)-1):
            if p_values[i+1] < p_values[i] - 1e-9:  # allow tiny numerical error
                return (False, f"Pressure decreased from {p_values[i]} to {p_values[i+1]} "
                               f"when T increased from {T_values[i]} to {T_values[i+1]}")
        # Check monotonic increase: ensure that p_values is strictly increasing
        # (if all equal, it's constant, still monotonic)
        # But we also want to ensure that large perturbations give larger pressure
        if p_values[-1] <= p_values[0]:
            return (False, "Pressure at +10% T is not greater than baseline (or equal)")
        
        return (True, f"Pressure increases monotonically with T: baseline={p0}, "
                      f"1%={p_values[1]}, 5%={p_values[2]}, 10%={p_values[3]}")
    
    # ----- Optional Weak Answer Sanity (max one) -----
    def test_answer_sanity_basic_values(self):
        """
        Weak sanity: check that a is exactly 0 (as derived) and d is exactly 0,
        and that dU, U, ideal_relation are non-empty strings or sympy expressions.
        """
        errors = []
        if self.a_val is None:
            errors.append("a is None")
        elif self.a_val != 0:
            errors.append(f"a is {self.a_val}, expected 0")
        if self.d_val is None:
            errors.append("d is None")
        elif self.d_val != 0:
            errors.append(f"d is {self.d_val}, expected 0")
        if self.e_val is None:
            errors.append("e is None")
        elif not isinstance(self.e_val, (int,float)):
            errors.append("e is not numeric")
        if self.dU_expr is None:
            errors.append("dU is None")
        if self.U_expr is None:
            errors.append("U is None")
        if self.ideal_relation is None:
            errors.append("ideal_relation is None")
        if errors:
            return (False, "; ".join(errors))
        return (True, "Basic answer sanity passed")
    
    # ----- Main runner -----
    def run_all_tests(self):
        """
        Discover all test_ methods, run them, collect results.
        Returns (all_pass, logs_list).
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

# Example usage (not part of final solution, but shows structure):
if __name__ == "__main__":
    # Dummy solution for testing
    class DummySolution:
        def __init__(self):
            self.results = {
                'dU': "(d*T**(1/2)*V + e*T**2*V + f*T**(1/2)) dT + (-(a/2)*T**(1/2) + 2*b*T**3 - c*V**(-2)) dV",
                'a': 0,
                'd': 0,
                'e': 6,
                'U': "2*b*T**3*V + (2/3)*f*T**(3/2) + c*V**(-1) + const",
                'ideal_relation': "3*b*T**3*V + 4*f*T**(3/2) = 3*c/V = const"
            }
            self.givens = {
                'p': "a*T**(1/2) + b*T**3 + c*V**(-2)",
                'Cv': "d*T**(1/2)*V + e*T**2*V + f*T**(1/2)",
                'a': 0,
                'b': 1,
                'c': 2,
                'd': 0,
                'e': 6,
                'f': 3
            }
    checker = PhysicsChecker(DummySolution())
    all_pass, logs = checker.run_all_tests()
    for log in logs:
        print(log)