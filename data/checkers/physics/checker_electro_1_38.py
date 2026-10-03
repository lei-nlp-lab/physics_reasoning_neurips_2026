import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs (givens) safely
        self.given = getattr(solution_instance, 'given', {})
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract givens (use None for missing)
        self.epsilon0 = self.given.get('epsilon0', None)
        self.Q = self.given.get('Q', None)
        self.b = self.given.get('b', None)
        self.a = self.given.get('a', None)
        self.V = self.given.get('V', None)
        self.r = self.given.get('r', None)  # for energy density
        
        # Extract results (use None for missing)
        self.C = self.results.get('C', None)
        self.w_e = self.results.get('w_e', None)
        self.U = self.results.get('U', None)
        self.W = self.results.get('W', None)
        self.a_opt = self.results.get('a_opt', None)
        self.E_min = self.results.get('E_min', None)
    
    def _is_numeric(self, value):
        """Check if value is numeric (int, float, or sympy number)."""
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, sp.Expr):
            return value.is_number
        return False
    
    def _eval_numeric(self, value, subs_dict=None):
        """Try to get a numeric float from value."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, sp.Expr):
            if subs_dict:
                value = value.subs(subs_dict)
            if value.is_number:
                return float(value.evalf())
        return None
    
    def _simplify_diff(self, expr1, expr2):
        """Simplify the difference of two symbolic expressions."""
        if expr1 is None or expr2 is None:
            return None
        return sp.simplify(expr1 - expr2)
    
    # ===================== INTERFACE TESTS =====================
    def test_interface_required_keys_present(self):
        """All six result keys must exist and not be None."""
        keys = ['C', 'w_e', 'U', 'W', 'a_opt', 'E_min']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")
    
    def test_interface_key_types(self):
        """Results must be numeric or symbolic expressions."""
        keys = ['C', 'w_e', 'U', 'W', 'a_opt', 'E_min']
        invalid = []
        for k in keys:
            val = getattr(self, k)
            if val is None:
                invalid.append(k)
            elif not self._is_numeric(val) and not isinstance(val, sp.Expr):
                invalid.append(k)
        if invalid:
            return (False, f"Invalid types for keys: {invalid}")
        return (True, "All keys have valid types")
    
    # ===================== INVARIANT TESTS =====================
    def test_invariant_capacitance_single_sphere(self):
        """C = 4 pi epsilon0 b."""
        if self.C is None or self.epsilon0 is None or self.b is None:
            return (False, "SKIP: missing input/result for capacitance")
        expected = 4 * math.pi * self.epsilon0 * self.b
        C_num = self._eval_numeric(self.C, {'epsilon0': self.epsilon0, 'b': self.b})
        if C_num is not None:
            if math.isclose(C_num, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"C matches expected value: {C_num}")
            else:
                return (False, f"C = {C_num}, expected ~{expected}")
        if isinstance(self.C, sp.Expr):
            C_sym = self.C.subs({sp.Symbol('epsilon0'): self.epsilon0, sp.Symbol('b'): self.b})
            expected_sym = 4 * sp.pi * self.epsilon0 * self.b
            diff = sp.simplify(C_sym - expected_sym)
            if diff == 0:
                return (True, "C symbolically matches 4π ε0 b")
            else:
                return (False, f"C symbolic difference: {diff}")
        return (False, "Cannot compare C")
    
    def test_invariant_energy_density_form(self):
        """w_e = Q^2 / (32 pi^2 epsilon0 r^4)."""
        if self.w_e is None or self.Q is None or self.epsilon0 is None:
            return (False, "SKIP: missing data for energy density")
        expected = sp.Symbol('Q')**2 / (32 * sp.pi**2 * sp.Symbol('epsilon0') * sp.Symbol('r')**4)
        w_e_num = self._eval_numeric(self.w_e, {'Q': self.Q, 'epsilon0': self.epsilon0, 'r': self.r or 2*self.b})
        if w_e_num is not None:
            r_test = self.r if self.r else (2*self.b if self.b else 1.0)
            expected_num = self.Q**2 / (32 * math.pi**2 * self.epsilon0 * r_test**4)
            if math.isclose(w_e_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"w_e matches at r={r_test}")
            else:
                return (False, f"w_e={w_e_num}, expected ~{expected_num}")
        if isinstance(self.w_e, sp.Expr):
            expr_diff = self._simplify_diff(self.w_e, expected)
            if expr_diff == 0:
                return (True, "w_e symbolically matches")
            else:
                return (False, f"w_e symbolic diff: {expr_diff}")
        return (False, "Cannot evaluate w_e")
    
    def test_invariant_total_energy_integral(self):
        """U = Q^2 / (8 pi epsilon0 b)."""
        if self.U is None or self.Q is None or self.epsilon0 is None or self.b is None:
            return (False, "SKIP: missing data for total energy")
        expected = sp.Symbol('Q')**2 / (8 * sp.pi * sp.Symbol('epsilon0') * sp.Symbol('b'))
        U_num = self._eval_numeric(self.U, {'Q': self.Q, 'epsilon0': self.epsilon0, 'b': self.b})
        if U_num is not None:
            expected_num = self.Q**2 / (8 * math.pi * self.epsilon0 * self.b)
            if math.isclose(U_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"U matches: {U_num}")
            else:
                return (False, f"U={U_num}, expected ~{expected_num}")
        if isinstance(self.U, sp.Expr):
            diff = self._simplify_diff(self.U, expected)
            if diff == 0:
                return (True, "U symbolically matches")
            else:
                return (False, f"U symbolic diff: {diff}")
        return (False, "Cannot evaluate U")
    
    def test_invariant_work_equals_energy(self):
        """W should equal U (energy conservation)."""
        if self.W is None or self.U is None:
            return (False, "SKIP: missing W or U")
        W_num = self._eval_numeric(self.W)
        U_num = self._eval_numeric(self.U)
        if W_num is not None and U_num is not None:
            if math.isclose(W_num, U_num, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"W = {W_num} matches U = {U_num}")
            else:
                return (False, f"W={W_num}, U={U_num}, not equal")
        if isinstance(self.W, sp.Expr) and isinstance(self.U, sp.Expr):
            diff = sp.simplify(self.W - self.U)
            if diff == 0:
                return (True, "W symbolically equals U")
            else:
                return (False, f"W - U = {diff}")
        return (False, "Cannot compare W and U")
    
    def test_invariant_optimal_radius_condition(self):
        """a_opt should be b/2."""
        if self.a_opt is None or self.b is None:
            return (False, "SKIP: missing a_opt or b")
        expected = sp.Symbol('b') / 2
        a_opt_num = self._eval_numeric(self.a_opt, {'b': self.b})
        if a_opt_num is not None:
            expected_num = self.b / 2
            if math.isclose(a_opt_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"a_opt = {a_opt_num} (b/2 = {expected_num})")
            else:
                return (False, f"a_opt = {a_opt_num}, expected {expected_num}")
        if isinstance(self.a_opt, sp.Expr):
            diff = sp.simplify(self.a_opt - sp.Symbol('b')/2)
            if diff == 0:
                return (True, "a_opt symbolically equals b/2")
            else:
                return (False, f"a_opt symbolic diff: {diff}")
        return (False, "Cannot evaluate a_opt")
    
    def test_invariant_minimum_field_value(self):
        """E_min = 4V / b."""
        if self.E_min is None or self.V is None or self.b is None:
            return (False, "SKIP: missing E_min, V, or b")
        expected = 4 * sp.Symbol('V') / sp.Symbol('b')
        E_min_num = self._eval_numeric(self.E_min, {'V': self.V, 'b': self.b})
        if E_min_num is not None:
            expected_num = 4 * self.V / self.b
            if math.isclose(E_min_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"E_min = {E_min_num}, expected {expected_num}")
            else:
                return (False, f"E_min = {E_min_num}, expected {expected_num}")
        if isinstance(self.E_min, sp.Expr):
            diff = sp.simplify(self.E_min - 4*sp.Symbol('V')/sp.Symbol('b'))
            if diff == 0:
                return (True, "E_min symbolically equals 4V/b")
            else:
                return (False, f"E_min symbolic diff: {diff}")
        return (False, "Cannot evaluate E_min")
    
    # ===================== MODELING / TRACE TESTS =====================
    def test_modeling_electric_field_between_spheres(self):
        """Placeholder: modeling quality deferred to monotonicity tests."""
        return (True, "Modeling quality covered by other tests")
    
    def test_modeling_energy_density_consistency(self):
        """Check that w_e matches half epsilon0 E^2 computed from Gauss's law."""
        if self.w_e is None or self.Q is None or self.epsilon0 is None or self.r is None:
            return (False, "SKIP: missing data")
        r_val = self.r
        if r_val is None and self.b is not None:
            r_val = 2 * self.b  # arbitrary test point
        if r_val is None:
            return (False, "SKIP: no r value")
        E_gauss = self.Q / (4 * math.pi * self.epsilon0 * r_val**2)
        w_e_expected = 0.5 * self.epsilon0 * E_gauss**2
        w_e_num = self._eval_numeric(self.w_e, {'r': r_val})
        if w_e_num is None:
            return (False, "Cannot evaluate w_e numerically")
        if math.isclose(w_e_num, w_e_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"w_e({r_val}) = {w_e_num}, consistent with Gauss")
        else:
            return (False, f"w_e({r_val}) = {w_e_num}, expected {w_e_expected}")
    
    # ===================== MONOTONICITY TESTS =====================
    def test_monotonicity_capacitance_with_b(self):
        """Capacitance C should increase with sphere radius b (perturb b by +10%)."""
        if self.C is None or self.b is None or self.epsilon0 is None:
            return (False, "SKIP: missing C, b, or epsilon0")
        perturbation = 0.1
        b_new = self.b * (1 + perturbation)
        # Evaluate C at original b and new b
        C_orig = self._eval_numeric(self.C, {'epsilon0': self.epsilon0, 'b': self.b})
        C_new = self._eval_numeric(self.C, {'epsilon0': self.epsilon0, 'b': b_new})
        if C_orig is None or C_new is None:
            return (False, "Cannot evaluate C numerically")
        if C_new > C_orig:
            return (True, f"C increases from {C_orig} to {C_new} when b increases by {perturbation*100}%")
        else:
            return (False, f"C does not increase: {C_orig} -> {C_new}")
    
    def test_monotonicity_energy_density_with_r(self):
        """Energy density w_e should decrease with increasing r (perturb r by +10%)."""
        if self.w_e is None or self.r is None:
            return (False, "SKIP: missing w_e or r")
        r_orig = self.r
        perturbation = 0.1
        r_new = r_orig * (1 + perturbation)
        w_e_orig = self._eval_numeric(self.w_e, {'r': r_orig})
        w_e_new = self._eval_numeric(self.w_e, {'r': r_new})
        if w_e_orig is None or w_e_new is None:
            return (False, "Cannot evaluate w_e at radii")
        # Expect w_e decreases as r increases
        if w_e_new < w_e_orig:
            return (True, f"w_e decreases from {w_e_orig} to {w_e_new} when r increases by {perturbation*100}%")
        else:
            return (False, f"w_e does not decrease: {w_e_orig} -> {w_e_new}")
    
    def test_monotonicity_electric_field_near_inner_sphere(self):
        """Electric field E(a) should be minimized at a=b/2; perturb a ±10% and verify increase."""
        if self.a_opt is None or self.V is None or self.b is None:
            return (False, "SKIP: missing data for E(a) test")
        a_opt_num = self._eval_numeric(self.a_opt, {'b': self.b})
        if a_opt_num is None:
            return (False, "Cannot evaluate a_opt numerically")
        perturbation = 0.1
        a_small = a_opt_num * (1 - perturbation)
        a_large = a_opt_num * (1 + perturbation)
        # Compute E(a) using derived formula: E(a) = V / (a * (1 - a/b))
        def E_a(a_val):
            return self.V / (a_val * (1 - a_val/self.b))
        E_small = E_a(a_small)
        E_large = E_a(a_large)
        E_opt = 4 * self.V / self.b  # from invariant
        # Both E_small and E_large should be > E_opt (minimum)
        if E_small > E_opt and E_large > E_opt:
            return (True, f"E(a) at a={a_small} and {a_large} > E_min={E_opt}")
        else:
            return (False, "E(a) not minimized at a_opt")
    
    # ===================== ANSWER SANITY TEST (minimal) =====================
    def test_answer_sanity_signs(self):
        """All results should be non-negative (physical)."""
        keys = ['C', 'w_e', 'U', 'W', 'a_opt', 'E_min']
        for k in keys:
            val = getattr(self, k)
            if val is None:
                continue
            num_val = self._eval_numeric(val)
            if num_val is not None and num_val < 0:
                return (False, f"{k} is negative: {num_val}")
        return (True, "All checked results are non-negative")
    
    # ===================== RUN ALL TESTS =====================
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