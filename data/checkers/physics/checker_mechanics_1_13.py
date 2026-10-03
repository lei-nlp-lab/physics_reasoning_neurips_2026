import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely
        self.h = None
        self.g = None
        self.b_over_m = None

        if isinstance(solution_instance, dict):
            self.h = solution_instance.get('h', None)
            self.g = solution_instance.get('g', None)
            self.b_over_m = solution_instance.get('b_over_m', None)
            results = solution_instance.get('results', {})
        else:
            self.h = getattr(solution_instance, 'h', None)
            self.g = getattr(solution_instance, 'g', None)
            self.b_over_m = getattr(solution_instance, 'b_over_m', None)
            results = getattr(solution_instance, 'results', {})

        # Extract results with None defaults
        self.V0 = results.get('V0', None)
        self.t_impact = results.get('t_impact', None)
        self.V_func = results.get('V_func', None)   # string like "V0*exp(-b/m*x)"
        self.x_depth = results.get('x_depth', None)
        self.x_t = results.get('x_t', None)         # string like "(m/b)*ln(1+b*V0/m*t)"

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        missing = []
        for key in ['V0', 't_impact', 'V_func', 'x_depth', 'x_t']:
            if getattr(self, key) is None:
                missing.append(key)
        if missing:
            return False, f"Missing result keys: {missing}"
        return True, "All five required result keys present"

    def test_interface_key_types(self):
        if self.V0 is None or self.t_impact is None or self.x_depth is None:
            return False, "Some numeric results are None (cannot check types)"
        if not isinstance(self.V0, (int, float)):
            return False, "V0 is not numeric"
        if not isinstance(self.t_impact, (int, float)):
            return False, "t_impact is not numeric"
        if not isinstance(self.x_depth, (int, float)):
            return False, "x_depth is not numeric"
        if not isinstance(self.V_func, str):
            return False, "V_func is not a string"
        if not isinstance(self.x_t, str):
            return False, "x_t is not a string"
        return True, "All key types are as expected"

    # ------------------------------------------------------------
    # Relational invariant checks (cross-validation between outputs)
    # ------------------------------------------------------------
    def test_invariant_V_func_at_zero_equals_V0(self):
        """Check V_func(x=0) == V0 from boundary condition."""
        if self.V_func is None or self.V0 is None:
            return False, "SKIP: V_func or V0 not available"
        try:
            x = sp.symbols('x')
            expr = sp.sympify(self.V_func)
            # Evaluate at x=0 by setting 'x' to 0
            subs = {'V0': self.V0, 'b/m': 0, 'b': 0, 'x': 0}
            val = float(expr.subs(subs).evalf())
            if math.isclose(val, self.V0, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"V_func(0) = {val} ≈ V0 = {self.V0}"
            return False, f"V_func(0) = {val} but should be V0 = {self.V0}"
        except Exception as e:
            return False, f"Failed to evaluate V_func at x=0: {e}"

    def test_invariant_x_t_at_zero_equals_zero(self):
        """Check x_t(t=0) == 0."""
        if self.x_t is None:
            return False, "SKIP: x_t not available"
        try:
            t = sp.symbols('t')
            expr = sp.sympify(self.x_t)
            subs = {'t': 0}
            val = float(expr.subs(subs).evalf())
            if math.isclose(val, 0.0, abs_tol=1e-2):
                return True, f"x_t(0) = {val} ≈ 0"
            return False, f"x_t(0) = {val} but should be 0"
        except Exception as e:
            return False, f"Failed to evaluate x_t at t=0: {e}"

    def test_invariant_V_func_at_x_depth_equals_V0_over_10(self):
        """Check V_func(x_depth) == V0/10 (self-consistency between V_func and x_depth)."""
        if self.V_func is None or self.x_depth is None or self.V0 is None:
            return False, "SKIP: V_func, x_depth, or V0 not available"
        if self.b_over_m is None:
            return False, "SKIP: b_over_m missing"
        try:
            x = sp.symbols('x')
            expr = sp.sympify(self.V_func)
            subs = {'V0': self.V0, 'b/m': self.b_over_m, 'b': self.b_over_m, 'x': self.x_depth}
            val = float(expr.subs(subs).evalf())
            expected = self.V0 / 10.0
            if math.isclose(val, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"V_func(x_depth) = {val} ≈ V0/10 = {expected}"
            return False, f"V_func(x_depth) = {val} but expected V0/10 = {expected}"
        except Exception as e:
            return False, f"Failed to evaluate V_func at x_depth: {e}"

    # ------------------------------------------------------------
    # Modeling / trace quality checks
    # ------------------------------------------------------------
    def test_modeling_ode_consistency(self):
        """Check that derivative of V_func equals - (b/m) * V_func."""
        if self.V_func is None:
            return False, "SKIP: V_func not available"
        if self.V0 is None or self.b_over_m is None:
            return False, "SKIP: V0 or b/m missing"
        x = sp.symbols('x')
        try:
            expr = sp.sympify(self.V_func)
            dVdx = sp.diff(expr, x)
            expected = -self.b_over_m * expr
            diff = sp.simplify(dVdx - expected)
            subs = {'V0': self.V0, 'b/m': self.b_over_m, 'b': self.b_over_m * 1.0, 'x': x}
            diff_num = diff.subs(subs)
            if diff_num == 0:
                return True, "Derivative satisfies dv/dx = - (b/m) v (symbolically)"
            for x_val in [1.0, 2.0]:
                val = diff_num.subs(x, x_val).evalf()
                if abs(float(val)) > 1e-2:
                    return False, f"At x={x_val}, derivative discrepancy: {val}"
            return True, "Derivative satisfies dv/dx ≈ - (b/m) v (numerically)"
        except Exception as e:
            return False, f"Failed to check ODE consistency: {e}"

    def test_modeling_x_t_derivative_equals_V_func_at_x_t(self):
        """Check that dx/dt of x_t equals V_func evaluated at x_t."""
        if self.x_t is None or self.V_func is None:
            return False, "SKIP: x_t or V_func not available"
        if self.V0 is None or self.b_over_m is None:
            return False, "SKIP: V0 or b/m missing"
        t = sp.symbols('t')
        x = sp.symbols('x')
        try:
            x_expr = sp.sympify(self.x_t)
            V_func_expr = sp.sympify(self.V_func)
            # Build substitution dictionaries
            subs_common = {'V0': self.V0, 'b/m': self.b_over_m, 'b': self.b_over_m * 1.0, 't': t}
            subs_V = {'V0': self.V0, 'b/m': self.b_over_m, 'b': self.b_over_m * 1.0, 'x': x}
            # LHS: dx/dt
            dx_dt = sp.diff(x_expr.subs(subs_common), t).simplify()
            # RHS: V_func evaluated at x = x_t(t)
            x_t_sub = x_expr.subs(subs_common)
            V_at_x = V_func_expr.subs(subs_V).subs(x, x_t_sub).simplify()
            for t_val in [0.5, 1.0, 2.0]:
                lhs_val = dx_dt.subs(t, t_val).evalf()
                rhs_val = V_at_x.subs(t, t_val).evalf()
                if not math.isclose(float(lhs_val), float(rhs_val), rel_tol=1e-2, abs_tol=1e-2):
                    return False, f"At t={t_val}, dx/dt = {lhs_val} but V(x_t) = {rhs_val}"
            return True, "dx/dt matches V_func evaluated at x_t(t) within tolerance"
        except Exception as e:
            return False, f"Failed to check x_t derivative: {e}"

    # ------------------------------------------------------------
    # Monotonicity check (explicit perturbation)
    # ------------------------------------------------------------
    def test_monotonicity_depth_decreases_with_b_over_m(self):
        """Increasing b/m decreases the depth where V = V0/10."""
        if self.x_depth is None:
            return False, "SKIP: x_depth not available"
        if self.b_over_m is None:
            return False, "SKIP: b_over_m missing"
        bm_base = self.b_over_m
        depth_base = math.log(10) / bm_base
        # Perturb upward by 10%
        bm_pert = bm_base * 1.1
        depth_pert = math.log(10) / bm_pert
        if depth_pert < depth_base:
            return True, f"Increasing b/m from {bm_base} to {bm_pert} decreases depth from {depth_base:.2f} to {depth_pert:.2f} (expected monotonic)"
        else:
            return False, f"Monotonicity violation: depth did not decrease when b/m increased (base {depth_base:.2f}, perturbed {depth_pert:.2f})"

    # ------------------------------------------------------------
    # Weak answer sanity (one test only)
    # ------------------------------------------------------------
    def test_answer_sanity_weak(self):
        issues = []
        if self.V0 is not None:
            if not math.isfinite(self.V0) or self.V0 <= 0:
                issues.append(f"V0 invalid (value={self.V0})")
        if self.t_impact is not None:
            if not math.isfinite(self.t_impact) or self.t_impact <= 0:
                issues.append(f"t_impact invalid (value={self.t_impact})")
        if self.x_depth is not None:
            if not math.isfinite(self.x_depth) or self.x_depth <= 0:
                issues.append(f"x_depth invalid (value={self.x_depth})")
        if issues:
            return False, "; ".join(issues)
        return True, "Numeric results are positive and finite"

    # ------------------------------------------------------------
    # Run all tests
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)