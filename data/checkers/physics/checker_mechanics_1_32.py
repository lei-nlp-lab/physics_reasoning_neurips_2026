import sympy as sp
import math
import sys

class PhysicsChecker:
    """
    Audits a solution for the phonograph turntable problem.
    The solution_instance should contain:
      - attributes: m, omega, c, x0, y0, vx0, vy0 (numeric or symbolic)
      - attribute: results (dict) with keys: k, v_x, v_y, x, y
    """
    def __init__(self, solution_instance):
        # extract numeric/symbolic parameters, default None
        self.m = getattr(solution_instance, 'm', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.c = getattr(solution_instance, 'c', None)
        self.x0 = getattr(solution_instance, 'x0', None)
        self.y0 = getattr(solution_instance, 'y0', None)
        self.vx0 = getattr(solution_instance, 'vx0', None)
        self.vy0 = getattr(solution_instance, 'vy0', None)

        # results dictionary
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # extract symbolic results safely
        self.k_expr = self._safe_expr('k')
        self.v_x_expr = self._safe_expr('v_x')
        self.v_y_expr = self._safe_expr('v_y')
        self.x_expr = self._safe_expr('x')
        self.y_expr = self._safe_expr('y')

        # symbol for time
        self.t = sp.symbols('t')

    def _safe_expr(self, key):
        """Return sympy expression or None if missing/invalid."""
        raw = self.results.get(key, None)
        if raw is None:
            return None
        try:
            return sp.sympify(raw)
        except (sp.SympifyError, TypeError, ValueError):
            return None

    def _all_params_numeric(self):
        """Check that all essential parameters are present and numeric."""
        params = [self.m, self.omega, self.c, self.x0, self.y0, self.vx0, self.vy0]
        return all(p is not None and isinstance(p, (int, float, sp.Number)) for p in params)

    # ----------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------
    def test_interface_required_parameters(self):
        """Check that all required physical parameters are provided."""
        missing = []
        for name in ['m', 'omega', 'c', 'x0', 'y0', 'vx0', 'vy0']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing parameters: {missing}")
        return (True, "All parameters present")

    def test_interface_results_keys_exist(self):
        """Check that all required result keys are present."""
        required_keys = ['k', 'v_x', 'v_y', 'x', 'y']
        missing = [k for k in required_keys if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All result keys present")

    def test_interface_results_not_none(self):
        """Check that each result expression is not None."""
        for name, expr in [('k', self.k_expr), ('v_x', self.v_x_expr),
                           ('v_y', self.v_y_expr), ('x', self.x_expr),
                           ('y', self.y_expr)]:
            if expr is None:
                return (False, f"Result '{name}' could not be parsed")
        return (True, "All results are valid symbolic expressions")

    # ----------------------------------------------------------
    # Invariant (theorem-based) tests
    # ----------------------------------------------------------
    def test_invariant_k_from_equilibrium(self):
        """Check that k equals m*omega^2 (derived from equilibrium condition)."""
        if self.k_expr is None:
            return (False, "SKIP: k expression not available")
        if not (self.m is not None and self.omega is not None and
                self._all_params_numeric()):
            return (False, "SKIP: numerical parameters needed for this test")
        expected_k = self.m * self.omega**2
        try:
            # evaluate expression (might be numeric or symbolic)
            computed = float(self.k_expr.evalf(subs={sp.symbols('m'): self.m,
                                                     sp.symbols('omega'): self.omega,
                                                     sp.symbols('c'): self.c}) if 
                            self.k_expr.is_Symbol else float(self.k_expr))
            # also if it's a numeric expression with parameters, substitute
            computed = sp.N(self.k_expr.subs({sp.symbols('m'): self.m,
                                              sp.symbols('omega'): self.omega,
                                              sp.symbols('c'): self.c}))
        except Exception:
            # fallback: try to substitute using free symbols
            try:
                replacements = {sp.Symbol('m'): self.m,
                                sp.Symbol('omega'): self.omega,
                                sp.Symbol('c'): self.c}
                computed = float(self.k_expr.subs(replacements).evalf())
            except Exception:
                return (False, "Could not evaluate k expression")
        if math.isclose(computed, float(expected_k), rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"k ≈ mω²: k={computed}, expected={expected_k}")
        else:
            return (False, f"k mismatch: {computed} vs {expected_k}")

    def test_invariant_v_x_initial_condition(self):
        """
        Check that v_x(t=0) equals vx0 (given initial condition).
        This uses the decay and oscillation form.
        """
        if self.v_x_expr is None or self.vx0 is None:
            return (False, "SKIP: v_x expression or vx0 missing")
        # Substitute t=0 and evaluate
        try:
            t0_val = self.v_x_expr.subs({self.t: 0})
            # if expression contains other symbols, substitute them
            for sym in t0_val.free_symbols:
                # try to get value from object
                val = getattr(self, sym.name, None)
                if val is not None:
                    t0_val = t0_val.subs(sym, val)
            computed = float(t0_val.evalf())
        except Exception as e:
            return (False, f"Could not evaluate v_x at t=0: {e}")
        # Compare numerically (fuzzy allowed because of potential symbolic simplification)
        if math.isclose(computed, float(self.vx0), rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v_x(0) ≈ vx0: {computed} ≈ {self.vx0}")
        else:
            return (False, f"v_x(0) mismatch: {computed} vs {self.vx0}")

    def test_invariant_v_y_initial_condition(self):
        """Check v_y(0) equals vy0."""
        if self.v_y_expr is None or self.vy0 is None:
            return (False, "SKIP: v_y expression or vy0 missing")
        try:
            t0_val = self.v_y_expr.subs({self.t: 0})
            for sym in t0_val.free_symbols:
                val = getattr(self, sym.name, None)
                if val is not None:
                    t0_val = t0_val.subs(sym, val)
            computed = float(t0_val.evalf())
        except Exception as e:
            return (False, f"Could not evaluate v_y at t=0: {e}")
        if math.isclose(computed, float(self.vy0), rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v_y(0) ≈ vy0: {computed} ≈ {self.vy0}")
        else:
            return (False, f"v_y(0) mismatch: {computed} vs {self.vy0}")

    def test_invariant_velocity_magnitude_decay(self):
        """
        For the correct solution, ||v||^2 = (vx0^2+vy0^2) * exp(-2 c t / m)
        This holds regardless of rotation.
        """
        if None in (self.v_x_expr, self.v_y_expr, self.m, self.c):
            return (False, "SKIP: v_x, v_y, m, c needed")
        # Pick a few time points (numeric)
        if not self._all_params_numeric():
            return (False, "SKIP: require numeric parameters")
        # compute squared magnitude at t=0
        t0_vx = self.v_x_expr.subs({self.t: 0})
        t0_vy = self.v_y_expr.subs({self.t: 0})
        try:
            for sym in t0_vx.free_symbols | t0_vy.free_symbols:
                val = getattr(self, sym.name, None)
                if val is not None:
                    t0_vx = t0_vx.subs(sym, val)
                    t0_vy = t0_vy.subs(sym, val)
            mag_sq_0 = float(t0_vx.evalf())**2 + float(t0_vy.evalf())**2
        except Exception as e:
            return (False, f"Could not compute initial magnitude: {e}")

        # test at a later time, e.g., t = 0.5 / (m??) choose arbitrary positive t
        t_test = 0.5  # seconds
        vx_t = self.v_x_expr.subs({self.t: t_test})
        vy_t = self.v_y_expr.subs({self.t: t_test})
        try:
            for sym in vx_t.free_symbols | vy_t.free_symbols:
                val = getattr(self, sym.name, None)
                if val is not None:
                    vx_t = vx_t.subs(sym, val)
                    vy_t = vy_t.subs(sym, val)
            mag_sq_t = float(vx_t.evalf())**2 + float(vy_t.evalf())**2
        except Exception as e:
            return (False, f"Could not compute magnitude at t: {e}")

        expected_mag_sq_t = mag_sq_0 * math.exp(-2 * self.c * t_test / self.m)
        if math.isclose(mag_sq_t, expected_mag_sq_t, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"||v||^2 decays correctly: {mag_sq_t} ≈ {expected_mag_sq_t}")
        else:
            return (False, f"||v||^2 mismatch: {mag_sq_t} vs {expected_mag_sq_t}")

    # ----------------------------------------------------------
    # Modeling/trace consistency tests
    # ----------------------------------------------------------
    def test_modeling_v_x_contains_decay_and_rotation(self):
        """
        Check that v_x expression contains exp(-c t/m) AND
        linear combination of cos(2ωt) and sin(2ωt) (if vy0 may be zero, still present).
        """
        if self.v_x_expr is None:
            return (False, "SKIP: v_x expression missing")
        expr_str = sp.sstr(self.v_x_expr, sympy_order=True)
        # Check for decay factor
        if 'exp' not in expr_str and 'exp' not in expr_str:
            return (False, "v_x lacks exponential decay factor")
        # Check for trigonometric terms with argument containing omega
        # Look for cos, sin and ensure omega appears
        # We'll parse as expression and check structure
        trig_terms = self.v_x_expr.atoms(sp.sin, sp.cos)
        if not trig_terms:
            # maybe they factorized, but check if expression contains cos/sin functions
            # Use count of trig functions
            trig_count = self.v_x_expr.count(sp.sin) + self.v_x_expr.count(sp.cos)
            if trig_count == 0:
                return (False, "v_x contains no sin/cos terms; rotation absent")
        # Check that omega appears somewhere
        omega_sym = sp.Symbol('omega')
        if omega_sym not in self.v_x_expr.free_symbols:
            # maybe omega is in argument of trig
            # search for omega in the expression
            if not any(str(omega_sym) in str(s) for s in self.v_x_expr.free_symbols):
                return (False, "omega does not appear in v_x")
        return (True, "v_x contains decay and rotation")

    def test_modeling_x_initial_condition(self):
        """Check that x(t=0) equals x0."""
        if self.x_expr is None or self.x0 is None:
            return (False, "SKIP: x expression or x0 missing")
        try:
            t0_val = self.x_expr.subs({self.t: 0})
            for sym in t0_val.free_symbols:
                val = getattr(self, sym.name, None)
                if val is not None:
                    t0_val = t0_val.subs(sym, val)
            computed = float(t0_val.evalf())
        except Exception as e:
            return (False, f"Could not evaluate x(0): {e}")
        if math.isclose(computed, float(self.x0), rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"x(0) ≈ x0: {computed} ≈ {self.x0}")
        else:
            return (False, f"x(0) mismatch: {computed} vs {self.x0}")

    def test_modeling_y_initial_condition(self):
        """Check that y(t=0) equals y0."""
        if self.y_expr is None or self.y0 is None:
            return (False, "SKIP: y expression or y0 missing")
        try:
            t0_val = self.y_expr.subs({self.t: 0})
            for sym in t0_val.free_symbols:
                val = getattr(self, sym.name, None)
                if val is not None:
                    t0_val = t0_val.subs(sym, val)
            computed = float(t0_val.evalf())
        except Exception as e:
            return (False, f"Could not evaluate y(0): {e}")
        if math.isclose(computed, float(self.y0), rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"y(0) ≈ y0: {computed} ≈ {self.y0}")
        else:
            return (False, f"y(0) mismatch: {computed} vs {self.y0}")

    # ----------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------
    def test_monotonicity_damping_strength_increases_with_c(self):
        """
        For fixed time, increasing c should reduce magnitude of velocity.
        Choose two positive c values, evaluate v_x^2+v_y^2 at a positive t.
        """
        if not self._all_params_numeric():
            return (False, "SKIP: require numeric parameters")
        if None in (self.v_x_expr, self.v_y_expr):
            return (False, "SKIP: velocity expressions missing")
        # baseline c
        c0 = float(self.c)
        t_test = 0.3  # seconds
        # compute magnitude at baseline
        try:
            vx_base = self.v_x_expr.subs({self.t: t_test})
            vy_base = self.v_y_expr.subs({self.t: t_test})
            for sym in vx_base.free_symbols | vy_base.free_symbols:
                val = getattr(self, sym.name, None)
                if val is not None:
                    vx_base = vx_base.subs(sym, val)
                    vy_base = vy_base.subs(sym, val)
            mag_base = math.sqrt(float(vx_base.evalf())**2 + float(vy_base.evalf())**2)
        except Exception as e:
            return (False, f"Could not compute baseline magnitude: {e}")

        # increase c by 20%
        c_high = 1.2 * c0
        try:
            vx_high = self.v_x_expr.subs({self.t: t_test, sp.Symbol('c'): c_high})
            vy_high = self.v_y_expr.subs({self.t: t_test, sp.Symbol('c'): c_high})
            for sym in vx_high.free_symbols | vy_high.free_symbols:
                val = getattr(self, sym.name, None)
                if val is not None and sym.name != 'c':
                    vx_high = vx_high.subs(sym, val)
                    vy_high = vy_high.subs(sym, val)
            mag_high = math.sqrt(float(vx_high.evalf())**2 + float(vy_high.evalf())**2)
        except Exception as e:
            return (False, f"Could not compute high-c magnitude: {e}")

        if mag_high < mag_base:
            return (True, f"Magnitude decreases with c: {mag_high:.4f} < {mag_base:.4f}")
        else:
            return (False, f"Monotonicity fails: {mag_high:.4f} >= {mag_base:.4f}")

    def test_monotonicity_x_increases_with_vx0(self):
        """
        For fixed initial x0 and other parameters, increasing initial velocity vx0
        should increase x(t) at a small positive time (since the solution is linear
        in initial velocities). We perturb vx0 by +10% and check that the displacement
        from x0 increases.
        """
        if not self._all_params_numeric():
            return (False, "SKIP: require numeric parameters")
        if self.x_expr is None or self.vx0 is None:
            return (False, "SKIP: x expression or vx0 missing")
        # Use a small time where the effect is monotonic
        t_test = 0.1
        # Baseline displacement: x(t) - x0
        try:
            x_base = self.x_expr.subs({self.t: t_test})
            # substitute all known symbols except maybe vx0 (which we will perturb)
            for sym in x_base.free_symbols:
                name = sym.name
                if name == 'vx0':
                    continue
                val = getattr(self, name, None)
                if val is not None:
                    x_base = x_base.subs(sym, val)
            x_base_val = float(x_base.evalf())
            delta_base = x_base_val - float(self.x0)
        except Exception as e:
            return (False, f"Could not compute baseline x(t): {e}")

        # increase vx0 by 10%
        vx0_high = 1.1 * float(self.vx0)
        try:
            x_high = self.x_expr.subs({self.t: t_test, sp.Symbol('vx0'): vx0_high})
            for sym in x_high.free_symbols:
                name = sym.name
                if name == 'vx0':
                    continue
                val = getattr(self, name, None)
                if val is not None:
                    x_high = x_high.subs(sym, val)
            x_high_val = float(x_high.evalf())
            delta_high = x_high_val - float(self.x0)
        except Exception as e:
            return (False, f"Could not compute high vx0 x(t): {e}")

        # For the solution to be correct, the displacement should increase with vx0
        if delta_high > delta_base:
            return (True, f"x(t)-x0 increases with vx0: {delta_high:.4f} > {delta_base:.4f}")
        else:
            return (False, f"Monotonicity fails: {delta_high:.4f} <= {delta_base:.4f}")

    # ----------------------------------------------------------
    # Answer sanity test (weak, no isclose, no derivation)
    # ----------------------------------------------------------
    def test_answer_sanity_k_positive(self):
        """Check that k is positive and finite."""
        if self.k_expr is None:
            return (False, "SKIP: k expression missing")
        try:
            # Evaluate numerically if possible
            subs_dict = {}
            for sym in self.k_expr.free_symbols:
                val = getattr(self, sym.name, None)
                if val is not None:
                    subs_dict[sym] = val
            if subs_dict:
                val = float(self.k_expr.subs(subs_dict).evalf())
            else:
                # it might be a pure number
                val = float(self.k_expr.evalf())
        except Exception:
            # Cannot evaluate; just check structure
            # At least ensure expression contains m and omega
            if not (sp.Symbol('m') in self.k_expr.free_symbols and
                    sp.Symbol('omega') in self.k_expr.free_symbols):
                return (False, "k expression does not contain m and omega")
            return (True, "k expression plausible (contains m and omega)")
        if val > 0 and math.isfinite(val):
            return (True, f"k = {val} is positive and finite")
        else:
            return (False, f"k = {val} is not positive or not finite")

    # ----------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------
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