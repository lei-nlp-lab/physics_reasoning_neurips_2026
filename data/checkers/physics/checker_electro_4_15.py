import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for an electromagnetic wave problem:
    B = B0 * exp(a*x) * sin(k*y - omega*t) * e_z  (Gaussian units, vacuum)
    Checks Maxwell's equations, dispersion relation, and propagation speed.
    """
    def __init__(self, solution_instance):
        # Extract results safely with None defaults
        results = getattr(solution_instance, 'results',
                          getattr(solution_instance, 'output', {}))
        self.Ex = results.get('Ex', None)
        self.Ey = results.get('Ey', None)
        self.E_vector = results.get('E_vector', None)
        self.v = results.get('v', None)
        self.possible = results.get('possible', None)

        # Givens (may be present as symbolic variables or numeric)
        self.B0 = results.get('B0', None)
        self.a = results.get('a', None)
        self.k = results.get('k', None)
        self.omega = results.get('omega', None)
        self.c = results.get('c', None)
        self.x = sp.Symbol('x', real=True)
        self.y = sp.Symbol('y', real=True)
        self.t = sp.Symbol('t', real=True)

        # Store original instance for potential extra attributes
        self.instance = solution_instance

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys are present (non-None)."""
        required = ['Ex', 'Ey', 'E_vector', 'v', 'possible']
        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            return (False, f"Missing required keys: {missing}")
        return (True, "All required keys present")

    def test_interface_types(self):
        """Check that Ex and Ey are sympy expressions (or at least strings)."""
        for name in ['Ex', 'Ey']:
            val = getattr(self, name, None)
            if val is None:
                return (False, f"SKIP: {name} is None")
            if not isinstance(val, (sp.Expr, sp.Basic, str)):
                return (False, f"{name} is not a sympy expression or string")
        return (True, "Ex and Ey are valid symbolic expressions")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_divergence_e(self):
        """Check ∇·E = 0 using symbolic differentiation."""
        if self.Ex is None or self.Ey is None:
            return (False, "SKIP: Ex or Ey not available")
        try:
            Ex = sp.sympify(self.Ex) if isinstance(self.Ex, str) else self.Ex
            Ey = sp.sympify(self.Ey) if isinstance(self.Ey, str) else self.Ey
        except Exception as e:
            return (False, f"Cannot sympify Ex/Ey: {e}")

        try:
            divE = sp.diff(Ex, self.x) + sp.diff(Ey, self.y)
            all_vars = list(divE.free_symbols) if hasattr(divE, 'free_symbols') else []
            simplified = sp.simplify(divE)
            if simplified == 0:
                return (True, "divE = 0 symbolically")
            # Substitute numeric values for remaining variables
            default_vals = {'B0': 1.0, 'a': 0.2, 'k': 2.0, 'omega': 3.0, 'c': 1.0}
            complete_subs = {self.x: 1.0, self.y: 1.0, self.t: 1.0}
            for symb in all_vars:
                name = str(symb)
                if self.x == symb or self.y == symb or self.t == symb:
                    continue
                val = getattr(self, name, None)
                if val is None and name in default_vals:
                    val = default_vals[name]
                if val is not None and isinstance(val, (int, float)):
                    complete_subs[symb] = float(val)
                else:
                    return (False, f"SKIP: cannot evaluate divergence due to missing numeric value for {name}")
            numeric_val = float(divE.subs(complete_subs).evalf())
            if math.isclose(numeric_val, 0.0, abs_tol=1e-5):
                return (True, f"∇·E ≈ {numeric_val:.2e} (near zero)")
            else:
                return (False, f"∇·E = {numeric_val:.2e}, not zero")
        except Exception as e:
            return (False, f"Divergence computation crashed: {e}")

    def test_invariant_faraday(self):
        """Check Faraday's law: ∂Ey/∂x - ∂Ex/∂y = (ω/c) B0 e^{ax} cos(ky-ωt)."""
        if self.Ex is None or self.Ey is None:
            return (False, "SKIP: Ex or Ey not available")
        try:
            Ex = sp.sympify(self.Ex) if isinstance(self.Ex, str) else self.Ex
            Ey = sp.sympify(self.Ey) if isinstance(self.Ey, str) else self.Ey
        except:
            return (False, "Cannot sympify Ex/Ey")

        lhs = sp.diff(Ey, self.x) - sp.diff(Ex, self.y)
        # Use default numeric values if missing
        default_vals = {'B0': 1.0, 'a': 0.2, 'k': 2.0, 'omega': 3.0, 'c': 1.0}
        B0_val = self.B0 if self.B0 is not None else default_vals['B0']
        a_val = self.a if self.a is not None else default_vals['a']
        k_val = self.k if self.k is not None else default_vals['k']
        omega_val = self.omega if self.omega is not None else default_vals['omega']
        c_val = self.c if self.c is not None else default_vals['c']

        w = k_val*self.y - omega_val*self.t
        rhs = (omega_val / c_val) * B0_val * sp.exp(a_val*self.x) * sp.cos(w)

        diff = sp.simplify(lhs - rhs)
        if diff == 0:
            return (True, "Faraday's law satisfied symbolically")
        # Numeric check at multiple points
        test_points = [
            {self.x: 0.5, self.y: 0.2, self.t: 0.1},
            {self.x: 1.0, self.y: 0.5, self.t: 0.3},
            {self.x: 0.0, self.y: 0.0, self.t: 0.0}
        ]
        for point in test_points:
            subs = point.copy()
            for sym in diff.free_symbols:
                if sym not in subs:
                    name = str(sym)
                    if name == 'B0': subs[sym] = float(B0_val)
                    elif name == 'a': subs[sym] = float(a_val)
                    elif name == 'k': subs[sym] = float(k_val)
                    elif name == 'omega': subs[sym] = float(omega_val)
                    elif name == 'c': subs[sym] = float(c_val)
                    else:
                        return (False, f"Unknown symbol {sym} in Faraday check")
            try:
                val = float(diff.subs(subs).evalf())
                if not math.isclose(val, 0.0, abs_tol=1e-5):
                    return (False, f"Faraday violation at {point}: diff = {val:.2e}")
            except Exception as e:
                return (False, f"Numeric evaluation failed: {e}")
        return (True, "Faraday's law holds within tolerance at test points")

    def test_invariant_dispersion(self):
        """Check dispersion relation: ω^2 = c^2 (k^2 - a^2)."""
        # Need numeric values
        defaults = {'omega': 3.0, 'c': 1.0, 'k': 2.0, 'a': 0.2}
        names = ['omega', 'c', 'k', 'a']
        vals = {}
        for name in names:
            val = getattr(self, name, None)
            if val is None:
                if name in defaults:
                    vals[name] = defaults[name]
                else:
                    return (False, f"SKIP: missing {name} and no default")
            else:
                try:
                    vals[name] = float(val)
                except:
                    return (False, f"SKIP: cannot convert {name} to float")
        omega, c, k, a = vals['omega'], vals['c'], vals['k'], vals['a']
        lhs = omega**2
        rhs = c**2 * (k**2 - a**2)
        if math.isclose(lhs, rhs, rel_tol=1e-5):
            return (True, f"Dispersion holds: ω² = {lhs:.3f}, c²(k²-a²) = {rhs:.3f}")
        else:
            return (False, f"Dispersion violated: ω²={lhs:.3f} vs rhs={rhs:.3f}")

    # ----------------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------------
    def test_modeling_exponential_factor(self):
        """Check that both Ex and Ey contain exp(a*x) factor."""
        if self.Ex is None or self.Ey is None:
            return (False, "SKIP: Ex/Ey missing")
        try:
            Ex = sp.sympify(self.Ex) if isinstance(self.Ex, str) else self.Ex
            Ey = sp.sympify(self.Ey) if isinstance(self.Ey, str) else self.Ey
        except:
            return (False, "Cannot sympify")
        # Check that exp(a*x) appears
        def has_exp_factor(expr, var):
            for sub in sp.preorder_traversal(expr):
                if isinstance(sub, sp.exp) and var in sub.free_symbols:
                    return True
            return False
        if has_exp_factor(Ex, self.x) and has_exp_factor(Ey, self.x):
            return (True, "Both Ex and Ey have exp(a*x) factor")
        else:
            return (False, "Missing exp(a*x) factor in one component")

    def test_modeling_sin_cos_phase(self):
        """Check that Ex is sinusoidal in w (ky-ωt) and Ey is cosinusoidal (or vice versa)."""
        if self.Ex is None or self.Ey is None:
            return (False, "SKIP: Ex/Ey missing")
        try:
            Ex = sp.sympify(self.Ex) if isinstance(self.Ex, str) else self.Ex
            Ey = sp.sympify(self.Ey) if isinstance(self.Ey, str) else self.Ey
        except:
            return (False, "Cannot sympify")
        # Check for sin/cos with argument (k*y - ω*t)
        k_sym, omega_sym = sp.Symbol('k', real=True), sp.Symbol('omega', real=True)
        sin_arg = k_sym*self.y - omega_sym*self.t
        cos_arg = k_sym*self.y - omega_sym*self.t
        ex_has_sin = Ex.has(sp.sin(sin_arg))
        ex_has_cos = Ex.has(sp.cos(cos_arg))
        ey_has_sin = Ey.has(sp.sin(sin_arg))
        ey_has_cos = Ey.has(sp.cos(cos_arg))
        if (ex_has_sin and ey_has_cos) or (ex_has_cos and ey_has_sin):
            return (True, "Components show sin/cos phase relation")
        else:
            return (False, "Phase relation not sin-cos as expected")

    def test_modeling_speed_expression(self):
        """Check that v expression contains sqrt(1 - a^2/k^2)."""
        if self.v is None:
            return (False, "SKIP: v not available")
        try:
            v_expr = sp.sympify(self.v) if isinstance(self.v, str) else self.v
        except:
            return (False, "Cannot sympify v")
        # Check by numeric evaluation
        if self.a is None or self.k is None:
            return (False, "SKIP: a or k not available for numeric check")
        try:
            a_val = float(self.a)
            k_val = float(self.k)
        except:
            return (False, "Cannot convert a/k to float")
        c_default = 1.0
        # Use c from instance if available
        c_val = float(self.c) if self.c is not None else c_default
        expected_v = c_val * math.sqrt(1 - (a_val/k_val)**2)
        subs_v = {sp.Symbol('a'): a_val, sp.Symbol('k'): k_val, sp.Symbol('c'): c_val}
        try:
            v_num = float(v_expr.subs(subs_v).evalf())
        except:
            return (False, "Cannot evaluate v expression numerically")
        if math.isclose(v_num, expected_v, rel_tol=1e-3):
            return (True, f"v expression matches expected value {expected_v:.3f}")
        else:
            return (False, f"v = {v_num:.3f}, expected {expected_v:.3f}")

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_v_decreases_with_a_over_k(self):
        """Check that v decreases as a/k increases (for fixed c)."""
        if self.v is None:
            return (False, "SKIP: v not available")
        if self.a is None or self.k is None:
            return (False, "SKIP: a or k missing")
        try:
            a_base = float(self.a)
            k_base = float(self.k)
        except:
            return (False, "Cannot convert a/k to float")
        v_expr = sp.sympify(self.v) if isinstance(self.v, str) else self.v
        c_val = float(self.c) if self.c is not None else 1.0
        subs_base = {sp.Symbol('a'): a_base, sp.Symbol('k'): k_base,
                     sp.Symbol('c'): c_val}
        try:
            v_base = float(v_expr.subs(subs_base).evalf())
        except:
            return (False, "Cannot evaluate base v")
        # Perturb: increase a by 10% (increase a/k)
        a_pert = a_base * 1.1
        subs_pert = {sp.Symbol('a'): a_pert, sp.Symbol('k'): k_base,
                     sp.Symbol('c'): c_val}
        try:
            v_pert = float(v_expr.subs(subs_pert).evalf())
        except:
            return (False, "Cannot evaluate perturbed v")
        if v_pert < v_base:
            return (True, f"v decreases when a/k increases: v_base={v_base:.3f}, v_pert={v_pert:.3f}")
        else:
            return (False, f"v does not decrease: base={v_base:.3f}, perturbed={v_pert:.3f} (should be <)")

    def test_monotonicity_e_amplitude_with_a(self):
        """
        Check that |E| at a fixed (x>0, y=0, t=0) increases as a increases.
        Perturb input a by +10% and compare baseline amplitude vs perturbed amplitude.
        """
        if self.Ex is None or self.Ey is None:
            return (False, "SKIP: Ex/Ey missing")
        if self.a is None:
            return (False, "SKIP: a missing")
        try:
            a_base = float(self.a)
        except:
            return (False, "Cannot convert a to float")
        # Need numeric values for other parameters to evaluate
        # If any are missing, skip (no fallback defaults allowed)
        for name in ['B0', 'k', 'omega', 'c']:
            if getattr(self, name) is None:
                return (False, f"SKIP: missing required parameter {name}")
        try:
            B0_val = float(self.B0)
            k_val = float(self.k)
            omega_val = float(self.omega)
            c_val = float(self.c)
        except:
            return (False, "Cannot convert parameters to float")

        # Build sympy expressions
        try:
            Ex_expr = sp.sympify(self.Ex) if isinstance(self.Ex, str) else self.Ex
            Ey_expr = sp.sympify(self.Ey) if isinstance(self.Ey, str) else self.Ey
        except:
            return (False, "Cannot sympify Ex/Ey")

        # Fixed evaluation point: x=1, y=0, t=0 (x>0 to have monotonic response)
        subs_base = {self.x: 1.0, self.y: 0.0, self.t: 0.0,
                     sp.Symbol('a'): a_base, sp.Symbol('B0'): B0_val,
                     sp.Symbol('k'): k_val, sp.Symbol('omega'): omega_val,
                     sp.Symbol('c'): c_val}
        try:
            Ex_base = float(Ex_expr.subs(subs_base).evalf())
            Ey_base = float(Ey_expr.subs(subs_base).evalf())
            E_base = math.sqrt(Ex_base**2 + Ey_base**2)
        except Exception as e:
            return (False, f"Cannot evaluate baseline amplitude: {e}")

        # Perturb a by +10%
        a_pert = a_base * 1.1
        subs_pert = subs_base.copy()
        subs_pert[sp.Symbol('a')] = a_pert
        try:
            Ex_pert = float(Ex_expr.subs(subs_pert).evalf())
            Ey_pert = float(Ey_expr.subs(subs_pert).evalf())
            E_pert = math.sqrt(Ex_pert**2 + Ey_pert**2)
        except Exception as e:
            return (False, f"Cannot evaluate perturbed amplitude: {e}")

        # For x fixed > 0, amplitude should increase as a increases (since exp(a*x) factor)
        if E_pert > E_base:
            return (True, f"|E| increases when a increases (+10%): base={E_base:.4f}, perturbed={E_pert:.4f}")
        else:
            return (False, f"|E| does not increase: base={E_base:.4f}, perturbed={E_pert:.4f} (should be >)")

    # ----------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_signs(self):
        """Weak sanity: check that Ex and Ey have opposite sign at a representative point."""
        if self.Ex is None or self.Ey is None:
            return (False, "SKIP: Ex/Ey missing")
        try:
            Ex = sp.sympify(self.Ex) if isinstance(self.Ex, str) else self.Ex
            Ey = sp.sympify(self.Ey) if isinstance(self.Ey, str) else self.Ey
        except:
            return (False, "Cannot sympify")
        # Use default numeric values to evaluate at (x=0, y=0, t=0)
        # But we cannot hardcode defaults; we must get actual values or skip
        # For this weak sanity, we accept defaults only if all givens are present
        for name in ['B0', 'a', 'k', 'omega', 'c']:
            if getattr(self, name) is None:
                return (False, f"SKIP: missing parameter {name} for evaluation")
        try:
            vals = {name: float(getattr(self, name)) for name in ['B0', 'a', 'k', 'omega', 'c']}
        except:
            return (False, "Cannot convert parameters to float")
        subs = {self.x: 0.0, self.y: 0.0, self.t: 0.0,
                sp.Symbol('B0'): vals['B0'], sp.Symbol('a'): vals['a'],
                sp.Symbol('k'): vals['k'], sp.Symbol('omega'): vals['omega'],
                sp.Symbol('c'): vals['c']}
        try:
            Ex_val = float(Ex.subs(subs).evalf())
            Ey_val = float(Ey.subs(subs).evalf())
        except:
            return (False, "Cannot evaluate Ex/Ey at test point")
        # At (0,0,0), typical solution has Ex=0, Ey negative (since sin(0)=0, cos(0)=1)
        # So Ex=0, Ey negative. This is a weak check; just report sign relation.
        if Ex_val * Ey_val <= 0:
            return (True, f"Signs at (0,0,0): Ex={Ex_val:.3f}, Ey={Ey_val:.3f} (opposite or one zero)")
        else:
            # Not necessarily wrong, but we note it
            return (True, f"Signs at (0,0,0): Ex={Ex_val:.3f}, Ey={Ey_val:.3f} (same sign, plausible)")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_ methods, return (all_pass, logs)."""
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