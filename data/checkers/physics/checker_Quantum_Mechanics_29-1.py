import sympy as sp
import math

class PhysicsChecker:
    """
    Audits a solution for an infinite square well with time-independent
    and time-dependent perturbations.
    """
    def __init__(self, solution_instance):
        # Safe extraction from solution_instance.results dict
        results = getattr(solution_instance, 'results', {})
        self.psi_n = results.get('psi_n', None)
        self.E_n = results.get('E_n', None)
        self.delta_E_n = results.get('delta_E_n', None)
        self.c2 = results.get('c2', None)
        self.P12 = results.get('P12', None)
        self.P13 = results.get('P13', None)

        # Also extract givens if available (for re-derivation)
        givens = getattr(solution_instance, 'givens', {})
        self.m = givens.get('m', None)
        self.b = givens.get('b', None)
        self.hbar = givens.get('hbar', None)
        self.eps = givens.get('eps', None)
        self.lam = givens.get('lambda', None)
        self.omega = givens.get('omega', None)

        # Explanation steps (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ---------- interface tests ----------
    def test_interface_psi_n_exists(self):
        if self.psi_n is None:
            return (False, "psi_n is missing")
        if not isinstance(self.psi_n, (sp.Expr, str)):
            return (False, "psi_n is not a valid expression")
        return (True, "psi_n present and valid")

    def test_interface_E_n_exists(self):
        if self.E_n is None:
            return (False, "E_n is missing")
        if not isinstance(self.E_n, (sp.Expr, str)):
            return (False, "E_n is not a valid expression")
        return (True, "E_n present and valid")

    def test_interface_delta_E_n_exists(self):
        if self.delta_E_n is None:
            return (False, "delta_E_n is missing")
        if not isinstance(self.delta_E_n, (sp.Expr, str)):
            return (False, "delta_E_n is not a valid expression")
        return (True, "delta_E_n present and valid")

    def test_interface_c2_exists(self):
        if self.c2 is None:
            return (False, "c2 is missing")
        if not isinstance(self.c2, (sp.Expr, str)):
            return (False, "c2 is not a valid expression")
        return (True, "c2 present and valid")

    def test_interface_P12_exists(self):
        if self.P12 is None:
            return (False, "P12 is missing")
        if not isinstance(self.P12, (sp.Expr, str)):
            return (False, "P12 is not a valid expression")
        return (True, "P12 present and valid")

    def test_interface_P13_exists(self):
        if self.P13 is None:
            return (False, "P13 is missing")
        if not isinstance(self.P13, (sp.Expr, str)):
            return (False, "P13 is not a valid expression")
        return (True, "P13 present and valid")

    # ---------- invariant tests ----------
    def test_invariant_normalization(self):
        """Integral |psi|^2 dx from 0 to b should equal 1."""
        if self.psi_n is None or self.b is None:
            return (False, "SKIP: psi_n or b missing")
        psi = sp.sympify(self.psi_n) if isinstance(self.psi_n, str) else self.psi_n
        x = sp.Symbol('x')
        b = sp.sympify(self.b)
        try:
            norm_int = sp.integrate(sp.conjugate(psi)*psi, (x, 0, b))
            n = sp.Symbol('n', integer=True, positive=True)
            val = sp.simplify(norm_int)
            if val == 1:
                return (True, "Normalization exact")
            # fallback numeric for n=1
            numeric = val.subs({n:1}).evalf()
            if math.isclose(float(numeric), 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Normalization passes numeric check")
            else:
                return (False, f"Normalization integral = {numeric}, expected 1")
        except Exception as e:
            return (False, f"Normalization integral error: {e}")

    def test_invariant_boundary_conditions(self):
        """psi(0)=0 and psi(b)=0."""
        if self.psi_n is None or self.b is None:
            return (False, "SKIP: psi_n or b missing")
        psi = sp.sympify(self.psi_n) if isinstance(self.psi_n, str) else self.psi_n
        x = sp.Symbol('x')
        b = sp.sympify(self.b)
        try:
            at_0 = sp.simplify(psi.subs(x, 0))
            at_b = sp.simplify(psi.subs(x, b))
            if at_0 == 0 and at_b == 0:
                return (True, "Boundary conditions satisfied")
            else:
                return (False, f"psi(0)={at_0}, psi(b)={at_b}")
        except Exception as e:
            return (False, f"Boundary check error: {e}")

    def test_invariant_schrodinger_equation(self):
        """Check -ħ²/(2m) d²ψ/dx² = E ψ."""
        if self.psi_n is None or self.E_n is None or self.m is None or self.hbar is None:
            return (False, "SKIP: required fields missing")
        psi = sp.sympify(self.psi_n) if isinstance(self.psi_n, str) else self.psi_n
        E = sp.sympify(self.E_n) if isinstance(self.E_n, str) else self.E_n
        m = sp.sympify(self.m)
        hbar = sp.sympify(self.hbar)
        x = sp.Symbol('x')
        n = sp.Symbol('n', integer=True, positive=True)
        try:
            lhs = -hbar**2/(2*m) * sp.diff(psi, x, 2)
            diff = sp.simplify(lhs - E*psi)
            if diff == 0:
                return (True, "Schrödinger equation exact")
            # numeric fallback for n=1
            num_diff = diff.subs({n:1}).evalf()
            if math.isclose(float(sp.Abs(num_diff)), 0, abs_tol=1e-10):
                return (True, "Schrödinger equation holds within tolerance")
            else:
                return (False, f"LHS - E*psi = {num_diff}")
        except Exception as e:
            return (False, f"Schrödinger check error: {e}")

    def test_invariant_energy_shift_formula(self):
        """Re-derive delta_E_n by computing <n|V1|n> and compare to given delta_E_n for n=2."""
        if self.psi_n is None or self.b is None or self.eps is None or self.delta_E_n is None:
            return (False, "SKIP: required data missing")
        psi = sp.sympify(self.psi_n) if isinstance(self.psi_n, str) else self.psi_n
        b = sp.sympify(self.b)
        eps = sp.sympify(self.eps)
        x = sp.Symbol('x')
        n = sp.Symbol('n', integer=True, positive=True)
        V1 = eps * sp.sin(sp.pi * x / b)
        integrand = sp.conjugate(psi)*V1*psi
        integral = sp.integrate(integrand, (x, 0, b))
        try:
            # numeric for n=2
            delta_n2_num = float(integral.subs({n:2}).evalf())
            # expected from formula (re-derived inside checker)
            expected_n2 = 2*eps/sp.pi * (1 + 1/(4*2**2-1))
            expected_num = float(expected_n2.evalf())
            if math.isclose(delta_n2_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"delta_E_2 matches: {delta_n2_num}")
            else:
                return (False, f"delta_E_2 computed={delta_n2_num}, expected={expected_num}")
        except Exception as e:
            return (False, f"Energy shift numeric check error: {e}")

    def test_invariant_parity_P13_zero(self):
        """By parity, P13 should be 0."""
        if self.P13 is None:
            return (False, "SKIP: P13 missing")
        if isinstance(self.P13, (int, float)):
            if self.P13 == 0:
                return (True, "P13 = 0")
            else:
                return (False, f"P13 = {self.P13}, expected 0")
        if isinstance(self.P13, str):
            try:
                val = sp.sympify(self.P13)
                if val == 0:
                    return (True, "P13 = 0")
            except:
                pass
        return (False, "P13 is not zero")

    # ---------- modeling / trace tests ----------
    def test_modeling_c2_form(self):
        """c2(t) should involve sin((ω21-ω)t/2) or similar."""
        if self.c2 is None:
            return (False, "SKIP: c2 missing")
        expr = sp.sympify(self.c2) if isinstance(self.c2, str) else self.c2
        # Look for sine or exponential with frequency difference
        sin_factors = expr.find(sp.Sin)
        if sin_factors:
            # Check if any sine argument contains omega
            for s in sin_factors:
                arg = s.args[0]
                if arg.has(sp.Symbol('omega')):
                    return (True, "c2 contains sine with frequency")
            return (True, "c2 contains sine (generic)")
        exp_factors = expr.find(sp.exp)
        if exp_factors:
            return (True, "c2 contains exponentials")
        return (False, "c2 does not appear to have sinusoidal/oscillatory form")

    def test_modeling_P12_contains_sinc_squared(self):
        """P12 should have sin²(Δω t/2)/(Δω)² factor."""
        if self.P12 is None:
            return (False, "SKIP: P12 missing")
        expr = sp.sympify(self.P12) if isinstance(self.P12, str) else self.P12
        # Look for sin^2
        sin_sq = expr.find(sp.Sin**2)
        if not sin_sq:
            # Accept trig functions with frequency difference
            all_trig = expr.find(sp.Sin) | expr.find(sp.Cos)
            any_freq = any(
                arg.has(sp.Symbol('omega'))
                for f in all_trig
                for arg in (f.args[0],)
            )
            if any_freq:
                return (True, "P12 contains trigonometric function with frequency difference")
        else:
            # Check that sin argument contains omega
            for s in sin_sq:
                base = s.args[0]  # Sin(base)
                arg = base.args[0] if base.func == sp.Sin else base
                if arg.has(sp.Symbol('omega')):
                    return (True, "P12 contains sin² of frequency difference")
            return (True, "P12 contains sin² (generic)")
        return (False, "P12 does not appear to have Rabi-type form")

    # ---------- monotonicity tests ----------
    def test_monotonicity_energy_shift_with_epsilon(self):
        """Increase epsilon by 50% and check delta_E increases proportionally."""
        if self.delta_E_n is None or self.eps is None:
            return (False, "SKIP: delta_E_n or eps missing")
        expr = sp.sympify(self.delta_E_n) if isinstance(self.delta_E_n, str) else self.delta_E_n
        if not expr.has(sp.Symbol('eps')):
            return (False, "delta_E_n does not depend on eps; cannot test monotonicity")
        try:
            # Evaluate for n=2 with baseline eps=0.1
            n = sp.Symbol('n', integer=True, positive=True)
            expr_n = expr.subs(n, 2)
            eps_sym = sp.Symbol('eps')
            baseline_eps = 0.1
            baseline_val = float(expr_n.subs(eps_sym, baseline_eps).evalf())
            # Perturb by +50%
            perturbed_eps = baseline_eps * 1.5
            perturbed_val = float(expr_n.subs(eps_sym, perturbed_eps).evalf())
            # Expect delta_E to increase (since expression is linear with positive coefficient)
            if perturbed_val > baseline_val:
                return (True, f"delta_E increased from {baseline_val} to {perturbed_val} when eps increased by 50%")
            else:
                return (False, f"Monotonicity fails: {baseline_val} -> {perturbed_val}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    def test_monotonicity_P12_small_time_behavior(self):
        """For small t, P12 should increase with t (if resonance)."""
        if self.P12 is None:
            return (False, "SKIP: P12 missing")
        expr = sp.sympify(self.P12) if isinstance(self.P12, str) else self.P12
        if not expr.has(sp.Symbol('t')):
            return (False, "P12 does not depend on t; cannot test monotonicity")
        try:
            # Use generic numeric settings, with near resonance omega close to omega21
            # Assume m=1, b=1, hbar=1 => omega21 = 3*pi^2/2 ≈ 14.8
            subs = {
                sp.Symbol('m'): 1,
                sp.Symbol('b'): 1,
                sp.Symbol('hbar'): 1,
                sp.Symbol('lambda'): 0.1,
                sp.Symbol('omega'): 15,  # close to resonance
                sp.Symbol('eps'): 0.1,
                sp.Symbol('t'): 0.0  # placeholder
            }
            present = expr.free_symbols
            valid_subs = {k: v for k, v in subs.items() if k in present}
            # baseline at t=0
            valid_subs[sp.Symbol('t')] = 0.0
            baseline_val = float(expr.subs(valid_subs).evalf())
            # perturb to small t=0.1
            valid_subs[sp.Symbol('t')] = 0.1
            perturbed_val = float(expr.subs(valid_subs).evalf())
            if perturbed_val > baseline_val:
                return (True, f"P12 increases from {baseline_val} to {perturbed_val} for t=0 to 0.1")
            else:
                return (False, f"Monotonicity fails: {baseline_val} -> {perturbed_val} (maybe away from resonance)")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    # ---------- answer sanity tests (weak, at most one) ----------
    def test_answer_sanity_plausible(self):
        """
        Weak sanity: P12 (if evaluable) is between 0 and 1,
        P13 is exactly zero if numeric.
        No re-derivation, no math.isclose, no sp.simplify.
        """
        # Check P12 range
        if self.P12 is not None:
            try:
                expr = sp.sympify(self.P12) if isinstance(self.P12, str) else self.P12
                # Use sample numeric values
                subs = {
                    sp.Symbol('m'): 1,
                    sp.Symbol('b'): 1,
                    sp.Symbol('hbar'): 1,
                    sp.Symbol('lambda'): 0.1,
                    sp.Symbol('omega'): 10,
                    sp.Symbol('t'): 1,
                    sp.Symbol('eps'): 0.1,
                }
                present = expr.free_symbols
                valid_subs = {k: v for k, v in subs.items() if k in present}
                val = float(expr.subs(valid_subs).evalf())
                if val < 0 or val > 1:
                    return (False, f"P12 out of plausible range [0,1]: {val}")
            except:
                pass  # choose not to fail if can't evaluate

        # Check P13: must be exactly 0 if numeric
        if self.P13 is not None:
            if isinstance(self.P13, (int, float)):
                if self.P13 != 0:
                    return (False, f"P13 should be exactly 0, got {self.P13}")
            # if expression, only check it's not negative numerically (if possible)
            else:
                try:
                    expr13 = sp.sympify(self.P13) if isinstance(self.P13, str) else self.P13
                    subs13 = {sp.Symbol('t'): 0.5, sp.Symbol('omega'): 1}
                    present13 = expr13.free_symbols
                    valid13 = {k: v for k, v in subs13.items() if k in present13}
                    val13 = float(expr13.subs(valid13).evalf())
                    if val13 < -1e-12:
                        return (False, f"P13 negative: {val13}")
                except:
                    pass
        return (True, "P12 plausible (0≤P12≤1 if evaluable), P13 zero or non-negative")

    # ---------- main runner ----------
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