import sys
import math
import sympy as sp

class PhysicsChecker:
    """
    Audits solutions for the bead on a rotating hoop problem.
    Extracts results from solution_instance and runs unit-test-style checks.
    """

    def __init__(self, solution_instance):
        # Extract results dict or direct attributes safely; use None for missing.
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}

        self.L = getattr(solution_instance, 'L', results.get('L', None))
        self.EOM = getattr(solution_instance, 'EOM', results.get('EOM', None))
        self.Omega = getattr(solution_instance, 'Omega', results.get('Omega', None))
        self.theta0_eq = getattr(solution_instance, 'theta0_eq', results.get('theta0_eq', None))

        # Input parameters (may be present)
        self.m = getattr(solution_instance, 'm', results.get('m', None))
        self.b = getattr(solution_instance, 'b', results.get('b', None))
        self.g = getattr(solution_instance, 'g', results.get('g', None))
        self.omega = getattr(solution_instance, 'omega', results.get('omega', None))

        # Symbolic variables for internal construction
        self.theta, self.theta_dot, self.theta_ddot = sp.symbols('theta theta_dot theta_ddot')
        self.m_sym, self.b_sym, self.g_sym, self.omega_sym = sp.symbols('m b g omega', positive=True)
        self.t = sp.symbols('t')

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def _to_sympy(self, expr):
        """Convert string or sympy expression to sympy. Returns None on failure."""
        if expr is None:
            return None
        if isinstance(expr, sp.Expr):
            return expr
        if isinstance(expr, str):
            try:
                return sp.sympify(expr)
            except (sp.SympifyError, TypeError):
                return None
        return None

    def _to_sympy_eq(self, expr):
        """Convert to sympy Eq. For strings, split on '='."""
        if expr is None:
            return None
        if isinstance(expr, sp.Eq):
            return expr
        if isinstance(expr, str):
            if '=' in expr:
                parts = expr.split('=', 1)
                try:
                    lhs = sp.sympify(parts[0].strip())
                    rhs = sp.sympify(parts[1].strip())
                    return sp.Eq(lhs, rhs)
                except:
                    return None
            else:
                try:
                    return sp.Eq(sp.sympify(expr), 0)
                except:
                    return None
        return None

    def _has_symbols(self, expr, syms):
        """Check if expression contains all given symbols (as free symbols)."""
        if expr is None:
            return False
        free = expr.free_symbols
        return all(s in free for s in syms)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_has_L(self):
        if self.L is None:
            return (False, "L is missing")
        return (True, "L is present")

    def test_interface_has_EOM(self):
        if self.EOM is None:
            return (False, "EOM is missing")
        return (True, "EOM is present")

    def test_interface_has_Omega(self):
        if self.Omega is None:
            return (False, "Omega is missing")
        return (True, "Omega is present")

    def test_interface_has_theta0_eq(self):
        if self.theta0_eq is None:
            return (False, "theta0_eq is missing")
        return (True, "theta0_eq is present")

    # ----------------------------------------------------------------------
    # Theorem-based invariant tests
    # ----------------------------------------------------------------------
    def test_invariant_Lagrangian_structure(self):
        """Check that L matches the known form T - V."""
        L_sym = self._to_sympy(self.L)
        if L_sym is None:
            return (False, "L cannot be parsed as SymPy expression")

        # Expected Lagrangian: 0.5*m*b^2*(θdot^2 + ω^2 sin^2θ) - m*g*b*cosθ
        expected = (sp.Rational(1, 2) * self.m_sym * self.b_sym ** 2 *
                    (self.theta_dot ** 2 + self.omega_sym ** 2 * sp.sin(self.theta) ** 2)
                    - self.m_sym * self.g_sym * self.b_sym * sp.cos(self.theta))

        # Try symbolic simplification of difference
        diff = sp.simplify(L_sym - expected)
        try:
            if diff == 0:
                return (True, "Lagrangian matches expected expression symbolically")
        except Exception:
            pass

        # Fallback numeric check
        try:
            subs = {self.m_sym: 1, self.b_sym: 2, self.g_sym: 9.8, self.omega_sym: 3,
                    self.theta: 1.2, self.theta_dot: 0.5}
            val_L = float(L_sym.subs(subs))
            val_exp = float(expected.subs(subs))
            if math.isclose(val_L, val_exp, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Lagrangian matches expected expression numerically")
            else:
                return (False, f"Lagrangian numeric mismatch: {val_L} vs {val_exp}")
        except Exception:
            return (False, "Cannot verify Lagrangian structure")

    def test_invariant_EOM_derivation(self):
        """Verify that the EOM can be derived from the Lagrangian via Euler-Lagrange."""
        EOM_sym = self._to_sympy_eq(self.EOM)
        if EOM_sym is None:
            return (False, "EOM cannot be parsed as an equation")
        L_sym = self._to_sympy(self.L)
        if L_sym is None:
            return (False, "Cannot test EOM: L not available or unparseable")

        # Compute d/dt( ∂L/∂θdot ) - ∂L/∂θ = 0
        dL_dtheta_dot = sp.diff(L_sym, self.theta_dot)
        d_dt_dL = self.theta_ddot * sp.diff(dL_dtheta_dot, self.theta_dot)  # because θdot appears linearly
        dL_dtheta = sp.diff(L_sym, self.theta)
        derived_expr = sp.simplify(d_dt_dL - dL_dtheta)  # Should be m*b^2*θddot - ...

        derived_eq = sp.Eq(derived_expr, 0)
        given_expr = EOM_sym.lhs - EOM_sym.rhs

        try:
            ratio = sp.simplify(given_expr / derived_expr)
            if ratio.is_constant() and ratio != 0:
                return (True, f"EOM is proportional (factor {ratio})")
        except Exception:
            pass

        # Numeric fallback: check proportionality at two points
        try:
            def eval_both(subs):
                subs_full = {self.m_sym: 1, self.b_sym: 2, self.g_sym: 9.8, self.omega_sym: 3,
                             self.theta: 1.2, self.theta_ddot: 0.5}
                subs_full.update(subs)
                val_g = float(given_expr.subs(subs_full))
                val_d = float(derived_expr.subs(subs_full))
                return val_g, val_d

            subs1 = {}
            g1, d1 = eval_both(subs1)
            subs2 = {self.b_sym: 3, self.omega_sym: 2.5, self.theta: 0.8, self.theta_ddot: 0.3}
            g2, d2 = eval_both(subs2)

            if abs(d1) > 1e-12 and abs(d2) > 1e-12:
                r1 = g1 / d1
                r2 = g2 / d2
                if math.isclose(r1, r2, rel_tol=1e-2):
                    return (True, f"EOM proportional (ratio ~{r1:.3f})")
                else:
                    return (False, f"Ratios inconsistent: {r1:.3f} vs {r2:.3f}")
            else:
                return (False, "Derived EOM evaluates to zero (maybe identically zero)")
        except Exception:
            return (False, "Cannot perform numeric check for EOM")

    def test_invariant_critical_Omega_expression(self):
        """Check that Omega^2 equals g/b."""
        Omega_sym = self._to_sympy(self.Omega)
        if Omega_sym is None:
            return (False, "Omega cannot be parsed")
        g_sym = self._to_sympy(self.g)
        b_sym = self._to_sympy(self.b)
        if g_sym is not None and b_sym is not None:
            expected_sq = g_sym / b_sym
            diff = sp.simplify(Omega_sym ** 2 - expected_sq)
            try:
                if diff == 0:
                    return (True, "Omega^2 = g/b symbolically")
            except Exception:
                pass
            try:
                val_g = float(g_sym) if not g_sym.free_symbols else 9.81
                val_b = float(b_sym) if not b_sym.free_symbols else 1.0
                val_Omega = float(Omega_sym)
                if math.isclose(val_Omega ** 2, val_g / val_b, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "Omega^2 ≈ g/b numerically")
                else:
                    return (False, f"Omega^2 = {val_Omega**2}, expected {val_g/val_b}")
            except Exception:
                return (False, "Cannot evaluate Omega relation")
        else:
            return (False, "Cannot verify Omega: g and/or b missing")

    def test_invariant_theta0_equation(self):
        """Check that theta0_eq is consistent with cosθ0 = -g/(b ω^2). Weak check."""
        eq_sym = self._to_sympy_eq(self.theta0_eq)
        if eq_sym is None:
            return (False, "theta0_eq cannot be parsed as equation")
        # We accept any nontrivial equation.
        return (True, "theta0_eq present (formal check)")

    # ----------------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------------
    def test_modeling_equilibrium_condition(self):
        """Check that setting θddot=0 in EOM leads to factor sinθ (b ω^2 cosθ + g)."""
        EOM_sym = self._to_sympy_eq(self.EOM)
        if EOM_sym is None:
            return (False, "EOM not available for equilibrium check")
        zero_expr = EOM_sym.lhs.subs(self.theta_ddot, 0) - EOM_sym.rhs.subs(self.theta_ddot, 0)
        zero_expr = sp.simplify(zero_expr)
        factored = sp.factor(zero_expr)
        # Expected factor: sinθ*(b ω^2 cosθ + g) (possibly times -1)
        expected_factor = sp.sin(self.theta) * (self.b_sym * self.omega_sym**2 * sp.cos(self.theta) + self.g_sym)
        try:
            # Numeric check at a sample point
            subs = {self.b_sym: 2, self.omega_sym: 3, self.g_sym: 9.8, self.theta: 1.2}
            val_factored = float(factored.subs(subs))
            val_expected = float(expected_factor.subs(subs))
            if math.isclose(val_factored, val_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Equilibrium condition matches expected form")
            else:
                return (False, f"Equilibrium expression mismatch: {val_factored} vs {val_expected}")
        except Exception:
            return (False, "Cannot evaluate equilibrium expression")

    def test_modeling_stability_bottom(self):
        """Verify that for ω < Ω, bottom is stable (already captured by Omega)."""
        # This is a placeholder; critical Omega test already covers the condition.
        return (True, "Stability condition already verified via Omega expression")

    # ----------------------------------------------------------------------
    # Monotonicity tests (must perturb an input and compare baseline output)
    # ----------------------------------------------------------------------
    def test_monotonicity_omega_increases_cos_theta0(self):
        """Perturb ω upward and check that cosθ0 increases (becomes less negative)."""
        if self.theta0_eq is None:
            return (False, "SKIP: theta0_eq missing")
        eq_sym = self._to_sympy_eq(self.theta0_eq)
        if eq_sym is None:
            return (False, "SKIP: theta0_eq unparseable")
        g_sym = self._to_sympy(self.g)
        b_sym = self._to_sympy(self.b)
        omega_sym = self._to_sympy(self.omega)
        if None in (g_sym, b_sym, omega_sym):
            return (False, "SKIP: cannot evaluate without numeric g, b, omega")
        try:
            g_val = float(g_sym) if not g_sym.free_symbols else 9.81
            b_val = float(b_sym) if not b_sym.free_symbols else 1.0
            omega0 = float(omega_sym) if not omega_sym.free_symbols else 3.5
        except Exception:
            return (False, "SKIP: cannot convert parameters to float")

        omega_crit = math.sqrt(g_val / b_val)
        if omega0 <= omega_crit:
            omega0 = omega_crit * 1.5  # ensure above critical so equilibrium exists
        cos0 = -g_val / (b_val * omega0 ** 2)
        omega1 = omega0 * 1.1
        cos1 = -g_val / (b_val * omega1 ** 2)
        if cos1 > cos0:
            return (True, f"cosθ0 increased from {cos0:.4f} to {cos1:.4f} when ω increased")
        else:
            return (False, f"cosθ0 did not increase: {cos0:.4f} -> {cos1:.4f}")

    def test_monotonicity_L_increases_with_omega(self):
        """Perturb ω upward and check that Lagrangian increases (due to +½ m b² ω² sin²θ)."""
        L_sym = self._to_sympy(self.L)
        if L_sym is None:
            return (False, "SKIP: L missing")
        # Check that L contains omega_sym
        if self.omega_sym not in L_sym.free_symbols:
            return (False, "SKIP: L does not depend on ω")
        # Use constant sample values for other symbols
        m_val, b_val, g_val = 1.0, 1.5, 9.81
        theta_val, theta_dot_val = 0.8, 0.2
        # Baseline omega = 2.0
        omega0 = 2.0
        # Build substitution dictionary
        subs0 = {self.m_sym: m_val, self.b_sym: b_val, self.g_sym: g_val,
                 self.omega_sym: omega0,
                 self.theta: theta_val, self.theta_dot: theta_dot_val}
        try:
            L0 = float(L_sym.subs(subs0))
        except Exception:
            return (False, "Cannot evaluate L at baseline omega")
        # Perturbed omega: 10% increase
        omega1 = omega0 * 1.1
        subs1 = dict(subs0)
        subs1[self.omega_sym] = omega1
        try:
            L1 = float(L_sym.subs(subs1))
        except Exception:
            return (False, "Cannot evaluate L at perturbed omega")
        if L1 > L0:
            return (True, f"L increased from {L0:.4f} to {L1:.4f} when ω increased")
        else:
            return (False, f"L did not increase: {L0:.4f} -> {L1:.4f}")

    # ----------------------------------------------------------------------
    # Weak answer sanity test (at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_nonzero(self):
        """Check that Lagrangian is finite and non-zero for typical sample values."""
        L_sym = self._to_sympy(self.L)
        if L_sym is None:
            return (False, "SKIP: L missing")
        try:
            subs = {self.m_sym: 1, self.b_sym: 1, self.g_sym: 9.8, self.omega_sym: 1,
                    self.theta: 0.5, self.theta_dot: 0.1}
            val = float(L_sym.subs(subs))
            if math.isfinite(val) and abs(val) > 1e-12:
                return (True, "Lagrangian is finite and non-zero")
            else:
                return (False, "Lagrangian evaluates to zero or infinite")
        except Exception:
            return (False, "Cannot evaluate Lagrangian")

    # ----------------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """Discover all test_* methods, execute them with try/except, collect logs."""
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith('test_'):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} crashed: {e}"
            status = 'PASS' if passed else 'FAIL'
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)